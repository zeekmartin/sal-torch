#!/usr/bin/env python
"""Probe the largest batch size I-JEPA ViT-H/14 trains at on one MI250X GCD.

This gates workstream 1b (~160 GPU-hours), so the number has to come from the
*real* training step, not an approximation of it. ``run_jepa_sal.make_step``
runs **two** forward passes per step -- a ``no_grad`` target pass and a
gradient-carrying prediction pass on patch-masked pixels -- then backward,
``clip_grad_norm_`` over 631M parameters, and an AdamW step. A probe that only
does one forward would report a batch size that then OOMs in epoch 1.

Two details that dominate the answer:

* **AdamW's moments appear on the first ``.step()``, not at construction.** The
  two fp32 moment buffers are ~5GB together for ViT-H/14, so peak VRAM after
  step 2 is several GB above peak after step 1. Each batch size therefore runs
  two steps and reports the later peak.
* **autocast casts activations, not parameters.** The weights, gradients and
  moments stay fp32 regardless of bf16 autocast, so ~10GB is committed before a
  single activation is stored.

Each candidate is probed in a freshly built model and optimizer, because an OOM
part-way through a step leaves allocator state that would distort the next
measurement. ``--batch-sizes`` are tried in the order given and the loop stops
at the first one that fits, since the question is "what is the largest".

Usage::

    python probe_jepa_vram.py                      # 64, then 32, then 16
    python probe_jepa_vram.py --grad-ckpt          # same, checkpointing on
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GB = 1e9


def probe(batch: int, image_size: int, patch_size: int, grad_ckpt: bool,
          steps: int, log) -> dict:
    """Run ``steps`` real training steps at ``batch``. Returns a result dict."""
    import run_jepa_sal as J

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out: dict = {"batch_size": batch, "grad_ckpt": grad_ckpt,
                 "steps_attempted": steps}
    model = optimizer = None
    try:
        model = J.load_model(torch.device("cuda"), grad_ckpt)
        optimizer = torch.optim.AdamW(model.parameters(), lr=J.LR)
        out["params"] = sum(p.numel() for p in model.parameters())
        out["weights_gb"] = round(torch.cuda.memory_allocated() / GB, 2)

        per_step = []
        for i in range(steps):
            t0 = time.time()
            pixels = torch.randn(batch, 3, image_size, image_size,
                                 device="cuda")
            visible = J.mask_patches(pixels, patch_size, J.PATCH_MASK_RATIO)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    target = model(pixel_values=pixels).last_hidden_state
                predicted = model(pixel_values=visible).last_hidden_state
                loss = torch.nn.functional.mse_loss(predicted, target.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            per_step.append(round(time.time() - t0, 2))
            out["peak_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
            out["reserved_gb"] = round(torch.cuda.max_memory_reserved() / GB, 2)
            out["steps_done"] = i + 1
            out["loss"] = float(loss.item())
            log(f"    step {i + 1}/{steps}  {per_step[-1]:5.2f}s  "
                f"peak {out['peak_gb']:5.2f} GB  reserved "
                f"{out['reserved_gb']:5.2f} GB  loss {out['loss']:.5f}")
            del pixels, visible, target, predicted, loss
        out["fits"] = True
        out["step_seconds"] = per_step
        # Step 1 pays MIOpen kernel compilation; the steady-state rate is what
        # a 160 GPU-hour estimate should be built from.
        out["steady_step_s"] = per_step[-1]
        out["img_per_s"] = round(batch / per_step[-1], 1)
    except torch.cuda.OutOfMemoryError as exc:
        out["fits"] = False
        out["error"] = "OutOfMemoryError"
        out["error_detail"] = str(exc)[:400]
        out["peak_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
        log(f"    OOM at batch {batch} after {out.get('steps_done', 0)} "
            f"complete step(s); peak {out['peak_gb']:.2f} GB")
    except RuntimeError as exc:
        # ROCm sometimes surfaces an allocation failure as a plain RuntimeError
        # rather than OutOfMemoryError, so treat that text as an OOM and let
        # anything else propagate -- a MIOpen failure must not look like a fit.
        if "out of memory" not in str(exc).lower():
            raise
        out["fits"] = False
        out["error"] = "RuntimeError(out of memory)"
        out["error_detail"] = str(exc)[:400]
        out["peak_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
        log(f"    OOM (RuntimeError) at batch {batch}; "
            f"peak {out['peak_gb']:.2f} GB")
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 32, 16],
                   help="tried in this order; the loop stops at the first fit")
    p.add_argument("--steps", type=int, default=2,
                   help="steps per candidate; 2 is the minimum that includes "
                        "AdamW's moment buffers in the peak")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="probe every size instead of stopping at the first fit")
    p.add_argument("--output", default="results/probe_jepa_vram")
    args = p.parse_args(argv)

    import run_jepa_sal as J

    log = J.log
    J.rule("I-JEPA ViT-H/14 VRAM probe — gates workstream 1b")
    if not torch.cuda.is_available():
        log("  no CUDA device; this probe is meaningless on CPU")
        return 2
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / GB
    log(f"  GPU:   {props.name}")
    log(f"  VRAM:  {vram_gb:.1f} GB")
    log(f"  model: {J.MODEL_ID}")
    log(f"  gradient checkpointing: {'on' if args.grad_ckpt else 'off'}")
    log(f"  candidates: {args.batch_sizes}  ({args.steps} real steps each)")

    # What the shipped heuristic would choose unaided, so the report can say
    # whether the measurement agrees with it.
    heuristic = next(b for floor, b in J._VRAM_BATCH if vram_gb >= floor)
    log(f"  run_jepa_sal._VRAM_BATCH would choose: batch {heuristic}")

    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(J.MODEL_ID)
    size = proc.crop_size["height"] if getattr(proc, "crop_size", None) \
        else proc.size["height"]
    cfg_patch = None
    try:
        from transformers import AutoConfig
        cfg_patch = AutoConfig.from_pretrained(J.MODEL_ID).patch_size
    except Exception as exc:                                   # noqa: BLE001
        log(f"  (could not read patch_size from config: {exc!r})")
    patch = cfg_patch or 14
    log(f"  image {size}x{size}, patch {patch} "
        f"-> {(size // patch) ** 2} patches/image")
    log(f"  patch mask ratio {J.PATCH_MASK_RATIO} (as in the real objective)")

    probes = []
    t0 = time.time()
    for batch in args.batch_sizes:
        J.rule(f"batch {batch}")
        r = probe(batch, size, patch, args.grad_ckpt, args.steps, log)
        probes.append(r)
        if r["fits"]:
            log(f"    FITS at batch {batch}: peak {r['peak_gb']:.2f} GB of "
                f"{vram_gb:.1f} GB ({100 * r['peak_gb'] / vram_gb:.0f}%), "
                f"{r['img_per_s']} img/s")
            if not args.all:
                break

    fitting = [r for r in probes if r["fits"]]
    best = max((r["batch_size"] for r in fitting), default=None)

    payload = {
        "experiment": "probe_jepa_vram",
        "model": J.MODEL_ID,
        "gpu": props.name,
        "vram_gb": round(vram_gb, 1),
        "image_size": size, "patch_size": patch,
        "patch_mask_ratio": J.PATCH_MASK_RATIO,
        "grad_ckpt": args.grad_ckpt,
        "steps_per_candidate": args.steps,
        "heuristic_batch": heuristic,
        "max_batch_that_fits": best,
        "probes": probes,
        "torch_version": torch.__version__,
        "wall_clock_s": round(time.time() - t0, 1),
        "caveats": {
            "random_pixels": "The probe feeds torch.randn rather than real "
                             "images. VRAM does not depend on pixel values, but "
                             "the loss numbers here are meaningless and the "
                             "throughput excludes data loading.",
            "no_sal_masker": "SAL's HeadMasker is not installed. It is a "
                             "forward pre-hook that zeroes slices of an "
                             "existing activation and allocates only the mask "
                             "vectors, so its cost is negligible -- but it is "
                             "not zero and it is not measured here.",
            "peak_is_allocated": "peak_gb is max_memory_allocated; reserved_gb "
                                 "is the allocator's larger reservation, which "
                                 "is what actually has to fit alongside "
                                 "fragmentation.",
            "first_step_compiles": "Step 1 includes MIOpen kernel compilation "
                                   "for the patch-embedding convolution, so "
                                   "only steady_step_s should feed a "
                                   "GPU-hour estimate.",
        },
    }
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "probe_jepa_vram.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    J.rule("Verdict")
    for r in probes:
        state = "fits" if r["fits"] else r.get("error", "failed")
        log(f"  batch {r['batch_size']:3d}  {state:28s} "
            f"peak {r.get('peak_gb', float('nan')):6.2f} GB"
            + (f"  {r['img_per_s']:6.1f} img/s" if r["fits"] else ""))
    log(f"\n  max batch that fits: {best}")
    if best is not None and best != heuristic:
        log(f"  NOTE: _VRAM_BATCH would have chosen {heuristic}. "
            f"Measured answer is {best}.")
    log(f"  metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
