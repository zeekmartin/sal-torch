#!/usr/bin/env python3
"""exp33 — SparseGPT / Wanda against DropHead / magnitude on ViT-B/16.

ViT-B/16 (``google/vit-base-patch16-224``) on ImageNet-1k
(``evanarlian/imagenet_1k_resized_256``), full 50 000-image validation split,
autocast bfloat16. The pruning cores are vendored in ``exp33_sparse_core.py``
and are equivalence-tested against the reference implementations
(``tests/test_exp33_sparse_equivalence.py``).

The question
------------
DropHead's gains are measured against *structured* baselines — magnitude head
pruning, with and without recovery. The strongest published one-shot pruners
are **unstructured**: SparseGPT and Wanda. At a matched count of removed
parameters, do they beat training-time head masking?

Two arms, as specified:

``attn``    SparseGPT / Wanda on the attention projections only (q, k, v, o),
            at a sparsity matched to what removing 33% / 50% of heads removes.
            Slicing a fraction *f* of heads uniformly per layer removes exactly
            *f* of the q/k/v/o weight parameters, so the matched sparsity on
            those layers **is** *f*. This is the comparison against DropHead.
``whole``   The same total parameter count removed, spread over every linear
            layer in the encoder (attention *and* MLP). Because the MLP holds
            ~3x the attention's parameters, the matched sparsity here is much
            lower — computed, not assumed, and recorded in the output.

Each arm is scored **one-shot** and **after 2 epochs of recovery** (exp29: the
recovery curve peaks at 2).

Two things this experiment must not get wrong
---------------------------------------------
**1. Recovery would otherwise destroy the sparsity.** Unstructured pruning
leaves zeros scattered inside the weight matrices; an ordinary fine-tune fills
them straight back in, and the arm silently stops being sparse while still
being labelled "50% pruned". Every recovery step here is followed by
re-applying the mask captured at prune time, and the final zero fraction is
re-measured and written to the JSON. If it has drifted, the run is wrong.

**2. There is no speedup to report for these arms.** The parameters are masked,
not removed: a dense kernel does exactly as much arithmetic as before. Latency
is measured only for the structured methods, at batch 32 (exp26: at batch 1
every arm on this model is 1.00x). The write-up must say this explicitly
rather than leaving an empty column to be misread.

Seeds
-----
SparseGPT and Wanda are deterministic given a calibration set, so the three
seeds vary **which 128 training images are used for calibration** — that is
the only stochastic input the one-shot arms have. The recovery arms additionally
vary the data order.

Usage::

    # gate: 1 seed, 1 config, tiny data -- exercises every code path
    python exp33_sparse_vs_structured.py --gate --output results/exp33_gate

    # full: one job per seed
    python exp33_sparse_vs_structured.py --seed 42 --data-cache $CACHE \\
        --output results/exp33_s42
"""
from __future__ import annotations

import argparse
import copy
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

from exp23_baselines import magnitude_heads, recover  # noqa: E402
from exp33_sparse_core import (attention_linear_layers, linear_layers,  # noqa: E402
                               prune_model, sparsity_report)

DATASET = "evanarlian/imagenet_1k_resized_256"
METHODS = ("wanda", "sparsegpt")
SCOPES = ("attn", "whole")
RATIOS = (0.33, 0.50)
CALIB_IMAGES = 128
RECOVER_EPOCHS = 2

# Structured references, measured on the identical model, dataset and eval
# path. exp27 (DropHead) and exp29 (magnitude + recovery, re-anchored at 2
# epochs by exp32). Printed beside this run's numbers; never reused as data.
REFERENCE = {
    "drophead_p33": {"accuracy": 0.7769, "source": "EXP27_DROPHEAD.md"},
    "drophead_p50": {"accuracy": 0.7254, "source": "EXP27_DROPHEAD.md"},
    "magnitude_recovery_p33_5ep": {"accuracy": 0.78845, "source": "EXP29"},
    "magnitude_recovery_p50_5ep": {"accuracy": 0.77265, "source": "EXP29"},
    "magnitude_recovery_p50_2ep": {"accuracy": 0.77570, "source": "EXP29"},
    "control_dense_5ep": {"accuracy": 0.80556, "source": "EXP31"},
}


def matched_sparsity(model, scope: str, head_ratio: float) -> tuple[float, dict]:
    """Sparsity that removes as many parameters as pruning ``head_ratio`` heads.

    Removing a fraction *f* of heads uniformly per layer removes *f* of the
    q/k/v/o weight parameters. ``attn`` therefore matches at exactly *f*;
    ``whole`` spreads the same absolute count over every linear layer.
    """
    attn = attention_linear_layers(model)
    attn_params = sum(l.weight.numel() for l in attn.values())
    target_removed = head_ratio * attn_params

    if scope == "attn":
        layers, sparsity = attn, head_ratio
    else:
        layers = {n: l for n, l in linear_layers(model).items()
                  if l.weight.dim() == 2 and l.out_features > 1}
        # The classifier head is tiny and task-critical; pruning it is not what
        # "whole model" is asking about, so it stays out of the target set.
        layers = {n: l for n, l in layers.items() if "classifier" not in n}
        total = sum(l.weight.numel() for l in layers.values())
        sparsity = target_removed / total
    detail = {"scope": scope, "head_ratio": head_ratio,
              "attn_weight_params": attn_params,
              "target_removed_params": int(target_removed),
              "target_layers": len(layers),
              "target_layer_params": sum(l.weight.numel() for l in layers.values()),
              "matched_sparsity": sparsity}
    return sparsity, detail


def capture_masks(layers: dict) -> dict:
    """Which weights are zero right after pruning — the mask recovery must keep."""
    return {n: (l.weight.data != 0) for n, l in layers.items()}


def reapply_masks(layers: dict, masks: dict):
    with torch.no_grad():
        for n, l in layers.items():
            l.weight.data *= masks[n]


def recover_sparse(model, layers, masks, train_split, epochs, args, device,
                   use_amp, P, log):
    """``recover()``, with the sparsity mask re-applied after every step.

    ``recover`` owns its optimizer and schedule, so the mask is enforced
    through a post-accumulate gradient hook on each pruned weight — the
    gradient of a pruned entry is zeroed before the optimizer sees it, and a
    final re-apply catches anything weight decay moved.
    """
    handles = []
    for n, l in layers.items():
        m = masks[n]

        def hook(grad, _m=m):
            return grad * _m
        handles.append(l.weight.register_hook(hook))

    t0 = time.time()
    losses = recover(model, train_split, epochs, args.batch_size, args.seed,
                     device, use_amp, label="sparse-recovery")
    for h in handles:
        h.remove()
    reapply_masks(layers, masks)
    return losses, time.time() - t0


def score(model, val_split, device, use_amp, P, skip_int4=False):
    out = {"accuracy": P.accuracy(model, val_split, device, use_amp)}
    if not skip_int4:
        out["int4_accuracy"] = P.accuracy(
            P.quantize_weights(copy.deepcopy(model), 4), val_split, device,
            use_amp)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--scopes", nargs="+", default=list(SCOPES))
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--calib-images", type=int, default=CALIB_IMAGES)
    p.add_argument("--recover-epochs", type=int, default=RECOVER_EPOCHS,
                   help="default 2 (exp29: the recovery curve peaks there)")
    p.add_argument("--skip-recovery", action="store_true",
                   help="one-shot cells only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp33")
    p.add_argument("--gate", action="store_true",
                   help="tiny end-to-end run: 1 method, 1 scope, 1 ratio, "
                        "2000 train / 1000 val, 1 recovery epoch")
    args = p.parse_args(argv)

    import run_sal_qat_prototype as P
    log, rule = P.log, P.rule

    if args.gate:
        args.n_train, args.n_val = 2000, 1000
        args.methods, args.scopes, args.ratios = ["wanda"], ["attn"], [0.50]
        args.recover_epochs, args.calib_images = 1, 32

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    # detect_device returns (device, vram_gb) — exp12 and the prototype
    # unpack it the same way.
    device, _vram = P.detect_device()
    use_amp = device.type == "cuda"

    rule("exp33 — SparseGPT / Wanda against DropHead / magnitude")
    log(f"  seed {args.seed} | {args.calib_images} calibration images | "
        f"recovery {args.recover_epochs} epoch(s)")
    log("  NOTE: these are UNSTRUCTURED methods. Parameters are masked, not "
        "removed; there is no dense-kernel speedup to report for them.")

    P.set_seed(args.seed)
    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
        train_split, val_split, num_classes = P.build_data(
            sizes, args.seed, args.workers, dataset_id=args.dataset,
            cache_dir=args.data_cache)
    else:
        train_split, val_split, num_classes = P.build_data(
            sizes, args.seed, args.workers, dataset_id=args.dataset)
    log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val, "
        f"{num_classes} classes")

    pretrained = P.load_model(num_classes).to(device).eval()
    dense = score(pretrained, val_split, device, use_amp, P, args.skip_int4)
    log(f"  pretrained: {dense['accuracy'] * 100:.2f}%")

    # Calibration set: the seed picks which images, and that is the only
    # stochastic input the one-shot arms have.
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(train_split[0].shape[0], generator=g)[:args.calib_images]
    calib_x = P.normalize(train_split[0][idx])
    calib = [(calib_x[i:i + 16],) for i in range(0, len(calib_x), 16)]
    log(f"  calibration: {len(calib_x)} images in {len(calib)} batches")

    payload = {
        "experiment": "exp33_sparse_vs_structured",
        "model": "google/vit-base-patch16-224",
        "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "seed": args.seed, "gate": args.gate,
        "n_train": int(train_split[0].shape[0]),
        "n_val": int(val_split[0].shape[0]),
        "num_classes": num_classes, "batch_size": args.batch_size,
        "calib_images": len(calib_x), "recover_epochs": args.recover_epochs,
        "pretrained": dense,
        "reference_structured": REFERENCE,
        "notes": {
            "unstructured": "SparseGPT and Wanda mask weights; they do not "
                            "remove them. No dense-kernel speedup exists for "
                            "these arms -- report accuracy at matched removed "
                            "parameter count only.",
            "matching": "Removing a fraction f of heads uniformly per layer "
                        "removes exactly f of the q/k/v/o weight parameters, "
                        "so scope=attn matches at sparsity f. scope=whole "
                        "spreads the same absolute count over all encoder "
                        "linear layers, giving a lower sparsity.",
            "mask_preserved": "Recovery re-applies the prune-time mask via a "
                              "gradient hook plus a final re-apply, and the "
                              "zero fraction is re-measured afterwards. "
                              "Without this an ordinary fine-tune fills the "
                              "zeros back in.",
            "seeds": "SparseGPT/Wanda are deterministic given a calibration "
                     "set; the seed varies which 128 training images calibrate.",
            "int4": "quantize_weights(model, 4): a simulated per-channel "
                    "round-trip in fp32, not a 4-bit kernel.",
        },
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cells": {},
    }

    def save():
        (outdir / "exp33_sparse.json").write_text(json.dumps(payload, indent=2))

    save()  # write before anything that could fail, per the expF lesson

    for method in args.methods:
        for scope in args.scopes:
            for ratio in args.ratios:
                key = f"{method}_{scope}_p{int(ratio * 100)}"
                rule(f"{method} | scope={scope} | matched to {ratio:.0%} of heads")
                model = copy.deepcopy(pretrained)
                sparsity, detail = matched_sparsity(model, scope, ratio)
                layers = (attention_linear_layers(model) if scope == "attn"
                          else {n: l for n, l in linear_layers(model).items()
                                if "classifier" not in n})
                log(f"    matched sparsity {sparsity:.4f} over "
                    f"{len(layers)} layers "
                    f"({detail['target_removed_params'] / 1e6:.1f}M params)")

                t0 = time.time()
                rep = prune_model(model, layers, calib, method, sparsity,
                                  device=device, log=log)
                prune_seconds = time.time() - t0
                masks = capture_masks(layers)
                one_shot = score(model, val_split, device, use_amp, P,
                                 args.skip_int4)
                log(f"    one-shot: {one_shot['accuracy'] * 100:.2f}%")

                cell = {"matched": detail, "prune_report": rep,
                        "prune_seconds": round(prune_seconds, 1),
                        "one_shot": one_shot}
                payload["cells"][key] = cell
                save()

                if not args.skip_recovery:
                    losses, secs = recover_sparse(
                        model, layers, masks, train_split,
                        args.recover_epochs, args, device, use_amp, P, log)
                    after = sparsity_report(model, layers)
                    rec = score(model, val_split, device, use_amp, P,
                                args.skip_int4)
                    drift = abs(after["target_zero_frac"]
                                - rep["target_zero_frac"])
                    log(f"    recovered ({args.recover_epochs} ep): "
                        f"{rec['accuracy'] * 100:.2f}% | "
                        f"zero frac {after['target_zero_frac']:.4f} "
                        f"(drift {drift:.2e})")
                    cell["recovered"] = rec
                    cell["recovered_losses"] = losses
                    cell["recover_seconds"] = round(secs, 1)
                    cell["sparsity_after_recovery"] = after
                    cell["sparsity_drift"] = drift
                    # A drifting mask means the arm is no longer sparse and the
                    # label on it is false. Record it loudly rather than
                    # quietly reporting a dense model as pruned.
                    cell["mask_preserved"] = bool(drift < 1e-6)
                    if drift >= 1e-6:
                        log(f"    WARNING: sparsity drifted by {drift:.2e} — "
                            "the recovery mask did not hold")
                    save()

    rule("done")
    log(f"  wrote {outdir / 'exp33_sparse.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
