#!/usr/bin/env python
"""1b — I-JEPA ViT-H/14 on ImageNet-1k, split into one job per arm plus an eval job.

Workstream 1b is ~53 GPU-hours. Running it as a single job was the obvious
thing and the wrong one: ``run_jepa_sal.py`` checkpoints the **SAL** arm once
per epoch and nothing else, so a failure in hour 45 — during the evaluation
matrix, which has no checkpointing at all — would lose the control and DropHead
arms entirely. This splits it:

===========  ==========================================================
``--arm X``  train one arm for ``--epochs`` and save ``X.pt`` (~2.5GB,
             weights only, no optimizer state) plus ``X.json`` with its
             losses, masker stats and, for SAL, the head set it trained
             without. ~16 GPU-hours each, independent.
``--eval``   load every ``X.pt`` found and run the whole table: dense,
             ``--prune-seeds`` random draws per ratio, magnitude, sliced,
             SAL's aligned set, the cross-arm fixed-set matrix, INT4 and
             latency. ~7 GPU-hours.
===========  ==========================================================

The evaluation calls ``run_jepa_sal.benchmark`` rather than carrying its own
copy of the loops. That function was lifted out of that script's ``main`` for
exactly this reason: two implementations of the same table drift apart and
nothing fails when they do.

Nothing here reimplements a sal-torch mechanism. Training reuses
``run_jepa_sal``'s ``load_model``, ``make_step``, ``train_control`` and
``train_drophead`` (which itself wraps the prototype's ``ResampleMasker``, which
wraps the real ``HeadMasker``), SAL goes through ``sal.trainer.SALTrainer``, and
the data comes from the shared memmap cache via ``build_data_memmap``.

**The accuracy column is a linear probe, not top-1.** I-JEPA is self-supervised
and has no 1000-class head, so these numbers are not comparable with exp23-27's
ImageNet top-1 and must never share a column with them.

Usage::

    python exp28_jepa_1b.py --arm baseline --epochs 2 --batch-size 64 \
        --data-cache ... --output results/jepa1b
    python exp28_jepa_1b.py --eval --output results/jepa1b \
        --prune-seeds 20 --int4 --aligned --fixed-set-matrix --slice
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

DATASET = "evanarlian/imagenet_1k_resized_256"
ARMS = ("baseline", "sal", "drophead")
# run_jepa_sal.benchmark names its rows from the SAL/control/drophead triple it
# was written around; "baseline" here is that script's "control".
CKPT_FOR = {"baseline": "baseline.pt", "sal": "sal.pt", "drophead": "drophead.pt"}


def train_one(arm, args, J, device, loaders, patch_size, use_amp, outdir):
    """Train a single arm and save its weights and metadata. Returns the info."""
    from sal import SALConfig
    from sal.trainer import SALTrainer

    model = J.load_model(device, grad_ckpt=False)
    t0 = time.time()
    pruned_heads, masker_stats, masks = None, None, {}

    if arm == "baseline":
        J.rule(f"baseline arm ({args.epochs} epochs, no head masking)")
        hist = J.train_control(model, loaders["train"], args.epochs,
                               patch_size, device, use_amp)
        losses = hist["losses"]
    elif arm == "drophead":
        J.rule(f"DropHead arm ({args.epochs} epochs, mask_ratio="
               f"{args.mask_ratio}, resampled every step)")
        hist = J.train_drophead(model, loaders["train"], args.epochs,
                                patch_size, device, use_amp, args.mask_ratio,
                                args.seed)
        losses, masker_stats = hist["losses"], hist["masker_stats"]
    elif arm == "sal":
        J.rule(f"SAL arm ({args.epochs} epochs, mask_ratio={args.mask_ratio})")
        cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio)
        J.log(f"  prune_fraction {cfg.prune_fraction} | "
              f"num_heads_to_prune {cfg.num_heads_to_prune} | "
              f"uniform_per_layer {cfg.uniform_per_layer}")
        opt = torch.optim.AdamW(model.parameters(), lr=J.LR)
        prog = J.Progress(len(loaders["train"]) * args.epochs, "sal training")
        step = J.make_step(patch_size, use_amp, prog)
        trainer = SALTrainer(model, cfg, opt, loaders["train"],
                             seed=args.seed, train_step=step)
        hist = trainer.train(num_epochs=args.epochs)
        prog.done()
        losses, masker_stats = hist["losses"], hist["masker_stats"]
        # The set SAL trained *without*: the aligned rows need it, and without
        # it the eval job can only score SAL on sets it never saw, which on
        # ViT-B/16 understated it by 9.6pp (EXP25_ALIGNED.md).
        pruned_heads = hist.get("pruned_heads")
        J.log(f"  pruned set: {len(pruned_heads) if pruned_heads else 0} heads")
    else:
        raise ValueError(arm)

    if not all(torch.isfinite(torch.tensor(float(x))) for x in losses):
        raise RuntimeError(f"{arm}: non-finite loss in {losses}. A diverged run "
                           "still yields a probe accuracy, and writing that to "
                           "JSON is worse than crashing.")

    info = {
        "arm": arm, "epochs": args.epochs, "losses": losses,
        "masker_stats": masker_stats,
        "pruned_heads": (sorted((int(a), int(b)) for a, b in pruned_heads)
                         if pruned_heads else None),
        "mask_ratio": args.mask_ratio, "seed": args.seed,
        "batch_size": args.batch_size, "n_train": args.n_train,
        "model": args.model, "dataset": args.dataset,
        "train_seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    # Weights only. A full checkpoint with AdamW's two moments is ~7.6GB per
    # arm; the eval job never resumes training, so it does not need them.
    J.save_checkpoint(outdir / CKPT_FOR[arm], model, None, args.epochs,
                      losses, masks, args, with_optimizer=False)
    (outdir / f"{arm}.json").write_text(json.dumps(info, indent=2),
                                        encoding="utf-8")
    J.log(f"  {arm}: losses {[round(x, 6) for x in losses]} "
          f"| {info['train_seconds'] / 3600:.2f} GPU-h")
    J.log(f"  metadata -> {outdir / f'{arm}.json'}")
    return info


def load_arm(path: Path, J, device):
    """Weights from a ``train_one`` checkpoint, into a fresh encoder."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = J.load_model(torch.device("cpu"), grad_ckpt=False)
    missing, unexpected = model.load_state_dict(state["model_state"],
                                                strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path.name}: state dict does not match the model "
                           f"(missing {list(missing)[:4]}, "
                           f"unexpected {list(unexpected)[:4]})")
    return model.to(device).eval(), state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=list(ARMS),
                      help="train this arm and save its checkpoint")
    mode.add_argument("--eval", action="store_true",
                      help="score every arm checkpoint found under --output")

    p.add_argument("--model", default="facebook/ijepa_vith14_1k")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--output", default="results/jepa1b")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64,
                   help="64 measured at 32.3GB of 68.7 on one MI250X GCD; the "
                        "shipped _VRAM_BATCH ladder would pick 16 "
                        "(PROBE_1B_VRAM.md)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--n-probe-train", type=int, default=None,
                   help="probe training images; default is run_jepa_sal's")
    p.add_argument("--n-probe-val", type=int, default=None)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--latency-runs", type=int, default=None)

    # Passed straight through to run_jepa_sal.benchmark, so the names match.
    p.add_argument("--prune-ratios", type=float, nargs="+", default=[0.33, 0.50])
    p.add_argument("--prune-seeds", type=int, default=20)
    p.add_argument("--int4", action="store_true")
    p.add_argument("--aligned", action="store_true")
    p.add_argument("--fixed-set-matrix", action="store_true")
    p.add_argument("--slice", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.epochs, args.n_train, args.n_val = 1, 2000, 1000
        args.prune_seeds, args.prune_ratios = 2, [0.5]
        args.n_probe_train, args.n_probe_val = 2000, 1000
        args.latency_runs = 5

    import run_jepa_sal as J
    import run_sal_qat_prototype as P

    J.MODEL_ID = args.model
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    J.rule("exp28 — 1b, " + (f"train {args.arm}" if args.arm else "evaluate"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    J.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    J.log(f"  model {args.model} | dataset {P.dataset_label(args.dataset)}")
    J.log(f"  seed {args.seed} | epochs {args.epochs} | batch {args.batch_size}")

    if not args.data_cache:
        p.error("--data-cache is required: the 193GB uint8 ImageNet-1k train "
                "split does not fit in a small-g job's memory")

    sizes = {
        "n_train": args.n_train, "n_val": args.n_val,
        "n_probe_train": args.n_probe_train or J.N_PROBE_TRAIN,
        "n_probe_val": args.n_probe_val or J.N_PROBE_VAL,
        "n_fi": J.N_FI_PROBE,
    }
    # The eval job never trains, so it does not need the 1.28M-row train
    # loader; it still needs the probe rows, which come off the same cache.
    if args.eval:
        sizes["n_train"] = max(sizes["n_probe_train"], 1)
    loaders, fi_batches, num_classes = J.build_data_memmap(
        sizes, args.batch_size, args.seed, args.dataset, args.data_cache,
        args.loader_workers, args.loader_block)
    fi_batches = [{k: v.to(device) for k, v in b.items()} for b in fi_batches]
    image_size = next(iter(loaders["cka"]))[0].shape[-1]
    J.log(f"  classes {num_classes} | probe image size {image_size}")

    started = time.time()

    # ---------------------------------------------------------------- train
    if args.arm:
        probe_model = J.load_model(torch.device("cpu"), grad_ckpt=False)
        patch_size = probe_model.config.patch_size
        del probe_model
        info = train_one(args.arm, args, J, device, loaders, patch_size,
                         use_amp, outdir)
        J.log(f"\n  wall clock {(time.time() - started) / 3600:.2f} h")
        return 0

    # ----------------------------------------------------------------- eval
    arms_found = {}
    for arm in ARMS:
        f = outdir / CKPT_FOR[arm]
        if f.exists():
            arms_found[arm] = f
        else:
            J.log(f"  {arm}: {f.name} not found — that arm will be missing")
    if "sal" not in arms_found:
        p.error(f"--eval needs at least {CKPT_FOR['sal']} under {outdir}: "
                "run_jepa_sal.benchmark is organised around the SAL arm")

    models, metas = {}, {}
    for arm, f in arms_found.items():
        J.log(f"  loading {arm} from {f.name} "
              f"({f.stat().st_size / 1e9:.1f}GB)")
        models[arm], _ = load_arm(f, J, device)
        mj = outdir / f"{arm}.json"
        metas[arm] = (json.loads(mj.read_text(encoding="utf-8"))
                      if mj.exists() else {})

    sal_pruned_heads = (metas.get("sal") or {}).get("pruned_heads")
    if args.aligned and not sal_pruned_heads:
        J.log("  --aligned but sal.json carries no pruned_heads; the aligned "
              "rows will be skipped and the matrix will lose that column")
    J.log(f"  SAL's trained-for set: "
          f"{len(sal_pruned_heads) if sal_pruned_heads else 0} heads")

    # The pretrained encoder is the CKA reference every arm is compared against.
    original = J.load_model(device, grad_ckpt=False).eval()
    lat_runs = args.latency_runs or J.LATENCY_RUNS
    J.log(f"  latency: {lat_runs} runs per device per row")

    J.rule("Compression benchmark")
    results, order, slicing, slicing_errors, aligned_rows = J.benchmark(
        original, models["sal"], models.get("baseline"),
        models.get("drophead"), loaders, fi_batches, num_classes, lat_runs,
        device, image_size, args, sal_pruned_heads)

    base = results["original"]
    for name, info in slicing.items():
        r = results[name]
        info["params"] = r["params"]
        info["param_fraction"] = r["params"] / base["params"]
        for dev in ("gpu", "cpu"):
            ms, ref = r.get(f"latency_{dev}_ms"), base.get(f"latency_{dev}_ms")
            info[f"speedup_{dev}"] = (ref / ms) if (ms and ref) else None

    payload = {
        "experiment": "exp28_jepa_1b",
        "variant": "1b",
        "model": args.model, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "accuracy_column": "sal.linear_probe on frozen features — I-JEPA is "
                           "self-supervised and has no 1000-class head, so "
                           "this is NOT ImageNet top-1 and is not comparable "
                           "with exp23-27.",
        "seed": args.seed, "epochs": args.epochs, "mask_ratio": args.mask_ratio,
        "batch_size": args.batch_size, "num_classes": num_classes,
        "n_probe_train": sizes["n_probe_train"],
        "n_probe_val": sizes["n_probe_val"],
        "prune_ratios": args.prune_ratios, "prune_seeds": args.prune_seeds,
        "int4": args.int4, "aligned": args.aligned,
        "fixed_set_matrix": aligned_rows or None,
        "arms_scored": sorted(models),
        "arm_metadata": metas,
        "sal_pruned_heads": sal_pruned_heads,
        "results": results, "order": order,
        "slicing": slicing, "slicing_errors": slicing_errors,
        "latency_runs": lat_runs,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "smoke": args.smoke,
        "caveats": {
            "linear_probe_not_top1": "See accuracy_column. The probe is fitted "
                                     f"on {sizes['n_probe_train']} images for "
                                     f"{num_classes} classes.",
            "int4_is_simulated": "INT4 is the prototype's per-channel "
                                 "quantize/dequantize round-trip in fp32, not a "
                                 "4-bit kernel. bitsandbytes NF4 is CUDA-only "
                                 "and absent on this ROCm image, so real INT4 "
                                 "is not available on this hardware. The column "
                                 "measures the accuracy cost of 4-bit weight "
                                 "rounding and nothing about its speed or "
                                 "memory.",
            "one_seed": "One training seed. It cannot estimate training-seed "
                        "variance, which on ViT-B/16 was SAL's largest term "
                        "(sd up to 17.55pp in exp27).",
            "masked_rows_keep_all_params": "Every row but the -sliced ones "
                                           "masks heads rather than removing "
                                           "them, so the parameter count and "
                                           "the arithmetic are unchanged and "
                                           "the latency columns should show no "
                                           "gain.",
        },
        "wall_clock_s": round(time.time() - started, 1),
    }
    out = outdir / "exp28_jepa_1b.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    J.log(f"\n  metrics -> {out}")

    try:
        J.rule("Summary")
        J.log(J.summary_table(results, order))
        J.log(f"\n  wall clock {(time.time() - started) / 3600:.2f} h")
        J.log("  The accuracy column is a linear probe, not ImageNet top-1.")
    except Exception as exc:                                   # noqa: BLE001
        J.log(f"  (summary table failed: {exc!r}; the JSON above is complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
