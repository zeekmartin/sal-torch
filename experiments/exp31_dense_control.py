#!/usr/bin/env python3
"""exp31 — the dense control: five epochs of fine-tuning, nothing pruned.

exp29's ratio sweep reads **79.41%** at 25% of heads removed, against the
pretrained checkpoint's 78.44%. Read that way it looks as though pruning a
quarter of the heads *improves* the model, which is not a claim anything
supports: a recovered arm has had five epochs of ImageNet-1k and the pretrained
checkpoint has had none. The reference that answers "does pruning cost
accuracy" is the same recipe with nothing pruned.

That control already exists -- exp25's 1a ``baseline`` arm, three seeds,
**80.44% / 79.01% INT4** (``results/lumi_v060/EXP25_ALIGNED.md``), and exp26
uses it for exactly this purpose. This script is a **replication of it through
the exp29 harness**, on seed 42, so that exp29's own numbers rest on a control
measured by the same code path rather than quoted across campaigns.

What it does: load the pretrained checkpoint, score it, run ``recover()`` for
five epochs with no slicing and no masking, score again, FP and simulated INT4,
on the full 50 000-image validation split. One arm, one seed, ~5 GPU-hours.

**Pass condition.** Agreement with 80.44% / 79.01% within the +-0.05pp seed
spread exp25 measured, allowing for this being one seed rather than three.
A result far from 80.44% does not revise exp25 -- it means this harness and
exp25's differ somewhere, and the exp29 ratio sweep's reference becomes
uncertain until that is found.

Usage::

    python exp31_dense_control.py --data-cache $CACHE --output results/exp31_s42
    python exp31_dense_control.py --gate --output results/exp31_gate
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

from exp23_baselines import recover  # noqa: E402

DATASET = "evanarlian/imagenet_1k_resized_256"

# exp25's 1a baseline arm: same images, epochs, batch size and LR schedule,
# nothing pruned. Three-seed mean. Printed beside this run's number as a
# cross-check; nothing here is reused from it.
EXP25_BASELINE = {"accuracy": 0.8044, "int4_accuracy": 0.7901,
                  "per_seed": {"42": 0.8048, "123": 0.8045, "456": 0.8038},
                  "sd_pp": 0.05,
                  "source": "results/lumi_v060/EXP25_ALIGNED.md"}

# exp29's ratio sweep, three-seed means at 5 epochs, for the restatement.
EXP29_RATIOS = {"0.25": 0.79407, "0.33": 0.78845, "0.50": 0.77265}


def score(model, val_split, device, use_amp, P, skip_int4):
    """FP32 and simulated-INT4 accuracy for one checkpoint."""
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
    p.add_argument("--epochs", type=int, default=5,
                   help="matched to the recovery arms it is the control for")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp31")
    p.add_argument("--gate", action="store_true",
                   help="1 epoch, 2000 images -- a small NON-smoke run")
    args = p.parse_args(argv)

    if args.gate:
        # The expF lesson: a smoke flag that also disables INT4 never exercises
        # the cell the table reads. Keep INT4 on and shrink the data instead.
        args.n_train, args.n_val, args.epochs = 2000, 1000, 1

    import run_sal_qat_prototype as P

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp31 — the dense control: five epochs, nothing pruned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | {args.epochs} epoch(s) | nothing pruned"
          + ("  [GATE]" if args.gate else ""))

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    train_split, val_split, num_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    P.log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val, "
          f"{num_classes} classes")

    P.set_seed(args.seed)
    pretrained = P.load_model(num_classes).to(device).eval()
    dense = score(pretrained, val_split, device, use_amp, P, args.skip_int4)
    P.log(f"  pretrained dense {dense['accuracy'] * 100:.2f}%"
          + (f" | INT4 {dense['int4_accuracy'] * 100:.2f}%"
             if "int4_accuracy" in dense else ""))

    out = {"experiment": "exp31_dense_control",
           "model": P.MODEL_ID, "dataset": args.dataset,
           "dataset_label": P.dataset_label(args.dataset),
           "seed": args.seed, "gate": args.gate,
           "num_classes": num_classes,
           "n_train": int(train_split[0].shape[0]),
           "n_val": int(val_split[0].shape[0]),
           "batch_size": args.batch_size,
           "epochs": args.epochs,
           "heads_removed": 0,
           "pretrained": dense,
           "exp25_baseline_reference": EXP25_BASELINE,
           "notes": {
               "arm": "no slicing, no masking — recover() on the pretrained "
                      "weights, identical optimiser and schedule to exp29's "
                      "recovery arms",
               "why": "a recovered arm has had five epochs of ImageNet-1k and "
                      "the pretrained checkpoint has had none; 78.44 answers "
                      "'better than what I downloaded', not 'does pruning cost "
                      "accuracy'",
               "int4": "quantize_weights(model, 4) — a simulated per-channel "
                       "quantize/dequantize round-trip in fp32, not a 4-bit "
                       "kernel; says nothing about speed or memory",
           },
           "gpu": (torch.cuda.get_device_name(0) if device.type == "cuda"
                   else "cpu"),
           "torch_version": torch.__version__}

    def save():
        (outdir / "exp31_dense_control.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")

    save()   # write before anything that presents the numbers (expF lesson)

    P.rule(f"A — {args.epochs} epoch(s) of fine-tuning, nothing pruned")
    P.set_seed(args.seed)
    model = copy.deepcopy(pretrained)
    t0 = time.time()
    losses = recover(model, train_split, args.epochs, args.batch_size,
                     args.seed, device, use_amp, label="dense-control")
    train_seconds = time.time() - t0
    row = score(model, val_split, device, use_amp, P, args.skip_int4)
    row.update(epochs=args.epochs, epoch_losses=losses,
               train_seconds=train_seconds)
    row["vs_pretrained_pp"] = (row["accuracy"] - dense["accuracy"]) * 100
    row["vs_exp25_baseline_pp"] = (
        row["accuracy"] - EXP25_BASELINE["accuracy"]) * 100
    out["control"] = row
    save()

    P.log(f"    dense + {args.epochs} ep:  {row['accuracy'] * 100:.2f}%"
          + (f" | INT4 {row['int4_accuracy'] * 100:.2f}%"
             if "int4_accuracy" in row else "")
          + f" | {row['vs_pretrained_pp']:+.2f}pp over pretrained"
          + f" | {train_seconds / 60:.0f} min")
    P.log(f"    exp25 1a baseline: {EXP25_BASELINE['accuracy'] * 100:.2f}%"
          f" | this run {row['vs_exp25_baseline_pp']:+.3f}pp against it")

    # The restatement this control exists to make possible. Only meaningful on
    # a full run: a gate's 2000-image control is not a reference for anything.
    if not args.gate:
        P.rule("B — exp29's ratio sweep against the matched control")
        ref = row["accuracy"]
        restated = {}
        for k, acc in EXP29_RATIOS.items():
            restated[k] = {
                "exp29_mean": acc,
                "vs_pretrained_pp": (acc - dense["accuracy"]) * 100,
                "vs_this_control_pp": (acc - ref) * 100,
            }
            P.log(f"    heads {float(k):.0%}: {acc * 100:.2f}%  "
                  f"{restated[k]['vs_pretrained_pp']:+.2f}pp vs pretrained  "
                  f"{restated[k]['vs_this_control_pp']:+.2f}pp vs this control")
        out["exp29_restated"] = {
            "control_accuracy": ref,
            "ratios": restated,
            "note": "exp29 ratio means are three-seed at 5 epochs; this "
                    "control is one seed. Treat the restated column as "
                    "indicative to ~0.05pp, exp25's measured seed spread.",
        }
        save()

    P.log(f"\n  wrote {outdir / 'exp31_dense_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
