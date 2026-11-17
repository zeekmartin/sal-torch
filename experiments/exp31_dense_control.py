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
# exp29's Block B ratio sweep was measured at FIVE epochs of recovery. Any
# pruning cost derived from it needs a five-epoch dense control.
EXP29_RATIO_EPOCHS = 5


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
    p.add_argument("--save-checkpoint", action="store_true",
                   help="keep the trained control's weights, "
                        "which an image-level comparison "
                        "against it needs")
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
    # The fine-tuning gain: what the epochs bought, with nothing pruned. It is
    # reported separately from any pruning cost, because an arm scored against
    # the pretrained checkpoint reports the SUM of the two and can show a model
    # pruning damaged as improved. See results/lumi_v060/CONTROL_CONVENTION.md.
    row["fine_tuning_gain_pp"] = (row["accuracy"] - dense["accuracy"]) * 100
    row["vs_pretrained_pp"] = row["fine_tuning_gain_pp"]   # same quantity
    if "int4_accuracy" in row and "int4_accuracy" in dense:
        row["fine_tuning_gain_int4_pp"] = (
            row["int4_accuracy"] - dense["int4_accuracy"]) * 100
    row["vs_exp25_baseline_pp"] = (
        row["accuracy"] - EXP25_BASELINE["accuracy"]) * 100
    out["control"] = row
    save()

    # The control's weights, kept only on request. An image-level
    # comparison of an arm against its matched control needs the
    # control's own per-image predictions, and a control that was not
    # saved has to be retrained for two epochs to produce them -- which
    # is how the first seven controls of wave 3 came to be unusable for
    # that question. The JSON is written before this, so a failure here
    # cannot lose the measurement.
    if args.save_checkpoint:
        models = outdir / "models"
        models.mkdir(parents=True, exist_ok=True)
        ckpt = models / "dense_control.pt"
        torch.save({"state_dict": model.state_dict(),
                    "config": model.config.to_dict(),
                    "arm": "dense_control", "seed": args.seed,
                    "epochs": args.epochs,
                    "accuracy": row["accuracy"]}, ckpt)
        out["control"]["checkpoint"] = str(ckpt)
        save()
        P.log(f"    saved {ckpt} "
              f"({ckpt.stat().st_size / 1e6:.0f} MB)")

    P.log(f"    dense + {args.epochs} ep:  {row['accuracy'] * 100:.2f}%"
          + (f" | INT4 {row['int4_accuracy'] * 100:.2f}%"
             if "int4_accuracy" in row else "")
          + f" | fine-tuning gain {row['fine_tuning_gain_pp']:+.2f}pp"
          + f" | {train_seconds / 60:.0f} min")
    P.log(f"    exp25 1a baseline: {EXP25_BASELINE['accuracy'] * 100:.2f}%"
          f" | this run {row['vs_exp25_baseline_pp']:+.3f}pp against it")

    # The restatement this control exists to make possible. Only meaningful on
    # a full run: a gate's 2000-image control is not a reference for anything.
    if not args.gate:
        P.rule("B — exp29's ratio sweep against the matched control")
        ref = row["accuracy"]
        # EXP29_RATIOS are measured at FIVE epochs of recovery. Restating them
        # against a control trained for a different number of epochs compares
        # unlike things: the dense arm alone moves 1.11pp between 2 and 5
        # epochs on this recipe, which is larger than most of the pruning
        # costs being reported. So the restatement runs only when the epoch
        # counts match, and says why when they do not.
        matched = args.epochs == EXP29_RATIO_EPOCHS
        restated = {}
        for k, acc in EXP29_RATIOS.items():
            cell = {
                "exp29_mean": acc,
                "arm_epochs": EXP29_RATIO_EPOCHS,
                "control_epochs": args.epochs,
                "epochs_matched": matched,
                "fine_tuning_gain_pp": row["fine_tuning_gain_pp"],
                "vs_pretrained_pp": (acc - dense["accuracy"]) * 100,
            }
            if matched:
                cell["pruning_cost_pp"] = (acc - ref) * 100
            restated[k] = cell
            P.log(f"    heads {float(k):.0%}: {acc * 100:.2f}%  "
                  f"({EXP29_RATIO_EPOCHS} ep)  "
                  + (f"pruning cost {cell['pruning_cost_pp']:+.2f}pp "
                     f"vs this control" if matched else
                     f"NOT COMPARABLE to this {args.epochs}-epoch control "
                     f"-- use the {EXP29_RATIO_EPOCHS}-epoch control"))
        if not matched:
            P.log(f"    NOTE: exp29's ratio arms ran {EXP29_RATIO_EPOCHS} "
                  f"epochs, this control {args.epochs}. No pruning cost is "
                  f"reported; see results/lumi_v060/CONTROL_CONVENTION.md.")
        out["exp29_restated"] = {
            "control_accuracy": ref,
            "control_epochs": args.epochs,
            "arm_epochs": EXP29_RATIO_EPOCHS,
            "epochs_matched": matched,
            "ratios": restated,
            "note": (
                "exp29 ratio means are three-seed at "
                f"{EXP29_RATIO_EPOCHS} epochs of recovery. A pruning cost is "
                "reported only against a control trained for the SAME number "
                "of epochs; this control ran "
                f"{args.epochs}. Where they match, the column is still "
                "indicative to ~0.05pp, exp25's measured seed spread."),
        }
        save()

    P.log(f"\n  wrote {outdir / 'exp31_dense_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
