#!/usr/bin/env python3
"""exp29 — how many epochs of recovery fine-tuning are actually needed?

exp23, exp24 and exp26 all recover for **five** epochs, and five was never
justified: it was picked once and inherited. Every number those experiments
rest on therefore carries an unexamined cost of 1.28M images x 5. The question
a reviewer asks first is whether three would have done, or two, or one.

ViT-B/16 / ImageNet-1k, three seeds, 50% of heads removed by magnitude — the
same starting point as exp23's ``magnitude_recovery`` row and exp24's arm 1.

**Why each curve point is its own run.** ``recover()`` drives
``make_optimizer``, which is a linear warmup followed by cosine decay over the
*total* step count it is given. The 3-epoch mark of a 10-epoch run is therefore
not a 3-epoch recovery: its learning rate is still high and its schedule has
not annealed, so it understates what 3 epochs can do, by an amount nobody has
measured. Reading a curve off one long run would answer a different question
from the one asked, and would not be comparable with exp24's 5-epoch figure
either. So each point in ``--epoch-points`` is a separate recovery with its own
completed schedule, from the same sliced checkpoint.

**Part two: the ratio sweep.** 25 / 33 / 50% of heads at a fixed epoch count.
Note for anyone reaching for exp26 here: exp26's 25% and 33% are **MLP neuron**
ratios, not head ratios — its only head arm is 50%. The head figures to check
against are exp24 arm 1 (``magnitude + recovery``, three seeds): **78.85%** at
P33 and **77.27%** at P50, 77.09 / 75.06 at INT4. Those are printed beside this
run's own numbers as a harness cross-check, not reused.

Usage::

    # gate: one seed, one epoch point, 2000 images
    python exp29_recovery_epochs.py --gate --output results/exp29_gate

    # full: one job per seed, every epoch point sequential inside it
    python exp29_recovery_epochs.py --seed 42 --data-cache $CACHE \\
        --output results/exp29_s42
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

DATASET = "evanarlian/imagenet_1k_resized_256"
EPOCH_POINTS = (1, 2, 3, 5, 10)
RATIOS = (0.25, 0.33, 0.50)

# exp24 arm 1 (magnitude + recovery), three-seed means, 5 epochs. Reference for
# the cross-check only; nothing here is reused from it.
EXP24_ARM1 = {"0.33": {"accuracy": 0.7885, "int4_accuracy": 0.7709},
              "0.50": {"accuracy": 0.7727, "int4_accuracy": 0.7506}}


def slice_at(pretrained, ratio, device, P, log):
    """Magnitude-prune ``ratio`` of heads, uniformly per layer."""
    from sal import slice_heads

    pairs = magnitude_heads(pretrained, ratio, per_layer=True)
    small = slice_heads(copy.deepcopy(pretrained), pairs).eval().to(device)
    log(f"    removed {len(pairs)} heads ({ratio:.0%}, lowest-L2, "
        "uniform per-layer quota)")
    return small, pairs


def score(model, val_split, device, use_amp, P, skip_int4):
    """FP32 and simulated-INT4 accuracy for one checkpoint."""
    out = {"accuracy": P.accuracy(model, val_split, device, use_amp)}
    if not skip_int4:
        out["int4_accuracy"] = P.accuracy(
            P.quantize_weights(copy.deepcopy(model), 4), val_split, device,
            use_amp)
    return out


def recover_for(small, train_split, epochs, args, device, use_amp, P, label):
    """One dedicated recovery with its own completed warmup+cosine schedule."""
    P.set_seed(args.seed)
    model = copy.deepcopy(small)
    t0 = time.time()
    losses = recover(model, train_split, epochs, args.batch_size, args.seed,
                     device, use_amp, label=label)
    return model, losses, time.time() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--epoch-points", type=int, nargs="+",
                   default=list(EPOCH_POINTS),
                   help="dedicated recovery runs, each with its own schedule")
    p.add_argument("--curve-ratio", type=float, default=0.50,
                   help="head ratio for the epoch curve")
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS),
                   help="head ratios for the fixed-epoch sweep")
    p.add_argument("--ratio-epochs", type=int, default=5)
    p.add_argument("--skip-curve", action="store_true")
    p.add_argument("--skip-ratios", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp29")
    p.add_argument("--gate", action="store_true",
                   help="1 seed, 1 epoch point, 2000 images")
    args = p.parse_args(argv)

    if args.gate:
        # A small NON-smoke run, which is the lesson from expF: a smoke flag
        # that also disables INT4 never exercises the cell the table reads.
        args.n_train, args.n_val = 2000, 1000
        args.epoch_points = [1]
        args.ratios = [0.50]
        args.ratio_epochs = 1

    import run_sal_qat_prototype as P

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp29 — how many epochs of recovery are needed?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | epoch points {args.epoch_points} | "
          f"curve ratio {args.curve_ratio} | ratios {args.ratios} @ "
          f"{args.ratio_epochs} ep" + ("  [GATE]" if args.gate else ""))

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

    out = {"experiment": "exp29_recovery_epochs",
           "model": P.MODEL_ID, "dataset": args.dataset,
           "dataset_label": P.dataset_label(args.dataset),
           "seed": args.seed, "gate": args.gate,
           "num_classes": num_classes,
           "n_train": int(train_split[0].shape[0]),
           "n_val": int(val_split[0].shape[0]),
           "batch_size": args.batch_size,
           "epoch_points": list(args.epoch_points),
           "curve_ratio": args.curve_ratio,
           "dense": dense,
           "curve": {}, "ratios": {},
           "exp24_arm1_reference": EXP24_ARM1,
           "notes": {
               "schedule": "each curve point is its own recovery with a "
                           "completed warmup+cosine schedule; a mid-run "
                           "checkpoint of a longer run is a different quantity",
               "int4": "quantize_weights(model, 4) — a simulated per-channel "
                       "quantize/dequantize round-trip in fp32, not a 4-bit "
                       "kernel; says nothing about speed or memory",
               "exp26": "exp26's 25% and 33% are MLP neuron ratios, not head "
                        "ratios; its only head arm is 50%",
           },
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "torch_version": torch.__version__}

    def save():
        (outdir / "exp29_recovery.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")

    save()

    # ------------------------------------------------- (A) the epoch curve
    if not args.skip_curve:
        P.rule(f"A — epoch curve at {args.curve_ratio:.0%} heads removed")
        small, pairs = slice_at(pretrained, args.curve_ratio, device, P, P.log)
        pre = score(small, val_split, device, use_amp, P, args.skip_int4)
        P.log(f"    sliced, before recovery: {pre['accuracy'] * 100:.2f}%")
        out["curve_pre_recovery"] = pre
        out["curve_head_set"] = sorted((int(a), int(b)) for a, b in pairs)
        out["curve_heads_removed"] = len(pairs)
        save()

        for ep in sorted(args.epoch_points):
            model, losses, secs = recover_for(
                small, train_split, ep, args, device, use_amp, P,
                f"recover {ep}ep")
            row = score(model, val_split, device, use_amp, P, args.skip_int4)
            row.update({"epochs": ep, "epoch_losses": losses,
                        "train_seconds": secs,
                        "gain_pp": (row["accuracy"] - pre["accuracy"]) * 100})
            out["curve"][str(ep)] = row
            P.log(f"    {ep:2d} epoch(s): {row['accuracy'] * 100:6.2f}%"
                  + (f" | INT4 {row['int4_accuracy'] * 100:.2f}%"
                     if "int4_accuracy" in row else "")
                  + f" | +{row['gain_pp']:.2f}pp over sliced"
                  + f" | {secs / 60:.0f} min")
            save()
            del model

        del small

    # ------------------------------------------------- (B) the ratio sweep
    if not args.skip_ratios:
        P.rule(f"B — ratio sweep at {args.ratio_epochs} epoch(s)")
        for ratio in args.ratios:
            small, pairs = slice_at(pretrained, ratio, device, P, P.log)
            pre = score(small, val_split, device, use_amp, P, args.skip_int4)
            model, losses, secs = recover_for(
                small, train_split, args.ratio_epochs, args, device, use_amp,
                P, f"recover p{int(round(ratio * 100))}")
            row = score(model, val_split, device, use_amp, P, args.skip_int4)
            row.update({"ratio": ratio, "epochs": args.ratio_epochs,
                        "heads_removed": len(pairs),
                        "accuracy_before_recovery": pre["accuracy"],
                        "epoch_losses": losses, "train_seconds": secs,
                        "head_set": sorted((int(a), int(b)) for a, b in pairs)})
            key = f"{ratio:.2f}"
            ref = EXP24_ARM1.get(key)
            if ref:
                row["exp24_arm1"] = ref
                row["vs_exp24_pp"] = (row["accuracy"] - ref["accuracy"]) * 100
            out["ratios"][key] = row
            P.log(f"    p{int(round(ratio * 100))}: {row['accuracy'] * 100:6.2f}%"
                  + (f" | INT4 {row['int4_accuracy'] * 100:.2f}%"
                     if "int4_accuracy" in row else "")
                  + (f" | exp24 arm1 read {ref['accuracy'] * 100:.2f}% "
                     f"({row['vs_exp24_pp']:+.2f}pp)" if ref else " | new cell"))
            save()
            del model, small

    # ------------------------------------------------------------ (C) summary
    try:
        P.rule("C — diminishing returns")
        curve = {int(k): v for k, v in out["curve"].items()}
        if curve and 5 in curve:
            pre_acc = out.get("curve_pre_recovery", {}).get("accuracy", 0.0)
            ref_gain = curve[5]["accuracy"] - pre_acc
            P.log(f"  gain at 5 epochs: {ref_gain * 100:.2f}pp "
                  "(the reference every other point is read against)")
            P.log(f"  {'epochs':>7s} {'accuracy':>9s} {'gain pp':>8s} "
                  f"{'% of 5ep':>9s}")
            knee = None
            for ep in sorted(curve):
                g = curve[ep]["accuracy"] - pre_acc
                frac = (g / ref_gain * 100) if ref_gain else float("nan")
                curve[ep]["fraction_of_5ep_gain"] = frac
                out["curve"][str(ep)]["fraction_of_5ep_gain"] = frac
                P.log(f"  {ep:7d} {curve[ep]['accuracy'] * 100:8.2f}% "
                      f"{g * 100:8.2f} {frac:8.1f}%")
                if knee is None and frac >= 95.0:
                    knee = ep
            out["diminishing_returns"] = {
                "reference_epochs": 5,
                "reference_gain_pp": ref_gain * 100,
                "first_epoch_at_95pct_of_5ep_gain": knee,
                "statement": (
                    f"{knee} epoch(s) reach 95% of the 5-epoch gain"
                    if knee else
                    "no tested epoch count reaches 95% of the 5-epoch gain")}
            P.log(f"  -> {out['diminishing_returns']['statement']}")
        else:
            P.log("  curve incomplete (no 5-epoch point); no knee computed")
    except Exception as e:                       # noqa: BLE001
        P.log(f"  [warn] summary failed: {type(e).__name__}: {e}")

    save()
    P.log(f"\n  wrote {outdir / 'exp29_recovery.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
