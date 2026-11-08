#!/usr/bin/env python
"""exp27 -- does DropHead's training help on any *fixed* head set? Inference only.

exp25 asked this of SAL and got a sharp answer: on SAL's own trained-for set,
SAL reads 76.51% against 20.33% for 1a's baseline checkpoint on the identical
set, and 34.52% for SAL on sets it never saw. The adaptation is real and
entirely set-specific.

DropHead resamples its mask every step, so it has no "own" set to align to. The
question is whether that still buys something on a fixed set, or only on the
random draws 1a scores it with (73.29% +- 0.71 at P50+INT4).

## Why the baseline checkpoint is evaluated too

A DropHead number on a magnitude set is uninterpretable on its own: magnitude
pruning of a trained ViT is already a decent selection rule, so a high score
could be the rule rather than the training. The comparison that means something
is **the identical head set removed from 1a's baseline checkpoint**, which saw
the same data, optimiser, schedule and epoch count with no masking. That is the
control that made exp25 decisive, and it costs one extra checkpoint's worth of
evaluation. The SAL checkpoint is included on the same sets for the same reason
-- it completes a {baseline, sal, drophead} x {magnitude, SAL-aligned} matrix in
which every cell is comparable with every other.

## The head-set provenance problem, and how this script handles it

``magnitude_heads`` ranks heads by the L2 norm of their slice of the attention
output projection, read off **whichever checkpoint is passed**. Three different
checkpoints therefore produce three different "magnitude sets", and exp23 and
exp24 do not even agree on which one to use: exp23 ranked the *pretrained*
checkpoint, exp24's ``drophead_magnitude`` arm ranked the drophead checkpoint
itself. "The same set as exp23/exp24" is not a well-defined object.

So both readings are measured rather than one being assumed:

``magnitude_pinned``  the set ranked on **one reference checkpoint** (default
                      ``baseline``) and removed from all three. Same
                      architecture everywhere, so the cross-checkpoint
                      differences are differences in weights alone. This is the
                      row the conclusion rests on.
``magnitude_self``    each checkpoint's own ranking, i.e. exp24's reading. This
                      is each model scored at the set *it* would choose, which
                      is the fairer question for a deployment decision and the
                      weaker one for a causal claim.

The overlap between each ``self`` set and the pinned set is recorded, so the
size of the discrepancy is a measurement and not a caveat. exp25 hit exactly
this problem late and had to report a one-to-two-head confound after the fact.

``aligned_p50`` additionally removes **SAL's** trained-for set from all three
checkpoints, which places DropHead on SAL's own ground.

No training, no draws: every row is one deterministic sliced architecture scored
on the full validation split. Accuracy is autocast bfloat16, the path 1a, exp24
and exp25 score under, not true FP32.

Usage::

    python exp27_drophead_eval.py --data-cache ... --seed 42 \
        --checkpoints results/in1k_a/seed42/models \
        --sal-json results/in1k_a/seed42/sal_qat_prototype.json \
        --output results/exp27/seed42
    python exp27_drophead_eval.py --smoke
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

from exp23_baselines import magnitude_heads                   # noqa: E402
from exp24_sal_recovery import aligned_heads, load_checkpoint  # noqa: E402

DATASET = "evanarlian/imagenet_1k_resized_256"
RATIOS = (0.33, 0.50)
CHECKPOINTS = ("drophead", "baseline", "sal")


def score(model, val_split, device, use_amp, skip_int4):
    """FP and INT4 accuracy for one architecture."""
    import run_sal_qat_prototype as P

    out = {"accuracy": P.accuracy(model, val_split, device, use_amp)}
    if not skip_int4:
        out["int4_accuracy"] = P.accuracy(
            P.quantize_weights(copy.deepcopy(model), 4), val_split, device, use_amp)
    return out


def fmt(entry):
    s = f"{entry['accuracy'] * 100:6.2f}%"
    if entry.get("int4_accuracy") is not None:
        s += f"   INT4 {entry['int4_accuracy'] * 100:6.2f}%"
    return s


def as_pairs(pairs):
    return sorted((int(a), int(b)) for a, b in pairs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--checkpoints", default=None,
                   help="directory holding drophead.pt / baseline.pt / sal.pt")
    p.add_argument("--sal-json", default=None,
                   help="1a JSON to read SAL's trained-for head set from; "
                        "without it the aligned_p50 rows are skipped")
    p.add_argument("--which", nargs="+", default=list(CHECKPOINTS),
                   choices=list(CHECKPOINTS))
    p.add_argument("--magnitude-from", default="baseline", choices=list(CHECKPOINTS),
                   help="the checkpoint whose magnitude ranking is pinned and "
                        "applied to all of them (default: baseline, the "
                        "unmasked control)")
    p.add_argument("--no-self-sets", action="store_true",
                   help="skip the per-checkpoint magnitude sets and evaluate "
                        "only the pinned one")
    p.add_argument("--n-train", type=int, default=2000,
                   help="inference only; the train split is built but unused, "
                        "so this stays small unless the cache forces otherwise")
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp27")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_val, args.ratios = 1000, [0.5]

    import run_sal_qat_prototype as P
    from sal import slice_heads

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp27 -- DropHead on fixed head sets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | ratios {args.ratios} | checkpoints {args.which}")
    P.log(f"  pinned magnitude ranking from: {args.magnitude_from}")

    # Everything validated before the data build, which is the expensive part.
    if not args.checkpoints:
        p.error("--checkpoints is required")
    ckdir = Path(args.checkpoints)
    needed = set(args.which) | {args.magnitude_from}
    for name in sorted(needed):
        f = ckdir / f"{name}.pt"
        if not f.exists():
            p.error(f"{name} needs {f}, which does not exist")

    pruned_heads = None
    if args.sal_json:
        meta = json.loads(Path(args.sal_json).read_text(encoding="utf-8"))
        pruned_heads = ((meta.get("training") or {}).get("sal")
                        or {}).get("pruned_heads")
        if not pruned_heads:
            p.error(f"{args.sal_json} has no training.sal.pruned_heads")
        P.log(f"  SAL's trained-for set: {len(pruned_heads)} heads")
    else:
        P.log("  no --sal-json: the aligned_p50 rows will be skipped")

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    _train_split, val_split, num_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    P.log(f"  {val_split[0].shape[0]} val images, {num_classes} classes")

    # The pinned sets come from the reference checkpoint, once, before the loop,
    # so every checkpoint below is sliced with the identical head indices.
    P.rule(f"pinned magnitude sets (from {args.magnitude_from})")
    ref = load_checkpoint(ckdir / f"{args.magnitude_from}.pt", num_classes)
    pinned = {}
    for ratio in args.ratios:
        tag = f"magnitude_p{int(round(ratio * 100))}"
        pinned[tag] = as_pairs(magnitude_heads(ref, ratio, per_layer=True))
        P.log(f"    {tag}: {len(pinned[tag])} heads")
    del ref

    results: dict = {}
    self_overlap: dict = {}
    t0 = time.time()
    for name in args.which:
        P.rule(name)
        model = load_checkpoint(ckdir / f"{name}.pt", num_classes).to(device)
        cell = {"dense": score(model, val_split, device, use_amp, args.skip_int4)}
        P.log(f"    {'dense':22s} {fmt(cell['dense'])}")
        dense_acc = cell["dense"]["accuracy"]

        def add(tag, pairs, note=""):
            small = slice_heads(copy.deepcopy(model), pairs).eval().to(device)
            entry = score(small, val_split, device, use_amp, args.skip_int4)
            entry.update(heads=len(pairs), head_set=as_pairs(pairs),
                         delta_vs_dense=entry["accuracy"] - dense_acc)
            cell[tag] = entry
            P.log(f"    {tag:22s} {fmt(entry)}"
                  f"   ({entry['delta_vs_dense'] * 100:+.2f}pp vs dense){note}")
            del small
            return entry

        for tag, pairs in pinned.items():
            add(tag, pairs)

        # Each checkpoint's own ranking, and how far it diverges from the pinned
        # one. This is exp24's reading of "the magnitude set".
        if not args.no_self_sets:
            for ratio in args.ratios:
                ptag = f"magnitude_p{int(round(ratio * 100))}"
                own = as_pairs(magnitude_heads(model, ratio, per_layer=True))
                shared = len(set(map(tuple, own)) & set(map(tuple, pinned[ptag])))
                self_overlap[f"{name}:{ptag}"] = {
                    "overlap": shared, "of": len(own),
                    "identical": own == pinned[ptag]}
                if own == pinned[ptag]:
                    P.log(f"    {ptag}_self: identical to the pinned set, "
                          f"not re-evaluated")
                    continue
                add(f"{ptag}_self", own,
                    f"   [own ranking, {shared}/{len(own)} shared with pinned]")

        if pruned_heads:
            pairs, subset = aligned_heads(model, pruned_heads, 0.50)
            e = add("aligned_p50", pairs)
            e["aligned_is_subset"] = subset

        cell["checkpoint"] = str(ckdir / f"{name}.pt")
        results[name] = cell
        del model

    payload = {
        "experiment": "exp27_drophead_eval",
        "model": P.MODEL_ID, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "seed": args.seed, "ratios": args.ratios, "checkpoints": args.which,
        "magnitude_from": args.magnitude_from,
        "pinned_head_sets": pinned,
        "self_set_overlap": self_overlap,
        "sal_pruned_heads": (as_pairs(pruned_heads) if pruned_heads else None),
        "n_val": int(val_split[0].shape[0]), "num_classes": num_classes,
        "smoke": args.smoke, "checkpoint_dir": str(ckdir),
        "sal_json": args.sal_json,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": results,
        "caveats": {
            "inference_only": "No training of any kind. Every row is one "
                              "deterministic sliced architecture scored on the "
                              "full validation split, so there are no draws and "
                              "no spread within a seed.",
            "not_true_fp32": "The accuracy column runs under autocast bfloat16, "
                             "the path 1a, exp24 and exp25 score under, so it is "
                             "comparable with them rather than with an FP32 "
                             "reference.",
            "drophead_has_no_own_set": "DropHead resamples its mask every step, "
                                       "so unlike SAL it has no trained-for set. "
                                       "Every set here is external to it, which "
                                       "is the point of the experiment and also "
                                       "means it is never scored on home ground.",
            "pinned_vs_self": "magnitude_pXX uses the ranking of the "
                              "--magnitude-from checkpoint for all of them, so "
                              "the comparison is at a fixed architecture. The "
                              "_self rows use each checkpoint's own ranking, "
                              "which is exp24's reading; self_set_overlap says "
                              "how far the two diverge.",
            "int4_is_simulated": "INT4 is quantize_weights(model, 4), the "
                                 "simulated per-channel round-trip, not "
                                 "bitsandbytes NF4.",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = outdir / "exp27_drophead_eval.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    P.log(f"\n  metrics -> {out}")

    try:
        P.rule("Summary")
        P.log(f"  dataset: {P.dataset_label(args.dataset)}  seed {args.seed}")
        rows = ["dense"] + sorted(
            {k for c in results.values() for k in c if k not in ("dense", "checkpoint")})
        P.log("  " + "row".ljust(22) + "".join(n[:12].rjust(22) for n in args.which))
        for row in rows:
            line = "  " + row.ljust(22)
            for name in args.which:
                e = results.get(name, {}).get(row)
                line += (fmt(e).rjust(22) if e else "-".rjust(22))
            P.log(line)
        P.log(f"\n  wall clock {(time.time() - t0) / 60:.1f} min")
    except Exception as exc:                                  # noqa: BLE001
        P.log(f"  (summary table failed: {exc!r}; the JSON above is complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
