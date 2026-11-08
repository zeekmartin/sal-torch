#!/usr/bin/env python
"""exp25 — score the 1a checkpoints on SAL's own head set. Inference only.

1a evaluated its pruned cells as the mean over 20 **random** head selections,
which is the right protocol for a dense model but means SAL was never scored at
the one architecture it was trained for. exp24 found this by accident: slicing
SAL's aligned set *raised* accuracy by +5.1 to +15.4pp over the dense SAL model,
before any fine-tuning. That number came out of a recovery experiment as an
intermediate, so this measures it directly, on all three seeds, with the
baseline control the comparison needs.

Three checkpoints' worth of question:

``sal`` dense                SAL's clean accuracy, as 1a reports it.
``sal`` aligned P50          SAL's own trained set, removed. 72 of 144 heads.
``sal`` aligned P33          the lowest-magnitude 48 of those 72 (a subset:
                             an architecture SAL never trained for).
``baseline`` aligned P50/33  the **same head set** removed from the 1a baseline
                             checkpoint, which saw identical data, optimiser and
                             schedule but no masking.

The baseline rows are what make the SAL rows mean anything. SAL's aligned set is
not an arbitrary 72 heads: it was drawn randomly, but SAL then trained for
thousands of steps with exactly those heads masked, so they are the heads whose
function was redistributed. If removing that set also helps the baseline, the
set is simply a good set and SAL's adaptation is doing nothing. If it helps only
SAL, the adaptation is real and this is the configuration that shows it.

No training, no draws: every row is one deterministic architecture scored on the
full 50k validation set. The accuracy column is autocast bfloat16, the same path
1a and exp24 score under, not true FP32 -- it is labelled FP in the output for
that reason.

Usage::

    python exp25_aligned_eval.py --data-cache ... --seed 42 \
        --checkpoints results/in1k_a/seed42/models \
        --sal-json results/in1k_a/seed42/sal_qat_prototype.json \
        --output results/exp25/seed42
    python exp25_aligned_eval.py --smoke
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

from exp24_sal_recovery import aligned_heads, load_checkpoint  # noqa: E402

DATASET = "evanarlian/imagenet_1k_resized_256"
RATIOS = (0.33, 0.50)
CHECKPOINTS = ("sal", "baseline")


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
    if "int4_accuracy" in entry:
        s += f"   INT4 {entry['int4_accuracy'] * 100:6.2f}%"
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--checkpoints", default=None,
                   help="directory holding sal.pt / baseline.pt from a 1a run")
    p.add_argument("--sal-json", default=None,
                   help="1a JSON to read SAL's pruned head set from")
    p.add_argument("--which", nargs="+", default=list(CHECKPOINTS),
                   choices=list(CHECKPOINTS))
    p.add_argument("--n-train", type=int, default=2000,
                   help="inference only; the train split is built but unused, "
                        "so this stays small on purpose")
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp25")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_val, args.ratios, args.skip_int4 = 1000, [0.5], True

    import run_sal_qat_prototype as P
    from sal import slice_heads

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp25 — the 1a checkpoints on SAL's own head set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | ratios {args.ratios} | checkpoints {args.which}")

    # Every input checked before any GPU time: this job is cheap, but a missing
    # file found after the data build still costs the build.
    if not args.checkpoints:
        p.error("--checkpoints is required")
    ckdir = Path(args.checkpoints)
    for name in args.which:
        f = ckdir / f"{name}.pt"
        if not f.exists():
            p.error(f"{name} needs {f}, which does not exist")
    if not args.sal_json:
        p.error("--sal-json is required: the aligned set comes from it")
    meta = json.loads(Path(args.sal_json).read_text(encoding="utf-8"))
    pruned_heads = ((meta.get("training") or {}).get("sal") or {}).get("pruned_heads")
    if not pruned_heads:
        p.error(f"{args.sal_json} has no training.sal.pruned_heads; there is "
                "nothing to align to")
    P.log(f"  SAL's pruned set: {len(pruned_heads)} heads")

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    _train_split, val_split, num_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    P.log(f"  {val_split[0].shape[0]} val images, {num_classes} classes")

    results: dict = {}
    head_sets: dict = {}
    t0 = time.time()
    for name in args.which:
        P.rule(name)
        model = load_checkpoint(ckdir / f"{name}.pt", num_classes).to(device)
        cell = {"dense": score(model, val_split, device, use_amp, args.skip_int4)}
        P.log(f"    {'dense':18s} {fmt(cell['dense'])}")
        for ratio in args.ratios:
            tag = f"aligned_p{int(round(ratio * 100))}"
            # The head set is derived from this checkpoint's own magnitudes when
            # it is a subset, so it can differ between sal and baseline at P33.
            # Recorded per checkpoint rather than assumed identical.
            pairs, subset = aligned_heads(model, pruned_heads, ratio)
            small = slice_heads(copy.deepcopy(model), pairs).eval().to(device)
            entry = score(small, val_split, device, use_amp, args.skip_int4)
            entry.update(heads=len(pairs), aligned_is_subset=subset,
                         head_set=sorted((int(a), int(b)) for a, b in pairs),
                         delta_vs_dense=entry["accuracy"] - cell["dense"]["accuracy"])
            cell[tag] = entry
            head_sets[f"{name}:{tag}"] = entry["head_set"]
            P.log(f"    {tag:18s} {fmt(entry)}"
                  f"   ({entry['delta_vs_dense'] * 100:+.2f}pp vs dense"
                  + (", subset" if subset else "") + ")")
            del small
        cell["checkpoint"] = str(ckdir / f"{name}.pt")
        results[name] = cell
        del model

    # Does the baseline get the same set at P33? If the magnitude ranking inside
    # SAL's 72 heads differs between the two checkpoints, the P33 rows are not
    # strictly the same architecture and the comparison has to say so.
    same_p33 = None
    k = [f"{n}:aligned_p33" for n in ("sal", "baseline")]
    if all(x in head_sets for x in k):
        same_p33 = head_sets[k[0]] == head_sets[k[1]]
        P.log(f"\n  P33 subsets identical across checkpoints: {same_p33}")

    payload = {
        "experiment": "exp25_aligned_eval",
        "model": P.MODEL_ID, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "seed": args.seed, "ratios": args.ratios, "checkpoints": args.which,
        "n_val": int(val_split[0].shape[0]), "num_classes": num_classes,
        "smoke": args.smoke, "checkpoint_dir": str(ckdir),
        "sal_json": args.sal_json,
        "sal_pruned_heads": sorted((int(a), int(b)) for a, b in pruned_heads),
        "p33_subsets_identical": same_p33,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": results,
        "caveats": {
            "inference_only": "No training of any kind. Every row is one "
                              "deterministic sliced architecture scored on the "
                              "full validation split, so there are no draws and "
                              "no spread within a seed.",
            "not_true_fp32": "The accuracy column runs under autocast bfloat16, "
                             "the path 1a and exp24 score under, so it is "
                             "comparable with them rather than with an FP32 "
                             "reference.",
            "aligned_subset": "SAL trained at mask_ratio 0.5, so the P33 row is "
                              "the lowest-magnitude subset of SAL's set -- an "
                              "architecture SAL never trained for.",
            "baseline_is_1a_not_pretrained": "The baseline checkpoint is 1a's "
                                             "own unmasked arm (same data, "
                                             "optimiser and schedule), not the "
                                             "raw pretrained checkpoint. That is "
                                             "the matched control for SAL.",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = outdir / "exp25_aligned_eval.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    P.log(f"\n  metrics -> {out}")

    try:
        P.rule("Summary")
        P.log(f"  dataset: {P.dataset_label(args.dataset)}  seed {args.seed}")
        rows = ["dense"] + [f"aligned_p{int(round(r * 100))}" for r in args.ratios]
        P.log("  " + "row".ljust(18) + "".join(n[:12].rjust(22) for n in args.which))
        for row in rows:
            line = "  " + row.ljust(18)
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
