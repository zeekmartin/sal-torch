#!/usr/bin/env python
"""Noise floor of the I-JEPA linear probe. Gates 1b (~53 GPU-hours).

1b has no top-1 accuracy to report: I-JEPA is self-supervised and carries no
1000-class head, so its accuracy column is ``sal.linear_probe`` -- a linear
classifier fitted on frozen features. The default fits it on 5000 images for
1000 classes, i.e. **5 images per class**, and the plan is to stop at one seed
if the arms separate by more than 3pp.

That only works if the probe's own run-to-run spread is well under 3pp, and
nothing measured it. This does: the same frozen pretrained checkpoint, scored
five times, with **only the probe's training subset changing**. Any spread in
the result is the probe's, because the weights, the features and the validation
set are identical across repeats.

The five subsets are **disjoint consecutive blocks** of the cached train split.
The cache is written in random order (see ``run_sal_qat_prototype.build_data``),
so consecutive blocks are independent random samples -- a stronger design than
resampling with replacement, which would share images between repeats and
understate the spread.

``sal.linear_probe`` and ``sal.knn_accuracy`` are called five times each rather
than extracting features once and refitting. Refitting would be ~5x cheaper,
but it means reimplementing the probe, and a noise floor measured with a
different probe than 1b will use is not the number anyone needs. Five honest
calls cost about ten minutes.

Usage::

    python probe_jepa_noise.py --data-cache ... --repeats 5
    python probe_jepa_noise.py --smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="facebook/ijepa_vith14_1k")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-probe-train", type=int, default=5000,
                   help="probe training images per repeat; the 1b default")
    p.add_argument("--n-probe-val", type=int, default=2500,
                   help="held-out validation images, identical across repeats")
    p.add_argument("--n-train", type=int, default=1_281_167,
                   help="cached train split to draw the disjoint blocks from")
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--no-knn", action="store_true")
    p.add_argument("--output", default="results/probe_jepa_noise")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.repeats, args.n_probe_train, args.n_probe_val = 2, 200, 200
        args.n_train, args.n_val = 2000, 1000

    import run_jepa_sal as J
    import run_sal_qat_prototype as P
    from torch.utils.data import DataLoader

    J.MODEL_ID = args.model
    log = J.log
    J.rule("I-JEPA linear-probe noise floor — gates 1b")
    if not torch.cuda.is_available():
        log("  no CUDA device; this is meaningless on CPU")
        return 2
    device = torch.device("cuda")
    log(f"  GPU:    {torch.cuda.get_device_name(0)}")
    log(f"  model:  {args.model}")
    log(f"  probe:  {args.n_probe_train} train / {args.n_probe_val} val, "
        f"{args.repeats} disjoint repeats")

    need = args.repeats * args.n_probe_train
    if need > args.n_train:
        p.error(f"{args.repeats} disjoint blocks of {args.n_probe_train} need "
                f"{need} train rows, but --n-train is {args.n_train}")

    P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
    train, val, num_classes = P.build_data(
        {"n_train": args.n_train, "n_val": args.n_val, "n_fi": 256},
        args.seed, args.loader_workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    xtr, ytr = train
    xva, yva = val
    log(f"  {xtr.shape[0]} train / {xva.shape[0]} val rows, {num_classes} classes")
    log(f"  probe train images per class: "
        f"{args.n_probe_train / max(num_classes, 1):.1f}")

    model = J.load_model(device, grad_ckpt=False)
    model.eval()

    kw = ({"num_workers": args.loader_workers, "persistent_workers": True,
           "prefetch_factor": 4} if args.loader_workers else {})
    # One validation set for every repeat: the whole point is that only the
    # probe's *training* subset varies.
    val_loader = DataLoader(
        J._NormalizedSplit(xva[:args.n_probe_val], yva[:args.n_probe_val]),
        batch_size=args.batch_size, **kw)

    from sal import knn_accuracy, linear_probe

    runs = []
    t0 = time.time()
    for r in range(args.repeats):
        lo = r * args.n_probe_train
        hi = lo + args.n_probe_train
        tr_loader = DataLoader(J._NormalizedSplit(xtr[lo:hi], ytr[lo:hi]),
                               batch_size=args.batch_size, **kw)
        t1 = time.time()
        row = {"repeat": r, "train_rows": [lo, hi],
               "linear_probe": linear_probe(model, tr_loader, val_loader,
                                            num_classes=num_classes)}
        if not args.no_knn:
            row["knn_accuracy"] = knn_accuracy(model, tr_loader, val_loader, k=20)
        row["seconds"] = round(time.time() - t1, 1)
        runs.append(row)
        log(f"    repeat {r}: rows [{lo}:{hi}]  probe "
            f"{row['linear_probe'] * 100:6.2f}%"
            + (f"   knn {row['knn_accuracy'] * 100:6.2f}%"
               if "knn_accuracy" in row else "")
            + f"   ({row['seconds']:.0f}s)")
        del tr_loader

    def stats(key):
        xs = [r[key] for r in runs if r.get(key) is not None]
        if not xs:
            return None
        out = {"values": xs, "mean": statistics.fmean(xs),
               "min": min(xs), "max": max(xs), "range": max(xs) - min(xs)}
        out["sd"] = statistics.stdev(xs) if len(xs) > 1 else None
        return out

    summary = {"linear_probe": stats("linear_probe")}
    if not args.no_knn:
        summary["knn_accuracy"] = stats("knn_accuracy")

    lp = summary["linear_probe"]
    sd_pp = (lp["sd"] * 100) if lp.get("sd") is not None else None
    # The 3pp decision rule 1b is planned around. A gap is only readable if the
    # measurement's own spread is small next to it; 1pp is the threshold the
    # plan set, so the verdict is stated rather than left to the reader.
    verdict = None
    if sd_pp is not None:
        verdict = ("PROBE IS STABLE ENOUGH" if sd_pp <= 1.0
                   else "PROBE TOO NOISY — enlarge it before committing 1b")

    payload = {
        "experiment": "probe_jepa_noise",
        "model": args.model, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "n_probe_train": args.n_probe_train, "n_probe_val": args.n_probe_val,
        "repeats": args.repeats, "num_classes": num_classes,
        "images_per_class": round(args.n_probe_train / max(num_classes, 1), 2),
        "disjoint_blocks": True,
        "seed": args.seed, "batch_size": args.batch_size, "smoke": args.smoke,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runs": runs, "summary": summary,
        "sd_pp": (round(sd_pp, 3) if sd_pp is not None else None),
        "verdict": verdict,
        "caveats": {
            "only_the_probe_subset_varies": "Same frozen pretrained "
                "checkpoint, same validation images, same code path. The only "
                "thing that changes between repeats is which 5000 training "
                "images the linear classifier is fitted on, so the spread is "
                "the probe's and nothing else's.",
            "disjoint_not_resampled": "The repeats are disjoint consecutive "
                "blocks of a cache written in random order, so they are "
                "independent samples. Resampling with replacement would share "
                "images between repeats and understate the spread.",
            "not_the_whole_noise_budget": "This is the probe's contribution "
                "alone. 1b also carries head-selection noise, which exp04 "
                "measured as the dominant term on ViT-B/16 and which the 20 "
                "draws exist to average down, plus training-seed noise that "
                "one seed cannot estimate at all.",
            "pretrained_not_trained": "Measured on the pretrained checkpoint. "
                "A checkpoint after two epochs of masked training may have a "
                "different feature geometry and so a different probe spread.",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "probe_jepa_noise.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    J.rule("Verdict")
    for key, s in summary.items():
        if s is None:
            continue
        log(f"  {key}")
        log(f"    values  " + "  ".join(f"{v * 100:.2f}" for v in s["values"]))
        log(f"    mean {s['mean'] * 100:.2f}%   sd "
            + (f"{s['sd'] * 100:.3f}pp" if s["sd"] is not None else "n/a")
            + f"   range {s['range'] * 100:.2f}pp")
    if verdict:
        log(f"\n  linear-probe sd = {sd_pp:.3f}pp  ->  {verdict}")
    log(f"  metrics -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
