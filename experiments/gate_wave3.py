#!/usr/bin/env python3
"""Wave-3 reference gate: the published number, through the patched harness.

Two things at once, which is why it is worth a job of its own.

**It is the reference gate.** The pretrained ViT-B/16 reads 78.44% top-1 on
the full 50 000-image validation split of this ImageNet copy. Every
accuracy in the campaign is anchored to that, so production does not start
until it reproduces.

**It is the regression test for the harness change.** ``accuracy()`` now
starts a memory probe and can dump per-image predictions. Both are meant to
be invisible to the number it returns. The only way to know is to re-measure
the anchor through the patched code, and to score the split twice -- once
with the prediction dump on and once off -- and require bit-identical
accuracies. An eye on the diff is not evidence.

    python experiments/gate_wave3.py --output results/gate_wave3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO / "scripts"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: What the campaign publishes for the pretrained checkpoint on this copy.
REFERENCE = {"accuracy": 0.7844, "tolerance": 0.0040,
             "source": "results/lumi_v060/PHASE3_IMAGENET1K.md and every "
                       "control ladder since"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="evanarlian/imagenet_1k_resized_256")
    p.add_argument("--data-cache", default=None)
    p.add_argument("--n-val", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--loader-block", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/gate_wave3")
    args = p.parse_args(argv)

    import run_sal_qat_prototype as P

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = (args.loader_workers,
                                            args.loader_block)
    device, vram = P.detect_device()
    print("=== wave-3 gate ===")
    print("  device %s (%.0f GB)" % (device, vram))

    sizes = {"n_train": 2000, "n_val": args.n_val, "n_fi": P.N_FI}
    _, val_split, n_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    model = P.load_model(n_classes).to(device).eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: predictions on.
    P.COLLECT_PREDICTIONS = str(out / "gate_predictions.npz")
    t0 = time.time()
    acc_with = P.accuracy(model, val_split, device, use_amp=True,
                          batch_size=args.batch_size)
    mem_with = dict(P.LAST_EVAL_MEMORY)
    t_with = time.time() - t0
    P.COLLECT_PREDICTIONS = None

    # Pass 2: predictions off, same model, same split, same order.
    t0 = time.time()
    acc_without = P.accuracy(model, val_split, device, use_amp=True,
                             batch_size=args.batch_size)
    mem_without = dict(P.LAST_EVAL_MEMORY)
    t_without = time.time() - t0

    identical = acc_with == acc_without
    delta = acc_with - REFERENCE["accuracy"]
    within = abs(delta) <= REFERENCE["tolerance"]

    # The dumped predictions must reproduce the accuracy they came from.
    import numpy as np
    z = np.load(out / "gate_predictions.npz")
    from_dump = float((z["pred"] == z["label"]).mean())
    dump_ok = abs(from_dump - acc_with) < 1e-12

    doc = {
        "experiment": "gate_wave3",
        "what": ("the published dense reference, re-measured through the "
                 "harness after adding the memory probe and the prediction "
                 "dump"),
        "dataset": args.dataset, "n_val": int(z["pred"].shape[0]),
        "batch_size": args.batch_size, "autocast": "float16",
        "reference": REFERENCE,
        "accuracy_with_predictions": acc_with,
        "accuracy_without_predictions": acc_without,
        "accuracies_identical": identical,
        "accuracy_from_the_dump": from_dump,
        "dump_reproduces_accuracy": dump_ok,
        "delta_vs_reference": delta,
        "within_tolerance": within,
        "seconds_with": round(t_with, 1), "seconds_without": round(t_without, 1),
        "memory_with": mem_with, "memory_without": mem_without,
        "gpu": P.gpu_name() if hasattr(P, "gpu_name") else None,
        "torch_version": __import__("torch").__version__,
        "passed": bool(identical and within and dump_ok),
    }
    (out / "gate_wave3.json").write_text(json.dumps(doc, indent=1),
                                         encoding="utf-8")

    print("  reference          %.4f +/- %.4f" % (REFERENCE["accuracy"],
                                                  REFERENCE["tolerance"]))
    print("  measured           %.5f  (delta %+.5f) -> %s"
          % (acc_with, delta, "within" if within else "OUT OF BAND"))
    print("  dump on vs off     %.5f vs %.5f -> %s"
          % (acc_with, acc_without,
             "identical" if identical else "DIFFER"))
    print("  dump reproduces it %s" % ("yes" if dump_ok else "NO"))
    mw = mem_with.get("max_memory_allocated_bytes")
    if mw:
        print("  peak allocated     %.2f GB (reserved %.2f GB)"
              % (mw / 1e9,
                 (mem_with.get("max_memory_reserved_bytes") or 0) / 1e9))
    rss = mem_with.get("rss_bytes_after")
    if rss:
        print("  RSS after          %.2f GB" % (rss / 1e9))
    print("\n  %s" % ("GATE PASSED" if doc["passed"] else "GATE FAILED"))
    print("wrote %s" % (out / "gate_wave3.json"))
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
