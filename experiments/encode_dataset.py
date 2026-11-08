#!/usr/bin/env python
"""Encode a dataset into the prototype's uint8 memmap cache, once, on CPU.

Every arm of every seed of workstreams 1a, 1c and 2 trains on the same
ImageNet-1k tensors. Decoding 1.28M JPEGs is pure CPU work, so doing it inside
the first GPU job would bill GPU-hours for something a CPU node does better and
would repeat it for every job that cannot find the cache.

This runs ``build_data`` for its side effect -- filling
``--data-cache`` -- and then reports the shapes, so a later GPU job starts with
a cache hit.

ImageNet-1k here is the ungated mirror
``evanarlian/imagenet_1k_resized_256``: the canonical 1 000 classes in
canonical order, pre-resized to 256px on the shortest side. ``ILSVRC/imagenet-1k``
is gated and needs an accepted-terms token.

Usage (CPU partition, no GPU)::

    python encode_dataset.py --dataset evanarlian/imagenet_1k_resized_256 \\
        --n-train 1281167 --n-val 50000 \\
        --data-cache /scratch/project_465003656/sal-torch/data_cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="evanarlian/imagenet_1k_resized_256")
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--data-cache", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import run_sal_qat_prototype as P

    t0 = time.time()
    print(f"=== encoding {args.dataset} -> {args.data_cache}", flush=True)
    sizes = {"n_train": args.n_train, "n_val": args.n_val}
    train, val, num_classes = P.build_data(sizes, args.seed, args.workers,
                                           dataset_id=args.dataset,
                                           cache_dir=args.data_cache)
    print(f"  train {tuple(train[0].shape)} labels {tuple(train[1].shape)}", flush=True)
    print(f"  val   {tuple(val[0].shape)} labels {tuple(val[1].shape)}", flush=True)
    print(f"  classes {num_classes}", flush=True)
    print(f"  label range train [{int(train[1].min())}, {int(train[1].max())}]",
          flush=True)
    cache = Path(args.data_cache)
    total = sum(f.stat().st_size for f in cache.glob("*.npy"))
    print(f"  cache {total / 1e9:.1f} GB in {cache}", flush=True)
    print(f"  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    print("ENCODE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
