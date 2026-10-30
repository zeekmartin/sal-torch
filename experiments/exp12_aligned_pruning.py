#!/usr/bin/env python3
"""exp12 — does SAL help when you prune exactly the heads it killed?

Track C found that plain per-step head dropout beats SAL by ~30pp under 50%
head pruning (`results/lumi_v060/PHASE2C.md`). The one interpretation that could
still justify SAL's accumulate-and-hold schedule is that it was being judged on
the wrong head set: SAL finishes training with a *specific* set of heads dead,
and the evaluation then removed a different, random set. So SAL was scored having
lost its trained-dead heads **plus** an unrelated half, while DropHead entered
evaluation with every head alive.

This script runs the missing comparison. For each seed it trains three arms —
``baseline``, ``sal`` and ``drophead`` — records which heads SAL actually killed
(via ``HeadMasker.pruned_heads()``, new in this commit), and then scores **every
arm** under two structural perturbations of the same size:

* ``aligned``   — exactly the heads SAL killed.
* ``random``    — a random set of the same total size, averaged over
                  ``--draws`` draws, identical draws for every arm.

Both are applied by **masking**, not slicing, and that is deliberate. SAL's
pruned set is drawn from the whole model, so it is not uniform per layer, and
``slice_heads()`` requires the same count in every layer. Masking is the only way
to apply the aligned set at all, so the random comparison is masked too — the
two cells then differ in *which* heads are gone and nothing else. Masking costs
no accuracy relative to slicing the same heads (sal-torch's own test suite checks
that equivalence), it simply does not make the model smaller.

How to read the result:

* If ``sal`` on ``aligned`` is much better than ``sal`` on ``random``, the
  adaptation is real and specific, and SAL's schedule is defensible for
  deployments that prune the heads SAL chose.
* If ``sal`` on ``aligned`` still loses to ``drophead``, accumulation has no
  remaining defence.
* ``baseline`` on ``aligned`` is the control that matters most: if an untrained
  model loses just as little from that set, then the set is simply made of
  unimportant heads and SAL's "adaptation" is selection, not reorganisation.

Usage::

    python exp12_aligned_pruning.py --seed 42 --mask-ratio 0.5 --output results/exp12/
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROTOTYPE = HERE.parent / "scripts" / "run_sal_qat_prototype.py"


def load_prototype():
    """Import the validated prototype as a module and reuse its machinery.

    Data pipeline, model loading, optimizer, training step, quantizer and
    accuracy function all come from the script that produced the published
    numbers, so this experiment cannot drift from it.
    """
    spec = importlib.util.spec_from_file_location("sal_qat_prototype", PROTOTYPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg=""):
    print(msg, flush=True)


class MaskSet:
    """Mask an explicit set of ``(layer, head)`` pairs, for evaluation only."""

    def __init__(self, model, heads, config):
        from sal import HeadMasker
        self.masker = HeadMasker(model, config, seed=0)
        self.heads = list(heads)

    def __enter__(self):
        self.masker.install()
        for layer, head in self.heads:
            self.masker._masks[layer][head] = 0.0
        self.masker.apply_mask()
        return self

    def __exit__(self, *exc):
        self.masker.remove()
        return False


def random_heads(num_layers, num_heads, count, seed):
    """``count`` distinct (layer, head) pairs drawn from the whole model.

    Matches how ``HeadMasker`` draws — globally, not per layer — so the random
    comparison has the same shape of perturbation as the aligned one, differing
    only in which heads it picks.
    """
    rng = random.Random(seed)
    allh = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    return sorted(rng.sample(allh, count))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-val", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--draws", type=int, default=20,
                   help="random head sets to average the 'random' cell over")
    p.add_argument("--output", default="results/exp12_aligned")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    P = load_prototype()
    from sal import SALConfig

    if args.smoke:
        args.n_train, args.n_val, args.epochs, args.draws = 200, 100, 1, 2

    device, vram = P.detect_device()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"

    P.rule("exp12 — aligned vs random head pruning")
    log(f"  device {device} ({vram:.1f} GB) | seed {args.seed} | "
        f"mask_ratio {args.mask_ratio} | {args.epochs} epochs")

    sizes = dict(n_train=args.n_train, n_val=args.n_val, n_fi=64,
                 workers=8)
    train_split, val_split, num_classes = P.build_data(sizes, args.seed)
    log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val, "
        f"{num_classes} classes")

    t_all = time.time()
    arms, pruned_by_sal = {}, None
    for key, mask_mode in (("baseline", "none"), ("sal", "sal"), ("drophead", "drophead")):
        P.rule(f"training {key} (masking={mask_mode})")
        model, info = P.train_variant(
            key, mask_mode, False, num_classes, train_split, args.epochs,
            args.batch_size, args.seed, device, use_amp, args.mask_ratio)
        arms[key] = model
        if key == "sal":
            pruned_by_sal = info.get("pruned_heads")
        log(f"  losses {[round(x, 4) for x in info['losses']]}")

    if not pruned_by_sal:
        log("ERROR: SALTrainer did not report pruned_heads — is sal up to date?")
        return 1

    cfg = SALConfig.auto(arms["baseline"], prune_fraction=args.mask_ratio)
    nl, nh = cfg.num_layers, cfg.num_heads_per_layer
    n_pruned = len(pruned_by_sal)
    per_layer = {}
    for layer, _ in pruned_by_sal:
        per_layer[layer] = per_layer.get(layer, 0) + 1

    P.rule("SAL's pruned set")
    log(f"  {n_pruned} of {nl * nh} heads ({n_pruned / (nl * nh):.1%})")
    log(f"  per layer: {[per_layer.get(i, 0) for i in range(nl)]}")
    log(f"  uniform per layer: {len(set(per_layer.values())) == 1 and len(per_layer) == nl}")

    draws = [random_heads(nl, nh, n_pruned, 10_000 + args.seed * 100 + i)
             for i in range(args.draws)]

    results = {}
    for key, model in arms.items():
        P.rule(f"scoring {key}")
        clean = P.accuracy(model, val_split, device, use_amp)
        with MaskSet(model, pruned_by_sal, cfg):
            aligned = P.accuracy(model, val_split, device, use_amp)
        rnd = []
        for heads in draws:
            with MaskSet(model, heads, cfg):
                rnd.append(P.accuracy(model, val_split, device, use_amp))
        # The same two perturbations with INT4 weights on top. The copy is made
        # and quantized *outside* any mask context: deepcopy would otherwise
        # duplicate the live pre-hooks, whose closures still point at the
        # original masker's mask tensors.
        q = P.quantize_weights(copy.deepcopy(model), 4)
        with MaskSet(q, pruned_by_sal, cfg):
            aligned_i4 = P.accuracy(q, val_split, device, use_amp)
        rnd_i4 = []
        for heads in draws[:max(1, args.draws // 4)]:
            with MaskSet(q, heads, cfg):
                rnd_i4.append(P.accuracy(q, val_split, device, use_amp))
        del q

        results[key] = {
            "clean": clean,
            "aligned": aligned,
            "random_mean": statistics.fmean(rnd),
            "random_std": statistics.stdev(rnd) if len(rnd) > 1 else None,
            "random_per_draw": rnd,
            "aligned_int4": aligned_i4,
            "random_int4_mean": statistics.fmean(rnd_i4),
            "aligned_minus_random": aligned - statistics.fmean(rnd),
        }
        log(f"    clean   {clean * 100:6.2f}%")
        log(f"    aligned {aligned * 100:6.2f}%")
        log(f"    random  {statistics.fmean(rnd) * 100:6.2f}%"
            + (f" ± {statistics.stdev(rnd) * 100:.2f}" if len(rnd) > 1 else ""))
        log(f"    aligned - random  {results[key]['aligned_minus_random'] * 100:+6.2f}pp")

    P.rule("Summary")
    log(f"  {'arm':<10}{'clean':>9}{'aligned':>9}{'random':>9}{'A-R':>9}"
        f"{'aln+I4':>9}{'rnd+I4':>9}")
    for key in ("baseline", "sal", "drophead"):
        r = results[key]
        log(f"  {key:<10}{r['clean'] * 100:8.2f}%{r['aligned'] * 100:8.2f}%"
            f"{r['random_mean'] * 100:8.2f}%{r['aligned_minus_random'] * 100:+8.2f} "
            f"{r['aligned_int4'] * 100:7.2f}%{r['random_int4_mean'] * 100:8.2f}%")

    payload = {
        "experiment": "exp12_aligned_pruning",
        "model": P.MODEL_ID, "dataset": P.DATASET_ID,
        "seed": args.seed, "mask_ratio": args.mask_ratio, "epochs": args.epochs,
        "n_train": train_split[0].shape[0], "n_val": val_split[0].shape[0],
        "draws": args.draws, "smoke": args.smoke,
        "device": str(device), "vram_gb": round(vram, 1),
        "num_layers": nl, "num_heads_per_layer": nh,
        "sal_pruned_heads": pruned_by_sal,
        "sal_pruned_count": n_pruned,
        "sal_pruned_per_layer": [per_layer.get(i, 0) for i in range(nl)],
        "results": results,
        "caveats": {
            "masking": "aligned and random cells are both applied by masking, not "
                       "slicing: SAL's pruned set is not uniform per layer so "
                       "slice_heads() cannot express it. Masking is accuracy-"
                       "equivalent to slicing the same heads and makes the two "
                       "cells differ only in which heads are removed.",
            "int4": "per-output-channel symmetric weight round-trip on every "
                    "nn.Linear except the classifier, as in the prototype.",
            "drophead_aligned": "drophead never had a pruned set of its own; it is "
                                "scored on SAL's set so every arm faces the same "
                                "perturbation.",
        },
        "wall_clock_s": round(time.time() - t_all, 1),
    }
    dest = outdir / "exp12_aligned.json"
    dest.write_text(json.dumps(payload, indent=2))
    log(f"  results -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
