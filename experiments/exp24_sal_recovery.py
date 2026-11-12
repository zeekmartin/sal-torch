#!/usr/bin/env python
"""exp24 — does SAL training make recovery fine-tuning better?

exp23 settled the comparison SAL was avoiding: post-hoc magnitude pruning
followed by five epochs of recovery fine-tuning reaches **75.1% at P50+INT4**
on ImageNet-1k, where SAL alone reaches 30.6%. SAL wins its own control by
+14.6pp and loses to the strong baseline by 44.5pp, so as a *substitute* for
recovery it is not competitive.

That leaves one position worth testing. SAL might be a **pre-conditioner**: a
model trained to operate without heads could recover faster, further, or more
reliably than one that was not. This experiment asks that, and separates it
from SAL's head *selection*, which is a different claim.

Four arms, ViT-B/16 on ImageNet-1k (resized 256px,
evanarlian/imagenet_1k_resized_256), each sliced then fine-tuned for
``--recover-epochs``:

``magnitude``           pretrained -> magnitude set. **Not run here** -- this is
                        exp23's ``magnitude_recovery``, read from its JSON with
                        ``--compare`` so nothing is retrained.
``sal_aligned``         the 1a SAL checkpoint -> SAL's own pruned set.
``sal_magnitude``       the 1a SAL checkpoint -> magnitude set.
``drophead_magnitude``  the 1a DropHead checkpoint -> magnitude set.

The three comparisons the arms exist for::

    sal_aligned        vs magnitude   does SAL help recovery?
    sal_magnitude      vs magnitude   does it help on a set SAL never saw?
    drophead_magnitude vs magnitude   does the control help too?

``sal_aligned`` minus ``sal_magnitude`` isolates the selection from the
pre-conditioning: if SAL helps only on its own set, the mechanism is which
heads it chose, not what it taught the weights.

**Three protocol notes.**

1. **No random draws here, unlike 1a.** 1a's pruned cells average 20 random
   head selections because it prunes a *dense* model and the draw is free. Every
   arm here is a single sliced architecture that was then fine-tuned, so a draw
   would mean a separate recovery run -- 20x the cost. Each arm reports one
   deterministic number, which is exactly what exp23's ``magnitude_recovery``
   reports, so the comparison is like-for-like. The spread across the three
   seeds is the only variability quoted.

2. **An aligned set below 50% is a subset of SAL's set.** The 1a checkpoints
   trained at ``--mask-ratio 0.5``, so SAL has no 33% set of its own. For a
   lower ratio this takes the lowest-magnitude heads *from among the ones SAL
   was trained without*, keeping the per-layer quota. That is the only coherent
   reading of "aligned at 33%", and it is flagged as a subset in the output.

3. **DropHead has no aligned set, by design.** It resamples every step, so the
   final mask is the last draw rather than a set the model learned to live
   without; ``run_sal_qat_prototype`` records ``pruned_heads=None`` for it
   deliberately. Hence only a magnitude arm for DropHead.

Usage::

    python exp24_sal_recovery.py --data-cache ... --seed 42 \
        --checkpoints results/in1k_a/seed42/models \
        --sal-json results/in1k_a/seed42/sal_qat_prototype.json \
        --compare results/exp23/seed42/exp23_baselines.json \
        --output results/exp24/seed42
    python exp24_sal_recovery.py --smoke
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

from exp23_baselines import head_magnitudes, magnitude_heads, recover  # noqa: E402

DATASET = "evanarlian/imagenet_1k_resized_256"
RATIOS = (0.33, 0.50)
ARMS = ("sal_aligned", "sal_magnitude", "drophead_magnitude")
CKPT_FOR = {"sal_aligned": "sal.pt", "sal_magnitude": "sal.pt",
            "drophead_magnitude": "drophead.pt"}


def aligned_heads(model, pruned_heads, ratio: float):
    """SAL's own pruned set, narrowed to ``ratio`` if SAL removed more.

    ``pruned_heads`` comes from 1a's JSON as a list of ``[layer, head]`` pairs.
    When the requested ratio needs fewer heads per layer than SAL removed, the
    lowest-magnitude ones among SAL's set are kept -- see note 2 in the module
    docstring. Raises when SAL's set is too small, rather than quietly padding
    it with heads SAL was never trained without, which would make the arm a
    different experiment under the same name.
    """
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    per = int(round(ratio * n_heads))
    by_layer: dict[int, list[int]] = {}
    for layer, head in pruned_heads:
        by_layer.setdefault(int(layer), []).append(int(head))

    mags = head_magnitudes(model)
    picked, subset = [], False
    for li in range(n_layers):
        row = by_layer.get(li, [])
        if len(row) < per:
            raise ValueError(
                f"layer {li}: SAL removed {len(row)} heads, a {ratio:.0%} quota "
                f"needs {per}. The aligned arm cannot run above SAL's own "
                "training ratio.")
        if len(row) > per:
            subset = True
            row = [h for h, _ in sorted(((h, mags[(li, h)]) for h in row),
                                        key=lambda kv: kv[1])][:per]
        picked += [(li, h) for h in sorted(row)]
    return picked, subset


def load_checkpoint(path: Path, num_classes: int):
    """A 1a checkpoint as a dense model, ready to be sliced.

    SAL masks through forward hooks and never removes weights, so the saved
    state dict is the full architecture -- which is why it can be sliced any way
    afterwards. Loaded strictly: a partially matched checkpoint would train and
    score perfectly happily while measuring something else.
    """
    import run_sal_qat_prototype as P

    model = P.load_model(num_classes)
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path.name}: state dict does not match the model "
            f"(missing {len(missing)}, unexpected {len(unexpected)}). "
            "Refusing to evaluate a partially loaded checkpoint.")
    return model.eval()


def run_arm(name, ckpt_path, selection, ratio, train_split, val_split, args,
            device, use_amp, num_classes, pruned_heads=None):
    """Slice one checkpoint, fine-tune it, and score it at FP32 and INT4."""
    import run_sal_qat_prototype as P
    from sal import slice_heads

    tag = f"p{int(round(ratio * 100))}"
    model = load_checkpoint(ckpt_path, num_classes).to(device)
    dense = P.accuracy(model, val_split, device, use_amp)

    subset = False
    if selection == "aligned":
        pairs, subset = aligned_heads(model, pruned_heads, ratio)
        how = ("SAL's own pruned set" if not subset
               else "lowest-magnitude subset of SAL's pruned set")
    else:
        pairs = magnitude_heads(model, ratio, per_layer=True)
        how = "lowest-L2 heads, uniform per-layer quota"

    P.set_seed(args.seed)
    small = slice_heads(copy.deepcopy(model), pairs).eval().to(device)
    before = P.accuracy(small, val_split, device, use_amp)
    losses = recover(small, train_split, args.recover_epochs, args.batch_size,
                     args.seed, device, use_amp, label=f"{name} {tag}")
    after = P.accuracy(small, val_split, device, use_amp)

    entry = {"accuracy": after, "accuracy_before_recovery": before,
             "dense_accuracy": dense, "epoch_losses": losses,
             "epochs": args.recover_epochs, "heads": len(pairs),
             "head_set": sorted((int(a), int(b)) for a, b in pairs),
             "aligned_is_subset": subset,
             "checkpoint": str(ckpt_path),
             "method": f"{ckpt_path.stem} checkpoint, {how}, sliced, then "
                       f"{args.recover_epochs} epochs of AdamW fine-tuning"}
    if not args.skip_int4:
        entry["int4_accuracy"] = P.accuracy(
            P.quantize_weights(copy.deepcopy(small), 4), val_split, device, use_amp)
    P.log(f"    {name:20s} {after * 100:6.2f}%  "
          f"(sliced {before * 100:.2f}% -> +{(after - before) * 100:.2f}pp, "
          f"dense was {dense * 100:.2f}%)"
          + (f", INT4 {entry['int4_accuracy'] * 100:.2f}%"
             if "int4_accuracy" in entry else ""))
    del model, small
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--checkpoints", default=None,
                   help="directory holding sal.pt / drophead.pt from a 1a run")
    p.add_argument("--sal-json", default=None,
                   help="1a JSON to read SAL's pruned head set from "
                        "(required for the sal_aligned arm)")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    # exp29 measured the curve: it peaks at 2 epochs and declines, with 5 below
    # 2 on 3/3 seeds in FP and INT4 and 10 below 1. Two is the default because
    # it is both better and 39% of the cost. **The published cells in this
    # file's own results were measured at 5** -- pass --recover-epochs 5 to
    # reproduce them, and do not compare a 2-epoch re-run against them.
    p.add_argument("--recover-epochs", type=int, default=2,
                   help="default 2 (exp29: the curve peaks there); the "
                        "published exp23/exp24/exp26 cells used 5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--compare", default=None,
                   help="exp23 JSON, for the magnitude+recovery arm")
    p.add_argument("--output", default="results/exp24")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_train, args.n_val = 2000, 1000
        args.recover_epochs, args.ratios = 1, [0.5]
        args.skip_int4 = True

    import run_sal_qat_prototype as P

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp24 — does SAL pre-condition recovery?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | ratios {args.ratios} | arms {args.arms} | "
          f"recovery {args.recover_epochs} epoch(s)")

    ckdir = Path(args.checkpoints) if args.checkpoints else None
    pruned_heads = None
    # Check every input before spending a single GPU-hour: an arm that dies on a
    # missing file after five epochs of fine-tuning costs six hours.
    if ckdir is None:
        p.error("--checkpoints is required")
    for arm in args.arms:
        f = ckdir / CKPT_FOR[arm]
        if not f.exists():
            p.error(f"arm {arm} needs {f}, which does not exist")
    if "sal_aligned" in args.arms:
        if not args.sal_json:
            p.error("the sal_aligned arm needs --sal-json")
        meta = json.loads(Path(args.sal_json).read_text(encoding="utf-8"))
        pruned_heads = ((meta.get("training") or {}).get("sal") or {}).get(
            "pruned_heads")
        if not pruned_heads:
            p.error(f"{args.sal_json} has no training.sal.pruned_heads; "
                    "the aligned arm has nothing to align to")
        P.log(f"  SAL's pruned set: {len(pruned_heads)} heads")

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    train_split, val_split, num_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    P.log(f"  {train_split[0].shape[0]} train / {val_split[0].shape[0]} val, "
          f"{num_classes} classes")

    baseline_rows = {}
    if args.compare:
        prev = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        for tag, cell in (prev.get("results") or {}).items():
            if isinstance(cell, dict) and "magnitude_recovery" in cell:
                baseline_rows[tag] = cell["magnitude_recovery"]
        P.log(f"  reusing exp23 magnitude+recovery for {sorted(baseline_rows)}")

    results: dict = {}
    t0 = time.time()
    for ratio in args.ratios:
        tag = f"prune{int(round(ratio * 100))}"
        P.rule(f"ratio {ratio:.0%}")
        cell: dict = {}
        if tag in baseline_rows:
            cell["magnitude"] = dict(baseline_rows[tag], reused_from=args.compare)
            b = cell["magnitude"]
            P.log(f"    {'magnitude (exp23)':20s} {b['accuracy'] * 100:6.2f}%"
                  + (f", INT4 {b['int4_accuracy'] * 100:.2f}%"
                     if "int4_accuracy" in b else ""))
        for arm in args.arms:
            sel = "aligned" if arm == "sal_aligned" else "magnitude"
            cell[arm] = run_arm(arm, ckdir / CKPT_FOR[arm], sel, ratio,
                                train_split, val_split, args, device, use_amp,
                                num_classes, pruned_heads=pruned_heads)
        results[tag] = cell

    payload = {
        "experiment": "exp24_sal_recovery",
        "model": P.MODEL_ID, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "seed": args.seed, "ratios": args.ratios, "arms": args.arms,
        "recover_epochs": args.recover_epochs,
        "n_train": int(train_split[0].shape[0]), "n_val": int(val_split[0].shape[0]),
        "num_classes": num_classes, "smoke": args.smoke,
        "checkpoints": str(ckdir) if ckdir else None,
        "compare": args.compare,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": results,
        "caveats": {
            "no_random_draws": "Every arm is one sliced architecture that was "
                               "then fine-tuned, so each reports a single "
                               "deterministic accuracy -- the same protocol as "
                               "exp23's magnitude_recovery, which it is compared "
                               "with. 1a's 20 random draws prune a dense model, "
                               "where a draw costs nothing; here a draw would "
                               "mean a separate recovery run.",
            "aligned_subset": "The 1a checkpoints trained at mask_ratio 0.5, so "
                              "an aligned set below 50% is the lowest-magnitude "
                              "subset of SAL's own set, flagged per arm as "
                              "aligned_is_subset.",
            "drophead_has_no_aligned_set": "DropHead resamples every step, so "
                                           "there is no set it was trained to "
                                           "live without; only a magnitude arm "
                                           "is defined for it.",
            "magnitude_arm_is_reused": "The magnitude+recovery row is read from "
                                       "exp23 rather than retrained, so it is "
                                       "the identical number, not a replication.",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    out = outdir / "exp24_sal_recovery.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    P.log(f"\n  metrics -> {out}")

    try:
        P.rule("Summary")
        P.log(f"  dataset: {P.dataset_label(args.dataset)}")
        names = (["magnitude"] if baseline_rows else []) + list(args.arms)
        P.log("  ratio  " + "".join(f"{n[:18]:>20s}" for n in names))
        for ratio in args.ratios:
            cell = results[f"prune{int(round(ratio * 100))}"]
            row = "".join(
                (f"{cell[n]['accuracy'] * 100:19.2f}%" if n in cell else f"{'-':>20s}")
                for n in names)
            P.log(f"  {ratio:5.0%}  " + row)
            if not args.skip_int4:
                row4 = "".join(
                    (f"{cell[n].get('int4_accuracy', float('nan')) * 100:19.2f}%"
                     if n in cell else f"{'-':>20s}") for n in names)
                P.log("   +INT4 " + row4)
        P.log(f"\n  wall clock {(time.time() - t0) / 60:.1f} min")
    except Exception as exc:                                  # noqa: BLE001
        P.log(f"  (summary table failed: {exc!r}; the JSON above is complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
