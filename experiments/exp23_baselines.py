#!/usr/bin/env python
"""exp23 — the pruning baselines a reviewer will ask for.

SAL's claim is comparative, so it is only as strong as what it is compared with.
exp04/exp14/1a compare SAL against a baseline *trained the same way and then
pruned*, which is the right control for "does SAL-style training help". It is
not the control for "is SAL better than just pruning well", and that is the
question a reviewer asks.

Three arms, on ViT-B/16 / ImageNet-1k (resized 256px,
evanarlian/imagenet_1k_resized_256):

``random``              uniform random heads removed from the pretrained model,
                        no training at all. The floor.
``magnitude``           the lowest-L2 heads removed, no recovery. The naive
                        baseline from the pruning literature.
``magnitude_recovery``  the same, then fine-tuned for ``--recover-epochs``. The
                        strong baseline: most of what post-hoc pruning papers
                        report comes from this step.

Compared at 33% and 50% against SAL and DropHead from the 1a runs, which are
read from their JSON when ``--compare`` points at them.

**Two protocol decisions worth knowing.**

1. **Magnitude selection is per-layer, with the same quota as the random draw.**
   The published naive baseline ranks heads globally, which gives an uneven
   per-layer count and is therefore *not* a valid ``slice_heads()`` argument.
   Since 1a's pruned cells are physically sliced, a globally-selected set would
   be compared at a different size, and size is the thing being held fixed.
   ``magnitude_global`` is reported alongside, evaluated by masking, so the cost
   of the constraint is visible rather than assumed away.

2. **``google/vit-base-patch16-224`` is already trained on ImageNet-1k.** With
   ``num_labels=1000`` its classifier is not reinitialised, so the "no training"
   arms start from a converged 80%-accurate model. That is the realistic setting
   -- someone holding the official checkpoint who wants it prunable -- and it is
   what makes a zero-training baseline meaningful at all.

Usage::

    python exp23_baselines.py --data-cache ... --seed 42 --output results/exp23/seed42
    python exp23_baselines.py --smoke
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
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DATASET = "evanarlian/imagenet_1k_resized_256"
RATIOS = (0.33, 0.50)


def head_magnitudes(model):
    """``{(layer, head): L2 norm of its slice of the output projection}``.

    A head's weight is the block of the attention output projection that reads
    it, so its norm is how loudly that head speaks into the residual stream.

    This delegates to ``sal.compare._head_magnitudes`` rather than carrying its
    own copy of the arithmetic. The body here used to be a near-verbatim
    duplicate, which meant two implementations of the same score that could
    drift apart silently -- and exp23, exp24 and exp27 all rank heads through
    this function, so a drift would move their tables and nothing would fail.
    Depending on a private name is the lesser risk: an experiment may reach into
    the package it is measuring, and if the private signature changes the import
    breaks loudly at startup instead of returning different numbers.

    The wrapper exists because the signatures differ: ``_head_magnitudes`` takes
    the head count and hidden size as arguments, and every caller here has only
    the model.
    """
    from sal.compare import _head_magnitudes

    cfg = model.config
    return _head_magnitudes(model, cfg.num_attention_heads, cfg.hidden_size)


def magnitude_heads(model, ratio: float, per_layer: bool = True):
    """The heads to remove: lowest magnitude, with a uniform quota per layer.

    ``per_layer=False`` ranks globally, which is the published method and is not
    sliceable -- see the module docstring.
    """
    cfg = model.config
    mags = head_magnitudes(model)
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    if not per_layer:
        total = int(round(ratio * n_layers * n_heads))
        return [hh for hh, _ in sorted(mags.items(), key=lambda kv: kv[1])[:total]]
    # The same quota the uniform random draw uses, so the two are the same size.
    per = int(round(ratio * n_heads))
    picked = []
    for li in range(n_layers):
        row = sorted(((h, mags[(li, h)]) for h in range(n_heads)), key=lambda kv: kv[1])
        picked += [(li, h) for h, _ in row[:per]]
    return picked


def masked_accuracy(model, pairs, val_split, device, use_amp):
    """Accuracy with ``pairs`` masked rather than sliced.

    Needed for the global-magnitude row, whose per-layer counts are uneven and
    therefore cannot be sliced. Masking and slicing agree to <=0.16pp where both
    are possible (PHASE2G), so the row is comparable to the sliced ones; it is
    labelled as masked regardless.
    """
    from sal import SALConfig
    from sal.masker import HeadMasker

    cfg = SALConfig.auto(model, prune_fraction=0.5)
    masker = HeadMasker(model, cfg, seed=0)
    masker.install()
    try:
        masker.activate()
        for layer, head in pairs:
            masker._masks[layer][head] = 0.0
        masker.apply_mask()
        import run_sal_qat_prototype as P
        return P.accuracy(model, val_split, device, use_amp)
    finally:
        masker.remove()


def recover(model, train_split, epochs, batch_size, seed, device, use_amp,
            label="recovery"):
    """Fine-tune a pruned model, with the SAL arms' exact training loop.

    Recovery fine-tuning is where most of the accuracy in a post-hoc pruning
    paper comes from, so giving it a weaker optimiser than the SAL arms would
    flatter SAL by precisely that margin. This reuses the prototype's
    ``make_step`` (bfloat16 autocast, horizontal flip, grad clip at 1.0) and
    ``make_optimizer`` (AdamW, 10x LR on the classifier, linear warmup then
    cosine decay) -- the same functions ``train_variant`` calls, driven by the
    same no-masking loop its ``mask_mode="none"`` branch uses.
    """
    import run_sal_qat_prototype as P

    model.to(device).train()
    loader = P.batches(train_split, batch_size, shuffle=True, seed=seed,
                       workers=P.LOADER_WORKERS, block=P.LOADER_BLOCK)
    total = len(loader) * epochs
    optimizer, scheduler = P.make_optimizer(model, total)
    prog = P.Progress(total, label)
    step = P.make_step(device, use_amp, prog)
    # make_step's progress line reads mask_module.stats["pruned_heads"]; the
    # prototype's own unmasked branch passes this same stand-in.
    no_mask = P._NoMask()
    losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for batch in loader:
            total_loss += step(model, batch, optimizer, no_mask)
            scheduler.step()
        losses.append(total_loss / max(len(loader), 1))
    prog.done()
    model.eval()
    return losses


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--prune-seeds", type=int, default=20)
    p.add_argument("--recover-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
    p.add_argument("--no-recovery", action="store_true",
                   help="skip the expensive arm (b); (a) and (c) are eval-only")
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--compare", default=None,
                   help="a 1a results directory to read sal/drophead numbers from")
    p.add_argument("--output", default="results/exp23")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        args.n_train, args.n_val = 2000, 1000
        args.prune_seeds, args.recover_epochs = 2, 1
        args.ratios, args.skip_int4 = [0.5], True

    import run_sal_qat_prototype as P

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    P.rule("exp23 — pruning baselines")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    P.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    P.log(f"  dataset: {P.dataset_label(args.dataset)}")
    P.log(f"  seed {args.seed} | ratios {args.ratios} | "
          f"{args.prune_seeds} random draws | recovery {args.recover_epochs} epoch(s)")

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
    results: dict = {}
    t0 = time.time()

    # ----------------------------------------------------------------- clean
    P.rule("pretrained, dense")
    clean = P.accuracy(pretrained, val_split, device, use_amp)
    results["dense"] = {"accuracy": clean, "trained": False}
    P.log(f"    dense FP16   {clean * 100:6.2f}%  (no training of any kind)")

    for ratio in args.ratios:
        tag = f"prune{int(round(ratio * 100))}"
        P.rule(f"ratio {ratio:.0%}")
        cell: dict = {}

        # --- (c) the floor: random heads, no training
        accs = []
        for s in range(args.prune_seeds):
            small = P.sliced(copy.deepcopy(pretrained), ratio, 9000 + s)
            accs.append(P.accuracy(small.to(device), val_split, device, use_amp))
            del small
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        cell["random"] = {"accuracy": mean, "std": sd, "per_seed": accs,
                          "method": "uniform random heads, sliced, no training"}
        P.log(f"    random        {mean * 100:6.2f}% ± {sd * 100:.2f}  "
              f"({args.prune_seeds} draws)")

        # --- (a) the naive baseline: lowest-magnitude heads, no recovery
        pairs = magnitude_heads(pretrained, ratio, per_layer=True)
        from sal import slice_heads
        small = slice_heads(copy.deepcopy(pretrained), pairs).eval().to(device)
        acc = P.accuracy(small, val_split, device, use_amp)
        cell["magnitude"] = {"accuracy": acc, "heads": len(pairs),
                             "method": "lowest-L2 heads, uniform per-layer quota, "
                                       "sliced, no training"}
        P.log(f"    magnitude     {acc * 100:6.2f}%  (deterministic, {len(pairs)} heads)")

        # --- (a') the published variant: global ranking, masked
        gpairs = magnitude_heads(pretrained, ratio, per_layer=False)
        gacc = masked_accuracy(copy.deepcopy(pretrained).to(device), gpairs,
                               val_split, device, use_amp)
        per_layer_counts: dict[int, int] = {}
        for layer, _ in gpairs:
            per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1
        cell["magnitude_global"] = {
            "accuracy": gacc, "heads": len(gpairs),
            "per_layer_counts": per_layer_counts,
            "method": "lowest-L2 heads ranked globally, MASKED (uneven per-layer "
                      "counts are not sliceable)"}
        P.log(f"    magnitude[G]  {gacc * 100:6.2f}%  (masked, per-layer counts "
              f"{sorted(per_layer_counts.values())})")

        # --- (b) the strong baseline: magnitude + recovery fine-tuning
        if not args.no_recovery:
            P.set_seed(args.seed)
            small = slice_heads(copy.deepcopy(pretrained), pairs).eval()
            losses = recover(small, train_split, args.recover_epochs,
                             args.batch_size, args.seed, device, use_amp,
                             label=f"recovery p{int(round(ratio * 100))}")
            racc = P.accuracy(small, val_split, device, use_amp)
            entry = {"accuracy": racc, "epoch_losses": losses,
                     "epochs": args.recover_epochs,
                     "method": f"magnitude, sliced, then {args.recover_epochs} "
                               "epochs of AdamW fine-tuning"}
            if not args.skip_int4:
                entry["int4_accuracy"] = P.accuracy(
                    P.quantize_weights(copy.deepcopy(small), 4), val_split,
                    device, use_amp)
            cell["magnitude_recovery"] = entry
            P.log(f"    mag+recovery  {racc * 100:6.2f}%  "
                  f"(+{(racc - acc) * 100:.2f}pp from recovery)")
            del small

        results[tag] = cell

    payload = {
        "experiment": "exp23_baselines",
        "model": P.MODEL_ID, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "seed": args.seed, "ratios": args.ratios,
        "prune_seeds": args.prune_seeds, "recover_epochs": args.recover_epochs,
        "n_train": int(train_split[0].shape[0]), "n_val": int(val_split[0].shape[0]),
        "num_classes": num_classes, "smoke": args.smoke,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "results": results,
        "caveats": {
            "already_trained": "google/vit-base-patch16-224 ships a 1000-class "
                               "classifier, so with num_labels=1000 nothing is "
                               "reinitialised and the 'no training' arms start "
                               "from a converged model at "
                               f"{clean * 100:.2f}% top-1.",
            "magnitude_quota": "The sliced magnitude arm uses a uniform per-layer "
                               "quota, because slice_heads() requires one and the "
                               "comparison holds size fixed. magnitude_global is "
                               "the published global ranking, evaluated by masking.",
            "recovery_is_the_strong_arm": "Most of what post-hoc pruning papers "
                                          "report comes from recovery fine-tuning, "
                                          "so it gets the same optimiser, schedule "
                                          "and data order as the SAL arms.",
        },
        "wall_clock_s": round(time.time() - t0, 1),
    }
    # Measurements before presentation: a formatting bug must never destroy a run.
    dest = outdir / "exp23_baselines.json"
    dest.write_text(json.dumps(payload, indent=2))
    P.log(f"\n  metrics -> {dest}")

    try:
        P.rule("Summary")
        P.log(f"  dense (no training): {clean * 100:.2f}%")
        P.log(f"  {'ratio':<8}{'random':>18}{'magnitude':>12}{'mag[global]':>13}"
              f"{'mag+recovery':>14}")
        for ratio in args.ratios:
            c = results[f"prune{int(round(ratio * 100))}"]
            rec = c.get("magnitude_recovery", {}).get("accuracy")
            P.log(f"  {ratio:<8.0%}"
                  f"{c['random']['accuracy'] * 100:12.2f}% ± {c['random']['std'] * 100:4.2f}"
                  f"{c['magnitude']['accuracy'] * 100:11.2f}%"
                  f"{c['magnitude_global']['accuracy'] * 100:12.2f}%"
                  + (f"{rec * 100:13.2f}%" if rec is not None else f"{'-':>14}"))
        P.log(f"\n  wall clock {(time.time() - t0) / 60:.1f} min")
    except Exception as exc:    # presentation only; the metrics are already safe
        P.log(f"  WARNING: summary failed ({type(exc).__name__}: {exc})")
        P.log(f"  the measurements are intact in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
