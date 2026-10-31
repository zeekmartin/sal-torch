#!/usr/bin/env python3
"""exp15 — does the v0.6.0 result hold on a 632M self-supervised encoder?

exp14 established the deployment claim on ViT-B/16 / ImageNet-100: with a
uniform-per-layer draw, a SAL model pruned on *its own* trained head set is
16.5% smaller and scores +1.4pp over the best alternative of the same size, on
8-10 of 10 seeds (`results/lumi_v060/PHASE2F.md`). The README records the matching
limitation: that path has been measured on **one architecture**, supervised, 86M
parameters.

This runs the same comparison on I-JEPA ViT-H/14 — 632M parameters, 32 layers x
16 heads, trained self-supervised, evaluated by linear probe rather than by its
own classifier head. Three arms, all trained identically apart from masking:

* ``control``   — no masking at all.
* ``sal``       — the accumulating schedule, held to the end, with
                  ``uniform_per_layer=True`` so the recorded set is a valid
                  ``slice_heads()`` argument.
* ``drophead``  — the same hook and ratio, resampled every step, no accumulation.

Every arm is then scored under perturbations of **identical size**:

* ``clean``            — nothing removed.
* ``aligned``          — exactly the heads SAL killed, masked.
* ``random``           — uniform-per-layer random sets of the same size, masked,
                         averaged over ``--draws`` draws shared across arms.
* ``sliced_aligned``   — the aligned set removed for real by ``slice_heads()``.
* ``sliced_random``    — a subset of the random draws, sliced.

each of those also with INT4 weights, using the prototype's quantizer so the
numbers are comparable to every other v0.6.0 cell.

The control that decides the interpretation is ``control`` on ``aligned``: if an
unmasked-trained model also loses little from SAL's set, the set is just made of
unimportant heads and this is selection, not adaptation. On ViT-B/16 it was not.

Usage::

    python exp15_ijepa_aligned.py --seed 42 --mask-ratio 0.5 --output results/exp15
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
JEPA = HERE.parent / "scripts" / "run_jepa_sal.py"
PROTOTYPE = HERE.parent / "scripts" / "run_sal_qat_prototype.py"

ARMS = ("control", "sal", "drophead")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg=""):
    print(msg, flush=True)


class MaskSet:
    """Mask an explicit set of ``(layer, head)`` pairs for the duration of a block."""

    def __init__(self, model, heads, cfg):
        from sal.masker import HeadMasker
        self.masker = HeadMasker(model, cfg, seed=0)
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


def uniform_random_heads(num_layers, num_heads, count, seed):
    """``count`` pairs spread equally over the layers.

    The aligned set is uniform per layer by construction, so the random
    comparison must be too — otherwise the two cells differ in the *shape* of
    the perturbation and not only in which heads it hits.
    """
    rng = random.Random(seed)
    per, extra = divmod(count, num_layers)
    bonus = set(rng.sample(range(num_layers), extra))
    out = []
    for layer in range(num_layers):
        k = min(num_heads, per + (1 if layer in bonus else 0))
        out += [(layer, h) for h in sorted(rng.sample(range(num_heads), k))]
    return sorted(out)


def train_arm(arm, J, device, grad_ckpt, loaders, epochs, mask_ratio,
              seed, uniform, use_amp):
    """Train one arm and return ``(model, info)``.

    ``make_step`` and the data come from ``run_jepa_sal.py``, so the three arms
    differ only in how heads are masked during training. The control arm goes
    through ``train_control`` rather than a null ratio: ``num_heads_to_prune``
    floors at 1, so "nearly off" is not off.
    """
    from sal import SALConfig
    from sal.masker import HeadMasker
    from sal.trainer import SALTrainer

    torch.manual_seed(seed)
    random.seed(seed)
    model = J.load_model(device, grad_ckpt)
    model.train()
    loader = loaders["train"]
    patch_size = model.config.patch_size
    t0 = time.time()
    pruned_heads, stats = None, None

    if arm == "control":
        losses = J.train_control(model, loader, epochs, patch_size, device,
                                 use_amp)["losses"]
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=J.LR)
        cfg = SALConfig.auto(model, prune_fraction=mask_ratio,
                             uniform_per_layer=uniform)
        step = J.make_step(patch_size, use_amp)
        if arm == "sal":
            # SALTrainer owns the accumulating schedule and moves batches itself.
            trainer = SALTrainer(model, cfg, optimizer, loader, seed=seed,
                                 train_step=step)
            hist = trainer.train(num_epochs=epochs)
            losses = hist["losses"]
            stats = hist.get("masker_stats")
            pruned_heads = hist.get("pruned_heads")
        else:
            masker = HeadMasker(model, cfg, seed=seed)
            masker.install()
            losses = []
            try:
                for epoch in range(epochs):
                    total = n = 0.0
                    for batch in loader:
                        batch = [b.to(device) if torch.is_tensor(b) else b
                                 for b in batch]
                        # Resample the whole draw every step: clear, then redraw.
                        for m in masker._masks.values():
                            m.fill_(1.0)
                        masker._active = True
                        masker._deactivate_random_heads(cfg.num_heads_to_prune)
                        total += step(model, batch, optimizer, masker)
                        n += 1
                    losses.append(total / max(n, 1))
                    log(f"    epoch {epoch + 1} mean loss {losses[-1]:.6f}")
                stats = dict(masker.stats)
            finally:
                masker.remove()

    model.eval()
    info = {"losses": losses, "masker_stats": stats, "pruned_heads": pruned_heads,
            "train_seconds": round(time.time() - t0, 1)}
    log(f"  {arm}: losses {[round(x, 4) for x in losses]} "
        f"| {info['train_seconds'] / 60:.1f} min"
        + (f" | masker {stats}" if stats else ""))
    return model, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--draws", type=int, default=5,
                   help="random head sets to average the 'random' cell over")
    p.add_argument("--sliced-draws", type=int, default=2,
                   help="of those, how many to also slice (each rebuilds a 632M model)")
    p.add_argument("--n-train", type=int, default=None,
                   help="override the pretext-training set size (for cheap gating)")
    p.add_argument("--n-probe", type=int, default=None,
                   help="override the probe-train size; probe-val is half of it")
    p.add_argument("--no-uniform", action="store_true",
                   help="use the pre-v0.6.0 global draw (then nothing can be sliced)")
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--output", default="results/exp15_ijepa")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    J = load_module(JEPA, "run_jepa_sal")
    P = load_module(PROTOTYPE, "sal_qat_prototype")
    from sal import SALConfig, count_params, linear_probe

    uniform = not args.no_uniform
    sizes = dict(n_train=J.N_TRAIN, n_probe_train=J.N_PROBE_TRAIN,
                 n_probe_val=J.N_PROBE_VAL, n_fi=J.N_FI_PROBE,
                 latency_runs=J.LATENCY_RUNS, epochs=J.DEFAULT_EPOCHS)
    if args.smoke:
        sizes.update(J.SMOKE)
        args.draws, args.sliced_draws = 2, 1
    if args.epochs is not None:
        sizes["epochs"] = args.epochs
    # Explicit sizes beat the smoke defaults, so `--smoke --n-train 2048` does
    # what it says instead of quietly running 100 images.
    if args.n_train is not None:
        sizes["n_train"] = args.n_train
    if args.n_probe is not None:
        sizes["n_probe_train"] = args.n_probe
        sizes["n_probe_val"] = max(1, args.n_probe // 2)
    epochs = sizes["epochs"]

    device, batch_size, grad_ckpt, vram = J.detect_device(args.batch_size)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"

    J.rule("exp15 — I-JEPA ViT-H/14, aligned vs random pruning")
    log(f"  device {device} ({vram:.1f} GB) | batch {batch_size} | "
        f"grad_ckpt {grad_ckpt}")
    log(f"  seed {args.seed} | mask_ratio {args.mask_ratio} | uniform {uniform} | "
        f"{epochs} epochs | draws {args.draws} (+{args.sliced_draws} sliced)")

    loaders, fi_batches, num_classes = J.build_data(sizes, batch_size, args.seed)
    log(f"  {num_classes} classes | probe {sizes['n_probe_train']}/"
        f"{sizes['n_probe_val']} | train {sizes['n_train']}")

    t_all = time.time()
    arms, infos = {}, {}
    for arm in ARMS:
        J.rule(f"training {arm}")
        model, info = train_arm(arm, J, device, grad_ckpt, loaders, epochs,
                                args.mask_ratio, args.seed, uniform, use_amp)
        arms[arm], infos[arm] = model, info

    aligned = infos["sal"]["pruned_heads"]
    if not aligned:
        log("ERROR: SALTrainer did not report pruned_heads — is sal up to date?")
        return 1

    cfg = SALConfig.auto(arms["control"], prune_fraction=args.mask_ratio,
                         uniform_per_layer=uniform)
    nl, nh = cfg.num_layers, cfg.num_heads_per_layer
    per_layer = {}
    for layer, _ in aligned:
        per_layer[layer] = per_layer.get(layer, 0) + 1
    counts = [per_layer.get(i, 0) for i in range(nl)]
    is_uniform = len(set(counts)) == 1 and len(per_layer) == nl

    J.rule("SAL's pruned set")
    log(f"  {len(aligned)} of {nl * nh} heads ({len(aligned) / (nl * nh):.1%}); "
        f"config target {cfg.num_heads_to_prune}, effective fraction "
        f"{cfg.effective_prune_fraction:.3f}")
    log(f"  per layer: {counts}")
    log(f"  uniform per layer: {is_uniform}")

    draws = [uniform_random_heads(nl, nh, len(aligned),
                                  10_000 + args.seed * 100 + i)
             for i in range(args.draws)]

    def probe(model):
        return linear_probe(model, loaders["probe_train"], loaders["probe_val"],
                            num_classes=num_classes)

    def sliced_probe(model, heads):
        """Slice ``heads`` out for real, probe, and report the size change."""
        if not is_uniform:
            return None
        from sal import slice_heads
        before = count_params(model)
        small = slice_heads(model, heads).to(device).eval()
        acc = probe(small)
        after = count_params(small)
        del small
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"accuracy": acc, "params_before": before, "params_after": after,
                "param_reduction": 1.0 - after / before}

    results = {}
    for arm in ARMS:
        model = arms[arm]
        J.rule(f"scoring {arm}")
        row = {"clean": probe(model)}
        with MaskSet(model, aligned, cfg):
            row["aligned"] = probe(model)
        rnd = []
        for heads in draws:
            with MaskSet(model, heads, cfg):
                rnd.append(probe(model))
        row["random_per_draw"] = rnd
        row["random_mean"] = statistics.fmean(rnd)
        row["random_std"] = statistics.stdev(rnd) if len(rnd) > 1 else None
        row["aligned_minus_random"] = row["aligned"] - row["random_mean"]
        row["sliced_aligned"] = sliced_probe(model, aligned)
        sr = [sliced_probe(model, h) for h in draws[:args.sliced_draws]]
        sr = [x for x in sr if x]
        row["sliced_random_mean"] = (statistics.fmean(x["accuracy"] for x in sr)
                                     if sr else None)

        if not args.skip_int4:
            # The INT4 copy is built outside every mask context: deepcopy would
            # otherwise duplicate live pre-hooks whose closures still point at
            # the original masker's tensors.
            q = P.quantize_weights(copy.deepcopy(model), 4)
            row["clean_int4"] = probe(q)
            with MaskSet(q, aligned, cfg):
                row["aligned_int4"] = probe(q)
            qr = []
            for heads in draws[:max(1, args.draws // 2)]:
                with MaskSet(q, heads, cfg):
                    qr.append(probe(q))
            row["random_int4_mean"] = statistics.fmean(qr)
            del q
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[arm] = row
        log(f"    clean   {row['clean'] * 100:6.2f}%")
        log(f"    aligned {row['aligned'] * 100:6.2f}%")
        log(f"    random  {row['random_mean'] * 100:6.2f}%"
            + (f" ± {row['random_std'] * 100:.2f}" if row["random_std"] else ""))
        log(f"    aligned - random  {row['aligned_minus_random'] * 100:+6.2f}pp")
        if row["sliced_aligned"]:
            log(f"    sliced aligned {row['sliced_aligned']['accuracy'] * 100:6.2f}%  "
                f"({row['sliced_aligned']['param_reduction']:.1%} fewer params)")

    payload = {
        "experiment": "exp15_ijepa_aligned",
        "model": J.MODEL_ID, "dataset": J.DATASET_ID,
        "seed": args.seed, "mask_ratio": args.mask_ratio, "epochs": epochs,
        "uniform_per_layer": uniform, "sal_set_is_uniform": is_uniform,
        "num_layers": nl, "num_heads_per_layer": nh,
        "config_target_heads": cfg.num_heads_to_prune,
        "effective_prune_fraction": cfg.effective_prune_fraction,
        "sal_pruned_heads": aligned, "sal_pruned_count": len(aligned),
        "sal_pruned_per_layer": counts,
        "batch_size": batch_size, "grad_ckpt": grad_ckpt,
        "draws": args.draws, "sliced_draws": args.sliced_draws,
        "skip_int4": args.skip_int4, "smoke": args.smoke,
        "device": str(device), "vram_gb": round(vram, 1),
        "num_classes": num_classes,
        "n_train": sizes["n_train"], "n_probe_train": sizes["n_probe_train"],
        "n_probe_val": sizes["n_probe_val"],
        "training": infos, "results": results,
        "caveats": {
            "metric": "linear probe top-1 on frozen features; I-JEPA has no "
                      "classifier of its own, so this is the headline metric "
                      "rather than fine-tuned accuracy.",
            "masking": "aligned and random cells are masked, which keeps every "
                       "weight and matmul and measures accuracy retention only. "
                       "The sliced_* cells apply the same sets with "
                       "slice_heads(), which is what makes the retention a real "
                       "size reduction.",
            "int4": "per-output-channel symmetric weight round-trip, the same "
                    "grid used by every other v0.6.0 cell.",
            "drophead_aligned": "drophead has no pruned set of its own; it is "
                                "scored on SAL's so every arm faces the same "
                                "perturbation.",
        },
        "wall_clock_s": round(time.time() - t_all, 1),
    }
    # Measurements before presentation.
    dest = outdir / "exp15_ijepa_aligned.json"
    dest.write_text(json.dumps(payload, indent=2))
    log(f"\n  metrics -> {dest}")

    J.rule("Summary")
    log(f"  {'arm':<10}{'clean':>9}{'aligned':>9}{'random':>9}{'A-R':>9}"
        f"{'slc-aln':>9}{'slc-rnd':>9}{'aln+I4':>9}")
    for arm in ARMS:
        r = results[arm]
        sa = r["sliced_aligned"]
        log(f"  {arm:<10}{r['clean'] * 100:8.2f}%{r['aligned'] * 100:8.2f}%"
            f"{r['random_mean'] * 100:8.2f}%{r['aligned_minus_random'] * 100:+8.2f} "
            + (f"{sa['accuracy'] * 100:7.2f}%" if sa else f"{'n/a':>8}")
            + (f"{r['sliced_random_mean'] * 100:8.2f}%"
               if r.get("sliced_random_mean") is not None else f"{'n/a':>9}")
            + (f"{r['aligned_int4'] * 100:8.2f}%" if "aligned_int4" in r
               else f"{'n/a':>9}"))
    log(f"\n  wall clock {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
