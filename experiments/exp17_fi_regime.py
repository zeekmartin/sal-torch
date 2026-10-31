#!/usr/bin/env python3
"""exp17 — does the fragility score predict which compression regime a model is in?

Two architectures measured so far behave in opposite ways under 50% head pruning:

* **ViT-B/16** (86M, FI=0.066) *collapses* — a standard fine-tune drops from
  93.77% to 31.56%, and the thing that rescues it is DropHead, which is robust on
  an *arbitrary* head set (`results/lumi_v060/PHASE2F.md`).
* **I-JEPA ViT-H/14** (631M, FI=0.004) *holds* — the control keeps 61-69% of an
  83% clean score, DropHead buys nothing measurable (+0.49pp, 2/3 seeds, costs
  1.08pp of clean accuracy), and SAL dominates every cell
  (`results/lumi_v060/PHASE2G.md`).

The hypothesis under test is that **FI, measured before any training, tells the
two regimes apart**: high FI (fragile, poorly triangulated head graph) means the
model needs DropHead's arbitrary-set robustness; low FI means it survives pruning
on its own and SAL's specialisation is what pays.

This is a *diagnostic* use of FI and nothing more — exactly the use the README
already endorses, "use FIScanner to decide whether to compress and how much".
**FI does not choose which heads to prune here, in any arm.** Every pruned set in
this script is a uniform random draw (sal-torch cardinal rules 1 and 2).

Four architectures are added to the two known points:

======================  =======  ========  ===================================
model                   params   task      why
======================  =======  ========  ===================================
deit-small-patch16-224     22M   IN-100    smallest; expected fragile
bert-base-uncased         110M   SST-2     encoder-only, language modality
dinov2-large              304M   IN-100    expected robust, self-supervised
phi-2                    2648M   SST-2     large decoder (1 seed, expensive)
======================  =======  ========  ===================================

**The confound to keep in sight.** FI and parameter count are not independent
across these six models: the two known points differ by 7x in size *and* 16x in
FI, so a correlation between FI and regime is also a correlation between size and
regime. This design cannot separate them on its own; what it can do is look for a
**dissociation** — a small model with low FI, or a large one with high FI. The
summary therefore reports params alongside FI and the rank correlation of each
against the regime, rather than claiming FI as the cause. See the caveats block
written into every JSON.

Per model, per seed:

a. ``FIScanner`` scan of the *untrained* pretrained model (FI before any arm)
b. ``baseline`` / ``sal`` / ``drophead`` trained at mask_ratio 0.5
c. each arm scored clean, on **SAL's own trained set** (``aligned``), and on
   ``--draws`` uniform random sets of identical shape shared across arms
d. the same three cells again under simulated INT4
e. the control cell that separates adaptation from selection: ``baseline``
   scored on SAL's aligned set versus the random sets

Reuses ``exp13_crossarch_drophead`` for data loading, training and scoring, so
the text/image split, the eager-attention fix for ROCm and the non-finite-loss
guard all come from one place.

Usage::

    python exp17_fi_regime.py --model facebook/dinov2-large --seed 42 \
        --mask-ratio 0.5 --draws 20 --output results/exp17/dinov2-large/seed42
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp13_crossarch_drophead as E  # noqa: E402

ARMS = ("baseline", "sal", "drophead")


def log(msg=""):
    print(msg, flush=True)


def rule(title):
    log()
    log(f"=== {title} " + "=" * max(0, 66 - len(title)))


# --------------------------------------------------------------------- FI scan
def fi_scan(model_id, num_classes, tok, probe_split, device, is_txt, normalize,
            num_samples=500):
    """FI of the *pretrained* model, before any arm trains.

    Deliberately measured on the untrained checkpoint: the hypothesis is that FI
    is usable as an up-front triage signal, which it only is if it can be read
    before committing to a training run.
    """
    from sal.fi import compute_fi, extract_activation_graph

    model = E.load_model(model_id, num_classes, tok).to(device).eval()
    probe = _probe_batches(probe_split, device, is_txt, normalize, num_samples)
    t0 = time.time()
    adj = extract_activation_graph(model, probe, num_samples=num_samples)
    fi = compute_fi(adj)
    out = {"fi": float(fi), "graph_nodes": int(adj.shape[0]),
           "edges": int((adj > 0).sum() // 2),
           "scan_seconds": round(time.time() - t0, 1)}
    log(f"  FI = {fi:.6f}  ({out['graph_nodes']} heads, {out['edges']} edges, "
        f"{out['scan_seconds']}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _probe_batches(split, device, is_txt, normalize, num_samples):
    """A list of dicts the FI extractor can iterate, capped at ``num_samples``."""
    out, taken = [], 0
    for batch in E.batches(split, 32):
        if is_txt:
            item = {"input_ids": batch[0], "attention_mask": batch[1]}
            n = batch[0].shape[0]
        else:
            x = batch[0]
            item = {"pixel_values": normalize(x) if normalize else x}
            n = x.shape[0]
        out.append(item)
        taken += n
        if taken >= num_samples:
            break
    return out


# ---------------------------------------------------------------- INT4 scoring
def int4_copy(model):
    """A weight-quantized INT4 copy, built outside any mask context.

    Quantizing inside a ``MaskHeads`` block would bake the zeros into the
    quantization scales and measure something else entirely.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_sal_qat_prototype import quantize_weights
    return quantize_weights(copy.deepcopy(model), 4)


def score_set(model, cfg, heads, eval_split, device, is_txt, normalize):
    """Accuracy with an explicit head set masked off."""
    with E.MaskHeads(model, cfg, heads):
        return E.accuracy(model, eval_split, device, is_txt, normalize)


def score_arm(model, cfg, aligned, draws, eval_split, device, is_txt, normalize,
              skip_int4=False):
    row = {"clean": E.accuracy(model, eval_split, device, is_txt, normalize)}
    row["aligned"] = score_set(model, cfg, aligned, eval_split, device, is_txt,
                               normalize)
    vals = [score_set(model, cfg, h, eval_split, device, is_txt, normalize)
            for h in draws]
    row["random_per_draw"] = vals
    row["random_mean"] = statistics.fmean(vals)
    row["random_std"] = statistics.stdev(vals) if len(vals) > 1 else None
    row["aligned_minus_random"] = row["aligned"] - row["random_mean"]
    log(f"    clean    {row['clean'] * 100:6.2f}%")
    log(f"    aligned  {row['aligned'] * 100:6.2f}%")
    log(f"    random   {row['random_mean'] * 100:6.2f}%"
        + (f" ± {row['random_std'] * 100:.2f}" if row["random_std"] else ""))
    log(f"    aligned - random   {row['aligned_minus_random'] * 100:+.2f}pp")

    if skip_int4:
        return row
    q = int4_copy(model)
    row["clean_int4"] = E.accuracy(q, eval_split, device, is_txt, normalize)
    row["aligned_int4"] = score_set(q, cfg, aligned, eval_split, device, is_txt,
                                    normalize)
    qv = [score_set(q, cfg, h, eval_split, device, is_txt, normalize)
          for h in draws]
    row["random_int4_mean"] = statistics.fmean(qv)
    row["random_int4_per_draw"] = qv
    log(f"    INT4: clean {row['clean_int4'] * 100:6.2f}%  "
        f"aligned {row['aligned_int4'] * 100:6.2f}%  "
        f"random {row['random_int4_mean'] * 100:6.2f}%")
    del q
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-eval", type=int, default=872)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--fi-samples", type=int, default=500)
    p.add_argument("--skip-int4", action="store_true")
    p.add_argument("--skip-fi", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing; needed for phi-2 on one GCD")
    p.add_argument("--output", default="results/exp17")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        (args.n_train, args.n_eval, args.epochs, args.draws, args.fi_samples) = (
            256, 128, 1, 2, 64)

    is_txt = E.is_text(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.grad_ckpt:
        # phi-2 at 2.6B does not fit one 64GB GCD with AdamW states and full
        # activations. Wrapping the loader rather than editing exp13 keeps the
        # shared training path identical for every other model in the series.
        _plain_load = E.load_model

        def _ckpt_load(*a, **kw):
            m = _plain_load(*a, **kw)
            if hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
                if hasattr(m, "config"):
                    m.config.use_cache = False
            return m

        E.load_model = _ckpt_load
        log("  gradient checkpointing: on")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    rule("exp17 — does FI predict the compression regime?")
    log(f"  model {args.model} | seed {args.seed} | mask_ratio {args.mask_ratio} "
        f"| {args.epochs} epochs | {args.draws} draws")
    log(f"  device {device}"
        + (f" ({torch.cuda.get_device_name(0)}, "
           f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)"
           if device.type == "cuda" else ""))

    tok, normalize = None, None
    if is_txt:
        train_split, eval_split, num_classes, tok = E.build_text(
            args.model, args.n_train, args.n_eval, args.seed)
    else:
        train_split, eval_split, num_classes, P = E.build_image(
            args.model, args.n_train, args.n_eval, args.seed)
        normalize = lambda x: P.normalize(x)  # noqa: E731
    log(f"  {train_split[0].shape[0]} train / {eval_split[0].shape[0]} eval, "
        f"{num_classes} classes")

    t_all = time.time()
    payload = {
        "experiment": "exp17_fi_regime", "model": args.model,
        "dataset": E.TEXT_DATASET if is_txt else E.IMAGE_DATASET,
        "seed": args.seed, "mask_ratio": args.mask_ratio, "epochs": args.epochs,
        "n_train": int(train_split[0].shape[0]),
        "n_eval": int(eval_split[0].shape[0]),
        "num_classes": num_classes, "batch_size": args.batch_size,
        "draws": args.draws, "skip_int4": args.skip_int4, "smoke": args.smoke,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    # ------------------------------------------------------------- (a) FI scan
    if not args.skip_fi:
        rule("FI scan of the pretrained model")
        payload["fi_scan"] = fi_scan(args.model, num_classes, tok, train_split,
                                     device, is_txt, normalize, args.fi_samples)

    from sal import SALConfig

    # --------------------------------------- (b) train, then (c)(d) score arms
    results, training = {}, {}
    aligned = draws = cfg0 = None

    for arm in ARMS:
        rule(f"training {arm}")
        model, info = E.train_arm(arm, args.model, num_classes, train_split,
                                  args.epochs, args.batch_size, args.seed,
                                  device, args.mask_ratio, is_txt, normalize, tok)
        training[arm] = info
        cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio,
                             uniform_per_layer=True)
        if cfg0 is None:
            cfg0 = cfg
            log(f"  graph: {cfg.num_layers} layers x {cfg.num_heads_per_layer} "
                f"heads; prune {cfg.num_heads_to_prune} "
                f"({cfg.effective_prune_fraction:.3f}), "
                f"{cfg.num_heads_to_prune // cfg.num_layers}/layer")

        if arm == "sal":
            # SAL's own trained set becomes the `aligned` set for every arm,
            # including the two already scored -- so the arms are re-scored
            # below rather than scored as they are trained.
            aligned = sorted(tuple(h) for h in (info.get("pruned_heads") or []))
            if not aligned:
                raise RuntimeError(
                    "SAL reported no pruned heads; the aligned set is the whole "
                    "point of this experiment and nothing downstream is valid "
                    "without it.")
            payload["sal_pruned_heads"] = [list(h) for h in aligned]
            payload["sal_pruned_count"] = len(aligned)
            per_layer = [sum(1 for l, _ in aligned if l == li)
                         for li in range(cfg.num_layers)]
            payload["sal_pruned_per_layer"] = per_layer
            payload["sal_set_is_uniform"] = len(set(per_layer)) == 1
            log(f"  SAL pruned {len(aligned)} heads, per layer {per_layer}, "
                f"uniform {payload['sal_set_is_uniform']}")
            if draws is None:
                draws = [E.uniform_heads(cfg.num_layers, cfg.num_heads_per_layer,
                                         len(aligned) / cfg.total_heads,
                                         9_000 + args.seed * 50 + i)
                         for i in range(args.draws)]
        # Keep the trained model on the host so every arm can be scored against
        # the same aligned set once SAL has produced it.
        results[arm] = model.cpu()

    if aligned is None:
        raise RuntimeError("no sal arm ran; there is no aligned set")

    scores = {}
    for arm in ARMS:
        rule(f"scoring {arm}")
        model = results[arm].to(device).eval()
        scores[arm] = score_arm(model, cfg0, aligned, draws, eval_split, device,
                                is_txt, normalize, args.skip_int4)
        results[arm] = model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------- (e) the headline five metrics
    def pp(x):
        return round(x * 100, 4)

    b, s, d = scores["baseline"], scores["sal"], scores["drophead"]
    payload["metrics"] = {
        "fi": payload.get("fi_scan", {}).get("fi"),
        # How much the untrained-for-pruning model loses: the regime marker.
        "baseline_drop": pp(b["random_mean"] - b["clean"]),
        "baseline_drop_aligned": pp(b["aligned"] - b["clean"]),
        # Same-set gains over the baseline: the comparison with no asymmetry.
        "sal_gain": pp(s["aligned"] - b["aligned"]),
        "drophead_gain": pp(d["aligned"] - b["aligned"]),
        "sal_vs_drophead_same_set": pp(s["aligned"] - d["aligned"]),
        # DropHead's real selling point: robustness on an arbitrary set.
        "drophead_gain_random": pp(d["random_mean"] - b["random_mean"]),
        "sal_aligned_vs_drophead_random": pp(s["aligned"] - d["random_mean"]),
        # Adaptation or selection? Positive means the set is intrinsically easy.
        "control_aligned_minus_random": pp(b["aligned_minus_random"]),
        "sal_aligned_minus_random": pp(s["aligned_minus_random"]),
        "drophead_aligned_minus_random": pp(d["aligned_minus_random"]),
        "clean_cost_sal": pp(s["clean"] - b["clean"]),
        "clean_cost_drophead": pp(d["clean"] - b["clean"]),
    }
    if not args.skip_int4:
        payload["metrics"].update({
            "sal_gain_int4": pp(s["aligned_int4"] - b["aligned_int4"]),
            "drophead_gain_int4": pp(d["aligned_int4"] - b["aligned_int4"]),
            "sal_vs_drophead_int4": pp(s["aligned_int4"] - d["aligned_int4"]),
        })

    payload["results"] = scores
    payload["training"] = training
    payload["num_layers"] = cfg0.num_layers
    payload["num_heads_per_layer"] = cfg0.num_heads_per_layer
    payload["effective_prune_fraction"] = cfg0.effective_prune_fraction
    payload["params"] = int(sum(p.numel() for p in results["baseline"].parameters()))
    payload["caveats"] = {
        "fi_is_a_diagnostic": "FI is measured before training and never selects "
                              "heads. Every pruned set here is a uniform random "
                              "draw. The hypothesis is about triage, not selection.",
        "fi_size_confound": "FI and parameter count are correlated across the "
                            "models in this series, so a relationship between FI "
                            "and the regime is also one between size and the "
                            "regime. Only a dissociation (small+low-FI or "
                            "large+high-FI) would separate them; the summary "
                            "reports both columns for that reason.",
        "quick_protocol": "fewer epochs than the ViT-B/16 and I-JEPA protocols, "
                          "so cell values are not comparable to PHASE2F/PHASE2G "
                          "in absolute terms. The regime marker and the signs are "
                          "what this measures.",
        "masking_not_slicing": "pruning is applied by masking, which removes no "
                               "weight and no matmul. Sizes and latencies are not "
                               "measured here.",
        "draw_noise": "random-set sampling, not the training seed, was the "
                      "dominant noise term in PHASE2A; --draws controls it and "
                      "defaults to 20.",
    }
    payload["wall_clock_s"] = round(time.time() - t_all, 1)

    # Measurements before presentation: a formatting bug must never be able to
    # throw away a finished run (it cost twenty expF jobs, 11.7 GPU-hours).
    dest = outdir / "exp17_fi_regime.json"
    dest.write_text(json.dumps(payload, indent=2))
    log(f"\n  metrics -> {dest}")

    try:
        rule("Summary")
        log(f"  FI = {payload.get('fi_scan', {}).get('fi', float('nan')):.6f}"
            f"   params {payload['params'] / 1e6:.1f}M")
        log(f"  {'arm':<10}{'clean':>9}{'aligned':>9}{'random':>9}{'A-R':>8}"
            f"{'clnI4':>8}{'alnI4':>8}{'rndI4':>8}")
        for arm in ARMS:
            r = scores[arm]
            log(f"  {arm:<10}{r['clean'] * 100:8.2f}%{r['aligned'] * 100:8.2f}%"
                f"{r['random_mean'] * 100:8.2f}%"
                f"{r['aligned_minus_random'] * 100:+8.2f}"
                + ("".join(f"{r[k] * 100:7.2f}%" for k in
                           ("clean_int4", "aligned_int4", "random_int4_mean"))
                   if not args.skip_int4 else f"{'n/a':>8}{'n/a':>8}{'n/a':>8}"))
        log()
        for k, v in payload["metrics"].items():
            if k != "fi":
                log(f"    {k:<34}{v:+8.2f}pp")
        log(f"\n  wall clock {(time.time() - t_all) / 60:.1f} min")
    except Exception as exc:  # presentation only; the metrics are already safe
        log(f"  WARNING: summary failed ({type(exc).__name__}: {exc})")
        log(f"  the measurements are intact in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
