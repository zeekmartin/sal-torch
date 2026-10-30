#!/usr/bin/env python3
"""exp13 — does DropHead's advantage over SAL hold on other architectures?

Track C showed that on ViT-B/16 / ImageNet-100, plain per-step head dropout
beats SAL's accumulate-and-hold schedule by ~30pp under 50% head pruning
(`results/lumi_v060/PHASE2C.md`). That was one architecture and one task. This
is the quick cross-architecture check: the same three arms on

  * **GPT-2 Medium / SST-2** — a different modality, and the architecture the
    README's *primary* evidence rests on (+2.45pp for SAL at 33% pruning on
    5/5 seeds). If DropHead beats SAL here too, the headline claim is affected
    directly.
  * **DINOv2-base / ImageNet-100** — a different vision backbone, trained with
    a different objective from ViT-B/16's supervised checkpoint.

Three arms, identical data order and step count:

  baseline   plain fine-tuning
  sal        SAL's accumulating schedule (sal.SALTrainer)
  drophead   per-step resampled head dropout at the same ratio

Scoring is by **masking** a uniform-random set of heads, averaged over
``--draws`` sets, with every arm seeing the identical sets. Masking rather than
slicing keeps this architecture-agnostic — GPT-2 stores attention in ``Conv1D``
and the two backbones would otherwise need different slicing paths — and the
question here is accuracy retention, not model size.

This is deliberately a *quick* check: fewer epochs and fewer seeds than the
published protocols, and the numbers are not comparable to them. What it can
settle is the **direction** on 3 seeds, which is what "does it generalise"
needs.

Usage::

    python exp13_crossarch_drophead.py --model gpt2-medium --seed 42
    python exp13_crossarch_drophead.py --model facebook/dinov2-base --seed 42
    python exp13_crossarch_drophead.py --model gpt2-medium --download-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

TEXT_DATASET = "stanfordnlp/sst2"
IMAGE_DATASET = "clane9/imagenet-100"
ARMS = ("baseline", "sal", "drophead")


def log(msg=""):
    print(msg, flush=True)


def rule(title):
    log()
    log(f"=== {title} " + "=" * max(0, 66 - len(title)))


def is_text(model_id: str) -> bool:
    return "gpt2" in model_id.lower() or "bert" in model_id.lower()


# ------------------------------------------------------------------ data
def build_text(model_id, n_train, n_eval, seed, max_len=64):
    """SST-2 as padded input_ids / attention_mask / labels tensors."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset(TEXT_DATASET)

    def enc(split, n):
        rows = split.shuffle(seed=seed).select(range(min(n, len(split))))
        out = tok(rows["sentence"], truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        return (out["input_ids"], out["attention_mask"],
                torch.tensor(rows["label"]))

    return enc(ds["train"], n_train), enc(ds["validation"], n_eval), 2, tok


def build_image(model_id, n_train, n_eval, seed):
    """ImageNet-100 as uint8 image tensors, reusing the prototype's encoder."""
    import importlib.util

    proto = Path(__file__).resolve().parent.parent / "scripts" / "run_sal_qat_prototype.py"
    spec = importlib.util.spec_from_file_location("sal_qat_prototype", proto)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    train, val, num_classes = P.build_data(
        {"n_train": n_train, "n_val": n_eval}, seed)
    return train, val, num_classes, P


def batches(split, batch_size, shuffle=False, seed=0):
    n = split[0].shape[0]
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        sel = torch.tensor(idx[i:i + batch_size])
        yield tuple(t[sel] for t in split)


# ------------------------------------------------------------------ model
def load_model(model_id, num_classes, tok=None):
    if is_text(model_id):
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_classes)
        if m.config.pad_token_id is None:
            m.config.pad_token_id = tok.pad_token_id if tok else 0
        return m
    from transformers import AutoModelForImageClassification
    return AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=num_classes, ignore_mismatched_sizes=True)


def forward_loss(model, batch, device, is_txt, normalize=None):
    if is_txt:
        ids, attn, y = (t.to(device) for t in batch)
        out = model(input_ids=ids, attention_mask=attn, labels=y)
        return out.loss, out.logits, y
    x, y = batch[0].to(device), batch[1].to(device)
    out = model(pixel_values=normalize(x), labels=y)
    return out.loss, out.logits, y


@torch.no_grad()
def accuracy(model, split, device, is_txt, normalize=None, batch_size=64):
    model.eval()
    correct = n = 0
    for batch in batches(split, batch_size):
        _, logits, y = forward_loss(model, batch, device, is_txt, normalize)
        correct += int((logits.argmax(-1) == y).sum())
        n += y.shape[0]
    return correct / max(n, 1)


# ------------------------------------------------------------------ training
def train_arm(arm, model_id, num_classes, train_split, epochs, batch_size, seed,
              device, mask_ratio, is_txt, normalize, tok):
    from sal import SALConfig, SALTrainer
    from sal.masker import HeadMasker

    torch.manual_seed(seed)
    random.seed(seed)
    model = load_model(model_id, num_classes, tok).to(device)
    model.train()

    steps_per_epoch = (train_split[0].shape[0] + batch_size - 1) // batch_size
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-5, total_steps=max(total, 1), pct_start=0.1)

    cfg = SALConfig.auto(model, prune_fraction=mask_ratio)
    masker = None
    if arm in ("sal", "drophead"):
        masker = HeadMasker(model, cfg, seed=seed)

    def one_step(m, batch, optimizer, mask_module):
        loss, _, _ = forward_loss(m, batch, device, is_txt, normalize)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    t0 = time.time()
    losses, pruned_heads = [], None

    if arm == "sal":
        loader = list(batches(train_split, batch_size, shuffle=True, seed=seed))
        trainer = SALTrainer(model, cfg, opt, loader, scheduler=sched, seed=seed,
                            train_step=one_step)
        hist = trainer.train(num_epochs=epochs)
        losses, pruned_heads = hist["losses"], hist.get("pruned_heads")
        stats = hist["masker_stats"]
    else:
        if masker is not None:
            masker.install()
        gstep = 0
        try:
            for ep in range(epochs):
                tot = cnt = 0.0
                for batch in batches(train_split, batch_size, shuffle=True,
                                     seed=seed * 100 + ep):
                    if masker is not None:
                        # Per-step resampled dropout: clear, then redraw.
                        for m_ in masker._masks.values():
                            m_.fill_(1.0)
                        masker._active = True
                        masker._deactivate_random_heads(cfg.num_heads_to_prune)
                    tot += one_step(model, batch, opt, masker)
                    sched.step()
                    cnt += 1
                    gstep += 1
                losses.append(tot / max(cnt, 1))
            stats = dict(masker.stats) if masker is not None else None
        finally:
            if masker is not None:
                masker.remove()

    model.eval()
    log(f"  {arm}: losses {[round(x, 4) for x in losses]} | {time.time() - t0:.0f}s"
        + (f" | masker {stats}" if stats else ""))
    return model, {"losses": losses, "masker_stats": stats,
                   "pruned_heads": pruned_heads,
                   "train_seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ scoring
class MaskHeads:
    """Mask an explicit (layer, head) set for the duration of the block."""

    def __init__(self, model, cfg, heads):
        from sal.masker import HeadMasker
        self.m = HeadMasker(model, cfg, seed=0)
        self.heads = heads

    def __enter__(self):
        self.m.install()
        for layer, head in self.heads:
            self.m._masks[layer][head] = 0.0
        self.m.apply_mask()
        return self

    def __exit__(self, *exc):
        self.m.remove()
        return False


def uniform_heads(num_layers, num_heads, ratio, seed):
    """``round(ratio * num_heads)`` heads from every layer — a uniform cut."""
    k = int(round(ratio * num_heads))
    rng = random.Random(seed)
    return [(l, h) for l in range(num_layers)
            for h in sorted(rng.sample(range(num_heads), k))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-eval", type=int, default=872)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--prune-ratios", type=float, nargs="+", default=[0.33, 0.50])
    p.add_argument("--output", default="results/exp13_crossarch")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args(argv)

    is_txt = is_text(args.model)

    if args.download_only:
        log(f"Pre-fetching {args.model} + "
            f"{TEXT_DATASET if is_txt else IMAGE_DATASET} into "
            f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
        if is_txt:
            tr, _, nc, tok = build_text(args.model, 8, 8, 0)
            load_model(args.model, nc, tok)
        else:
            tr, _, nc, _ = build_image(args.model, 8, 8, 0)
            load_model(args.model, nc)
        log("  OK — model and dataset cached")
        return 0

    if args.smoke:
        args.n_train, args.n_eval, args.epochs, args.draws = 256, 128, 1, 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule("exp13 — cross-architecture DropHead check")
    log(f"  model {args.model} | seed {args.seed} | mask_ratio {args.mask_ratio} "
        f"| {args.epochs} epochs | device {device}")
    if device.type == "cuda":
        log(f"  GPU {torch.cuda.get_device_name(0)}")

    tok, normalize = None, None
    if is_txt:
        train_split, eval_split, num_classes, tok = build_text(
            args.model, args.n_train, args.n_eval, args.seed)
    else:
        train_split, eval_split, num_classes, P = build_image(
            args.model, args.n_train, args.n_eval, args.seed)
        normalize = lambda x: P.normalize(x)  # noqa: E731
    log(f"  {train_split[0].shape[0]} train / {eval_split[0].shape[0]} eval, "
        f"{num_classes} classes")

    from sal import SALConfig

    t_all = time.time()
    results, training = {}, {}
    draws, cfg0 = None, None

    for arm in ARMS:
        rule(f"training {arm}")
        model, info = train_arm(arm, args.model, num_classes, train_split,
                                args.epochs, args.batch_size, args.seed, device,
                                args.mask_ratio, is_txt, normalize, tok)
        training[arm] = info
        cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio)
        if draws is None:
            cfg0 = cfg
            draws = {r: [uniform_heads(cfg.num_layers, cfg.num_heads_per_layer, r,
                                       9_000 + args.seed * 50 + i)
                         for i in range(args.draws)]
                     for r in args.prune_ratios}
            log(f"  graph: {cfg.num_layers} layers x {cfg.num_heads_per_layer} heads")

        row = {"clean": accuracy(model, eval_split, device, is_txt, normalize)}
        log(f"    clean {row['clean'] * 100:6.2f}%")
        for r in args.prune_ratios:
            vals = []
            for heads in draws[r]:
                with MaskHeads(model, cfg, heads):
                    vals.append(accuracy(model, eval_split, device, is_txt, normalize))
            row[f"prune{int(r * 100)}"] = {
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else None,
                "per_draw": vals,
            }
            log(f"    prune{int(r * 100):<3d} {statistics.fmean(vals) * 100:6.2f}%"
                + (f" ± {statistics.stdev(vals) * 100:.2f}" if len(vals) > 1 else ""))
        results[arm] = row
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rule("Summary")
    cols = ["clean"] + [f"prune{int(r * 100)}" for r in args.prune_ratios]
    log("  {:<10s}".format("arm") + "".join(f"{c:>11s}" for c in cols))
    for arm in ARMS:
        cells = []
        for c in cols:
            v = results[arm][c]
            cells.append(f"{(v if c == 'clean' else v['mean']) * 100:10.2f}%")
        log(f"  {arm:<10s}" + "".join(cells))
    for c in cols[1:]:
        d = (results["drophead"][c]["mean"] - results["sal"][c]["mean"]) * 100
        log(f"  DropHead - SAL at {c}: {d:+.2f}pp")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exp13_crossarch_drophead",
        "model": args.model,
        "dataset": TEXT_DATASET if is_txt else IMAGE_DATASET,
        "seed": args.seed, "mask_ratio": args.mask_ratio, "epochs": args.epochs,
        "n_train": train_split[0].shape[0], "n_eval": eval_split[0].shape[0],
        "num_classes": num_classes, "batch_size": args.batch_size,
        "draws": args.draws, "prune_ratios": args.prune_ratios,
        "smoke": args.smoke, "device": str(device),
        "num_layers": cfg0.num_layers, "num_heads_per_layer": cfg0.num_heads_per_layer,
        "training": training, "results": results,
        "caveats": {
            "quick_check": "fewer epochs and seeds than the published protocols; "
                           "these numbers are not comparable to them. It is the "
                           "direction that is being tested.",
            "masking": "pruning is applied by masking a uniform random head set, "
                       "averaged over --draws sets identical across arms. Masking "
                       "keeps this architecture-agnostic (GPT-2 keeps attention in "
                       "Conv1D) and measures accuracy retention, not model size.",
        },
        "wall_clock_s": round(time.time() - t_all, 1),
    }
    dest = outdir / f"exp13_{args.model.replace('/', '_')}_seed{args.seed}.json"
    dest.write_text(json.dumps(payload, indent=2))
    log(f"  results -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
