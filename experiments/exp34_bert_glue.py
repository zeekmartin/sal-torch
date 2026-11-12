#!/usr/bin/env python3
"""exp34 — DropHead against magnitude + recovery on BERT-base / GLUE.

``bert-base-uncased`` on SST-2, MRPC, QNLI and MNLI. Everything the project
has measured on head methods at ImageNet scale is one architecture family and
one modality; exp17 found BERT-base sits at the *other* end of the routing
table (a 5.15pp probe drop against ViT-B/16's 62.21), so it is the natural
test of whether the DropHead recommendation survives a move to text.

Order of business, per task
---------------------------
1. **Dense fine-tune, and check it against the published reference.** A head
   method's gain is measured against this; if the dense control is off, every
   number downstream is meaningless (the exp28 lesson). The run **aborts** when
   the control lands outside ``--control-tolerance`` of the reference rather
   than carrying on and producing a table.
2. **DropHead** at the deployment ratio, then sliced to exactly that ratio.
3. **Magnitude + recovery**: slice first, then fine-tune for
   ``--recover-epochs`` (default 2, exp29).

Both head arms are scored on the **same head set** — the magnitude set — so
neither gets a perturbation the other did not face, which is the protocol
exp27 settled on.

Reference values are the ones in the BERT paper and the GLUE leaderboard for
``bert-base-uncased``: SST-2 92-93, MRPC ~88 (accuracy), QNLI ~91, MNLI ~84
(matched). They are *targets for the control*, not baselines to beat.

Usage::

    # gate: one task, tiny subset, 1 epoch -- exercises every path
    python exp34_bert_glue.py --gate --output results/exp34_gate

    # full: one job per seed
    python exp34_bert_glue.py --seed 42 --output results/exp34_s42
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

MODEL_ID = "bert-base-uncased"
GLUE = "nyu-mll/glue"

# task -> (text keys, num_labels, validation split, reference ACCURACY)
#
# These are ``bert-base-uncased`` dev **accuracies**. The distinction matters:
# the number usually quoted for MRPC is **88-89 F1**, not accuracy, and its
# accuracy is ~84. The first smoke of this script gated MRPC against 88.0,
# read 83.82% and reported a 4.18pp control failure that was nothing of the
# kind -- the control was fine and the reference was the wrong metric. MRPC
# therefore also reports F1 below, against ``REFERENCE_F1``.
TASKS = {
    "sst2": (("sentence", None), 2, "validation", 92.5),
    "mrpc": (("sentence1", "sentence2"), 2, "validation", 84.0),
    "qnli": (("question", "sentence"), 2, "validation", 91.0),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched", 84.0),
}
# Tasks whose customary headline figure is F1, reported alongside accuracy.
REFERENCE_F1 = {"mrpc": 88.0}
RATIOS = (0.33, 0.50)
RECOVER_EPOCHS = 2
EPOCHS = 3
LR = 2e-5
BATCH = 32
MAX_LEN = 128


def log(msg: str = ""):
    print(msg, flush=True)


def rule(title: str):
    log()
    log(f"=== {title} " + "=" * max(0, 72 - len(title)))


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_task(task: str, tokenizer, n_train=None, n_val=None, seed=42):
    from datasets import load_dataset

    (k1, k2), n_labels, val_split, ref = TASKS[task]
    ds = load_dataset(GLUE, task)

    def enc(rows):
        args = (rows[k1],) if k2 is None else (rows[k1], rows[k2])
        return tokenizer(*args, truncation=True, max_length=MAX_LEN,
                         padding="max_length")

    cols = [c for c in (k1, k2, "idx") if c]
    train = ds["train"].shuffle(seed=seed)
    val = ds[val_split]
    if n_train:
        train = train.select(range(min(n_train, len(train))))
    if n_val:
        val = val.select(range(min(n_val, len(val))))
    train = train.map(enc, batched=True, remove_columns=cols)
    val = val.map(enc, batched=True, remove_columns=cols)
    train.set_format("torch")
    val.set_format("torch")
    return train, val, n_labels, ref


def loaders(train, val, batch_size, seed):
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(val, batch_size=batch_size * 2))


def load_model(n_labels, device):
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=n_labels, attn_implementation="eager")
    return m.to(device)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, want_f1: bool = False):
    """Accuracy, and F1 of the positive class when ``want_f1``.

    Returns a float, or ``(accuracy, f1)`` when F1 was asked for, so existing
    call sites keep their shape.
    """
    model.eval()
    right = total = tp = fp = fn = 0
    for batch in loader:
        labels = batch.pop("label").to(device)
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids", "attention_mask", "token_type_ids")}
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else torch.autocast("cpu", enabled=False))
        with ctx:
            logits = model(**batch).logits
        pred = logits.float().argmax(-1)
        right += int((pred == labels).sum())
        total += labels.numel()
        if want_f1:
            tp += int(((pred == 1) & (labels == 1)).sum())
            fp += int(((pred == 1) & (labels == 0)).sum())
            fn += int(((pred == 0) & (labels == 1)).sum())
    acc = right / max(total, 1)
    if not want_f1:
        return acc
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, f1


def train_loop(model, loader, epochs, device, use_amp, masker=None,
               label="train", lr=LR):
    """Fine-tune. ``masker`` is a DropHead/SAL masker driven per step."""
    from torch.optim.lr_scheduler import LambdaLR

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = max(1, len(loader) * epochs)
    warm = max(1, int(0.1 * total))

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))

    sched = LambdaLR(opt, lr_at)
    if masker is not None:
        masker.install()
    model.train()
    losses, step = [], 0
    for epoch in range(epochs):
        running, n = 0.0, 0
        for batch in loader:
            labels = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ("input_ids", "attention_mask", "token_type_ids")}
            if masker is not None:
                masker.step(step, total)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                out = model(**batch, labels=labels)
                loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{label}: non-finite loss at step {step} — refusing to "
                    "continue, a diverged run still yields a majority-class "
                    "accuracy and writing that would be worse than crashing")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += float(loss)
            n += 1
            step += 1
        losses.append(running / max(n, 1))
        log(f"    {label} epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f}")
    if masker is not None:
        masker.remove()
    return losses


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--recover-epochs", type=int, default=RECOVER_EPOCHS,
                   help="default 2 (exp29: the recovery curve peaks there)")
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--control-tolerance", type=float, default=2.5,
                   help="pp below the published reference the dense control "
                        "may land before the run aborts")
    p.add_argument("--no-abort", action="store_true",
                   help="record a failing control instead of aborting")
    p.add_argument("--output", default="results/exp34")
    p.add_argument("--gate", action="store_true")
    args = p.parse_args(argv)

    if args.gate:
        args.tasks = ["mrpc"]
        args.n_train, args.n_val = 512, 256
        args.epochs, args.recover_epochs = 1, 1
        args.ratios = [0.50]
        args.no_abort = True

    from transformers import AutoTokenizer
    from sal import SALConfig, slice_heads
    from exp23_baselines import magnitude_heads
    import run_sal_qat_prototype as P

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    rule("exp34 — BERT-base / GLUE: DropHead vs magnitude + recovery")
    log(f"  seed {args.seed} | tasks {args.tasks} | ratios {args.ratios} | "
        f"{args.epochs} epochs, recovery {args.recover_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    payload = {
        "experiment": "exp34_bert_glue", "model": MODEL_ID, "dataset": GLUE,
        "seed": args.seed, "gate": args.gate, "epochs": args.epochs,
        "recover_epochs": args.recover_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "max_len": MAX_LEN, "ratios": args.ratios,
        "control_tolerance_pp": args.control_tolerance,
        "notes": {
            "control": "The dense fine-tune is the matched control; the gain "
                       "of every head arm is measured against it, not against "
                       "the published reference. The reference only gates "
                       "whether the control is sane.",
            "head_set": "Both head arms are scored on the same magnitude set, "
                        "uniform per layer, so neither faces a perturbation "
                        "the other did not (exp27's protocol).",
            "recovery": "2 epochs by default (exp29: the curve peaks at 2 and "
                        "5 is worse on 3/3 seeds).",
        },
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "tasks": {},
    }

    def save():
        (outdir / "exp34_bert_glue.json").write_text(json.dumps(payload, indent=2))

    save()

    for task in args.tasks:
        rule(f"{task}")
        set_seed(args.seed)
        train, val, n_labels, ref = build_task(
            task, tokenizer, args.n_train, args.n_val, args.seed)
        tl, vl = loaders(train, val, args.batch_size, args.seed)
        log(f"  {len(train)} train / {len(val)} val | {n_labels} labels | "
            f"reference {ref:.1f}%")
        entry = {"n_train": len(train), "n_val": len(val),
                 "num_labels": n_labels, "reference_accuracy": ref / 100}
        payload["tasks"][task] = entry
        save()

        # 1 ------------------------------------------------ dense control
        set_seed(args.seed)
        dense = load_model(n_labels, device)
        t0 = time.time()
        entry["dense_losses"] = train_loop(dense, tl, args.epochs, device,
                                           use_amp, label=f"{task}/dense",
                                           lr=args.lr)
        entry["dense_seconds"] = round(time.time() - t0, 1)
        want_f1 = task in REFERENCE_F1
        if want_f1:
            acc, f1 = evaluate(dense, vl, device, use_amp, want_f1=True)
            entry["dense_f1"] = f1
            entry["reference_f1"] = REFERENCE_F1[task] / 100
            entry["dense_f1_vs_reference_pp"] = round(
                (f1 - REFERENCE_F1[task] / 100) * 100, 3)
        else:
            acc = evaluate(dense, vl, device, use_amp)
        entry["dense_accuracy"] = acc
        gap = ref / 100 - acc
        entry["dense_vs_reference_pp"] = round(-gap * 100, 3)
        log(f"  dense control {acc * 100:.2f}% accuracy "
            f"(reference {ref:.1f}%, {-gap * 100:+.2f}pp)"
            + (f" | F1 {entry['dense_f1'] * 100:.2f}% "
               f"(reference {REFERENCE_F1[task]:.1f}%, "
               f"{entry['dense_f1_vs_reference_pp']:+.2f}pp)" if want_f1 else ""))
        save()
        if gap * 100 > args.control_tolerance:
            msg = (f"{task}: dense control {acc * 100:.2f}% is "
                   f"{gap * 100:.2f}pp below the {ref:.1f}% reference, outside "
                   f"the {args.control_tolerance}pp tolerance")
            entry["control_gate"] = {"passed": False, "reason": msg}
            save()
            if not args.no_abort:
                raise RuntimeError(
                    msg + " — aborting before the head arms, because their "
                    "gains would be measured against a broken control")
            log(f"  WARNING: {msg}")
        else:
            entry["control_gate"] = {"passed": True}
        save()

        for ratio in args.ratios:
            rk = f"p{int(ratio * 100)}"
            log(f"  --- {task} at {ratio:.0%} of heads ---")
            pairs = magnitude_heads(dense, ratio, per_layer=True)
            log(f"    magnitude set: {len(pairs)} heads, uniform per layer")
            cell = {"heads_removed": len(pairs)}

            # 2 ------------------------------------------------- DropHead
            set_seed(args.seed)
            dh = load_model(n_labels, device)
            cfg = SALConfig.auto(dh, prune_fraction=ratio,
                                 uniform_per_layer=True)
            masker = P.ResampleMasker(dh, cfg, seed=args.seed, mode="drophead")
            t0 = time.time()
            cell["drophead_losses"] = train_loop(
                dh, tl, args.epochs, device, use_amp, masker=masker,
                label=f"{task}/drophead-{rk}", lr=args.lr)
            cell["drophead_seconds"] = round(time.time() - t0, 1)
            cell["drophead_dense_accuracy"] = evaluate(dh, vl, device, use_amp)
            dh_small = slice_heads(copy.deepcopy(dh), pairs).to(device)
            cell["drophead_accuracy"] = evaluate(dh_small, vl, device, use_amp)
            log(f"    DropHead: dense {cell['drophead_dense_accuracy']*100:.2f}%"
                f" | sliced {cell['drophead_accuracy'] * 100:.2f}%")
            save()

            # 3 --------------------------------- magnitude + recovery
            small = slice_heads(copy.deepcopy(dense), pairs).to(device)
            cell["magnitude_oneshot_accuracy"] = evaluate(small, vl, device,
                                                          use_amp)
            set_seed(args.seed)
            t0 = time.time()
            cell["recovery_losses"] = train_loop(
                small, tl, args.recover_epochs, device, use_amp,
                label=f"{task}/magnitude-recover-{rk}", lr=args.lr)
            cell["recovery_seconds"] = round(time.time() - t0, 1)
            cell["magnitude_recovery_accuracy"] = evaluate(small, vl, device,
                                                           use_amp)
            log(f"    magnitude: one-shot "
                f"{cell['magnitude_oneshot_accuracy'] * 100:.2f}% | "
                f"+{args.recover_epochs}ep "
                f"{cell['magnitude_recovery_accuracy'] * 100:.2f}%")

            cell["drophead_minus_magnitude_pp"] = round(
                (cell["drophead_accuracy"]
                 - cell["magnitude_recovery_accuracy"]) * 100, 3)
            cell["drophead_vs_control_pp"] = round(
                (cell["drophead_accuracy"] - entry["dense_accuracy"]) * 100, 3)
            entry[rk] = cell
            save()

    rule("done")
    log(f"  wrote {outdir / 'exp34_bert_glue.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
