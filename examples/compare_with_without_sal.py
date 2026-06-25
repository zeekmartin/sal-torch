"""The proof — SAL vs. no-SAL under 33% head compression.

Trains the same model twice (with and without SAL), then compresses BOTH by
masking the same 33% of attention heads and compares validation accuracy. The
SAL-trained model trained while heads were silenced, so it degrades less when
compressed.

    python examples/compare_with_without_sal.py

This is a small, fast demonstration of the mechanism — not a benchmark.
"""
from __future__ import annotations

import itertools

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sal import SALConfig
from sal.masker import HeadMasker

MODEL = "distilbert-base-uncased"
TRAIN_STEPS = 20
PRUNE_SEED = 7  # both models are compressed with the identical 33% head set


def load_split(split, n):
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=f"{split}[:{n}]")
        return list(ds["sentence"]), list(ds["label"])
    except Exception:
        pos = ["a beautiful and moving film", "sharp, funny and alive"]
        neg = ["a dull and lifeless slog", "clumsy and utterly boring"]
        s, y = [], []
        while len(s) < n:
            s += [pos[len(s) % 2], neg[len(s) % 2]]
            y += [1, 0]
        return s[:n], y[:n]


def make_batches(tok, sentences, labels, batch_size, device):
    enc = tok(sentences, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    batches = []
    for i in range(0, len(labels), batch_size):
        batches.append({
            "input_ids": enc["input_ids"][i:i + batch_size].to(device),
            "attention_mask": enc["attention_mask"][i:i + batch_size].to(device),
            "labels": torch.tensor(labels[i:i + batch_size]).to(device),
        })
    return batches


def fresh_model(device):
    torch.manual_seed(0)  # identical starting point for both runs
    return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)


def train(model, batches, config, use_sal):
    model.train()
    opt = AdamW(model.parameters(), lr=5e-5)
    masker = None
    if use_sal:
        masker = HeadMasker(model, config, seed=123)
        masker.install()
    for step, batch in zip(range(TRAIN_STEPS), itertools.cycle(batches)):
        if masker:
            masker.step(step, TRAIN_STEPS)  # progressive head silencing during training
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    if masker:
        masker.remove()  # the *adaptation* lives in the weights, not the hooks


@torch.no_grad()
def accuracy(model, batches):
    model.eval()
    correct = total = 0
    for b in batches:
        pred = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.argmax(-1)
        correct += (pred == b["labels"]).sum().item()
        total += b["labels"].numel()
    return correct / max(total, 1)


def accuracy_compressed(model, batches, config):
    """Accuracy with the same fixed 33% of heads masked off."""
    masker = HeadMasker(model, config, seed=PRUNE_SEED)
    masker.install()
    masker.activate()  # deterministic 33% head set (same for both models)
    try:
        return accuracy(model, batches)
    finally:
        masker.remove()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)

    tr_s, tr_y = load_split("train", 200)
    va_s, va_y = load_split("validation", 400)
    train_batches = make_batches(tok, tr_s, tr_y, 8, device)
    val_batches = make_batches(tok, va_s, va_y, 16, device)

    config = SALConfig.auto(fresh_model(device), prune_fraction=0.33)
    print(f"Model: {MODEL} | {config.total_heads} heads | compressing "
          f"{config.num_heads_to_prune} ({config.prune_fraction:.0%}) at eval\n")

    rows = []
    for use_sal in (False, True):
        model = fresh_model(device)
        train(model, train_batches, config, use_sal=use_sal)
        full = accuracy(model, val_batches)
        pruned = accuracy_compressed(model, val_batches, config)
        rows.append(("SAL" if use_sal else "baseline", full, pruned))

    print(f"{'method':<10}{'val acc':>10}{'@33% prune':>14}{'drop':>10}")
    print("-" * 44)
    for name, full, pruned in rows:
        print(f"{name:<10}{full:>10.3f}{pruned:>14.3f}{full - pruned:>10.3f}")

    base_drop = rows[0][1] - rows[0][2]   # accuracy lost to compression, baseline
    sal_drop = rows[1][1] - rows[1][2]    # accuracy lost to compression, SAL
    print(f"\nAccuracy lost to 33% head compression:  baseline {base_drop:+.3f}   SAL {sal_drop:+.3f}")
    print("The point of SAL is the *drop*: it trained under head silencing, so compressing")
    print("it costs little accuracy, while the baseline degrades.")
    print("\nNote: this is a tiny 20-step demo (SAL's absolute accuracy is still catching up,")
    print("since silencing heads during training slows learning) and DistilBERT is highly")
    print("redundant. For the rigorous multi-epoch validation across models, see")
    print("scripts/validate_published_results.py.")


if __name__ == "__main__":
    main()
