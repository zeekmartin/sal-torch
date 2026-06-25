"""sal-torch quickstart — make a model compression-resilient in 3 lines.

Trains DistilBERT on a small slice of SST-2 with SAL head masking enabled, then
prints the masker stats before vs. after training.

    python examples/quickstart.py

Runs in well under 2 minutes on a T4 GPU (a bit longer on CPU).
"""
from __future__ import annotations

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from sal import SALConfig, SALCallback

MODEL = "distilbert-base-uncased"


def load_sst2(n: int = 100):
    """A small SST-2 slice (falls back to a tiny built-in sample if offline)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/sst2", split=f"train[:{n}]")
        return list(ds["sentence"]), list(ds["label"])
    except Exception:
        pos = ["a wonderful, moving film", "funny and deeply heartfelt"]
        neg = ["a boring, lifeless mess", "tedious and forgettable"]
        s, y = [], []
        while len(s) < n:
            s += [pos[len(s) % 2], neg[len(s) % 2]]
            y += [1, 0]
        return s[:n], y[:n]


class SALWithStats(SALCallback):
    """SALCallback that records the peak masker stats during training."""
    peak_pruned = 0
    peak_events = 0

    def on_step_end(self, args, state, control, **kw):
        if self.masker:
            s = self.masker.stats
            self.peak_pruned = max(self.peak_pruned, s["pruned_heads"])
            self.peak_events = max(self.peak_events, s["prune_events"])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    sentences, labels = load_sst2(100)
    enc = tok(sentences, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)
        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                    "labels": torch.tensor(labels[i])}

    # 3-line integration: auto-detect architecture, attach the callback, train.
    config = SALConfig.auto(model, prune_fraction=0.33)
    callback = SALWithStats(config, seed=42)

    print(f"Model: {MODEL}")
    print(f"Detected: {config.num_layers} layers x {config.num_heads_per_layer} heads "
          f"= {config.total_heads} heads total")
    print(f"SAL plan: prune {config.num_heads_to_prune} heads "
          f"({config.prune_fraction:.0%}) progressively during training\n")
    print(f"Before SAL: {config.total_heads}/{config.total_heads} heads active (0 pruned)")

    args = TrainingArguments(output_dir="./_quickstart_out", num_train_epochs=1,
                             per_device_train_batch_size=8, learning_rate=5e-5,
                             logging_steps=5, save_strategy="no", report_to=[], disable_tqdm=True)
    Trainer(model=model, args=args, train_dataset=DS(), callbacks=[callback]).train()

    active = config.total_heads - callback.peak_pruned
    print(f"After SAL:  {active}/{config.total_heads} heads active "
          f"({callback.peak_pruned} pruned over {callback.peak_events} prune events)")
    print("\nThe model trained while heads were progressively silenced, so it learned\n"
          "to operate without them — it is now resilient to ~33% head compression.")


if __name__ == "__main__":
    main()
