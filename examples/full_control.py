"""Full control — manual SALConfig + standalone SALTrainer (no HF Trainer).

Shows every SAL knob explicitly, trains GPT-2 small for ~50 steps with a custom
prune fraction, and reports the masker stats plus the Fragility Index before and
after training (FI "history").

    python examples/full_control.py
"""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal import SALConfig, FIScanner
from sal.trainer import SALTrainer

MODEL = "gpt2"


def load_sentences(n: int = 100):
    try:
        from datasets import load_dataset
        return list(load_dataset("glue", "sst2", split=f"train[:{n}]")["sentence"])
    except Exception:
        base = ["the film was a quiet triumph", "an utterly tedious experience",
                "bright, funny and warm", "a joyless and clumsy mess"]
        return [base[i % len(base)] for i in range(n)]


def build_loader(tok, sentences, batch_size=4):
    enc = tok(sentences, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    labels = ids.clone()
    labels[mask == 0] = -100  # ignore padding in the LM loss

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return ids.size(0)
        def __getitem__(self, i):
            return {"input_ids": ids[i], "attention_mask": mask[i], "labels": labels[i]}

    return DataLoader(DS(), batch_size=batch_size, shuffle=True)


def make_probe(tok, sentences, n_batches=2, batch_size=8):
    probe = []
    for b in range(n_batches):
        chunk = sentences[b * batch_size:(b + 1) * batch_size] or sentences[:batch_size]
        enc = tok(chunk, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        probe.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return probe


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

    sentences = load_sentences(100)
    loader = build_loader(tok, sentences, batch_size=4)  # 25 batches x 2 epochs = ~50 steps

    # Every parameter spelled out (no SALConfig.auto here).
    config = SALConfig(
        num_layers=12,
        num_heads_per_layer=12,
        prune_fraction=0.40,          # custom: silence 40% of heads
        prune_start_ratio=0.10,       # start pruning at 10% of training
        prune_end_ratio=0.80,         # reach full prune fraction by 80%
        prune_interval=2,             # re-evaluate the pruned set every 2 steps
        schedule="random",            # validated default
        fi_interval=10,               # FIMonitor would scan every 10 steps (HF Trainer)
        attention_pattern="h.{}.attn",
        head_dim=64,
    )
    print("SALConfig:")
    for k in ("num_layers", "num_heads_per_layer", "prune_fraction", "prune_start_ratio",
              "prune_end_ratio", "prune_interval", "schedule", "fi_interval"):
        print(f"  {k:<20} = {getattr(config, k)}")
    print(f"  total_heads          = {config.total_heads}")
    print(f"  num_heads_to_prune   = {config.num_heads_to_prune}\n")

    probe = make_probe(tok, sentences)
    fi_history = []
    fi_history.append(FIScanner(model, probe, num_samples=16, batch_size=8).scan().fi_score)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    trainer = SALTrainer(model, config, optimizer, loader, seed=42)
    result = trainer.train(num_epochs=2, log_interval=10)

    fi_history.append(FIScanner(model, probe, num_samples=16, batch_size=8).scan().fi_score)

    print(f"Training: {result['total_steps']} optimizer steps, "
          f"{result['masker_stats']['prune_events']} prune events")
    print(f"Loss per epoch: {[round(x, 4) for x in result['losses']]}")
    print(f"FI history [before -> after]: {fi_history[0]:.4f} -> {fi_history[1]:.4f}")


if __name__ == "__main__":
    main()
