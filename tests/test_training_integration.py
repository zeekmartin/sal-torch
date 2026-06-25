"""End-to-end SAL training plumbing test (DistilBERT + SST-2).

This is NOT a validation run — it verifies the pipeline wires together:
SALConfig.auto -> SALCallback -> HF Trainer, plus FIScanner before/after.

Marked ``integration`` (network + model downloads); skipped unless
``pytest --run-integration`` is passed. The ``check_sal_training`` helper is
plain so the Modal runner can call the same logic remotely.
"""
from __future__ import annotations
import math
import statistics
import tempfile
import pytest
import torch

MODEL = "distilbert-base-uncased"


def _load_sst2_subset(n: int = 50):
    """50 (sentence, label) pairs — from the GLUE SST-2 train split if available."""
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=f"train[:{n}]")
        return list(ds["sentence"]), list(ds["label"])
    except Exception:
        # Offline fallback: a tiny hand-built SST-2-style sample.
        pos = ["a triumph of storytelling", "warm, funny and deeply moving",
               "a beautiful, tender film", "wonderfully acted and paced"]
        neg = ["a dull, lifeless slog", "painfully boring and overlong",
               "a charmless, joyless mess", "tedious and utterly forgettable"]
        sents, labels = [], []
        while len(sents) < n:
            sents.append(pos[len(sents) % len(pos)]); labels.append(1)
            sents.append(neg[len(sents) % len(neg)]); labels.append(0)
        return sents[:n], labels[:n]


def _make_probe(dataset, batch_size=8, n_batches=2):
    probe = []
    for b in range(n_batches):
        sl = range(b * batch_size, (b + 1) * batch_size)
        probe.append({
            "input_ids": torch.stack([dataset[i]["input_ids"] for i in sl]),
            "attention_mask": torch.stack([dataset[i]["attention_mask"] for i in sl]),
        })
    return probe


def check_sal_training() -> dict:
    """Run the end-to-end plumbing test. Raises on failure; returns a report."""
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    from sal import SALConfig, SALCallback, FIScanner

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    sentences, labels = _load_sst2_subset(50)
    enc = tok(sentences, truncation=True, padding="max_length", max_length=32, return_tensors="pt")

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, i):
            return {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                    "labels": torch.tensor(labels[i])}
    dataset = DS()

    config = SALConfig.auto(model, prune_fraction=0.33)
    probe = _make_probe(dataset, batch_size=8, n_batches=2)

    # FI before training
    fi_before = FIScanner(model, probe, num_samples=16, batch_size=8).scan().fi_score

    # Record masker stats live (HF Trainer appends to log_history *before* on_log,
    # so we read the masker directly rather than via injected log fields).
    class RecordingSAL(SALCallback):
        peak_events = 0
        peak_pruned = 0
        was_active = False

        def on_step_end(self, args, state, control, **kw):
            if self.masker:
                s = self.masker.stats
                self.peak_events = max(self.peak_events, s["prune_events"])
                self.peak_pruned = max(self.peak_pruned, s["pruned_heads"])
                self.was_active = self.was_active or s["active"]

    with tempfile.TemporaryDirectory() as out:
        args = TrainingArguments(
            output_dir=out, max_steps=20, per_device_train_batch_size=8,
            learning_rate=5e-5, logging_steps=1, save_strategy="no",
            report_to=[], disable_tqdm=True,
        )
        callback = RecordingSAL(config, seed=42)
        trainer = Trainer(model=model, args=args, train_dataset=dataset, callbacks=[callback])
        trainer.train()

    log = trainer.state.log_history
    losses = [e["loss"] for e in log if "loss" in e]
    prune_events = callback.peak_events
    pruned_peak = callback.peak_pruned

    # FI after training
    fi_after = FIScanner(model, probe, num_samples=16, batch_size=8).scan().fi_score

    # ---- assertions ----
    assert len(losses) >= 2, f"expected logged losses, got {losses}"
    assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
    assert callback.was_active, "masker never activated during training"
    assert prune_events > 0, "masker never pruned (prune_events == 0)"
    assert pruned_peak > 0, "masker never activated (no heads pruned)"
    k = min(5, len(losses) // 2)
    first, last = statistics.mean(losses[:k]), statistics.mean(losses[-k:])
    assert last < first, f"training loss did not decrease: first={first:.4f} last={last:.4f}"
    assert 0.0 <= fi_before <= 1.0, f"fi_before out of range: {fi_before}"
    assert 0.0 <= fi_after <= 1.0, f"fi_after out of range: {fi_after}"

    return {"model": MODEL, "steps": 20, "prune_events": prune_events,
            "pruned_peak": pruned_peak, "loss_first": round(first, 4), "loss_last": round(last, 4),
            "fi_before": round(fi_before, 4), "fi_after": round(fi_after, 4), "passed": True}


@pytest.mark.integration
def test_sal_training_pipeline():
    report = check_sal_training()
    assert report["passed"]


def run(verbose: bool = True) -> dict:
    """Run the training plumbing test (used by the Modal runner)."""
    try:
        r = check_sal_training()
    except Exception as e:  # noqa: BLE001
        r = {"model": MODEL, "passed": False, "error": f"{type(e).__name__}: {e}"}
    if verbose:
        print(f"[training] {r}", flush=True)
    return r
