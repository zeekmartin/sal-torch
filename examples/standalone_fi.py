"""Standalone Fragility Index scan — diagnostic only, no training.

Scans a pretrained model's attention graph and reports a structural fragility
score plus a per-layer IMMUNE / BUFFER / CRITICAL map, then saves a JSON report.

    python examples/standalone_fi.py

Runs in well under 30 seconds on CPU. SAL is not involved here — FI is an
independent diagnostic.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sal import FIScanner

MODEL = "distilbert-base-uncased"


def make_probe(tok, n_batches: int = 3, batch_size: int = 8):
    """A few batches of generic text to drive activation extraction."""
    text = [
        "the market opened higher on strong earnings",
        "she walked the dog along the quiet river path",
        "the committee will review the proposal next week",
        "rain is expected across the region tomorrow",
        "the new engine improves fuel efficiency markedly",
        "students gathered in the hall for the lecture",
        "the recipe calls for two cups of flour",
        "investors remain cautious ahead of the report",
    ]
    probe = []
    for _ in range(n_batches):
        enc = tok(text[:batch_size], truncation=True, padding="max_length",
                  max_length=32, return_tensors="pt")
        probe.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return probe


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.eval()

    probe = make_probe(tok)
    result = FIScanner(model, probe, num_samples=24, batch_size=8).scan()

    print(f"Model: {MODEL}")
    print(f"Fragility Index: {result.fi_score:.4f}   (0 = robust, 1 = fragile)")
    print(result.summary)
    print(f"\nPer-layer map:")
    for layer, klass in result.layer_map.items():
        print(f"  layer {layer:>2}: {klass.value}")
    if result.critical_layers:
        print(f"\nCritical layers (handle with care when compressing): {result.critical_layers}")

    out = "fi_report.json"
    result.save(out)
    print(f"\nSaved JSON report -> {out}")


if __name__ == "__main__":
    main()
