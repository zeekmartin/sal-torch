"""Head-selection experiment — is the v0.5.0 regression a selection problem?

The v0.5.0 pipeline run put a SAL-trained GPT-2 Medium *behind* a no-SAL control
under compression (0.8398 vs 0.8633), reversing the v0.4.0 result on the same
model and task. The one nameable difference between the two runs is **which
heads get removed**: `CompressionPipeline.compress()` takes the lowest-magnitude
heads, while the v0.4.0 battery took random ones — and random removal is exactly
what SAL trains against.

So: both arms, three selection strategies, everything else held fixed.

  ``magnitude``   lowest output-projection slice norm within each layer
  ``random``      a random set within each layer — what v0.4.0 used
  ``fi_guided``   IMMUNE layers first, then BUFFER, then CRITICAL only if the
                  budget cannot be met without them

All three remove the **same number of heads**, and every cell uses masking
rather than slicing, so the only variable is *which* heads. Masking also makes
this directly comparable to the v0.4.0 battery. Quantization to INT4 follows, so
each cell is the full compression treatment.

If SAL wins under ``random`` and loses under ``magnitude``, the v0.5.0 regression
is a default-selection problem and the fix is a better default. If SAL loses
under all three, it is not about selection.

One caveat the run itself will report: on GPT-2 few layers classify as
IMMUNE/BUFFER, so ``fi_guided`` has to spill into CRITICAL layers to match the
budget. The spill count is printed — read that row with it in mind.

Usage::

    modal run scripts/modal_selection_experiment.py
"""
import json

import modal

app = modal.App("sal-torch-selection")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "numpy", "accelerate>=1.1.0",
                 "bitsandbytes")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

MODEL = "gpt2-medium"
N_TRAIN = 2048
N_EVAL = 512
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
PRUNE_FRACTION = 0.33
STRATEGIES = ("magnitude", "random", "fi_guided")
RESULTS_PATH = "scripts/selection_experiment_results.json"


def _load():
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2ForSequenceClassification

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id

    sst = load_dataset("stanfordnlp/sst2", split="train")
    train, ev = sst.select(range(N_TRAIN)), sst.select(range(N_TRAIN, N_TRAIN + N_EVAL))

    def batches(split):
        out = []
        sents, labels = list(split["sentence"]), list(split["label"])
        for i in range(0, len(sents), BATCH_SIZE):
            enc = tok(sents[i:i + BATCH_SIZE], padding="max_length", truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out.append({"input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": torch.tensor(labels[i:i + BATCH_SIZE])})
        return out

    return model, batches(train), batches(ev)


def _to_dev(batches, device):
    return [{k: v.to(device) for k, v in b.items()} for b in batches]


def _train_arm(arm, model, train_b, eval_b, device):
    """Train one arm and return (trained_model, dense_accuracy)."""
    import time

    import torch
    from torch.optim import AdamW

    from sal import CompressionPipeline
    from sal.robustness import _evaluate

    model.to(device)
    t0 = time.time()
    if arm == "SAL":
        pipe = CompressionPipeline(model, eval_b, metric="accuracy",
                                   batch_size=BATCH_SIZE)
        pipe.sal_train(train_b, epochs=EPOCHS, prune_fraction=PRUNE_FRACTION, lr=LR)
        trained = pipe.model
    else:
        torch.manual_seed(0)
        model.train()
        opt = AdamW(model.parameters(), lr=LR)
        for _ in range(EPOCHS):
            for batch in train_b:
                model(**batch).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        model.eval()
        trained = model

    dense = _evaluate(trained, eval_b, "accuracy", BATCH_SIZE)
    print(f"[{arm}] trained in {time.time() - t0:.0f}s, dense accuracy {dense:.4f}",
          flush=True)
    return trained, dense


def _run_cell(arm, strategy, master, eval_b, device):
    """One (arm, strategy) cell: mask the selected heads, quantize, evaluate."""
    import copy
    import time

    import torch

    from sal import CompressionPipeline

    t0 = time.time()
    model = copy.deepcopy(master).to(device)
    pipe = CompressionPipeline(model, eval_b, metric="accuracy", batch_size=BATCH_SIZE)
    try:
        pipe.compress(pruning=PRUNE_FRACTION, quantization="int4",
                      slice_heads=False, strategy=strategy)
        report = pipe.validate()
        cell = {"accuracy": report.last.accuracy, "size_mb": report.last.size_mb,
                "heads_removed": len(pipe._pruned_heads),
                "spill": getattr(pipe, "_fi_guided_spill", None),
                "seconds": round(time.time() - t0, 1), "error": None}
        print(f"  [{arm}/{strategy}] acc={cell['accuracy']:.4f}  "
              f"{cell['heads_removed']} heads  {cell['size_mb']:.0f}MB"
              + (f"  (spill={cell['spill']})" if cell["spill"] else ""), flush=True)
    except Exception as e:  # noqa: BLE001 — one dead cell must not sink the grid
        cell = {"accuracy": None, "error": str(e), "seconds": round(time.time() - t0, 1)}
        print(f"  [{arm}/{strategy}] FAILED: {e}", flush=True)
    finally:
        del pipe, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cell


@app.function(image=image, gpu="T4", timeout=3600, memory=32768)
def selection_grid():
    import copy

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== head-selection experiment on {MODEL} ({device}) ===", flush=True)
    print(f"    2 arms x {len(STRATEGIES)} strategies, pruning={PRUNE_FRACTION}, "
          f"then INT4; masked (not sliced) so only head choice varies", flush=True)

    torch.manual_seed(0)
    model, train_b, eval_b = _load()
    train_b, eval_b = _to_dev(train_b, device), _to_dev(eval_b, device)

    results = {}
    for arm in ("standard", "SAL"):
        fresh = copy.deepcopy(model)
        trained, dense = _train_arm(arm, fresh, train_b, eval_b, device)
        master = trained.cpu()
        del trained, fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results[arm] = {"dense": dense, "cells": {}}
        for strategy in STRATEGIES:
            results[arm]["cells"][strategy] = _run_cell(arm, strategy, master, eval_b,
                                                        device)
        del master
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _report(results)
    return results


def _report(results: dict):
    print("\n========== 2 x 3 SELECTION GRID ==========", flush=True)
    head = f"{'arm':<10}{'dense':>9}" + "".join(f"{s:>13}" for s in STRATEGIES)
    print(head, flush=True)
    print("-" * len(head), flush=True)
    for arm in ("standard", "SAL"):
        row = f"{arm:<10}{results[arm]['dense']:>9.4f}"
        for s in STRATEGIES:
            a = results[arm]["cells"][s].get("accuracy")
            row += f"{(f'{a:.4f}' if a is not None else 'FAILED'):>13}"
        print(row, flush=True)

    print("\nretention (compressed / that arm's own dense):", flush=True)
    print(head, flush=True)
    print("-" * len(head), flush=True)
    for arm in ("standard", "SAL"):
        d = results[arm]["dense"]
        row = f"{arm:<10}{1.0:>9.2%}"
        for s in STRATEGIES:
            a = results[arm]["cells"][s].get("accuracy")
            row += f"{(f'{a / d:.2%}' if a and d else '-'):>13}"
        print(row, flush=True)

    print("\nSAL minus standard, per strategy:", flush=True)
    verdict = {}
    for s in STRATEGIES:
        sa = results["SAL"]["cells"][s].get("accuracy")
        ba = results["standard"]["cells"][s].get("accuracy")
        if sa is None or ba is None:
            print(f"  {s:<12} n/a", flush=True)
            continue
        verdict[s] = sa - ba
        winner = "SAL" if sa > ba else ("standard" if sa < ba else "tie")
        print(f"  {s:<12} {sa - ba:+.4f}   winner: {winner}", flush=True)

    spill = results["SAL"]["cells"].get("fi_guided", {}).get("spill")
    if spill:
        print(f"\n  note: fi_guided placed {spill} heads in CRITICAL layers to match "
              "the budget; it is not a pure 'never CRITICAL' run.", flush=True)
    print(f"  noise floor: 1 example = {1 / N_EVAL:.3%}", flush=True)
    print("  (single seed, one model)", flush=True)

    wins = [s for s, d in verdict.items() if d > 0]
    if not verdict:
        print("\nverdict: no cell completed on both arms.", flush=True)
    elif len(wins) == len(verdict):
        print("\nverdict: SAL wins under every strategy — the v0.5.0 regression was "
              "not about selection.", flush=True)
    elif wins:
        print(f"\nverdict: SAL wins only under {wins} — the v0.5.0 regression is a "
              "SELECTION problem, and the default should change.", flush=True)
    else:
        print("\nverdict: SAL loses under every strategy — the regression is NOT a "
              "selection problem.", flush=True)


@app.local_entrypoint()
def main():
    results = selection_grid.remote()
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
