"""Validate CompressionPipeline end to end on a real model via Modal.

The v0.4.0 benchmarks established the recipe — fully fine-tune with SAL, then
compress — by wiring the pieces together by hand. v0.5.0 puts that path behind
one object. This checks the object reproduces the result.

GPT-2 Medium on SST-2, one T4:

  1. ``scan()``      — fragility, absorption map, projected sizes
  2. ``sal_train()`` — 3 epochs, full fine-tuning, prune_fraction=0.33
  3. ``compress()``  — 33% of heads physically sliced out, then INT4
  4. ``validate()``  — the measured waterfall, size and accuracy per stage
  5. ``export()``    — write it out and reload it
  6. reload check    — the exported model scores what the in-memory one scored

A standard (no-SAL) arm runs the same compression so the final number has
something to be compared against; without it "0.9 accuracy at 300MB" is
unfalsifiable.

Sequence classification is used rather than the log-likelihood protocol from
``modal_robustness_scale.py`` because CompressionPipeline's built-in accuracy
metric is argmax-over-logits — for a 2-class head that is exactly right, and it
keeps the pipeline under test rather than a bespoke harness.

Usage::

    modal run scripts/modal_v050_test.py
"""
import modal

app = modal.App("sal-torch-v050")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "numpy", "accelerate>=1.1.0",
                 "bitsandbytes", "matplotlib", "fpdf2")
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


def _load():
    """GPT-2 Medium with a 2-class head, plus tokenized SST-2 batches."""
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2ForSequenceClassification

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id

    sst = load_dataset("stanfordnlp/sst2", split="train")
    train = sst.select(range(N_TRAIN))
    ev = sst.select(range(N_TRAIN, N_TRAIN + N_EVAL))

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


def _run_arm(arm: str, model, train_b, eval_b, device, use_sal: bool):
    """Take one model through the pipeline. Returns (report, export_result)."""
    import time

    from sal import CompressionPipeline

    print(f"\n{'=' * 70}\n[{arm}] CompressionPipeline\n{'=' * 70}", flush=True)
    model.to(device)
    pipe = CompressionPipeline(model, eval_b, metric="accuracy", batch_size=BATCH_SIZE,
                               model_name=f"{MODEL} ({arm})")

    print(f"[{arm}] scan()...", flush=True)
    print(pipe.scan().recommendation, flush=True)

    if use_sal:
        print(f"\n[{arm}] sal_train({EPOCHS} epochs, full fine-tune, "
              f"prune_fraction={PRUNE_FRACTION})...", flush=True)
        t0 = time.time()
        pipe.sal_train(train_b, epochs=EPOCHS, prune_fraction=PRUNE_FRACTION, lr=LR)
        print(f"[{arm}] trained in {time.time() - t0:.0f}s", flush=True)
    else:
        # Same budget, no masking — otherwise the arms differ in training too.
        print(f"\n[{arm}] plain fine-tune ({EPOCHS} epochs, no SAL)...", flush=True)
        t0 = time.time()
        _plain_train(pipe, train_b, device)
        print(f"[{arm}] trained in {time.time() - t0:.0f}s", flush=True)

    print(f"\n[{arm}] compress(pruning={PRUNE_FRACTION}, quantization='int4', "
          f"slice_heads=True)...", flush=True)
    pipe.compress(pruning=PRUNE_FRACTION, quantization="int4", slice_heads=True)

    report = pipe.validate()
    print(f"\n[{arm}] waterfall:", flush=True)
    print(report.table, flush=True)

    print(f"\n[{arm}] export()...", flush=True)
    export = pipe.export(f"/tmp/{arm}_compressed")
    print(f"[{arm}] export: {export}", flush=True)

    return pipe, report, export


def _plain_train(pipe, batches, device):
    """The no-SAL control: identical loop, no masker."""
    import torch
    from torch.optim import AdamW

    torch.manual_seed(pipe.seed)
    model = pipe.model
    model.train()
    opt = AdamW(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for batch in batches:
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    pipe._record("fine_tuned", detail=f"{EPOCHS} epochs, full fine-tune, no SAL")


@app.function(image=image, gpu="T4", timeout=3600, memory=32768)
def pipeline_end_to_end():
    import copy

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== v0.5.0 CompressionPipeline check on {MODEL} ({device}) ===", flush=True)

    torch.manual_seed(0)
    model, train_b, eval_b = _load()
    train_b = _to_dev(train_b, device)
    eval_b = _to_dev(eval_b, device)

    sal_model = copy.deepcopy(model)
    base_model = copy.deepcopy(model)
    del model

    sal_pipe, sal_report, sal_export = _run_arm("SAL", sal_model, train_b, eval_b,
                                                device, use_sal=True)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    base_pipe, base_report, base_export = _run_arm("standard", base_model, train_b, eval_b,
                                                   device, use_sal=False)

    # --- reload check: does the exported artifact score what we measured? -------
    reload_acc = None
    try:
        from sal.robustness import _evaluate
        reloaded = type(sal_pipe.model).from_pretrained(sal_export["path"])
        reloaded.to(device).eval()
        reload_acc = _evaluate(reloaded, eval_b, "accuracy", BATCH_SIZE)
        print(f"\nreloaded SAL model accuracy: {reload_acc:.4f}", flush=True)
    except Exception as e:  # noqa: BLE001 — report the failure, do not lose the run
        print(f"\nreload check unavailable: {e}", flush=True)

    try:
        sal_report.save("/tmp/compression_report.pdf")
        print("compression PDF written", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PDF skipped: {e}", flush=True)

    def summarize(name, rep):
        return {"arm": name, "stages": [s.name for s in rep.stages],
                "original_mb": rep.first.size_mb, "final_mb": rep.last.size_mb,
                "original_acc": rep.first.accuracy, "final_acc": rep.last.accuracy,
                "size_ratio": rep.size_ratio, "summary": rep.summary}

    s, b = summarize("SAL", sal_report), summarize("standard", base_report)
    print("\n========== v0.5.0 SUMMARY ==========", flush=True)
    for r in (b, s):
        print(f"  {r['arm']:<9} {r['original_mb']:.0f}MB/{r['original_acc']:.4f} -> "
              f"{r['final_mb']:.0f}MB/{r['final_acc']:.4f}  ({r['size_ratio']:.1f}x)",
              flush=True)
    print(f"\n  compressed accuracy: SAL {s['final_acc']:.4f} vs "
          f"standard {b['final_acc']:.4f} "
          f"({s['final_acc'] - b['final_acc']:+.4f})", flush=True)
    print(f"  noise floor: 1 example = {1 / N_EVAL:.3%}", flush=True)
    print("  (single seed, one model)", flush=True)

    return {"sal": s, "standard": b, "reloaded_accuracy": reload_acc,
            "sal_export": sal_export, "standard_export": base_export}


@app.local_entrypoint()
def main():
    res = pipeline_end_to_end.remote()
    print("\n========== RESULT ==========")
    import json
    print(json.dumps(res, indent=2, default=str))
