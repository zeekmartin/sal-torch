"""Validate the v0.4.0 robustness suite on a real model via Modal.

The question v0.4.0 exists to answer:

    SAL trains models to survive head pruning. Does that resilience also
    show up under **quantization** — which is how 39% of the practitioners
    we polled actually compress?

The experiment, on DistilBERT / SST-2:

  1. Fine-tune DistilBERT on SST-2 **without** SAL (baseline arm).
  2. Fine-tune an identical copy **with** SAL (prune_fraction=0.33).
     Both arms start from the same weights, see the same batches in the same
     order, and train for the same number of steps. The only difference is
     that the SAL arm progressively silences a third of its attention heads
     during training.
  3. Run `RobustnessTest` on both dense models: INT8, INT4, head pruning at
     33% and 50%, and FFN neuron dropout at 10% and 20%.
  4. Run `robustness_compare` and print the head-to-head table.

The SAL masker is removed before evaluation, so both arms are dense at test
time and the comparison is about the *weights* SAL produced, not about heads
that are still switched off.

This is a single-seed plumbing-and-signal check on one small model, not a
benchmark. Treat a win here as "worth pursuing", not as proof.

T4 is intentional (small model, short runs). bitsandbytes is installed so the
INT4 row uses real NF4 when it works; the suite falls back to simulated INT4
and says so if it does not.

Usage:
    modal run scripts/modal_v040_test.py
"""
import modal

app = modal.App("sal-torch-v040")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "numpy",
                 "accelerate>=1.1.0", "bitsandbytes")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

MODEL = "distilbert-base-uncased"
METHODS = ["int8", "int4", "head_pruning_33", "head_pruning_50",
           "neuron_dropout_10", "neuron_dropout_20"]

N_TRAIN = 2048
N_EVAL = 512
MAX_LEN = 64
BATCH_SIZE = 16
STEPS = 250
LR = 2e-5


def _tokenize(tok, sentences, labels, max_len, batch_size):
    import torch
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tok(sentences[i:i + batch_size], padding="max_length", truncation=True,
                  max_length=max_len, return_tensors="pt")
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "labels": torch.tensor(labels[i:i + batch_size])})
    return out


def _load():
    """DistilBERT (2-class) plus SST-2 train/eval batches, tokenized on CPU."""
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, attn_implementation="eager")

    sst = load_dataset("stanfordnlp/sst2", split="train")
    train = sst.select(range(N_TRAIN))
    evalset = sst.select(range(N_TRAIN, N_TRAIN + N_EVAL))
    train_batches = _tokenize(tok, list(train["sentence"]), list(train["label"]),
                              MAX_LEN, BATCH_SIZE)
    eval_batches = _tokenize(tok, list(evalset["sentence"]), list(evalset["label"]),
                             MAX_LEN, BATCH_SIZE)
    return model, train_batches, eval_batches


def _to_dev(batches, device):
    return [{k: v.to(device) for k, v in b.items()} for b in batches]


def _train(model, batches, device, steps=STEPS, lr=LR, sal_config=None, seed=0):
    """Fine-tune ``model``. With ``sal_config``, drive a HeadMasker alongside.

    The masker is always removed before returning, so the model comes back
    dense — SAL's effect lives in the adapted weights.
    """
    import itertools
    import torch
    from torch.optim import AdamW

    torch.manual_seed(seed)
    # Place the model first: HeadMasker builds its per-layer masks on the model's
    # device at install() time, so installing before .to(device) strands them on CPU.
    model.to(device)

    masker = None
    if sal_config is not None:
        from sal.masker import HeadMasker
        masker = HeadMasker(model, sal_config, seed=seed)
        masker.install()

    model.train()
    opt = AdamW(model.parameters(), lr=lr)
    it = itertools.cycle(batches)
    try:
        for step in range(steps):
            if masker is not None:
                masker.step(step, steps)
            out = model(**next(it))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    finally:
        if masker is not None:
            stats = masker.stats
            masker.remove()
            print(f"    SAL: {stats['pruned_heads']}/{stats['total_heads']} heads "
                  f"silenced by end of training, over {stats['prune_events']} prune events",
                  flush=True)
    model.eval()
    return model


@app.function(image=image, gpu="T4", timeout=3600)
def sal_vs_baseline_robustness():
    import copy
    import torch
    from sal import SALConfig, robustness_compare

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== v0.4.0 robustness check on {MODEL} ({device}) ===", flush=True)

    torch.manual_seed(0)
    model, train_batches, eval_batches = _load()
    train_batches = _to_dev(train_batches, device)
    # Eval batches stay on CPU: RobustnessTest moves each batch to whichever
    # device the model under test is on, and INT8 runs on CPU.

    baseline_model = copy.deepcopy(model)
    sal_model = copy.deepcopy(model)

    print(f"\n[1/4] fine-tuning WITHOUT SAL ({STEPS} steps)...", flush=True)
    _train(baseline_model, train_batches, device, seed=0)

    print(f"[2/4] fine-tuning WITH SAL, prune_fraction=0.33 ({STEPS} steps)...", flush=True)
    sal_config = SALConfig.auto(sal_model, prune_fraction=0.33)
    _train(sal_model, train_batches, device, sal_config=sal_config, seed=0)

    print("\n[3/4] running RobustnessTest on both models...", flush=True)
    # robustness_compare runs both suites internally, so take the per-model
    # reports off it rather than paying for the same evaluations twice.
    comparison = robustness_compare(sal_model, baseline_model, eval_batches,
                                    methods=METHODS, metric="accuracy",
                                    batch_size=BATCH_SIZE, dropout_trials=3, seed=0)
    base_report = comparison.baseline_report
    sal_report = comparison.sal_report

    print("\n--- standard (no SAL) ---", flush=True)
    print(base_report.table, flush=True)
    print("\n--- SAL-trained ---", flush=True)
    print(sal_report.table, flush=True)

    for r in sal_report.results:
        if r.backend or r.note:
            print(f"    {r.method}: backend={r.backend} {r.note}", flush=True)

    print("\n[4/4] head-to-head comparison...", flush=True)
    print(comparison.table, flush=True)

    # --- the question this experiment exists to answer -----------------------
    quant_rows = [r for r in comparison.comparable if r.method in ("int8", "int4")]
    quant_wins = [r for r in quant_rows if r.winner == "SAL"]
    prune_rows = [r for r in comparison.comparable if r.method.startswith("head_pruning")]
    prune_wins = [r for r in prune_rows if r.winner == "SAL"]

    print("\n========== v0.4.0 SUMMARY ==========", flush=True)
    print(f"clean accuracy   standard={base_report.results[0].baseline:.4f}  "
          f"SAL={sal_report.results[0].baseline:.4f}", flush=True)
    print(f"robustness_score standard={base_report.robustness_score:.4f}  "
          f"SAL={sal_report.robustness_score:.4f}", flush=True)
    print(f"\n{comparison.summary}", flush=True)
    print(f"\nquantization (int8/int4): SAL wins {len(quant_wins)}/{len(quant_rows)}", flush=True)
    print(f"head pruning:             SAL wins {len(prune_wins)}/{len(prune_rows)}", flush=True)

    if quant_rows:
        verdict = ("SAL resilience EXTENDS to quantization"
                   if len(quant_wins) * 2 > len(quant_rows)
                   else "SAL resilience does NOT clearly extend to quantization")
    else:
        verdict = "no quantization row completed — inconclusive"
    print(f"\nverdict: {verdict}", flush=True)
    print("(single seed, one model — signal, not proof)", flush=True)

    return {
        "clean_standard": base_report.results[0].baseline,
        "clean_sal": sal_report.results[0].baseline,
        "robustness_standard": base_report.robustness_score,
        "robustness_sal": sal_report.robustness_score,
        "summary": comparison.summary,
        "quant_wins": f"{len(quant_wins)}/{len(quant_rows)}",
        "prune_wins": f"{len(prune_wins)}/{len(prune_rows)}",
        "verdict": verdict,
        "rows": {r.method: r.winner for r in comparison.rows},
        "int4_backend": next((r.backend for r in sal_report.results
                              if r.method == "int4"), None),
    }


@app.local_entrypoint()
def main():
    res = sal_vs_baseline_robustness.remote()
    print("\n========== RESULT ==========")
    for k, v in res.items():
        print(f"  {k}: {v}")
