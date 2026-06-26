"""Validate the v0.2.0 features on a real model + dataset via Modal.

Runs two things on DistilBERT + SST-2:
  1. PlasticityScanner.scan() — the three-axis absorption map + a recommendation.
  2. sal.compare() — SAL vs. magnitude vs. random_posthoc at 33% compression.

T4 is intentional: this is a plumbing/validation run on a small model, not a
benchmark. DistilBERT is loaded with attn_implementation="eager" so it returns
attention weights (the routing axis needs them).

Usage:
    modal run scripts/modal_v020_test.py                  # both
    modal run scripts/modal_v020_test.py::plasticity
    modal run scripts/modal_v020_test.py::compare_methods
"""
import modal

app = modal.App("sal-torch-v020")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "scipy", "numpy",
                 "accelerate>=1.1.0")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

MODEL = "distilbert-base-uncased"


def _load_model_and_data(n_train=256, n_eval=128, max_len=64, batch_size=16):
    """DistilBERT-for-classification + tokenized SST-2 batches (input_ids,
    attention_mask, labels). Returns (model, train_batches, eval_batches)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, attn_implementation="eager")  # eager -> attentions exposed

    raw = load_dataset("stanfordnlp/sst2", split="train")

    def batches(rows, n):
        rows = rows.select(range(min(n, len(rows))))
        out = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok(chunk["sentence"], padding="max_length", truncation=True,
                      max_length=max_len, return_tensors="pt")
            out.append({"input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "labels": torch.tensor(chunk["label"])})
        return out

    train_batches = batches(raw, n_train)
    eval_batches = batches(raw.select(range(n_train, n_train + n_eval)), n_eval)
    return model, train_batches, eval_batches


@app.function(image=image, gpu="T4", timeout=900)
def plasticity():
    import torch
    from sal import PlasticityScanner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== PlasticityScanner on {MODEL} ({device}) ===", flush=True)
    model, train_batches, _ = _load_model_and_data()
    model.to(device)
    for b in train_batches:
        for k in b:
            b[k] = b[k].to(device)

    pmap = PlasticityScanner(model, train_batches, num_samples=128).scan()
    print("\nsummary:", pmap.summary, flush=True)
    print("\nabsorption map (layer -> class):", flush=True)
    for li in range(pmap.num_layers):
        r = pmap.routing.get(li, float("nan"))
        m = pmap.mutual_info.get(li, float("nan"))
        print(f"  L{li}: {pmap.absorption_map[li]:<10} routing={r:.3f} MI={m:.3f}", flush=True)
    print("\nCKA (adjacent layers):", flush=True)
    for (a, b), v in pmap.cka_similarity.items():
        print(f"  {a}-{b}: {v:.3f}", flush=True)

    rec = pmap.recommend(target_compression=0.33)
    print(f"\nrecommendation @33%: prune {len(rec.safe_to_prune)} heads, "
          f"never-touch {len(rec.never_touch)} heads, "
          f"est. impact {rec.expected_impact:+.3f}", flush=True)
    return pmap.to_dict()


@app.function(image=image, gpu="T4", timeout=1800)
def compare_methods():
    import torch
    from sal import compare

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== sal.compare() on {MODEL} ({device}) ===", flush=True)
    model, train_batches, eval_batches = _load_model_and_data()
    model.to(device)
    for batch_list in (train_batches, eval_batches):
        for b in batch_list:
            for k in b:
                b[k] = b[k].to(device)

    result = compare(model, train_batches, eval_batches,
                     methods=["sal", "magnitude", "random_posthoc"],
                     compression=0.33, sal_epochs=2, metric="accuracy", batch_size=16)
    print("\n" + result.table, flush=True)
    print(f"\nwinner: {result.winner}", flush=True)
    return result.to_dict()


@app.local_entrypoint()
def main():
    pmap = plasticity.remote()
    cmp = compare_methods.remote()
    print("\n========== v0.2.0 SUMMARY ==========")
    print("plasticity summary:", pmap.get("summary"))
    print("absorption_map:", pmap.get("absorption_map"))
    print("comparison winner:", cmp.get("winner"))
    for r in cmp.get("results", []):
        print("  ", r)
