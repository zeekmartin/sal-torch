"""Validate the v0.3.0 continual-learning features on real models via Modal.

Plumbing validation for StructuralGuard + DriftMonitor on DistilBERT:

  1. Fine-tune DistilBERT on SST-2 (task A, 50 steps) and record SST-2 accuracy.
  2. Snapshot the task-A model, then fine-tune on MNLI (task B, 50 steps) two
     ways from identical starting weights:
        * WITH a StructuralGuard built from the task-A model,
        * WITHOUT any guard (baseline).
  3. Re-evaluate both on SST-2 (task A) and measure structural drift.
  4. Verify the guarded model forgets less: lower forgetting_score and higher
     retained SST-2 accuracy.

The SST-2 readout head is frozen during task B in *both* arms, so the only
moving part that differs between them is encoder head protection — which is
exactly what StructuralGuard governs. This isolates the mechanism; it is a
plumbing check, not a continual-learning benchmark.

T4 is intentional (small model, short runs). DistilBERT is loaded with
attn_implementation="eager" so it returns attention weights (the routing axis
and hub detection need them).

Usage:
    modal run scripts/modal_v030_test.py
"""
import modal

app = modal.App("sal-torch-v030")

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


def _tokenize_single(tok, sentences, labels, max_len, batch_size):
    import torch
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tok(sentences[i:i + batch_size], padding="max_length", truncation=True,
                  max_length=max_len, return_tensors="pt")
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "labels": torch.tensor(labels[i:i + batch_size])})
    return out


def _tokenize_pairs(tok, premises, hypotheses, labels, max_len, batch_size):
    import torch
    out = []
    for i in range(0, len(premises), batch_size):
        enc = tok(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                  padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "labels": torch.tensor(labels[i:i + batch_size])})
    return out


def _load(n_train=512, n_eval=256, n_taskb=512, max_len=64, batch_size=16):
    """DistilBERT-2class + SST-2 (task A) and MNLI-binarized (task B) batches."""
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, attn_implementation="eager")

    sst = load_dataset("stanfordnlp/sst2", split="train")
    sst_train = sst.select(range(n_train))
    sst_eval = sst.select(range(n_train, n_train + n_eval))
    a_train = _tokenize_single(tok, list(sst_train["sentence"]), list(sst_train["label"]),
                               max_len, batch_size)
    a_eval = _tokenize_single(tok, list(sst_eval["sentence"]), list(sst_eval["label"]),
                              max_len, batch_size)

    # MNLI binarized: entailment (0) -> 1, everything else -> 0. Keeps the 2-class head.
    mnli = load_dataset("nyu-mll/multi_nli", split="train").select(range(n_taskb))
    b_labels = [1 if lbl == 0 else 0 for lbl in mnli["label"]]
    b_train = _tokenize_pairs(tok, list(mnli["premise"]), list(mnli["hypothesis"]),
                              b_labels, max_len, batch_size)
    return model, a_train, a_eval, b_train


def _to_dev(batches, device):
    for b in batches:
        for k in b:
            b[k] = b[k].to(device)
    return batches


def _train(model, batches, steps, lr, device, freeze_head=False):
    import itertools
    from torch.optim import AdamW

    if freeze_head:
        for name in ("pre_classifier", "classifier"):
            mod = getattr(model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr)
    it = itertools.cycle(batches)
    for _ in range(steps):
        batch = next(it)
        out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad()


def _accuracy(model, batches, device):
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in batches:
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == b["labels"]).sum().item()
            total += b["labels"].numel()
    return correct / max(total, 1)


@app.function(image=image, gpu="T4", timeout=2400)
def guarded_vs_unguarded():
    import copy
    import torch
    from sal import StructuralGuard, DriftMonitor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== StructuralGuard continual-learning check on {MODEL} ({device}) ===", flush=True)
    model, a_train, a_eval, b_train = _load()
    model.to(device)
    a_train = _to_dev(a_train, device); a_eval = _to_dev(a_eval, device)
    b_train = _to_dev(b_train, device)

    # --- Task A: learn SST-2 -------------------------------------------------
    print("\n[task A] fine-tuning on SST-2 (50 steps)...", flush=True)
    _train(model, a_train, steps=50, lr=2e-5, device=device)
    acc_a = _accuracy(model, a_eval, device)
    print(f"[task A] SST-2 accuracy after task A: {acc_a:.4f}", flush=True)

    # Two identical copies of the task-A model.
    model_guarded = copy.deepcopy(model)
    model_unguarded = copy.deepcopy(model)

    # --- Build the guard from the task-A model -------------------------------
    print("\n[guard] scanning task-A model to build StructuralGuard...", flush=True)
    guard = StructuralGuard.from_model(model_guarded, a_train, protection_level=0.5,
                                       num_samples=256)
    print(f"[guard] protecting {len(guard.protected_heads)} / "
          f"{guard.num_layers * guard.num_heads} heads "
          f"across layers {sorted(guard.protection_map.keys())}", flush=True)

    # Baseline snapshot for the unguarded arm (task-A structural state).
    mon = DriftMonitor(model_unguarded, a_train, num_samples=256)
    mon.snapshot("before")

    # --- Task B WITH guard ---------------------------------------------------
    print("\n[task B | guarded] fine-tuning on MNLI (50 steps) with guard...", flush=True)
    guard.protect(model_guarded)
    _train(model_guarded, b_train, steps=50, lr=2e-5, device=device, freeze_head=True)
    guard.release()
    drift_guarded = guard.measure_drift(model_guarded, probe_dataset=a_train)
    acc_a_guarded = _accuracy(model_guarded, a_eval, device)

    # --- Task B WITHOUT guard (baseline) ------------------------------------
    print("[task B | unguarded] fine-tuning on MNLI (50 steps) without guard...", flush=True)
    _train(model_unguarded, b_train, steps=50, lr=2e-5, device=device, freeze_head=True)
    mon.snapshot("after")
    drift_unguarded = mon.compare("before", "after")
    acc_a_unguarded = _accuracy(model_unguarded, a_eval, device)

    # --- Report --------------------------------------------------------------
    print("\n========== v0.3.0 CONTINUAL-LEARNING SUMMARY ==========", flush=True)
    print(f"SST-2 accuracy after task A:            {acc_a:.4f}", flush=True)
    print(f"SST-2 retained (guarded task B):        {acc_a_guarded:.4f}", flush=True)
    print(f"SST-2 retained (unguarded task B):      {acc_a_unguarded:.4f}", flush=True)
    print(f"forgetting_score (guarded):             {drift_guarded.forgetting_score:.4f}", flush=True)
    print(f"forgetting_score (unguarded):           {drift_unguarded.forgetting_score:.4f}", flush=True)
    print(f"protected_integrity (guarded):          {drift_guarded.protected_integrity}", flush=True)
    print(f"\nguarded drift:   {drift_guarded.summary}", flush=True)
    print(f"unguarded drift: {drift_unguarded.summary}", flush=True)

    forgets_less = drift_guarded.forgetting_score <= drift_unguarded.forgetting_score
    retains_more = acc_a_guarded >= acc_a_unguarded
    print(f"\nguarded forgets less (structural):      {forgets_less}", flush=True)
    print(f"guarded retains more SST-2 accuracy:    {retains_more}", flush=True)

    return {
        "acc_a": acc_a,
        "acc_a_guarded": acc_a_guarded,
        "acc_a_unguarded": acc_a_unguarded,
        "forgetting_guarded": drift_guarded.forgetting_score,
        "forgetting_unguarded": drift_unguarded.forgetting_score,
        "protected_integrity": drift_guarded.protected_integrity,
        "guarded_forgets_less": forgets_less,
        "guarded_retains_more": retains_more,
    }


@app.local_entrypoint()
def main():
    res = guarded_vs_unguarded.remote()
    print("\n========== RESULT ==========")
    for k, v in res.items():
        print(f"  {k}: {v}")
