"""Real-scale robustness benchmark — does SAL survive compression where it hurts?

The v0.4.0 DistilBERT result was clear on one point and useless on another: SAL
roughly halved head-pruning damage, but every quantization margin sat inside the
noise floor because INT4 barely dented an 86%-accuracy sentiment classifier.
This script re-asks the question at scales where quantization actually costs
something.

Three tiers, three GPU classes:

  ================  ==========  ==========  ==============================
  tier              model       GPU         task
  ================  ==========  ==========  ==============================
  ``mistral``       Mistral-7B  A100-40GB   HellaSwag (4-way, reasoning)
  ``phi2``          Phi-2 2.7B  A10G        MMLU (4-way, knowledge)
  ``gpt2``          GPT-2 Med   T4          SST-2 (2-way, sentiment)
  ================  ==========  ==========  ==============================

Protocol, identical per tier
----------------------------
**Phase A — training.** Fine-tune twice from the same checkpoint with LoRA
r=16 for 3 epochs: once plain, once with SAL at ``prune_fraction=0.33``. Both
arms see the same data in the same order with the same seed. Adapters are merged
before evaluation, so the battery compresses a plain dense model.

**Phase B — compression battery.** Seven variants per arm::

    dense, int8, int4, prune33, prune50, prune33+int8, prune33+int4

Quantization prefers bitsandbytes on CUDA (LLM.int8() and NF4 — the paths people
actually deploy) and falls back to ``torch.ao`` dynamic INT8 on CPU. The output
head (``lm_head``/``score``) is never quantized, matching standard practice.
Combined variants quantize first, then install the head mask, so the pruning
hooks sit on the quantized modules rather than being deep-copied through them.

**Phase C — evaluation.** Every variant is scored by length-normalized
log-likelihood over the answer choices — the lm-eval-harness protocol, no
classification head anywhere — plus serialized size in MB and forward
throughput in tokens/sec.

Questions
---------
1. Does the SAL arm survive INT4 better once INT4 has something to break?
2. Do structural and precision compression compound, or does pruning eat the
   headroom quantization needs?
3. What does the accuracy-vs-size Pareto frontier look like across all 14
   variants?

Read the results with these caveats
-----------------------------------
* **LoRA weakens SAL.** SAL's mechanism is the model reorganizing around missing
  heads, but LoRA freezes the base weights — only the adapters can redistribute.
  A null result here is evidence about *SAL-under-LoRA*, not about SAL. Full
  fine-tuning would be the stronger test and the more expensive one.
* **Throughput is not comparable across backends.** A CPU INT8 fallback and a
  CUDA NF4 variant are different machines; the device is recorded per variant
  and printed. Compare sizes across variants, compare speeds only within a
  device.
* **Single seed, one run per tier.** Same limitation as v0.4.0. Margins under a
  point are noise at these eval sizes.

Usage::

    SAL_TIERS=gpt2 modal run scripts/modal_robustness_scale.py
    SAL_TIERS=mistral,phi2 modal run scripts/modal_robustness_scale.py
    modal run scripts/modal_robustness_scale.py                  # all three tiers

Tier selection is an **environment variable, not a CLI flag**, because Modal
validates every ``@app.function`` in the app when the app is created — including
GPU entitlement. A registered A100 function your account cannot provision fails
the whole run before a single T4 function starts. ``SAL_TIERS`` controls which
functions get registered at all, so cheap tiers run on accounts that cannot
touch the expensive ones.

Mistral-7B is a gated repo. Export ``HF_TOKEN`` locally before running; it is
forwarded to the container. Results are written to
``scripts/robustness_scale_results.json``.

This is a research experiment, not a product test.
"""
from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("sal-torch-robustness-scale")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "numpy", "accelerate>=1.1.0",
                 "peft>=0.8", "bitsandbytes", "sentencepiece", "protobuf")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

# Forwarded from the local environment so gated repos work without a Modal secret.
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

VARIANTS = ["dense", "int8", "int4", "prune33", "prune50", "prune33+int8", "prune33+int4"]

# Sized so a tier finishes inside the 1800s timeout. Raise n_train/n_eval for a
# real measurement once the plumbing is confirmed.
TIERS = {
    "mistral": {
        "model": "mistralai/Mistral-7B-v0.3",
        "task": "hellaswag",
        "gpu": "A100-40GB",
        "memory": 65536,
        "dtype": "bfloat16",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "n_train": 512, "n_eval": 256, "max_len": 256,
        "train_bs": 2, "grad_accum": 4, "eval_bs": 8, "lr": 1e-4,
    },
    "phi2": {
        "model": "microsoft/phi-2",
        "task": "mmlu",
        "gpu": "A10G",
        "memory": 32768,
        "dtype": "bfloat16",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "dense"],
        "n_train": 512, "n_eval": 256, "max_len": 256,
        "train_bs": 2, "grad_accum": 4, "eval_bs": 8, "lr": 1e-4,
    },
    "gpt2": {
        "model": "gpt2-medium",
        "task": "sst2",
        "gpu": "T4",
        "memory": 16384,
        "dtype": "float32",
        "lora_targets": ["c_attn"],
        "n_train": 1024, "n_eval": 512, "max_len": 96,
        "train_bs": 8, "grad_accum": 1, "eval_bs": 16, "lr": 2e-4,
    },
}

EPOCHS = 3
LORA_R = 16
PRUNE_FRACTION = 0.33
# Output projections stay in full precision — quantizing them is not standard
# practice and would confound the comparison.
QUANT_SKIP = ("lm_head", "score", "classifier", "embed_out")


# ------------------------------------------------------------------ task loading
def load_task(name: str, n_train: int, n_eval: int, seed: int = 0):
    """Return (train, eval) lists of {prompt, choices, gold} examples."""
    from datasets import load_dataset

    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag")
        def conv(r):
            return {"prompt": r["ctx"], "choices": [" " + e.strip() for e in r["endings"]],
                    "gold": int(r["label"])}
        train = [conv(r) for r in ds["train"].shuffle(seed=seed).select(range(n_train))]
        ev = [conv(r) for r in ds["validation"].shuffle(seed=seed).select(range(n_eval))]
        return train, ev

    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        def conv(r):
            return {"prompt": f"{r['question'].strip()}\nAnswer:",
                    "choices": [" " + str(c).strip() for c in r["choices"]],
                    "gold": int(r["answer"])}
        train = [conv(r) for r in ds["validation"].shuffle(seed=seed).select(range(n_train))]
        ev = [conv(r) for r in ds["test"].shuffle(seed=seed).select(range(n_eval))]
        return train, ev

    if name == "sst2":
        ds = load_dataset("stanfordnlp/sst2")
        def conv(r):
            return {"prompt": f'Review: "{r["sentence"].strip()}"\nSentiment:',
                    "choices": [" negative", " positive"], "gold": int(r["label"])}
        train = [conv(r) for r in ds["train"].shuffle(seed=seed).select(range(n_train))]
        ev = [conv(r) for r in ds["validation"].shuffle(seed=seed).select(range(n_eval))]
        return train, ev

    raise ValueError(f"Unknown task '{name}'")


# --------------------------------------------------------- multiple-choice scoring
def _encode_pair(tok, prompt: str, continuation: str, max_len: int):
    """Token ids for prompt+continuation, plus how many trailing tokens are the
    continuation (after truncation from the left)."""
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    c = tok(continuation, add_special_tokens=False)["input_ids"]
    ids = (p + c)[-max_len:]
    n_cont = min(len(c), len(ids) - 1)     # need >= 1 context token to predict from
    return ids, max(n_cont, 1)


def _score_pairs(model, tok, pairs, max_len: int, device):
    """Length-normalized log P(continuation | prompt) for each (prompt, cont) pair."""
    import torch

    encoded = [_encode_pair(tok, p, c, max_len) for p, c in pairs]
    width = max(len(ids) for ids, _ in encoded)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, (ids, _) in enumerate(encoded):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1

    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)

    scores = []
    for i, (ids, n_cont) in enumerate(encoded):
        n = len(ids)
        target = input_ids[i, n - n_cont:n].to(logprobs.device)
        # Token at position t is predicted by the logits at position t-1.
        window = logprobs[i, n - n_cont - 1:n - 1, :]
        lp = window.gather(-1, target.unsqueeze(-1)).sum()
        scores.append(float(lp) / n_cont)
    return scores


def evaluate_accuracy(model, tok, examples, max_len: int, batch_size: int, device):
    """Multiple-choice accuracy by length-normalized log-likelihood."""
    model.eval()
    flat = [(i, j, ex["prompt"], ch)
            for i, ex in enumerate(examples) for j, ch in enumerate(ex["choices"])]
    scored = {}
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        vals = _score_pairs(model, tok, [(p, c) for _, _, p, c in chunk], max_len, device)
        for (i, j, _, _), v in zip(chunk, vals):
            scored.setdefault(i, {})[j] = v
    correct = 0
    for i, ex in enumerate(examples):
        per_choice = scored.get(i, {})
        if not per_choice:
            continue
        pred = max(per_choice, key=per_choice.get)
        correct += int(pred == ex["gold"])
    return correct / max(len(examples), 1)


# --------------------------------------------------------------------- training
def _train_batch(tok, examples, idxs, max_len: int):
    """Causal-LM batch over prompt+gold continuation, loss masked to the continuation."""
    import torch

    rows = []
    for j in idxs:
        ex = examples[j]
        ids, n_cont = _encode_pair(tok, ex["prompt"], ex["choices"][ex["gold"]], max_len)
        labels = [-100] * len(ids)
        for t in range(len(ids) - n_cont, len(ids)):
            labels[t] = ids[t]
        rows.append((ids, labels))

    width = max(len(ids) for ids, _ in rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (ids, labs) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)
        attn[i, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def train_lora(base_model, tok, examples, cfg: dict, device, use_sal: bool, seed: int = 0):
    """LoRA fine-tune, optionally with SAL head masking. Returns the merged model.

    SAL runs against ``base_model`` (the object PEFT wraps in place), so the head
    masks sit on the real attention output projections while the LoRA adapters
    take the gradient. The masker is removed before merging.
    """
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    from sal.config import SALConfig
    from sal.masker import HeadMasker

    torch.manual_seed(seed)

    sal_config = SALConfig.auto(base_model, prune_fraction=PRUNE_FRACTION) if use_sal else None

    peft_model = get_peft_model(base_model, LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["lora_targets"]))
    peft_model.to(device)

    # Install after .to(device): masks are allocated on the model's device.
    masker = HeadMasker(base_model, sal_config, seed=seed) if use_sal else None
    if masker is not None:
        masker.install()

    bs, accum = cfg["train_bs"], cfg["grad_accum"]
    order = np.random.RandomState(seed).permutation(len(examples))
    batches = [order[i:i + bs] for i in range(0, len(examples), bs)]
    total_steps = max(1, (len(batches) * EPOCHS) // accum)

    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg["lr"])
    peft_model.train()

    step, micro = 0, 0
    for _ in range(EPOCHS):
        for idxs in batches:
            if masker is not None:
                masker.step(step, total_steps)
            batch = {k: v.to(device) for k, v in
                     _train_batch(tok, examples, idxs, cfg["max_len"]).items()}
            loss = peft_model(**batch).loss / accum
            loss.backward()
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1

    if masker is not None:
        stats = masker.stats
        masker.remove()
        print(f"    SAL: {stats['pruned_heads']}/{stats['total_heads']} heads silenced "
              f"by end of training over {stats['prune_events']} prune events", flush=True)

    peft_model.eval()
    merged = peft_model.merge_and_unload()
    del peft_model
    return merged


# ----------------------------------------------------------------- quantization
def conv1d_to_linear(model, skip=QUANT_SKIP):
    """Rewrite GPT-2-style ``Conv1D`` layers as equivalent ``nn.Linear``.

    GPT-2 stores attention and MLP projections as ``Conv1D`` (weight ``[in, out]``,
    ``y = x @ W + b``) rather than ``nn.Linear`` (weight ``[out, in]``,
    ``y = x @ W.T + b``). Every quantization path here — bitsandbytes and
    ``torch.ao`` alike — matches on ``nn.Linear``, so without this the whole
    GPT-2 tier would report quantization numbers for a model nothing was
    quantized in. Transposing the weight makes the two exactly equivalent.
    """
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Conv1D):
                if any(s in path for s in skip):
                    continue
                w = child.weight.data                      # [in, out]
                lin = nn.Linear(w.shape[0], w.shape[1], bias=child.bias is not None)
                lin.weight = nn.Parameter(w.t().contiguous(), requires_grad=False)
                if child.bias is not None:
                    lin.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(parent, name, lin.to(w.device, w.dtype))
            else:
                walk(child, path)

    walk(model, "")
    return model


def _swap_linears(model, make_new, skip=QUANT_SKIP):
    """Replace every nn.Linear (except ``skip``-named ones) via ``make_new``."""
    import torch.nn as nn

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(s in path for s in skip):
                    continue
                new = make_new(child)
                if new is not None:
                    setattr(parent, name, new)
            else:
                walk(child, path)

    walk(model, "")
    return model


def quantize_int8(model, device):
    """LLM.int8() on CUDA via bitsandbytes; torch.ao dynamic INT8 on CPU.

    Returns (model, backend, device_used).
    """
    import torch
    import torch.nn as nn

    conv1d_to_linear(model)
    if device == "cuda":
        try:
            import bitsandbytes as bnb

            def make(old):
                new = bnb.nn.Linear8bitLt(old.in_features, old.out_features,
                                          bias=old.bias is not None, has_fp16_weights=False,
                                          threshold=6.0)
                new.weight = bnb.nn.Int8Params(old.weight.data.clone(),
                                               requires_grad=False, has_fp16_weights=False)
                if old.bias is not None:
                    new.bias = nn.Parameter(old.bias.data.clone(), requires_grad=False)
                return new

            return _swap_linears(model, make).to("cuda"), "bitsandbytes-int8", "cuda"
        except Exception as e:  # noqa: BLE001 — record the fallback rather than lose the row
            print(f"    bitsandbytes INT8 unavailable ({e}); using torch.ao on CPU", flush=True)

    q = torch.ao.quantization.quantize_dynamic(model.cpu().float().eval(), {nn.Linear},
                                               dtype=torch.qint8)
    return q, "torch-dynamic-int8", "cpu"


def quantize_int4(model, device):
    """NF4 via bitsandbytes. Returns (model, backend, device_used) or raises."""
    import bitsandbytes as bnb
    import torch
    import torch.nn as nn

    if device != "cuda":
        raise RuntimeError("bitsandbytes 4-bit needs CUDA")
    conv1d_to_linear(model)

    def make(old):
        new = bnb.nn.Linear4bit(old.in_features, old.out_features,
                                bias=old.bias is not None,
                                compute_dtype=torch.bfloat16, quant_type="nf4")
        new.weight = bnb.nn.Params4bit(data=old.weight.data.clone(),
                                       requires_grad=False, quant_type="nf4")
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.clone(), requires_grad=False)
        return new

    return _swap_linears(model, make).to("cuda"), "bitsandbytes-nf4", "cuda"


# -------------------------------------------------------------------- measurement
def model_size_mb(model) -> float:
    """Total tensor bytes in MB — what the artifact costs to store.

    Summed analytically rather than by serializing to a buffer: at 2.7B+ this is
    called once per variant, and round-tripping several GB fourteen times eats a
    meaningful slice of the run's timeout for a number that matches to within a
    fraction of a percent (GPT-2 Medium fp32: 1419.4MB serialized vs 1420.0MB
    analytic). Quantized parameters report their packed storage, which is the
    point of the measurement.
    """
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total / 1e6


def tokens_per_second(model, tok, examples, max_len: int, batch_size: int, device,
                      n_batches: int = 4) -> float:
    """Forward throughput. Only comparable between variants on the same device."""
    import torch

    model.eval()
    pairs = [(ex["prompt"], ex["choices"][0]) for ex in examples[:batch_size * (n_batches + 1)]]
    if not pairs:
        return 0.0
    chunks = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    if not chunks:
        return 0.0

    _score_pairs(model, tok, chunks[0], max_len, device)     # warmup
    if device == "cuda":
        torch.cuda.synchronize()

    tokens, t0 = 0, time.time()
    for chunk in chunks[1:n_batches + 1]:
        _score_pairs(model, tok, chunk, max_len, device)
        tokens += sum(len(_encode_pair(tok, p, c, max_len)[0]) for p, c in chunk)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return tokens / elapsed if elapsed > 0 else 0.0


# --------------------------------------------------------------------- battery
def _build_variant(master, variant: str, device):
    """Materialize one compression variant. Returns (model, backend, device, prune_ctx).

    Quantization runs before masking so the pruning hooks land on the quantized
    modules — deep-copying a model that already carries hooks is a trap.
    """
    import copy

    from sal.config import SALConfig
    from sal.masker import HeadMasker

    model = copy.deepcopy(master)
    backend, dev = None, device

    if "int8" in variant:
        model, backend, dev = quantize_int8(model, device)
    elif "int4" in variant:
        model, backend, dev = quantize_int4(model, device)
    else:
        model = model.to(device)

    masker = None
    if variant.startswith("prune"):
        pct = int(variant.split("+")[0].replace("prune", ""))
        config = SALConfig.auto(model, prune_fraction=pct / 100.0)
        masker = HeadMasker(model, config, seed=0)
        masker.install()
        masker.activate()

    return model, backend, dev, masker


def run_battery(master, tok, examples, cfg: dict, device, arm: str) -> dict:
    """Evaluate every variant of one trained model."""
    import torch

    out = {}
    for variant in VARIANTS:
        t0 = time.time()
        model = masker = None
        try:
            model, backend, dev, masker = _build_variant(master, variant, device)
            acc = evaluate_accuracy(model, tok, examples, cfg["max_len"], cfg["eval_bs"], dev)
            size = model_size_mb(model)
            tps = tokens_per_second(model, tok, examples, cfg["max_len"], cfg["eval_bs"], dev)
            out[variant] = {"accuracy": acc, "size_mb": round(size, 1),
                            "tokens_per_sec": round(tps, 1), "backend": backend,
                            "device": dev, "seconds": round(time.time() - t0, 1)}
            print(f"  [{arm}] {variant:<14} acc={acc:.4f}  size={size:7.1f}MB  "
                  f"{tps:8.1f} tok/s  ({dev}{'/' + backend if backend else ''})", flush=True)
        except Exception as e:  # noqa: BLE001 — a dead variant shouldn't sink the tier
            out[variant] = {"error": str(e), "seconds": round(time.time() - t0, 1)}
            print(f"  [{arm}] {variant:<14} FAILED: {e}", flush=True)
        finally:
            if masker is not None:
                masker.remove()
            del model, masker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------- reporting
def pareto_front(points: list) -> list:
    """Names of the non-dominated (size_mb, accuracy) points — smaller and better."""
    front = []
    for name, size, acc in points:
        dominated = any(s <= size and a >= acc and (s < size or a > acc)
                        for other, s, a in points if other != name)
        if not dominated:
            front.append(name)
    return front


def comparison_table(base: dict, sal: dict, metric: str = "accuracy") -> str:
    head = (f"{'variant':<16}{'baseline':>10}{'SAL':>10}{'delta':>10}"
            f"{'base_size':>11}{'sal_size':>10}{'winner':>10}")
    lines = [head, "-" * len(head)]
    for v in VARIANTS:
        b, s = base.get(v, {}), sal.get(v, {})
        if metric not in b or metric not in s:
            lines.append(f"{v:<16}{'-':>10}{'-':>10}{'-':>10}{'-':>11}{'-':>10}{'n/a':>10}")
            continue
        delta = s[metric] - b[metric]
        winner = "SAL" if delta > 0 else ("baseline" if delta < 0 else "tie")
        lines.append(f"{v:<16}{b[metric]:>10.4f}{s[metric]:>10.4f}{delta:>+10.4f}"
                     f"{b['size_mb']:>11.1f}{s['size_mb']:>10.1f}{winner:>10}")
    return "\n".join(lines)


def retention_table(base: dict, sal: dict) -> str:
    """Accuracy retained relative to each arm's own dense variant."""
    head = f"{'variant':<16}{'base_ret':>10}{'sal_ret':>10}{'winner':>10}"
    lines = [head, "-" * len(head)]
    b0 = base.get("dense", {}).get("accuracy")
    s0 = sal.get("dense", {}).get("accuracy")
    if not b0 or not s0:
        return "dense baseline missing — retention not computable"
    for v in VARIANTS:
        if v == "dense":
            continue
        b, s = base.get(v, {}), sal.get(v, {})
        if "accuracy" not in b or "accuracy" not in s:
            lines.append(f"{v:<16}{'-':>10}{'-':>10}{'n/a':>10}")
            continue
        br, sr = b["accuracy"] / b0, s["accuracy"] / s0
        winner = "SAL" if sr > br else ("baseline" if sr < br else "tie")
        lines.append(f"{v:<16}{br:>10.2%}{sr:>10.2%}{winner:>10}")
    return "\n".join(lines)


def summarize(tier: str, base: dict, sal: dict, n_eval: int = 0) -> dict:
    """Print the tables and return the machine-readable verdict for this tier."""
    print(f"\n===== {tier}: accuracy and size =====", flush=True)
    print(comparison_table(base, sal), flush=True)
    print(f"\n===== {tier}: retention vs each arm's own dense model =====", flush=True)
    print(retention_table(base, sal), flush=True)

    points = ([(f"baseline/{v}", d["size_mb"], d["accuracy"])
               for v, d in base.items() if "accuracy" in d]
              + [(f"SAL/{v}", d["size_mb"], d["accuracy"])
                 for v, d in sal.items() if "accuracy" in d])
    front = pareto_front(points)
    print(f"\n===== {tier}: Pareto frontier (accuracy vs size) =====", flush=True)
    for name, size, acc in sorted(points, key=lambda p: p[1]):
        mark = "  <-- frontier" if name in front else ""
        print(f"  {name:<24} {size:8.1f}MB  acc={acc:.4f}{mark}", flush=True)

    b0 = base.get("dense", {}).get("accuracy")
    s0 = sal.get("dense", {}).get("accuracy")

    # Two scorings, because they can disagree and only one of them is a
    # deployment number. Absolute asks "which model would I ship?"; retention
    # asks "which model degrades more gracefully?" — and retention flatters
    # whichever arm started lower, so it must never be reported alone.
    cats = {"quantization": [], "pruning": [], "combined": []}
    for v in VARIANTS:
        b, s = base.get(v, {}), sal.get(v, {})
        if v == "dense" or "accuracy" not in b or "accuracy" not in s:
            continue
        key = "combined" if "+" in v else ("pruning" if v.startswith("prune") else "quantization")
        abs_win = s["accuracy"] > b["accuracy"]
        ret_win = (b0 and s0) and (s["accuracy"] / s0) > (b["accuracy"] / b0)
        cats[key].append((bool(abs_win), bool(ret_win)))

    def tally(rows, idx):
        return f"{sum(1 for r in rows if r[idx])}/{len(rows)}"

    absolute = {k: tally(v, 0) for k, v in cats.items()}
    retention = {k: tally(v, 1) for k, v in cats.items()}
    clean_delta = (s0 - b0) if (b0 is not None and s0 is not None) else None

    print(f"\n===== {tier}: SAL wins by category =====", flush=True)
    print(f"  {'category':<16}{'absolute':>10}{'retention':>12}", flush=True)
    for k in ("quantization", "pruning", "combined"):
        print(f"  {k:<16}{absolute[k]:>10}{retention[k]:>12}", flush=True)
    if clean_delta is not None:
        print(f"\n  clean accuracy: baseline {b0:.4f} -> SAL {s0:.4f} ({clean_delta:+.4f})",
              flush=True)
    print("  absolute = which model scores higher, the deployment question.", flush=True)
    print("  retention = degradation vs each arm's own dense model; it flatters", flush=True)
    print("              whichever arm started lower, so read it alongside absolute.",
          flush=True)
    if n_eval:
        print(f"\n  noise floor: {n_eval} eval examples, so one example is "
              f"{1.0 / n_eval:.3%}; margins below that are not evidence.", flush=True)

    return {
        "n_eval": n_eval,
        "noise_floor": (1.0 / n_eval) if n_eval else None,
        "absolute": absolute,
        "retention": retention,
        "dense_baseline": b0,
        "dense_sal": s0,
        "clean_accuracy_delta": clean_delta,
        "pareto_front": front,
    }


# ------------------------------------------------------------------- the runner
def run_tier(tier: str) -> dict:
    """Train both arms and run the full battery. Executed inside the container."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TIERS[tier]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["dtype"])
    print(f"=== tier '{tier}': {cfg['model']} on {cfg['task']} ({device}, {cfg['dtype']}) ===",
          flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ex, eval_ex = load_task(cfg["task"], cfg["n_train"], cfg["n_eval"])
    print(f"    {len(train_ex)} train / {len(eval_ex)} eval examples, "
          f"{len(eval_ex[0]['choices'])}-way", flush=True)

    results = {}
    for arm, use_sal in (("baseline", False), ("SAL", True)):
        print(f"\n[{tier}/{arm}] LoRA r={LORA_R}, {EPOCHS} epochs"
              f"{', SAL prune_fraction=0.33' if use_sal else ''}...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)
        t0 = time.time()
        merged = train_lora(base, tok, train_ex, cfg, device, use_sal=use_sal, seed=0)
        print(f"    trained in {time.time() - t0:.0f}s", flush=True)

        # Park the master on CPU: each variant deep-copies from here, and holding
        # two 7B models on the GPU at once will not fit.
        master = merged.cpu()
        del merged, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[{tier}/{arm}] compression battery...", flush=True)
        results[arm] = run_battery(master, tok, eval_ex, cfg, device, f"{tier}/{arm}")
        del master
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verdict = summarize(tier, results["baseline"], results["SAL"], n_eval=cfg["n_eval"])
    return {"tier": tier, "model": cfg["model"], "task": cfg["task"],
            "gpu": cfg["gpu"], "n_train": cfg["n_train"], "n_eval": cfg["n_eval"],
            "epochs": EPOCHS, "lora_r": LORA_R, "prune_fraction": PRUNE_FRACTION,
            "baseline": results["baseline"], "SAL": results["SAL"], "verdict": verdict}


def _selected_tiers() -> list:
    """Tiers to register, from ``SAL_TIERS`` (default: all)."""
    raw = os.environ.get("SAL_TIERS", "all").strip()
    if raw in ("", "all"):
        return list(TIERS)
    names = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [n for n in names if n not in TIERS]
    if unknown:
        raise SystemExit(f"Unknown tier(s) {unknown} in SAL_TIERS. "
                         f"Choose from {list(TIERS)} or 'all'.")
    return names


def tier_mistral():
    return run_tier("mistral")


def tier_phi2():
    return run_tier("phi2")


def tier_gpt2():
    return run_tier("gpt2")


_ENTRYPOINTS = {"mistral": tier_mistral, "phi2": tier_phi2, "gpt2": tier_gpt2}


def _register(name: str):
    """Decorate one tier's entrypoint as a Modal function sized for its GPU.

    The decorator is applied programmatically rather than with ``@`` syntax so
    only the selected tiers are registered. Modal checks GPU entitlement for
    every function in the app at creation time, so an unregistered A100 tier is
    the difference between a T4 run that works and one that never starts. The
    functions themselves stay at module scope — Modal requires that.
    """
    cfg = TIERS[name]
    return app.function(image=image, gpu=cfg["gpu"], timeout=1800,
                        memory=cfg["memory"], secrets=[hf_secret])(_ENTRYPOINTS[name])


_FUNCS = {name: _register(name) for name in _selected_tiers()}
RESULTS_PATH = "scripts/robustness_scale_results.json"


@app.local_entrypoint()
def main():
    """Runs whichever tiers ``SAL_TIERS`` selected (default: all three)."""
    names = list(_FUNCS)
    print(f"tiers registered: {names}  (set SAL_TIERS to change)")

    results = {}
    for name in names:
        print(f"\n########## launching tier '{name}' on {TIERS[name]['gpu']} ##########")
        try:
            results[name] = _FUNCS[name].remote()
        except Exception as e:  # noqa: BLE001 — a failed tier shouldn't lose the others
            print(f"tier '{name}' failed: {e}")
            results[name] = {"tier": name, "error": str(e)}

    # Merge into whatever is already on disk: tiers run one at a time (they need
    # different GPUs and different budget approvals), so a fresh write would
    # silently discard every tier not in this run.
    merged = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, encoding="utf-8") as fh:
                merged = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"could not read existing {RESULTS_PATH} ({e}); starting fresh")
    merged.update(results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH} (tiers on file: {sorted(merged)})")

    print("\n########## CROSS-TIER SUMMARY ##########")
    print(f"{'':<10}{'':<24}{'quant':>13}{'prune':>13}{'combined':>13}{'clean':>10}")
    print(f"{'tier':<10}{'model':<24}{'abs / ret':>13}{'abs / ret':>13}"
          f"{'abs / ret':>13}{'delta':>10}")
    print("-" * 83)
    for name, res in results.items():
        if "verdict" not in res:
            print(f"{name:<10}{'(failed)':<24}{'-':>13}{'-':>13}{'-':>13}{'-':>10}")
            continue
        v = res["verdict"]
        a, r = v["absolute"], v["retention"]
        delta = v.get("clean_accuracy_delta")
        cells = [f"{a[k]} / {r[k]}" for k in ("quantization", "pruning", "combined")]
        print(f"{name:<10}{res['model']:<24}{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}"
              f"{(f'{delta:+.4f}' if delta is not None else '-'):>10}")
    print("\nSAL wins per category. abs = higher accuracy outright (the deployment")
    print("question); ret = less degradation from its own dense model. They disagree")
    print("whenever SAL trades clean accuracy for a flatter curve — see 'clean delta'.")
    print("Single seed per tier — treat sub-noise-floor margins as noise.")
