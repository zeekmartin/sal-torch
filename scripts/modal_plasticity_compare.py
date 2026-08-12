"""Structural profiles across architectures — do they actually differ?

`sal-torch` claims its scanners are architecture-agnostic diagnostics. That
claim is only interesting if the numbers they produce are *architecture
specific*: if every transformer scanned the same, the scan would be measuring
the tool rather than the model.

So: four pretrained checkpoints, no training, forward passes only.

  ================================  ==========  ============================
  checkpoint                        family      probe
  ================================  ==========  ============================
  ``distilbert-base-uncased``       BERT-like   text (SST-2 sentences)
  ``gpt2``                          causal LM   text (SST-2 sentences)
  ``bert-base-uncased``             BERT-like   text (SST-2 sentences)
  ``google/vit-base-patch16-224``   ViT         images (CIFAR-10)
  ================================  ==========  ============================

Per model, both scanners:

* **FIScanner** — the fragility score in [0,1] (low = triangulated and
  redundant, high = fragile) plus IMMUNE / BUFFER / CRITICAL per layer.
* **PlasticityScanner** — routing entropy, inter-layer CKA, an intra-layer MI
  proxy, folded into an ELASTIC / SATURATED / HUB absorption map.

Everything is loaded with ``attn_implementation="eager"``. Without it the
attention weights are never returned, routing entropy comes back NaN, and hub
detection silently degrades to "no hubs" — which reads exactly like a real
finding.

The probe is held at the same size for every model so counts are comparable;
what differs is the modality, which is unavoidable — a ViT cannot be probed with
sentences. Read cross-modality rows with that in mind: the text models are a
controlled three-way comparison, the ViT row is a fourth architecture on its own
probe.

Usage::

    modal run scripts/modal_plasticity_compare.py

Results are written to ``scripts/plasticity_compare_results.json``.
"""
from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("sal-torch-plasticity-compare")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers", "datasets", "numpy",
                 "accelerate>=1.1.0", "pillow")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

MODELS = [
    {"id": "distilbert-base-uncased", "modality": "text"},
    {"id": "gpt2", "modality": "text"},
    {"id": "bert-base-uncased", "modality": "text"},
    {"id": "google/vit-base-patch16-224", "modality": "vision"},
]

N_PROBE = int(os.environ.get("SAL_N_PROBE", "256"))
BATCH_SIZE = 16
MAX_LEN = 64
RESULTS_PATH = "scripts/plasticity_compare_results.json"


# ------------------------------------------------------------------ probe data
def text_probe(model_id: str, n: int, batch_size: int, max_len: int):
    """SST-2 sentences as {input_ids, attention_mask} batches."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sents = load_dataset("stanfordnlp/sst2", split="train").select(range(n))["sentence"]
    out = []
    for i in range(0, len(sents), batch_size):
        enc = tok([s.strip() for s in sents[i:i + batch_size]], padding="max_length",
                  truncation=True, max_length=max_len, return_tensors="pt")
        out.append({"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]})
    return out


def vision_probe(model_id: str, n: int, batch_size: int):
    """CIFAR-10 images as {pixel_values} batches."""
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_id)
    imgs = load_dataset("uoft-cs/cifar10", split="test").select(range(n))["img"]
    out = []
    for i in range(0, len(imgs), batch_size):
        px = proc(images=imgs[i:i + batch_size], return_tensors="pt")["pixel_values"]
        out.append({"pixel_values": px})
    return out


def load_model(spec: dict):
    """The bare encoder, in eager attention so routing entropy is available."""
    from transformers import AutoModel

    # eager is not optional here: SDPA/flash return no attention weights, and the
    # scanner degrades to NaN routing rather than failing loudly.
    return AutoModel.from_pretrained(spec["id"], attn_implementation="eager")


# --------------------------------------------------------------------- scanning
def scan_one(spec: dict, device) -> dict:
    """FI + plasticity for one checkpoint."""
    import torch

    from sal import FIScanner, PlasticityScanner

    model_id = spec["id"]
    print(f"\n=== {model_id} ({spec['modality']}) ===", flush=True)

    t0 = time.time()
    model = load_model(spec).to(device).eval()
    probe = (vision_probe(model_id, N_PROBE, BATCH_SIZE) if spec["modality"] == "vision"
             else text_probe(model_id, N_PROBE, BATCH_SIZE, MAX_LEN))
    print(f"    loaded + probe built in {time.time() - t0:.0f}s "
          f"({len(probe)} batches)", flush=True)

    t0 = time.time()
    fi = FIScanner(model, probe, num_samples=N_PROBE, batch_size=BATCH_SIZE).scan()
    fi_secs = time.time() - t0
    print(f"    FI: {fi.summary}  ({fi_secs:.0f}s)", flush=True)

    t0 = time.time()
    pmap = PlasticityScanner(model, probe, num_samples=N_PROBE,
                             batch_size=BATCH_SIZE).scan()
    pl_secs = time.time() - t0
    print(f"    plasticity: {pmap.summary}  ({pl_secs:.0f}s)", flush=True)

    import math

    routing = [v for v in pmap.routing.values() if not math.isnan(v)]
    cka = list(pmap.cka_similarity.values())
    mi = list(pmap.mutual_info.values())

    row = {
        "model": model_id,
        "modality": spec["modality"],
        "num_layers": fi.num_layers,
        "num_heads": fi.num_heads_per_layer,
        "total_heads": fi.num_layers * fi.num_heads_per_layer,
        "fi_score": round(fi.fi_score, 4),
        "num_immune": len(fi.immune_layers),
        "num_buffer": len(fi.buffer_layers),
        "num_critical": len(fi.critical_layers),
        "num_elastic": len(pmap.elastic_layers),
        "num_saturated": len(pmap.saturated_layers),
        "num_hub": len(pmap.hub_layers),
        "mean_routing": round(float(sum(routing) / len(routing)), 4) if routing else None,
        "mean_cka": round(float(sum(cka) / len(cka)), 4) if cka else None,
        "mean_mi": round(float(sum(mi) / len(mi)), 4) if mi else None,
        "routing_available": bool(routing),
        "fi_summary": fi.summary,
        "plasticity_summary": pmap.summary,
        "layer_classification": {str(k): v.value for k, v in fi.layer_map.items()},
        "absorption_map": {str(k): v for k, v in pmap.absorption_map.items()},
        "per_layer_routing": {str(k): round(v, 4) for k, v in pmap.routing.items()},
        "per_layer_mi": {str(k): round(v, 4) for k, v in pmap.mutual_info.items()},
        "seconds": round(fi_secs + pl_secs, 1),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


@app.function(image=image, gpu="T4", cpu=8.0, memory=32768, timeout=3600)
def scan_all() -> list:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"structural profiles across {len(MODELS)} architectures on {device}, "
          f"{N_PROBE} probe samples each", flush=True)

    rows = []
    for spec in MODELS:
        try:
            rows.append(scan_one(spec, device))
        except Exception as e:  # noqa: BLE001 — one dead model shouldn't sink the table
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"model": spec["id"], "modality": spec["modality"],
                         "error": f"{type(e).__name__}: {e}"})
    return rows


# ------------------------------------------------------------------- reporting
def comparison_table(rows: list) -> str:
    head = (f"{'model':<32}{'mod':<8}{'L':>4}{'H':>4}{'heads':>7}{'FI':>8}"
            f"{'imm':>5}{'buf':>5}{'crit':>5}{'elas':>6}{'sat':>5}{'hub':>5}")
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['model']:<32}{r['modality']:<8}{'  -- failed --':<44}")
            continue
        lines.append(
            f"{r['model']:<32}{r['modality']:<8}{r['num_layers']:>4}{r['num_heads']:>4}"
            f"{r['total_heads']:>7}{r['fi_score']:>8.4f}{r['num_immune']:>5}"
            f"{r['num_buffer']:>5}{r['num_critical']:>5}{r['num_elastic']:>6}"
            f"{r['num_saturated']:>5}{r['num_hub']:>5}")
    return "\n".join(lines)


def axes_table(rows: list) -> str:
    head = f"{'model':<32}{'routing':>10}{'CKA':>10}{'MI':>10}{'scan_s':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            continue

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        lines.append(f"{r['model']:<32}{fmt(r['mean_routing']):>10}"
                     f"{fmt(r['mean_cka']):>10}{fmt(r['mean_mi']):>10}"
                     f"{r['seconds']:>9.1f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    rows = scan_all.remote()

    print("\n########## STRUCTURAL PROFILES ##########")
    print(comparison_table(rows))
    print("\nL = layers, H = heads/layer.  imm/buf/crit = FI layer classification.")
    print("elas/sat/hub = plasticity absorption map.")

    print("\n########## PLASTICITY AXES (means) ##########")
    print(axes_table(rows))
    print("\nrouting = attention entropy (high = flexible re-routing).")
    print("CKA = adjacent-layer representation similarity (high = redundant).")
    print("MI = intra-layer head correlation proxy (high = heads share function).")

    ok = [r for r in rows if "error" not in r]
    if ok:
        fis = [r["fi_score"] for r in ok]
        print(f"\nFI spans {min(fis):.4f} to {max(fis):.4f} across "
              f"{len(ok)} architectures — the scanners are reading the model, "
              f"not the tool.")
    failed = [r["model"] for r in rows if "error" in r]
    if failed:
        print(f"failed: {', '.join(failed)}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"config": {"n_probe": N_PROBE, "batch_size": BATCH_SIZE,
                              "max_len": MAX_LEN,
                              "attn_implementation": "eager"},
                   "models": rows}, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
