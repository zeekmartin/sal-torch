"""I-JEPA structural scan — what does a self-supervised ViT look like inside?

I-JEPA (ViT-H/14, `facebook/ijepa_vith14_1k`) is trained by predicting
representations of masked image regions rather than by supervised
classification. The question this run exists to answer is *structural*: does
that objective leave a different attention topology than supervised ViT
pretraining, and if so, on which axis — routing entropy, inter-layer CKA,
intra-layer redundancy, or all three?

**On the ρ figures.** The task framing cites ρ=0.46 for I-JEPA against ρ=0.81
for ViT. Nothing in this repository defines, computes, or records a ρ, so this
script cannot confirm, refute, or explain those numbers — it does not know what
they measure. What it *can* do is put both models through the shipped scanners
on an identical probe and report where they actually differ. Read the output as
a structural comparison on its own terms, not as an explanation of ρ.

**The confound, stated up front.** I-JEPA ViT-H/14 is 32 layers × 16 heads at
patch 14; ViT-base is 12 × 12 at patch 16. Objective, depth, width and patch
size all differ at once, so a raw I-JEPA-vs-ViT-base gap cannot be attributed to
self-supervision. To narrow that, this run also scans **ViT-large-patch16-224**
(24 × 16, supervised) as a control: it sits between the two on depth and matches
I-JEPA on head count. If I-JEPA differs from *both* supervised models in the
same direction, the objective is a live explanation; if it merely sits on a
depth trend that ViT-large is already on, it is not.

All three are probed with the **same CIFAR-10 images** used in
`scripts/modal_plasticity_compare.py`, at the same sample count, so the numbers
are directly comparable to that table.

Everything loads with ``attn_implementation="eager"`` — without it no attention
weights come back, routing entropy is NaN, and hub detection silently reports
"no hubs", which reads exactly like a finding.

Needs A10G: ViT-H/14 is ~630M parameters and the scan holds 32 layers of
attention maps.

Usage::

    modal run scripts/modal_jepa_scan.py

Results are written to ``scripts/jepa_scan_results.json``.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import modal

app = modal.App("sal-torch-jepa-scan")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers", "datasets", "numpy",
                 "accelerate>=1.1.0", "pillow")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

# The subject, plus two supervised controls at different depths. ViT-base is the
# same checkpoint scanned in modal_plasticity_compare.py.
MODELS = [
    {"id": "facebook/ijepa_vith14_1k", "label": "I-JEPA ViT-H/14",
     "objective": "self-supervised", "role": "subject"},
    {"id": "google/vit-large-patch16-224", "label": "ViT-large/16",
     "objective": "supervised", "role": "depth control"},
    {"id": "google/vit-base-patch16-224", "label": "ViT-base/16",
     "objective": "supervised", "role": "reference"},
]

N_PROBE = int(os.environ.get("SAL_N_PROBE", "256"))
BATCH_SIZE = int(os.environ.get("SAL_BATCH", "8"))
RESULTS_PATH = "scripts/jepa_scan_results.json"


# ------------------------------------------------------------------ probe data
def vision_probe(processor_id: str, n: int, batch_size: int):
    """CIFAR-10 images as {pixel_values} batches — the same probe as Task B."""
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(processor_id)
    imgs = load_dataset("uoft-cs/cifar10", split="test").select(range(n))["img"]
    out = []
    for i in range(0, len(imgs), batch_size):
        px = proc(images=imgs[i:i + batch_size], return_tensors="pt")["pixel_values"]
        out.append({"pixel_values": px})
    return out


def load_model(model_id: str):
    """The bare encoder in eager attention.

    ``AutoModel`` resolves ``ijepa`` on transformers >= 4.47; the explicit
    fallback is here so an older image reports *why* rather than dying inside
    the auto-class lookup.
    """
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(model_id, attn_implementation="eager")
    except (KeyError, ValueError) as e:
        try:
            from transformers import IJepaModel
        except ImportError:
            raise RuntimeError(
                f"{model_id} needs transformers with I-JEPA support "
                f"(>= 4.47); AutoModel said: {e}") from e
        return IJepaModel.from_pretrained(model_id, attn_implementation="eager")


# --------------------------------------------------------------------- scanning
def scan_one(spec: dict, device) -> dict:
    """FI + plasticity for one checkpoint, with per-layer detail retained."""
    import torch

    from sal import FIScanner, PlasticityScanner

    model_id = spec["id"]
    print(f"\n=== {spec['label']} ({model_id}, {spec['objective']}) ===", flush=True)

    t0 = time.time()
    model = load_model(model_id).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    probe = vision_probe(model_id, N_PROBE, BATCH_SIZE)
    print(f"    {n_params / 1e6:.0f}M params, loaded + probe built in "
          f"{time.time() - t0:.0f}s ({len(probe)} batches of {BATCH_SIZE})", flush=True)

    t0 = time.time()
    fi = FIScanner(model, probe, num_samples=N_PROBE, batch_size=BATCH_SIZE).scan()
    print(f"    FI: {fi.summary}  ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    pmap = PlasticityScanner(model, probe, num_samples=N_PROBE,
                             batch_size=BATCH_SIZE).scan()
    print(f"    plasticity: {pmap.summary}  ({time.time() - t0:.0f}s)", flush=True)

    routing = [v for v in pmap.routing.values() if not math.isnan(v)]
    cka = list(pmap.cka_similarity.values())
    mi = list(pmap.mutual_info.values())

    def mean(xs):
        return round(float(sum(xs) / len(xs)), 4) if xs else None

    row = {
        "model": model_id,
        "label": spec["label"],
        "objective": spec["objective"],
        "role": spec["role"],
        "params_m": round(n_params / 1e6, 1),
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
        "mean_routing": mean(routing),
        "mean_cka": mean(cka),
        "mean_mi": mean(mi),
        "routing_available": bool(routing),
        "fi_summary": fi.summary,
        "plasticity_summary": pmap.summary,
        "layer_classification": {str(k): v.value for k, v in fi.layer_map.items()},
        "absorption_map": {str(k): v for k, v in pmap.absorption_map.items()},
        "per_layer_routing": {str(k): (None if math.isnan(v) else round(v, 4))
                              for k, v in pmap.routing.items()},
        "per_layer_mi": {str(k): round(v, 4) for k, v in pmap.mutual_info.items()},
        "per_layer_cka": {f"{a}-{b}": round(v, 4)
                          for (a, b), v in pmap.cka_similarity.items()},
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


@app.function(image=image, gpu="A10G", cpu=8.0, memory=65536, timeout=3600)
def scan_all() -> list:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"I-JEPA structural scan on {device}, {N_PROBE} CIFAR-10 probe images, "
          f"batch {BATCH_SIZE}", flush=True)

    rows = []
    for spec in MODELS:
        try:
            rows.append(scan_one(spec, device))
        except Exception as e:  # noqa: BLE001 — a dead model shouldn't sink the rest
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({**{k: spec[k] for k in ("id", "label", "objective", "role")},
                         "model": spec["id"], "error": f"{type(e).__name__}: {e}"})
    return rows


# ------------------------------------------------------------------- reporting
def profile_table(rows: list) -> str:
    head = (f"{'model':<18}{'objective':<18}{'params':>8}{'L':>4}{'H':>4}{'FI':>8}"
            f"{'imm':>5}{'buf':>5}{'crit':>5}{'elas':>6}{'sat':>5}{'hub':>5}")
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['label']:<18}{r['objective']:<18}  -- failed --")
            continue
        lines.append(
            f"{r['label']:<18}{r['objective']:<18}{r['params_m']:>7.0f}M"
            f"{r['num_layers']:>4}{r['num_heads']:>4}{r['fi_score']:>8.4f}"
            f"{r['num_immune']:>5}{r['num_buffer']:>5}{r['num_critical']:>5}"
            f"{r['num_elastic']:>6}{r['num_saturated']:>5}{r['num_hub']:>5}")
    return "\n".join(lines)


def axes_table(rows: list) -> str:
    head = f"{'model':<18}{'objective':<18}{'routing':>10}{'CKA':>10}{'MI':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            continue

        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"

        lines.append(f"{r['label']:<18}{r['objective']:<18}{fmt(r['mean_routing']):>10}"
                     f"{fmt(r['mean_cka']):>10}{fmt(r['mean_mi']):>10}")
    return "\n".join(lines)


def depth_profile(rows: list) -> str:
    """Per-axis means by relative depth, so models of different depth compare."""
    bands = [(0.0, 0.25, "first quarter"), (0.25, 0.5, "second quarter"),
             (0.5, 0.75, "third quarter"), (0.75, 1.01, "final quarter")]
    head = f"{'depth band':<18}" + "".join(
        f"{r['label']:>20}" for r in rows if "error" not in r)
    out = []
    for axis, key in (("routing entropy", "per_layer_routing"), ("intra-layer MI", "per_layer_mi")):
        out.append(f"\n{axis} by relative depth:")
        out.append(head)
        out.append("-" * len(head))
        for lo, hi, name in bands:
            line = f"{name:<18}"
            for r in rows:
                if "error" in r:
                    continue
                nl = r["num_layers"]
                vals = [v for k, v in r[key].items()
                        if v is not None and lo <= int(k) / max(nl - 1, 1) < hi]
                line += f"{(f'{sum(vals) / len(vals):.4f}' if vals else '-'):>20}"
            out.append(line)
    return "\n".join(out)


def _band_means(row: dict, key: str) -> list:
    """Mean of ``key`` per depth quartile, so models of different depth compare."""
    bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    nl = row["num_layers"]
    out = []
    for lo, hi in bands:
        vals = [v for k, v in row[key].items()
                if v is not None and lo <= int(k) / max(nl - 1, 1) < hi]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _slope(bands: list) -> Optional[float]:
    """Last-quartile minus first-quartile — the sign is the shape of the axis."""
    if bands[0] is None or bands[-1] is None:
        return None
    return bands[-1] - bands[0]


def interpret(rows: list) -> str:
    """Which axis, if any, separates the self-supervised model.

    Two tests, because they disagree and the second is the one that matters.

    **Level.** Does the supervised depth control sit between I-JEPA and
    ViT-base? If it does, depth may explain the gap. But "between" is a weak
    test: a control that is 1% of the way across while spanning 60% of the depth
    gap is not evidence of a trend, it is evidence that I-JEPA is an outlier. So
    the position is reported as a *fraction* of the gap and compared against the
    fraction of the depth gap the control covers.

    **Shape.** Levels can match while profiles run in opposite directions. The
    sign of (last depth-quartile − first) is compared across models; a flipped
    sign is a qualitative difference that no amount of depth-matching explains.
    """
    ok = {r["role"]: r for r in rows if "error" not in r}
    jepa, ref = ok.get("subject"), ok.get("reference")
    ctrl = ok.get("depth control")
    if not jepa or not ref:
        return "Not enough models completed to compare."

    out = ["\nWhich axis separates I-JEPA? (1) level"]
    depth_frac = None
    if ctrl:
        span = jepa["num_layers"] - ref["num_layers"]
        if span:
            depth_frac = (ctrl["num_layers"] - ref["num_layers"]) / span

    for axis, key in (("routing entropy", "mean_routing"), ("inter-layer CKA", "mean_cka"),
                      ("intra-layer MI", "mean_mi"), ("fragility (FI)", "fi_score")):
        j, b = jepa.get(key), ref.get(key)
        if j is None or b is None:
            out.append(f"  {axis:<18} n/a")
            continue
        c = ctrl.get(key) if ctrl else None
        note, ctrl_s = "", "n/a"
        if c is not None:
            ctrl_s = f"{c:.4f}"
            frac = (c - b) / (j - b) if abs(j - b) > 1e-12 else 0.0
            if not (0.0 <= frac <= 1.0):
                note = f"  -> control outside the gap ({frac:.0%}): NOT a depth trend"
            elif depth_frac is not None and frac < 0.25 * depth_frac:
                note = (f"  -> control only {frac:.0%} across while spanning "
                        f"{depth_frac:.0%} of the depth gap: I-JEPA is an OUTLIER")
            else:
                note = f"  -> control {frac:.0%} across: consistent with a depth trend"
        out.append(f"  {axis:<18} I-JEPA {j:.4f}  ViT-large {ctrl_s}  "
                   f"ViT-base {b:.4f}{note}")

    out.append("\nWhich axis separates I-JEPA? (2) shape, first vs last depth quartile")
    for axis, key in (("routing entropy", "per_layer_routing"), ("intra-layer MI", "per_layer_mi")):
        line = f"  {axis:<18}"
        signs = {}
        for role, r in (("I-JEPA", jepa), ("ViT-large", ctrl), ("ViT-base", ref)):
            if r is None:
                continue
            s = _slope(_band_means(r, key))
            signs[role] = s
            line += f"  {role} {('n/a' if s is None else f'{s:+.4f}')}"
        js, cs = signs.get("I-JEPA"), signs.get("ViT-large")
        if js is not None and cs is not None and js * cs < 0:
            line += "  -> OPPOSITE SIGN to the supervised control"
        out.append(line)

    out.append("\n  Reminder: depth, width, patch size and objective all differ between")
    out.append("  I-JEPA ViT-H/14 and ViT-base. The ViT-large control is what separates")
    out.append("  a depth trend from an objective effect; read the flags above with it.")
    out.append("  These scanners do not compute the rho cited in the task framing, so")
    out.append("  nothing here confirms or explains that number.")
    return "\n".join(out)


def report(rows: list):
    """Print every table. Pure function of ``rows`` — no GPU, no Modal."""
    print("\n########## STRUCTURAL PROFILES ##########")
    print(profile_table(rows))
    print("\nL = layers, H = heads/layer.  imm/buf/crit = FI layer classification.")
    print("elas/sat/hub = plasticity absorption map.")

    print("\n########## PLASTICITY AXES (means) ##########")
    print(axes_table(rows))

    print(depth_profile(rows))
    print(interpret(rows))

    failed = [r["label"] for r in rows if "error" in r]
    if failed:
        print(f"\nfailed: {', '.join(failed)}")
        for r in rows:
            if "error" in r:
                print(f"  {r['label']}: {r['error']}")


@app.local_entrypoint()
def main():
    # The analysis is a pure function of the scan output, so re-reading it costs
    # nothing while re-scanning costs A10G minutes. Set SAL_REPORT_ONLY=1 to
    # re-derive every table from the saved JSON after changing the reporting.
    if os.environ.get("SAL_REPORT_ONLY", "").strip() not in ("", "0", "false"):
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            rows = json.load(fh)["models"]
        print(f"report-only: re-deriving tables from {RESULTS_PATH} (no scan)")
        report(rows)
        return

    rows = scan_all.remote()
    report(rows)

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"config": {"n_probe": N_PROBE, "batch_size": BATCH_SIZE,
                              "probe": "CIFAR-10 test images",
                              "attn_implementation": "eager",
                              "note": "rho values cited in the task framing are not "
                                      "computed by these scanners"},
                   "models": rows}, fh, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
