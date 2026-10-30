#!/usr/bin/env python3
"""exp01 - reproduce the v0.5.1 FI/plasticity scan on an AMD MI250X (ROCm).

Phase 1 of the v0.6.0 LUMI plan: confirm sal-torch's structural scanners give
the *same numbers* on ROCm that they gave on NVIDIA, before any GPU-hours go
into training experiments.

The probe is deliberately identical to the one behind the published v0.5.1
figures (``scripts/modal_jepa_scan.py`` -> ``scripts/jepa_scan_results.json``):
**256 CIFAR-10 test images**, batch size 8, encoder in eager attention. FI is a
function of the activation correlation structure, so random-noise input would
produce a different, unusable number - the comparison only means something
against the same probe.

Reference values to reproduce (NVIDIA, v0.5.1):

    facebook/ijepa_vith14_1k   FI=0.0040   1 immune /  4 buffer / 27 critical

Usage::

    python exp01_fi_scan.py --model facebook/dinov2-base
    python exp01_fi_scan.py --model facebook/ijepa_vith14_1k   # needs transformers>=4.47

Offline on compute nodes: everything must already be in HF_HOME. Pre-fetch from
a login node with ``--download-only``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

N_PROBE = 256          # v0.5.1 reference probe size
BATCH_SIZE = 8         # v0.5.1 reference batch size
PROBE_DATASET = "uoft-cs/cifar10"
DEFAULT_MODEL = "facebook/dinov2-base"
RESULTS_DIR = "/scratch/project_465003656/sal-torch/results"

# Published v0.5.1 numbers, for an automatic pass/fail on the reproduction.
REFERENCE = {
    "facebook/ijepa_vith14_1k": {
        "fi_score": 0.004, "num_layers": 32, "num_heads": 16,
        "num_immune": 1, "num_buffer": 4, "num_critical": 27,
        "mean_routing": 0.8656, "mean_cka": 0.9728, "mean_mi": 0.0609,
    },
}


def log(msg):
    print(msg, flush=True)


def load_model(model_id):
    """The bare encoder in eager attention, with a clear error on old transformers."""
    import transformers
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(model_id, attn_implementation="eager")
    except (KeyError, ValueError) as e:
        if "ijepa" in model_id.lower():
            raise RuntimeError(
                "{} needs transformers >= 4.47 for I-JEPA support; this "
                "environment has {}. AutoModel said: {}".format(
                    model_id, transformers.__version__, e)) from e
        raise


def vision_probe(processor_id, n, batch_size):
    """CIFAR-10 test images as {pixel_values} batches - the v0.5.1 probe."""
    from datasets import load_dataset
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(processor_id)
    imgs = load_dataset(PROBE_DATASET, split="test").select(range(n))["img"]
    out = []
    for i in range(0, len(imgs), batch_size):
        px = proc(images=imgs[i:i + batch_size], return_tensors="pt")["pixel_values"]
        out.append({"pixel_values": px})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF encoder id (default {})".format(DEFAULT_MODEL))
    p.add_argument("--n-probe", type=int, default=N_PROBE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--output", default=RESULTS_DIR)
    p.add_argument("--no-plasticity", action="store_true",
                   help="FI only; skip the routing/CKA/MI scan")
    p.add_argument("--download-only", action="store_true",
                   help="fetch model + probe into HF_HOME and exit (run on a login node)")
    args = p.parse_args(argv)

    slug = args.model.split("/")[-1].replace(".", "-")

    if args.download_only:
        log("Pre-fetching {} + {} into HF_HOME={}".format(
            args.model, PROBE_DATASET,
            os.environ.get("HF_HOME", "~/.cache/huggingface")))
        load_model(args.model)
        vision_probe(args.model, args.batch_size, args.batch_size)
        log("  OK - model and probe dataset are cached")
        return 0

    import torch
    import transformers
    import sal
    from sal import FIScanner, PlasticityScanner

    if not torch.cuda.is_available():
        log("ERROR: no GPU visible - this experiment must run on a GPU node")
        return 1

    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
    log("exp01 FI scan | {} | torch {} | sal {}".format(
        dev_name, torch.__version__, sal.__version__))
    log("  model={}  probe={} CIFAR-10 images, batch {}".format(
        args.model, args.n_probe, args.batch_size))

    t0 = time.time()
    model = load_model(args.model).to(device).eval()
    n_params = sum(q.numel() for q in model.parameters())
    probe = vision_probe(args.model, args.n_probe, args.batch_size)
    t_load = time.time() - t0
    log("  {:.1f}M params, loaded + probe built in {:.0f}s ({} batches)".format(
        n_params / 1e6, t_load, len(probe)))

    t0 = time.time()
    fi = FIScanner(model, probe, num_samples=args.n_probe,
                   batch_size=args.batch_size).scan()
    t_fi = time.time() - t0
    log("  FI: {}  ({:.0f}s)".format(fi.summary, t_fi))

    out = {
        "experiment": "exp01_fi_scan",
        "model": args.model,
        "params_m": round(n_params / 1e6, 1),
        "probe": {"dataset": PROBE_DATASET, "split": "test", "n": args.n_probe,
                  "batch_size": args.batch_size, "attn_implementation": "eager"},
        "fi_score": fi.fi_score,
        "num_layers": fi.num_layers,
        "num_heads_per_layer": fi.num_heads_per_layer,
        "num_immune": len(fi.immune_layers),
        "num_buffer": len(fi.buffer_layers),
        "num_critical": len(fi.critical_layers),
        "fi_summary": fi.summary,
        "layer_classification": {str(k): v.value for k, v in fi.layer_map.items()},
        "timing_s": {"load_and_probe": round(t_load, 1), "fi_scan": round(t_fi, 1)},
        "env": {
            "device": dev_name,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sal": sal.__version__,
            "python": platform.python_version(),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        },
    }

    if not args.no_plasticity:
        t0 = time.time()
        pmap = PlasticityScanner(model, probe, num_samples=args.n_probe,
                                 batch_size=args.batch_size).scan()
        t_pl = time.time() - t0
        log("  plasticity: {}  ({:.0f}s)".format(pmap.summary, t_pl))
        routing = [v for v in pmap.routing.values() if not math.isnan(v)]
        cka = list(pmap.cka_similarity.values())
        mi = list(pmap.mutual_info.values())
        out["plasticity"] = {
            "summary": pmap.summary,
            "mean_routing": round(sum(routing) / len(routing), 4) if routing else None,
            "mean_cka": round(sum(cka) / len(cka), 4) if cka else None,
            "mean_mi": round(sum(mi) / len(mi), 4) if mi else None,
            "routing_available": bool(routing),
        }
        out["timing_s"]["plasticity_scan"] = round(t_pl, 1)
        out["env"]["peak_gpu_mem_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)

    # Automatic reproduction check against the published NVIDIA numbers.
    ref = REFERENCE.get(args.model)
    if ref:
        delta = abs(out["fi_score"] - ref["fi_score"])
        same_class = (out["num_immune"] == ref["num_immune"]
                      and out["num_buffer"] == ref["num_buffer"]
                      and out["num_critical"] == ref["num_critical"])
        out["reproduction"] = {
            "reference": ref,
            "fi_abs_delta": round(delta, 6),
            "fi_matches": delta < 5e-4,
            "layer_classification_matches": same_class,
            "verdict": "MATCH" if (delta < 5e-4 and same_class) else "DIVERGED",
        }
        log("  reproduction vs v0.5.1 (NVIDIA): {} (FI {:.4f} vs {:.4f}, delta={:.6f})".format(
            out["reproduction"]["verdict"], out["fi_score"], ref["fi_score"], delta))
    else:
        out["reproduction"] = {
            "reference": None,
            "note": "no published v0.5.1 figure for this model; this run "
                    "establishes the ROCm baseline",
        }
        log("  no published reference for this model - recording ROCm baseline")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "exp01_fi_scan_{}.json".format(slug)
    dest.write_text(json.dumps(out, indent=2))
    log("  results -> {}".format(dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
