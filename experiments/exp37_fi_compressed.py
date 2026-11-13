#!/usr/bin/env python3
"""exp37 - does the FI diagnostic detect compression, and how far does each move it?

FI is the product's certification signal. Every FI number the project has
published is read off an **uncompressed** checkpoint, so nothing says whether a
certification run can tell a compressed model from its parent -- or whether the
three compression families the wave-1 campaign measured look alike to it.

The question has a structural asymmetry that makes it worth asking, and that
the design has to respect:

- **Structured head pruning removes nodes.** A ViT-B/16 at P50 has 72 heads
  where the parent had 144, so FI is computed on a graph half the size. The
  number is comparable (FI is a fraction of edges) but the object is not the
  same graph, and a shift could be the compression or could be the smaller
  graph's different edge budget. **A random-head-set control at the same size
  separates those two**, and is why the `head50_random` arms exist below.
- **Unstructured and 2:4 keep every node.** The graph has 144 heads either way
  and only the activations change, so any FI shift is attributable.

So the comparison is not one ladder but two, and conflating them would answer
the wrong question.

## Tiers

**Tier A (this file's default): one-shot, no training, ~0.5 GPU-hours.**
Pruning from the pretrained checkpoint takes 1-29 seconds (exp33 measured
`prune_seconds`), so every arm is built in-process and scanned. Answers the
question for the *compression operation*.

**Tier B (`--checkpoints DIR`): the recovered models.** Whether two epochs of
recovery move FI back toward the parent is a different question, and it needs
the recovered weights. Wave 1 saved none -- hence `--save-checkpoint` in every
wave-2 script. Point this at those directories once exp35 and the re-runs land.

    python experiments/exp37_fi_compressed.py --gate
    sbatch --job-name=exp37a --time=01:30:00 experiments/run_exp.sh \\
           experiments/exp37_fi_compressed.py \\
           --data-cache /scratch/project_465003656/sal-torch/data_cache \\
           --output results/exp37
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent / "scripts"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_sal_qat_prototype as P                                  # noqa: E402
from exp23_baselines import magnitude_heads                        # noqa: E402
from exp33_sparse_core import (attention_linear_layers,            # noqa: E402
                               linear_layers, prune_model, sparsity_report)

DATASET = "evanarlian/imagenet_1k_resized_256"
# FI's own probe size. 500 is sal.fi's default and what every published scan
# used; keeping it means these numbers sit on the same scale as PHASE1's.
N_PROBE = 500
EDGE_DENSITY = 0.10


def fi_of(model, probe, device, n_probe=N_PROBE, density=EDGE_DENSITY) -> dict:
    """FI plus the graph shape it was computed on."""
    from sal.fi import compute_fi, extract_activation_graph

    model = model.to(device).eval()
    adj = extract_activation_graph(model, probe, num_samples=n_probe,
                                   batch_size=16, density=density)
    A = (np.asarray(adj) != 0).astype(np.int64)
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    return {
        "fi": float(compute_fi(adj)),
        "nodes": int(n),
        "edges": int(A.sum() // 2),
        "edge_density": float(A.sum() / max(n * (n - 1), 1)),
        "n_probe": n_probe,
        "requested_density": density,
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

def _sliced(pretrained, pairs):
    from sal import slice_heads
    return slice_heads(copy.deepcopy(pretrained), pairs).eval()


def build_tier_a(pretrained, calib, device, log, skip=()):
    """Every one-shot arm, yielded as ``(name, model, detail)``.

    Node-removing and node-preserving arms are tagged, because a shift in FI
    means a different thing in each.
    """
    cfg = pretrained.config
    nl, nh = cfg.num_hidden_layers, cfg.num_attention_heads

    yield "dense_pretrained", pretrained, {
        "family": "none", "nodes_removed": False, "params_zeroed": 0}

    # ---- structured: fewer nodes --------------------------------------- #
    for ratio in (0.25, 0.33, 0.50):
        name = f"head{int(ratio * 100)}_magnitude"
        if name in skip:
            continue
        pairs = magnitude_heads(pretrained, ratio, per_layer=True)
        yield name, _sliced(pretrained, pairs), {
            "family": "structured", "nodes_removed": True,
            "selection": "magnitude", "head_ratio": ratio,
            "heads_removed": len(pairs)}

    # The control that makes the structured rows readable: the same NUMBER of
    # heads removed at random. Without it, an FI shift at P50 cannot be told
    # apart from "a 72-head graph has a different edge budget than a 144-head
    # one". Two seeds, because the draw is the only stochastic input.
    for seed in (0, 1):
        name = f"head50_random_s{seed}"
        if name in skip:
            continue
        pairs = P.uniform_random_heads(nl, nh, 0.50, seed)
        yield name, _sliced(pretrained, pairs), {
            "family": "structured_control", "nodes_removed": True,
            "selection": f"random seed {seed}", "head_ratio": 0.50,
            "heads_removed": len(pairs)}

    # ---- sparse: every node kept, weights zeroed ----------------------- #
    attn = attention_linear_layers(pretrained)
    whole = {n: l for n, l in linear_layers(pretrained).items()
             if l.weight.dim() == 2 and "classifier" not in n}
    # The matched sparsities are exp33's: attn 0.50 and whole 0.1667 both
    # remove 16.35% of the model, the same share structured P50 removes.
    for method in ("wanda", "sparsegpt"):
        for scope, sparsity, nm in (("attn", 0.50, None),
                                    ("whole", 0.1667, None),
                                    ("attn_nm", 0.50, (2, 4))):
            name = f"{method}_{scope}"
            if name in skip:
                continue
            model = copy.deepcopy(pretrained).to(device)
            target = (attention_linear_layers(model) if "attn" in scope else
                      {n: l for n, l in linear_layers(model).items()
                       if "classifier" not in n})
            prunen, prunem = nm if nm else (0, 0)
            t0 = time.time()
            report = prune_model(model, target, calib, method, sparsity,
                                 device=device, log=log, prunen=prunen,
                                 prunem=prunem)
            yield name, model.eval(), {
                "family": "semi_structured" if nm else "unstructured",
                "nodes_removed": False, "method": method, "scope": scope,
                "nm": list(nm) if nm else None,
                "sparsity": sparsity,
                "prune_report": report,
                "params_zeroed": report.get("zeroed"),
                "zero_frac": report.get("target_zero_frac"),
                "prune_seconds": round(time.time() - t0, 1)}

    # ---- quantization: every node and every position kept -------------- #
    if "int4_roundtrip" not in skip:
        yield "int4_roundtrip", P.quantize_weights(
            copy.deepcopy(pretrained), 4).eval(), {
            "family": "quantization", "nodes_removed": False,
            "note": "simulated per-channel round-trip; no weight is zeroed, "
                    "every one is rounded"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--n-probe", type=int, default=N_PROBE)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--density", type=float, default=EDGE_DENSITY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--skip", nargs="*", default=[],
                   help="arm names to leave out")
    p.add_argument("--checkpoints", default=None,
                   help="tier B: a directory of *.pt written by a wave-2 "
                        "script's --save-checkpoint, scanned alongside tier A")
    p.add_argument("--output", default="results/exp37")
    p.add_argument("--gate", action="store_true",
                   help="3 arms, 64 probe images -- exercises the structured, "
                        "sparse and quantized paths and the FI call")
    args = p.parse_args(argv)

    if args.gate:
        args.n_probe, args.n_calib = 64, 16
        args.skip = ["head25_magnitude", "head33_magnitude",
                     "head50_random_s1", "wanda_whole", "sparsegpt_whole",
                     "sparsegpt_attn", "sparsegpt_attn_nm"]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device, _ = P.detect_device()
    use_amp = device.type == "cuda"
    P.log("\n=== exp37 - FI on compressed models (tier A: one-shot) ===")
    P.log(f"  probe {args.n_probe} images | calib {args.n_calib} | "
          f"edge density {args.density}")

    P.set_seed(args.seed)
    # build_data(sizes, seed, workers, dataset_id, cache_dir) -- the dataset
    # is a KEYWORD. Passed positionally it silently binds to `sizes` and the
    # run loads the default ImageNet-100 instead, which is how the first gate
    # failed; had the shapes been compatible it would have produced plausible
    # ImageNet-100 numbers under an ImageNet-1k label.
    sizes = {"n_train": max(args.n_calib * 8, 2000), "n_val": args.n_probe,
             "n_fi": args.n_probe}
    train_split, val_split, num_classes = P.build_data(
        sizes, args.seed, args.workers, dataset_id=args.dataset,
        cache_dir=args.data_cache)
    if num_classes != 1000 and not args.gate:
        raise RuntimeError(
            f"loaded {num_classes} classes, expected 1000 -- this is not "
            f"ImageNet-1k. Check --dataset ({args.dataset}).")
    pretrained = P.load_model(num_classes).to(device).eval()

    # FI's probe and the pruners' calibration set are deliberately different
    # data: calibrating and then measuring on the same images would let a
    # pruner look structurally intact on exactly what it was fitted to.
    probe = (val_split[0][:args.n_probe], val_split[1][:args.n_probe])
    # exp33's calibration shape exactly: normalised, in batches of 16, drawn
    # with the run's seed. prune_model pushes these through the model once.
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(train_split[0].shape[0], generator=g)[:args.n_calib]
    calib_x = P.normalize(train_split[0][idx])
    calib = [(calib_x[i:i + 16],) for i in range(0, len(calib_x), 16)]

    out = {
        "experiment": "exp37_fi_compressed",
        "tier": "A" + ("+B" if args.checkpoints else ""),
        "model": "google/vit-base-patch16-224",
        "dataset": args.dataset,
        "seed": args.seed,
        "gate": args.gate,
        "n_probe": args.n_probe,
        "n_calib": args.n_calib,
        "edge_density": args.density,
        "fi_definition": "fraction of head-graph edges with zero triangle "
                         "support (sal.fi.compute_fi); NOT 1 - lambda2/lambda_max",
        "notes": {
            "asymmetry": "Structured pruning removes graph NODES (144 heads "
                         "-> 72 at P50), so its FI is computed on a smaller "
                         "graph; unstructured, 2:4 and INT4 keep every node "
                         "and change only the activations. The two are "
                         "reported as separate ladders, and the "
                         "head50_random_* arms are the control that separates "
                         "'the compression moved FI' from 'a smaller graph "
                         "has a different edge budget'.",
            "probe_vs_calib": "The FI probe and the pruners' calibration set "
                              "are different images, so no pruner is measured "
                              "on what it was fitted to.",
            "no_accuracy": "No accuracy here. This asks what the diagnostic "
                           "sees, not what the compression costs; the costs "
                           "are in EXP32/EXP33 and exp35.",
            "tier_b": "The recovered models need saved weights. Wave 1 saved "
                      "none; --save-checkpoint in the wave-2 scripts fixes "
                      "that, and --checkpoints reads them here.",
        },
        "gpu": torch.cuda.get_device_name(0) if use_amp else None,
        "torch_version": torch.__version__,
        "arms": {},
    }
    dest = out_dir / "exp37_fi_compressed.json"

    def save():
        dest.write_text(json.dumps(out, indent=1), encoding="utf-8")

    save()
    baseline_fi = None
    for name, model, detail in build_tier_a(pretrained, calib, device, P.log,
                                            skip=tuple(args.skip)):
        t0 = time.time()
        scan = fi_of(model, probe, device, args.n_probe, args.density)
        detail.update(scan)
        detail["scan_seconds"] = round(time.time() - t0, 1)
        if name == "dense_pretrained":
            baseline_fi = scan["fi"]
        if baseline_fi:
            detail["fi_delta"] = scan["fi"] - baseline_fi
            detail["fi_relative_change"] = (scan["fi"] - baseline_fi) / baseline_fi
        P.log(f"  {name:<24} FI {scan['fi']:.6f}  "
              f"{scan['nodes']:>4} nodes {scan['edges']:>5} edges"
              + (f"  {detail['fi_relative_change']:+7.1%} vs dense"
                 if baseline_fi and name != "dense_pretrained" else ""))
        out["arms"][name] = detail
        save()
        if model is not pretrained:
            model.to("cpu")
            del model
            if use_amp:
                torch.cuda.empty_cache()

    if args.checkpoints:
        out["arms"].update(scan_checkpoints(
            Path(args.checkpoints), probe, device, args, baseline_fi, P.log))
        save()

    P.log(f"\nwrote {dest}")
    try:
        P.log("\n" + ladders(out))
    except Exception as e:                                        # noqa: BLE001
        P.log(f"table failed (data is saved): {type(e).__name__}: {e}")
    return 0


def scan_checkpoints(directory: Path, probe, device, args, baseline_fi, log):
    """Tier B: FI of recovered models written by --save-checkpoint."""
    from transformers import ViTConfig, ViTForImageClassification

    found = sorted(directory.rglob("*.pt"))
    if not found:
        raise RuntimeError(f"--checkpoints {directory} holds no *.pt")
    log(f"\n=== tier B: {len(found)} checkpoint(s) ===")
    rows = {}
    for ckpt in found:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = ViTForImageClassification(ViTConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        scan = fi_of(model.eval(), probe, device, args.n_probe, args.density)
        name = f"recovered_{blob.get('arm', ckpt.stem)}_s{blob.get('seed')}"
        scan.update(family="recovered", nodes_removed=None,
                    checkpoint=str(ckpt),
                    recover_epochs=blob.get("recover_epochs"),
                    accuracy=blob.get("accuracy"))
        if baseline_fi:
            scan["fi_delta"] = scan["fi"] - baseline_fi
            scan["fi_relative_change"] = (scan["fi"] - baseline_fi) / baseline_fi
        log(f"  {name:<30} FI {scan['fi']:.6f}  {scan['nodes']} nodes")
        rows[name] = scan
        model.to("cpu")
        del model
    return rows


def ladders(out: dict) -> str:
    """Two ladders, because a node-removing shift is not a node-preserving one."""
    groups = [("node-preserving (same 144-head graph)",
               ("none", "unstructured", "semi_structured", "quantization")),
              ("node-removing (smaller graph)",
               ("structured", "structured_control")),
              ("recovered (tier B)", ("recovered",))]
    lines = []
    for title, fams in groups:
        rows = [(n, d) for n, d in out["arms"].items()
                if d.get("family") in fams]
        if not rows:
            continue
        head = f"{'arm':<26}{'FI':>10}{'nodes':>7}{'edges':>7}{'vs dense':>10}"
        lines += [f"\n{title}", head, "-" * len(head)]
        for n, d in rows:
            rel = (f"{d['fi_relative_change']:+9.1%}"
                   if d.get("fi_relative_change") is not None else "        -")
            lines.append(f"{n:<26}{d['fi']:>10.6f}{d['nodes']:>7}"
                         f"{d['edges']:>7}{rel}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
