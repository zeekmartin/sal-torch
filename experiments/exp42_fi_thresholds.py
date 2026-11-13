#!/usr/bin/env python3
"""exp42 - what threshold makes the layer classification non-degenerate?

``sal.fi.classify_layers`` calls a layer CRITICAL when removing its heads moves
FI by at least ``critical_thr`` *relative*. The shipped default is 0.25, and
until now the docstring justified that number from a run that is not published
in this repository -- so the figures were removed and the default stood on an
assertion. This run measures the same thing on material that is entirely
generic, so the number can carry its evidence again.

**The problem the threshold has, stated as arithmetic.** On ViT-B/16 the
attention graph has 144 nodes and ~1030 edges at the default 0.10 edge density,
and the pretrained checkpoint's FI is ~0.13. Five percent of that is ~0.0066,
which on a 1030-edge graph is about **seven edges**. Nearly every layer moves FI
by more than seven edges when its twelve heads are removed, so at
``critical_thr=0.05`` almost every layer classifies CRITICAL and the
classification carries no information. The question is which threshold leaves a
usable split, and whether the answer is stable across probes.

**What this measures.** Nothing is trained and no model is modified. For each of
three independent probe draws:

* the global FI of the pretrained checkpoint, which also gives the
  **probe-to-probe noise floor** -- the quantity every other FI result in this
  project is missing, including exp37's;
* ``layer_fi_deltas``, the relative FI change from removing each layer's heads;
* the IMMUNE / BUFFER / CRITICAL split at each threshold in the sweep.

**The deltas are saved, not just the splits.** They are the quantity every
threshold is compared against, so once they are on disk any future threshold
question is answerable from the JSON with no GPU and no re-scan.

A usable baseline is **3-8 of 12 layers CRITICAL**: enough headroom that a layer
can move in either direction. A threshold that puts 11 or 12 there is not
classifying anything.

Usage::

    python experiments/exp42_fi_thresholds.py --gate
    sbatch --job-name=exp42 --time=00:40:00 experiments/run_exp.sh \\
           experiments/exp42_fi_thresholds.py \\
           --data-cache /scratch/project_465003656/sal-torch/data_cache \\
           --output results/exp42
"""
from __future__ import annotations

import argparse
import json
import platform
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

DATASET = "evanarlian/imagenet_1k_resized_256"
EDGE_DENSITY = 0.10
N_PROBE = 500
PROBE_SEEDS = (0, 1, 2)
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
IMMUNE_THR = 0.01
USABLE = (3, 8)        # inclusive CRITICAL count that leaves room to move


def make_probe(images_uint8, device, batch_size=16):
    """The probe in the shape ``sal.fi`` actually expects.

    A raw ``(images, labels)`` tuple walks into two traps at once, and neither
    one raises:

    * ``sal.fi._iter_data`` treats any iterable as a sequence of **batches**,
      so a 2-tuple yields the whole image tensor and then the label tensor.
      The images arrive as one N-image batch, and the labels are spared only
      because ``num_samples`` is already reached and the loop breaks first.
    * ``extract_activation_graph`` does **not** normalise, and ``build_data``
      returns **uint8 in [0, 255]**. ``nn.Conv2d`` promotes uint8 silently
      instead of raising, so the scan runs on inputs two orders of magnitude
      outside the ViT's [-1, 1] range and reports a perfectly plausible number.

    ``run_sal_qat_prototype`` normalises on its own FI path (``fi_x =
    normalize(val_split[0][:n_fi])``), and so does the certification harness.
    Building the probe here the same way is what makes the three comparable.
    """
    return [{"pixel_values": P.normalize(
        images_uint8[i:i + batch_size]).to(device)}
        for i in range(0, images_uint8.shape[0], batch_size)]


def layer_deltas(adjacency, num_heads_per_layer):
    """Relative FI change from removing each layer's heads.

    ``sal.fi`` exports no function for this: ``classify_layers`` computes the
    delta and then discards it, returning only the IMMUNE / BUFFER / CRITICAL
    label. The arithmetic below is its arithmetic -- same ``abs()``, same
    ``max(base, 1e-10)`` guard -- so a delta here classifies exactly as the
    shipped function would. It needs only the adjacency, so no model and no
    second forward pass are involved, which is why the sweep is free once the
    graph exists.
    """
    from sal.fi import compute_fi

    total = adjacency.shape[0]
    base = compute_fi(adjacency)
    out = {}
    for li in range(total // num_heads_per_layer):
        s, e = li * num_heads_per_layer, (li + 1) * num_heads_per_layer
        keep = np.ones(total, dtype=bool)
        keep[s:e] = False
        red = compute_fi(adjacency[np.ix_(keep, keep)])
        out[li] = abs(red - base) / max(base, 1e-10)
    return out


def scan_once(model, probe, n_probe, density):
    """Global FI plus the per-layer relative deltas, for one probe draw.

    Signature note: ``extract_activation_graph`` takes ``num_samples`` and
    ``density``, and reads the device off the model -- it has no ``device``
    argument. exp37 calls it the same way.
    """
    from sal.fi import compute_fi, extract_activation_graph

    adjacency = extract_activation_graph(
        model, probe, num_samples=n_probe, density=density)
    fi = float(compute_fi(adjacency))
    nh = model.config.num_attention_heads
    deltas = layer_deltas(adjacency, nh)
    return fi, {int(k): float(v) for k, v in deltas.items()}, adjacency


def split_at(deltas, critical_thr, immune_thr=IMMUNE_THR):
    """IMMUNE / BUFFER / CRITICAL counts from stored deltas. No model needed."""
    immune = sum(1 for v in deltas.values() if v < immune_thr)
    critical = sum(1 for v in deltas.values() if v >= critical_thr)
    return {"immune": immune, "buffer": len(deltas) - immune - critical,
            "critical": critical}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--n-probe", type=int, default=N_PROBE)
    p.add_argument("--probe-seeds", type=int, nargs="+", default=list(PROBE_SEEDS))
    p.add_argument("--density", type=float, default=EDGE_DENSITY)
    p.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--output", default="results/exp42")
    p.add_argument("--gate", action="store_true",
                   help="one probe of 64 images, two thresholds")
    args = p.parse_args(argv)

    if args.gate:
        args.n_probe, args.probe_seeds = 64, [0]
        args.thresholds = [0.05, 0.25]

    device, _ = P.detect_device()
    P.log("\n=== exp42 - FI layer-classification thresholds ===")
    P.log(f"  {len(args.probe_seeds)} probe(s) x {args.n_probe} images | "
          f"edge density {args.density} | thresholds {args.thresholds}")

    P.set_seed(0)
    # build_data(sizes, seed, workers, dataset_id, cache_dir) -- dataset is a
    # KEYWORD; positionally it binds to `sizes` and silently loads ImageNet-100.
    if args.data_cache:
        P.LOADER_WORKERS = args.loader_workers
    sizes = {"n_train": 2000, "n_val": args.n_probe * len(args.probe_seeds) + args.n_probe,
             "n_fi": args.n_probe}
    _, val_split, num_classes = P.build_data(
        sizes, 0, args.workers, dataset_id=args.dataset, cache_dir=args.data_cache)
    if num_classes != 1000 and not args.gate:
        raise RuntimeError(
            f"loaded {num_classes} classes, expected 1000 -- this is not "
            f"ImageNet-1k. Check --dataset ({args.dataset}).")

    model = P.load_model(num_classes).to(device).eval()

    out = {
        "experiment": "exp42_fi_thresholds",
        "model": P.MODEL_ID, "dataset": args.dataset,
        "dataset_label": P.dataset_label(args.dataset),
        "num_classes": num_classes,
        "n_probe": args.n_probe, "probe_seeds": args.probe_seeds,
        "edge_density": args.density, "thresholds": args.thresholds,
        "immune_thr": IMMUNE_THR, "gate": args.gate,
        "fi_definition": "fraction of edges with zero triangle support",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__, "platform": platform.platform(),
        "notes": {
            "scope": "pretrained checkpoint only; nothing trained, nothing "
                     "compressed. The thresholds are a property of the "
                     "classification arithmetic, not of any downstream task.",
            "deltas_saved": "per-layer relative FI deltas are stored, so any "
                            "future threshold question is answerable from this "
                            "JSON with no GPU.",
        },
        "probes": {},
    }

    dest = Path(args.output); dest.mkdir(parents=True, exist_ok=True)
    path = dest / "exp42_fi_thresholds.json"

    n_val = val_split[0].shape[0]
    for ps in args.probe_seeds:
        t0 = time.time()
        g = torch.Generator().manual_seed(ps)
        idx = torch.randperm(n_val, generator=g)[:args.n_probe]
        probe = make_probe(val_split[0][idx], device)
        fi, deltas, adjacency = scan_once(
            model, probe, args.n_probe, args.density)
        edges = int(adjacency.sum()) // 2
        out["probes"][str(ps)] = {
            "fi": float(fi), "edges": edges,
            "nodes": int(adjacency.shape[0]),
            "layer_deltas": deltas,
            "splits": {f"{t:.2f}": split_at(deltas, t) for t in args.thresholds},
            "seconds": round(time.time() - t0, 1),
        }
        P.log(f"  probe seed {ps}: FI {fi:.6f} over {edges} edges "
              f"[{time.time() - t0:.0f}s]")
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- the tables
    fis = [v["fi"] for v in out["probes"].values()]
    edges = out["probes"][str(args.probe_seeds[0])]["edges"]
    spread = max(fis) - min(fis)
    out["fi_noise"] = {
        "probes": len(fis), "min": min(fis), "max": max(fis),
        "spread": spread, "spread_pct_of_mean": spread / (sum(fis) / len(fis)) * 100,
        "meaning": ("the probe-to-probe spread of global FI on an UNCHANGED "
                    "model. Any FI difference smaller than this is not "
                    "evidence of anything."),
    }
    P.log(f"\nprobe-to-probe FI noise on an unchanged model: "
          f"{min(fis):.6f}-{max(fis):.6f}, spread {spread:.6f} "
          f"({out['fi_noise']['spread_pct_of_mean']:.1f}% of the mean)")
    P.log("  Any FI delta smaller than that is not evidence. This is the "
          "number every other FI result in the project was missing.")

    P.log(f"\nCRITICAL layers of {len(out['probes'][str(args.probe_seeds[0])]['layer_deltas'])}"
          f", by threshold and probe (usable band {USABLE[0]}-{USABLE[1]}):")
    header = "  " + f"{'thr':>6s}" + "".join(f"{'p' + str(s):>6s}" for s in args.probe_seeds) + "   verdict"
    P.log(header)
    usable = []
    for t in args.thresholds:
        counts = [out["probes"][str(s)]["splits"][f"{t:.2f}"]["critical"]
                  for s in args.probe_seeds]
        ok = all(USABLE[0] <= c <= USABLE[1] for c in counts)
        if ok:
            usable.append(t)
        P.log("  " + f"{t:6.2f}" + "".join(f"{c:6d}" for c in counts)
              + ("   usable on every probe" if ok else ""))
    out["usable_thresholds"] = usable
    out["shipped_default"] = 0.25
    out["shipped_default_usable"] = 0.25 in usable
    P.log(f"\nusable on every probe: {usable or 'none'}")
    P.log(f"shipped default 0.25 is "
          f"{'inside' if out['shipped_default_usable'] else 'OUTSIDE'} that set")

    # Five percent of FI, in edges -- the arithmetic the docstring states.
    mean_fi = sum(fis) / len(fis)
    out["five_percent_in_edges"] = round(0.05 * mean_fi * edges, 1)
    P.log(f"\n5% of FI={mean_fi:.6f} on {edges} edges is "
          f"{out['five_percent_in_edges']} edges -- which is why 0.05 "
          f"saturates: almost any layer clears a bar that low.")

    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    P.log(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
