"""SAL on a self-supervised vision encoder (I-JEPA) for edge-ready perception.

The pitch this example exists to test: a robot's perception model has to fit on
the robot. Compression is how it fits, and SAL is a claim about *surviving*
compression. For a supervised model you check that with accuracy. For I-JEPA
there is no accuracy — there is no head — so the question has to be asked about
the representations, which is what ``sal.evaluation`` is for.

What it does::

    1. structural scan of the pretrained encoder (FI + plasticity)
    2. SAL training with a self-supervised objective, via SALTrainer(train_step=)
    3. structural scan again
    4. prune at 33% and 50%, SAL-trained vs standard, and score both on
       linear probe / kNN / CKA / latency / parameters

Two things this file is honest about, because both affect how the numbers read:

**There is no I-JEPA ViT-B/16.** Meta released I-JEPA at ViT-H/14 and ViT-g/16
only (``facebook/ijepa_vith14_1k``, ``facebook/ijepa_vitg16_22k``). The default
here is the H/14 — 632M parameters, which is a real GPU. ``--model`` takes any
HF vision encoder if you want a smaller stand-in, but a ViT-B/16 loaded from
somewhere else is a different model trained a different way, and calling its
result an I-JEPA result would be wrong.

**The training objective is self-distillation, not I-JEPA's.** Real I-JEPA has
a separate predictor network and a target encoder updated by EMA, and it
predicts *masked patch* representations from visible context. Reproducing that
needs the pretraining apparatus. What runs here is the simplification the
benchmark asks for: the unmasked model's representation is the target, the
SAL-perturbed model's is the prediction, MSE between them. That is a legitimate
self-supervised objective and it exercises exactly what v0.5.1 added — a loss
computed outside the model, with SAL masking around it — but it is not I-JEPA
pretraining and the results should not be described as such.

Usage::

    python examples/jepa_sal.py --smoke                  # CPU, tiny, ~1 min
    python examples/jepa_sal.py --epochs 5 --mask-ratio 0.3

Requires: ``pip install sal-torch[reports] transformers datasets``
(no timm — the checkpoint is on the Hub as an ``IJepaModel``).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sal import (FIScanner, PlasticityScanner, SALConfig, cka_similarity,
                 count_params, knn_accuracy, linear_probe, measure_latency)
from sal.trainer import SALTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("jepa_sal")

DEFAULT_MODEL = "facebook/ijepa_vith14_1k"
PRUNE_RATIOS = (0.33, 0.50)


# ------------------------------------------------------------------ the model
def load_jepa(model_id: str = DEFAULT_MODEL, device="cpu"):
    """Load a pretrained I-JEPA encoder.

    ``attn_implementation="eager"`` is not optional if you want to look at
    attention maps afterwards — SDPA returns none, and the plots come out empty
    with no error to tell you why.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    return model.to(device)


def load_tiny_stand_in(device="cpu"):
    """A 2-layer I-JEPA with random weights — for --smoke, on CPU, offline."""
    from transformers import IJepaConfig, IJepaModel
    torch.manual_seed(0)
    cfg = IJepaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      intermediate_size=128, image_size=32, patch_size=16)
    return IJepaModel(cfg).to(device)


# ----------------------------------------------------------------- the objective
def jepa_train_step(model, batch, optimizer, mask_module):
    """Self-supervised step: predict the unmasked model's own representations.

    The target comes from the model with SAL masking **suspended**, under
    ``no_grad``; the prediction comes from the same weights with the pruned
    heads zeroed. Minimizing the gap is what forces the surviving heads to take
    over the removed ones' function.

    ``mask_module.unmasked()`` rather than ``remove_mask()`` / ``apply_mask()``
    because it restores the previous state even if the forward pass raises, and
    — the part that matters — it does not disturb the accumulated pruned set.
    ``deactivate()`` would reset it and quietly undo the schedule.
    """
    pixels = batch[0] if isinstance(batch, (list, tuple)) else batch

    with torch.no_grad(), mask_module.unmasked():
        target = model(pixel_values=pixels).last_hidden_state

    predicted = model(pixel_values=pixels).last_hidden_state
    loss = torch.nn.functional.mse_loss(predicted, target.detach())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# -------------------------------------------------------------------- pruning
def prune_heads(model, ratio: float, method: str = "random", seed: int = 0):
    """Return a copy of ``model`` with ``ratio`` of its heads masked off.

    ``random`` is SAL's own selection — the validated default, and the one a
    SAL-trained model was trained to survive. ``magnitude`` removes the heads
    with the smallest output-projection norm, the standard post-hoc baseline.

    Heads are *masked*, not sliced out, so both arms keep the same parameter
    count and the comparison isolates function from size. ``sal.slice_heads()``
    is what makes the saving real once a ratio has been chosen.
    """
    import random as _random

    from sal import arch_support
    from sal.masker import HeadMasker

    pruned = copy.deepcopy(model)
    info = arch_support.detect_architecture(pruned)
    cfg = SALConfig.auto(pruned, prune_fraction=max(ratio, 1e-6), prune_start_ratio=0.0)

    masker = HeadMasker(pruned, cfg, seed=seed)
    masker.install()
    if method == "random":
        masker.activate()
    elif method == "magnitude":
        masker.apply_mask()
        for layer_idx, order in enumerate(_magnitude_order(pruned, info)):
            n_drop = int(round(ratio * info.num_heads))
            for head in order[:n_drop]:
                masker._masks[layer_idx][head] = 0.0
    else:
        raise ValueError(f"method must be 'random' or 'magnitude', got {method!r}")

    # The handles stay installed on purpose: the returned model carries its mask.
    pruned._sal_masker = masker
    return pruned


def _magnitude_order(model, info):
    """Per layer, head indices ordered weakest-first by output-projection norm."""
    from sal import arch_support

    orders = []
    for proj in arch_support.get_output_projections(model, info.attention_pattern):
        w = proj.weight.detach()                  # [hidden_out, hidden_in]
        per_head = w.view(w.shape[0], info.num_heads, -1).norm(dim=(0, 2))
        orders.append(torch.argsort(per_head).tolist())
    return orders


# ----------------------------------------------------------------- benchmarking
def benchmark_compression(model, train_loader, val_loader, probe_loader,
                          ratios=PRUNE_RATIOS, arm: str = "sal", image_shape=None):
    """Score each prune ratio under random (SAL) and magnitude selection."""
    results = {}
    for ratio in ratios:
        for method in ("random", "magnitude"):
            name = f"{arm}_{method}_{int(ratio * 100)}"
            log.info("  scoring %s ...", name)
            pruned = prune_heads(model, ratio, method=method)
            results[name] = {
                "linear_probe": linear_probe(pruned, train_loader, val_loader),
                "knn_accuracy": knn_accuracy(pruned, train_loader, val_loader, k=20),
                "cka_similarity": cka_similarity(model, pruned, probe_loader),
                "params": count_params(pruned),
                "latency_cpu_ms": measure_latency(pruned, input_shape=image_shape,
                                                  device="cpu", runs=20),
            }
            if torch.cuda.is_available():
                results[name]["latency_gpu_ms"] = measure_latency(
                    pruned, input_shape=image_shape, device="cuda")
    return results


# ------------------------------------------------------------------ probe data
def synthetic_loaders(image_size: int, n_train=64, n_val=32, num_classes=4, bs=8):
    """Labelled synthetic images — enough to exercise the path without a dataset.

    Each class is a fixed random pattern plus noise, so the probes have
    something real to find. Swap in ImageNet-100 for a result worth quoting; see
    ``scripts/modal_jepa_sal.py``.
    """
    g = torch.Generator().manual_seed(0)
    prototypes = torch.randn(num_classes, 3, image_size, image_size, generator=g)

    def make(n, seed):
        gg = torch.Generator().manual_seed(seed)
        y = torch.arange(n) % num_classes
        x = prototypes[y] + 0.5 * torch.randn(n, 3, image_size, image_size, generator=gg)
        return DataLoader(TensorDataset(x, y), batch_size=bs)

    return make(n_train, 1), make(n_val, 2)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF vision encoder id (default: {DEFAULT_MODEL})")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--mask-ratio", type=float, default=0.3,
                    help="fraction of attention heads SAL prunes during training")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random-weight model, 1 epoch, CPU — validates the path")
    ap.add_argument("--out", default=None, help="write results to this JSON file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    if args.smoke:
        model = load_tiny_stand_in(device)
        epochs, image_size = 1, 32
        log.info("SMOKE: 2-layer random-weight I-JEPA, CPU. Numbers mean nothing.")
    else:
        model = load_jepa(args.model, device)
        epochs = args.epochs
        image_size = model.config.image_size

    train_loader, val_loader = synthetic_loaders(image_size, bs=args.batch_size)
    probe_loader, _ = synthetic_loaders(image_size, n_train=32, bs=args.batch_size)
    probe_batches = [{"pixel_values": x} for x, _ in probe_loader]
    image_shape = (1, 3, image_size, image_size)

    # 1 -------------------------------------------------------- pre-SAL scan
    log.info("\n=== Pre-SAL structural scan ===")
    fi_before = FIScanner(model, probe_batches, num_samples=32).scan()
    plasticity_before = PlasticityScanner(model, probe_batches, num_samples=32).scan()
    log.info("FI: %.4f", fi_before.fi_score)
    log.info("%s", fi_before.summary)
    log.info("absorption: %s", plasticity_before.absorption_map)

    # Keep an untouched copy: the standard arm, and the CKA reference.
    baseline = copy.deepcopy(model)

    # 2 ------------------------------------------------------- SAL training
    log.info("\n=== SAL training (self-supervised) ===")
    cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio)
    trainer = SALTrainer(
        model, cfg, AdamW(model.parameters(), lr=args.lr), train_loader,
        seed=42, train_step=jepa_train_step,          # <- the v0.5.1 hook
    )
    history = trainer.train(num_epochs=epochs)
    log.info("losses: %s", [f"{x:.5f}" for x in history["losses"]])
    log.info("masker: %s", history["masker_stats"])

    # 3 ------------------------------------------------------- post-SAL scan
    log.info("\n=== Post-SAL structural scan ===")
    fi_after = FIScanner(model, probe_batches, num_samples=32).scan()
    log.info("FI: %.4f (was %.4f)", fi_after.fi_score, fi_before.fi_score)

    # 4 ---------------------------------------------------------- benchmark
    log.info("\n=== Compression benchmark ===")
    results = {}
    results.update(benchmark_compression(model, train_loader, val_loader,
                                         probe_loader, arm="sal",
                                         image_shape=image_shape))
    results.update(benchmark_compression(baseline, train_loader, val_loader,
                                         probe_loader, arm="standard",
                                         image_shape=image_shape))

    log.info("\n%-26s %8s %8s %8s", "arm", "probe", "knn", "cka")
    log.info("%s", "-" * 54)
    for name, m in results.items():
        log.info("%-26s %8.4f %8.4f %8.4f",
                 name, m["linear_probe"], m["knn_accuracy"], m["cka_similarity"])

    # Each arm's CKA is against its *own* clean model, so "sal_random_33" vs
    # "standard_random_33" is a fair question: which model changed less when the
    # same fraction of its heads was removed.
    log.info("\nCKA is measured against each arm's own unpruned model.")

    if args.out:
        payload = {
            "model": args.model if not args.smoke else "smoke-tiny-ijepa",
            "epochs": epochs, "mask_ratio": args.mask_ratio,
            "fi_before": fi_before.fi_score, "fi_after": fi_after.fi_score,
            "losses": history["losses"], "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("\nwrote %s", args.out)


if __name__ == "__main__":
    main()
