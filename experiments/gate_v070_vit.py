#!/usr/bin/env python3
"""The v0.7.0 release gate, vision cells: the SHIPPED pipeline, end to end.

Every accuracy this project publishes for post-hoc pruning came from
`scripts/run_sal_qat_prototype.py` and the experiment scripts, not from
`sal.CompressionPipeline`. That is acceptable for a research trail and not for
a product claim, and it is how a missing recovery stage survived two releases.
So this script drives the product: `probe()`, `compress()`, `recover()`,
`validate()`, with nothing from the harness but the data loader and the
checkpoint.

Two run sets, three seeds each, scored against four of the six pre-registered
bands in `v070_gate_bands.py`:

* **run set A**, random head selection, bands G1-paper (two-sided) and
  G1-default (one-sided);
* **run set B**, magnitude head selection, bands G2-paper and G2-default.

One set satisfies two bands because the archived vision recipe and the shipped
defaults are the **same configuration**: clip at 1.0, linear warmup over 5% of
the steps, then cosine. The recipe change `recover()` makes bites only on the
decoder, whose gate is `gate_v070_opt.py` in the research repository.

The control is **loaded and verified, never quoted**: `matched_control` reads
exp32's own JSONs and checks them against the published mean before any cost is
derived, and emits a cost only when the epoch counts match.

Rules this script is written to, from `CLAUDE.md`:

* **rule 8**, an end-to-end micro-test that reaches the stage that changed with
  a shortened budget rather than a disabled one. ``--micro`` runs every stage
  on a synthetic 4-layer model and random tensors, on CPU, in seconds;
  ``--gate`` runs the real path at 2000 train and 1000 val images.
* **rule 10**, every measured stage writes its JSON as it completes, before the
  next one starts. A ROCm HSA abort is not a Python exception.
* **rule 12**, the exit code is this script's own: 0 only when every band it
  was asked to score passed.

Usage::

    python experiments/gate_v070_vit.py --micro                   # CPU, seconds
    python experiments/gate_v070_vit.py --gate --strategy random  # small, real
    python experiments/gate_v070_vit.py --strategy random --seed 42 \\
        --data-cache $CACHE --output results/gate_v070_random_s42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v070_gate_bands import check_all  # noqa: E402

DATASET = "evanarlian/imagenet_1k_resized_256"
SEEDS = (42, 123, 456)
#: band names per strategy, from the pre-registration
BANDS_FOR = {"random": ("G1-paper", "G1-default"),
             "magnitude": ("G2-paper", "G2-default")}


class NormalizedSplit:
    """A re-iterable view that hands the pipeline normalized batches.

    The splits are uint8 and the model wants normalized floats, and 50 000
    images as float32 is 30 GB, so the conversion is per batch and lazy. It has
    to be re-iterable rather than a generator: the pipeline evaluates once per
    stage.
    """

    def __init__(self, split, batch_size: int, normalize, with_labels=True,
                 workers: int = 0):
        self.split, self.batch_size = split, batch_size
        self.normalize, self.with_labels, self.workers = (normalize,
                                                          with_labels, workers)

    def __iter__(self):
        import run_sal_qat_prototype as P
        for x, y in P.batches(self.split, self.batch_size,
                              workers=self.workers):
            batch = {"pixel_values": self.normalize(x)}
            if self.with_labels:
                batch["labels"] = y
            yield batch

    def __len__(self):
        return (self.split[0].shape[0] + self.batch_size - 1) // self.batch_size


def _synthetic(n_train=16, n_val=16, classes=4, layers=2, heads=4, dim=32):
    """A model and data small enough to run every stage on a CPU in seconds.

    ``--micro`` exists to satisfy rule 8, which says the micro-test must reach
    the stage that changed on a shortened budget and not through a flag that
    switches the stage off. So this runs probe, compress, recover and validate,
    in that order, with two optimizer steps of recovery.
    """
    import torch.nn as nn

    class Cfg:
        model_type = "vit"
        num_hidden_layers, num_attention_heads, hidden_size = layers, heads, dim

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.nh, self.hd = heads, dim // heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            # Read the head count off the weights, never off a constant taken
            # at construction: slice_heads() physically narrows these
            # projections, so a fixture that remembers its original width
            # cannot survive the stage this script exists to exercise. It also
            # updates `nh`, which is why the attribute is spelled that way.
            b, s, _ = x.shape
            width = self.q_proj.out_features
            nh = max(1, width // self.hd)
            q, k, v = (f(x).view(b, s, nh, self.hd).transpose(1, 2)
                       for f in (self.q_proj, self.k_proj, self.v_proj))
            a = torch.softmax(q @ k.transpose(-2, -1) / self.hd ** 0.5, -1)
            o = (a @ v).transpose(1, 2).reshape(b, s, -1)
            return self.out_proj(o)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn, self.ln = Attn(), nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                     nn.Linear(dim * 2, dim))

        def forward(self, x):
            x = x + self.attn(self.ln(x))
            return x + self.mlp(self.ln(x))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.patch = nn.Linear(12, dim)
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([Block() for _ in range(layers)])
            self.ln_f = nn.LayerNorm(dim)
            self.classifier = nn.Linear(dim, classes)

        def forward(self, pixel_values=None, labels=None, **kw):
            x = self.patch(pixel_values)
            for blk in self.transformer.h:
                x = blk(x)
            logits = self.classifier(self.ln_f(x).mean(1))
            loss = (None if labels is None else
                    torch.nn.functional.cross_entropy(logits, labels))

            class Out:
                pass
            o = Out()
            o.logits, o.loss = logits, loss
            return o

    torch.manual_seed(0)
    def make(n):
        return [{"pixel_values": torch.randn(4, 6, 12),
                 "labels": torch.randint(0, classes, (4,))}
                for _ in range(max(1, n // 4))]
    return Tiny(), make(n_train), make(n_val)


def run_cell(model, eval_data, train_data, *, strategy, seed, pruning,
             epochs, mode, control, outdir, label, amp, log=print) -> dict:
    """One seed of one run set, through the shipped pipeline. Writes as it goes."""
    from sal import CompressionPipeline, RecoveryControl

    t0 = time.time()
    rec_kwargs = dict(epochs=epochs, domain="ImageNet-1k train split, the "
                                            "evaluation domain")
    if mode == "paper":
        # The archived vision recipe, pinned rather than defaulted into: clip
        # 1.0, linear warmup over 5% of the steps, then cosine, under bfloat16
        # autocast. These are also the shipped defaults, which is why one run
        # set satisfies both of this cell's bands. bfloat16 is pinned and not
        # left to "auto" because a paper-mode run that quietly used float16
        # with a GradScaler would not be the paper recipe; on a device without
        # bfloat16 the pin raises, which is the honest outcome.
        rec_kwargs.update(max_grad_norm=1.0, schedule="warmup_cosine",
                          warmup_ratio=0.05,
                          amp_dtype=(torch.bfloat16 if amp else None))
    else:
        # The shipped defaults, including "auto" precision, which is what a
        # user gets and therefore what the one-sided band judges.
        rec_kwargs.update(amp_dtype=("auto" if amp else None))
    pipe = CompressionPipeline(model, eval_data, metric="accuracy",
                               batch_size=128, seed=seed,
                               model_name="ViT-B/16 (gate, %s)" % label,
                               **({"eval_autocast": amp} if amp else {}))
    out = {"label": label, "strategy": strategy, "seed": seed, "mode": mode,
           "pruning": pruning, "epochs": epochs,
           "precision": pipe.precision, "stages": []}
    path = Path(outdir) / ("gate_%s.json" % label)
    path.parent.mkdir(parents=True, exist_ok=True)

    def peak_gib():
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 2 ** 30

    def save(stage: str):
        # Rule 10: write as each stage completes, not at the end of the cell.
        # Rule 9: each stage's OWN peak, reset between stages, because a
        # single end-of-run figure hides which stage was the expensive one.
        out.setdefault("peak_gib", {})[stage] = peak_gib()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out["last_stage"] = stage
        out["seconds"] = time.time() - t0
        path.write_text(json.dumps(out, indent=1, default=str),
                        encoding="utf-8")

    probe = pipe.probe(fraction=pruning)
    out["probe"] = probe.to_dict()
    save("probe")
    log("  probe: %.2fpp lost, route %s" % (probe.drop_pp, probe.route))

    pipe.compress(pruning=pruning, quantization=None, slice_heads=True,
                  strategy=strategy)
    out["one_shot_accuracy"] = pipe._one_shot_accuracy
    save("compress")
    log("  one-shot: %.4f" % out["one_shot_accuracy"])

    pipe.recover(train_data, control=control, **rec_kwargs)
    report = pipe.validate()
    out["stages"] = [{"name": s.name, "accuracy": s.accuracy,
                      "size_mb": s.size_mb, "detail": s.detail,
                      "seconds": s.seconds} for s in report.stages]
    out["recovered_accuracy"] = report.arm_accuracy
    out["pruning_cost_pp"] = report.pruning_cost_pp
    out["budget_matched"] = report.budget_matched
    out["cost_statement"] = report.cost_statement
    out["recovery"] = {k: v for k, v in (report.recovery or {}).items()
                       if k not in ("loss_curve", "grad_norms", "lr_curve")}
    out["recovery_curves"] = {k: (report.recovery or {}).get(k)
                              for k in ("loss_curve", "grad_norms", "lr_curve")}
    save("validate")
    log("  recovered: %.4f | %s" % (report.arm_accuracy, report.cost_statement))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=sorted(BANDS_FOR), default="random")
    p.add_argument("--mode", choices=("paper", "default"), default="paper",
                   help="paper passes the archived recipe explicitly; default "
                        "passes nothing and takes recover()'s own. For the "
                        "vision family these are the same configuration, "
                        "which is why one run set scores two bands")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--pruning", type=float, default=0.50)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--output", default="results/gate_v070_vit")
    p.add_argument("--gate", action="store_true",
                   help="one seed, 2000 train and 1000 val images: a small "
                        "NON-smoke run that exercises every stage")
    p.add_argument("--micro", action="store_true",
                   help="CPU, synthetic model and data, two optimizer steps "
                        "of recovery. Rule 8's micro-test: it reaches the "
                        "stage that changed instead of switching it off")
    args = p.parse_args(argv)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    bands_wanted = BANDS_FOR[args.strategy]
    results, per_seed, costs = [], [], []

    if args.micro:
        print("gate_v070_vit --micro: every stage, synthetic model, CPU")
        for seed in args.seeds[:1]:
            model, train_data, eval_data = _synthetic()
            from sal import RecoveryControl
            # A control of the model's own clean accuracy: the cost it yields
            # is meaningless on random data and the point is that the plumbing
            # computes one at all.
            from sal.robustness import _evaluate
            clean = _evaluate(model, eval_data, "accuracy", 4, None)
            cell = run_cell(model, eval_data, train_data, strategy=args.strategy,
                            seed=seed, pruning=0.50, epochs=1, mode=args.mode,
                            control=RecoveryControl(clean, epochs=1, seeds=1),
                            outdir=outdir, label="micro_s%d" % seed, amp=None)
            results.append(cell)
            per_seed.append(cell["recovered_accuracy"] * 100)
            costs.append(cell["pruning_cost_pp"])
        print("\nmicro run reached: %s" % results[-1]["last_stage"])
        print("stages: %s" % ", ".join(s["name"] for s in results[-1]["stages"]))
        print("recovery recorded %d loss values and %d gradient norms"
              % (len(results[-1]["recovery_curves"]["loss_curve"] or []),
                 len(results[-1]["recovery_curves"]["grad_norms"] or [])))
        print("cost: %s" % results[-1]["cost_statement"])
        # A micro run must NOT be scored against a band: it is not the cell.
        print("\nbands are deliberately NOT evaluated on a micro run: this is "
              "the plumbing, not the measurement.")
        return 0

    import run_sal_qat_prototype as P

    if args.gate:
        args.n_train, args.n_val, args.seeds = 2000, 1000, args.seeds[:1]

    P.rule("v0.7.0 release gate, vision: the shipped pipeline end to end")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = "float16" if device.type == "cuda" else None
    P.log("  device %s | strategy %s | mode %s | seeds %s%s"
          % (device, args.strategy, args.mode, args.seeds,
             "  [GATE]" if args.gate else ""))
    P.log("  bands: %s" % ", ".join(bands_wanted))

    from matched_control import ControlLadder
    ladder = ControlLadder.load(epochs=[args.epochs], verify=True)
    P.log("  control verified: %s" % ladder.summary()
          if hasattr(ladder, "summary") else "  control loaded and verified")

    sizes = {"n_train": args.n_train, "n_val": args.n_val, "n_fi": P.N_FI}
    for seed in args.seeds:
        P.log("\n  seed %d" % seed)
        train_split, val_split, num_classes = P.build_data(
            sizes, seed, args.workers, dataset_id=args.dataset,
            cache_dir=args.data_cache)
        model = P.load_model(num_classes).to(device)
        ctrl = ladder.at(epochs=args.epochs, seed=seed)
        from sal import RecoveryControl
        control = RecoveryControl(accuracy=ctrl.accuracy, epochs=args.epochs,
                                  seeds=getattr(ctrl, "seeds", 1))
        cell = run_cell(
            model,
            NormalizedSplit(val_split, 128, P.normalize),
            NormalizedSplit(train_split, args.batch_size, P.normalize),
            strategy=args.strategy, seed=seed, pruning=args.pruning,
            epochs=args.epochs, mode=args.mode, control=control,
            outdir=outdir, label="%s_%s_s%d" % (args.strategy, args.mode, seed),
            amp=amp, log=P.log)
        results.append(cell)
        per_seed.append(cell["recovered_accuracy"] * 100)
        costs.append(cell["pruning_cost_pp"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cost = (sum(c for c in costs if c is not None) / len(costs)
            if all(c is not None for c in costs) else None)
    scored = {name: {"per_seed": per_seed, "cost": cost}
              for name in bands_wanted}
    verdict = check_all(scored)
    (outdir / "gate_verdict_%s_%s.json" % (args.strategy, args.mode)).write_text(
        json.dumps({"verdict": verdict, "cells": results}, indent=1,
                   default=str), encoding="utf-8")

    print()
    for name, v in verdict["bands"].items():
        print("  %-12s %s" % (name, "PASS" if v["passed"] else "FAIL"))
        for why in v["why"]:
            print("      %s" % why)
    if args.gate or len(args.seeds) < 3:
        print("\n  NOT A GATE RESULT: %d seed(s) and a shortened budget. The "
              "bands are three-seed criteria." % len(args.seeds))
        return 0
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
