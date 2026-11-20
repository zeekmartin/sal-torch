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

from v070_gate_bands import (BANDS_BY_CELL, CELL_ROLES,  # noqa: E402
                             check_all, package_provenance)

DATASET = "evanarlian/imagenet_1k_resized_256"
#: What the archived vision recipe augments with, recorded in every cell so a
#: reader does not have to infer it from the accuracy.
AUGMENTATION = ("horizontal flip, p=0.5 per sample, drawn on the device as "
                "run_sal_qat_prototype.make_step draws it")
SEEDS = (42, 123, 456)
#: band names per strategy, from the pre-registration itself rather than a
#: second copy of it: the scorer reads the same map.
BANDS_FOR = {k: v for k, v in BANDS_BY_CELL.items()
             if k in ("random", "magnitude")}


class NormalizedSplit:
    """A re-iterable view that hands the pipeline normalized batches.

    The splits are uint8 and the model wants normalized floats, and 50 000
    images as float32 is 30 GB, so the conversion is per batch and lazy. It has
    to be re-iterable rather than a generator: the pipeline evaluates once per
    stage.
    """

    def __init__(self, split, batch_size: int, normalize, with_labels=True,
                 workers: int = 0, shuffle: bool = False, seed: int = 0,
                 block: int = 0, flip: bool = False, device=None):
        self.split, self.batch_size = split, batch_size
        self.normalize, self.with_labels, self.workers = (normalize,
                                                          with_labels, workers)
        # The archived recipe AUGMENTS, and `recover()` does not: augmentation
        # is a property of the data a caller hands in, not of a recovery
        # stage that has to work on tokens as well as images. The gate's
        # vision cell therefore carries exp40's augmentation here, in the data
        # view, in `make_step`'s own three lines and its own order: move to
        # the device, normalize, then flip half the batch horizontally.
        #
        # Measured cost of not doing this: training loss 0.049 lower in epoch
        # one and 0.039 in epoch two, on 6 of 6 vision runs with a spread of
        # 0.0006 across three seeds, and 0.15 to 0.26pp of validation
        # accuracy, which is what failed G1.
        self.flip, self.device = flip, device
        # The training pass is SHUFFLED, with its own seed, and block-shuffled
        # when the split is memory-mapped. That is what the archived recipe
        # does (exp23's recover calls batches(shuffle=True, seed=...) and a
        # full shuffle of the 193 GB file measured 3.6x slower than the
        # cached read), and an unshuffled recovery is a different run.
        self.shuffle, self.seed, self.block = shuffle, seed, block
        self._loader = None
        self._peeked = False

    def _build(self):
        import run_sal_qat_prototype as P
        return P.batches(self.split, self.batch_size, shuffle=self.shuffle,
                         seed=self.seed, workers=self.workers,
                         block=self.block)

    def _peek_batch(self) -> dict:
        """One batch, built by slicing, for the schedule's step count only.

        Never trained on and never scored: `recover()` takes it to learn the
        batch shape and, for a token budget, the tokens per batch. Its
        CONTENT is irrelevant, which is what makes it safe to serve without
        touching the sampler or the global generator. This view reports its
        own length, so no caller has to walk this pass to count it.
        """
        x, y = self.split[0][:self.batch_size], self.split[1][:self.batch_size]
        batch = {"pixel_values": self._prepare(x, augment=False)}
        if self.with_labels:
            batch["labels"] = (y.to(self.device) if self.device is not None
                               else y)
        return batch

    def _prepare(self, x, augment: bool):
        """`make_step`'s input pipeline: to the device, normalize, then flip.

        The draw is ``torch.rand(n, device=...) < 0.5`` from the device's own
        generator, which is the generator and the shape `make_step` uses. The
        order matters twice over: normalizing on the device is what the
        archive did, and one draw per training batch in sequence is what keeps
        this stream the archive's stream.
        """
        if self.device is not None:
            x = x.to(self.device)
        x = self.normalize(x)
        if not augment:
            return x
        dev = x.device
        picked = torch.rand(x.shape[0], device=dev) < 0.5
        return torch.where(picked.view(-1, 1, 1, 1), x.flip(-1), x)

    def __iter__(self):
        # ONE loader, iterated once per epoch, because `_BlockShuffle` creates
        # its generator once and advances it across epochs: that is how exp23
        # gets a different order in its second epoch, and rebuilding the
        # loader each epoch would repeat the first epoch's order instead.
        #
        # The first call is the schedule's one-batch peek, not an epoch, and
        # exp23 has no such peek, so it must leave no trace.
        #
        # The peek consumes NO randomness of any kind: it is one batch the
        # schedule counts and the loop never trains on, and this pass is the
        # whole reason the gate's order and augmentation can be the archive's.
        #
        # It used to be served from a throwaway loader, which was wrong in a
        # way only the equivalence test found: `iter(DataLoader)` draws a
        # base seed from the GLOBAL generator when the loader has none of its
        # own, so a throwaway pass left the global stream one draw ahead and
        # every flip decision after it differed from exp40's. exp40 builds one
        # loader and iterates it twice, so the gate must create exactly two
        # iterators as well. Slicing the split directly creates none.
        if self.shuffle and not self._peeked:
            self._peeked = True
            yield self._peek_batch()
            return
        augment = self.flip
        if self.shuffle:
            if self._loader is None:
                self._loader = self._build()
            source = self._loader
        else:
            source = self._build()
        for x, y in source:
            batch = {"pixel_values": self._prepare(x, augment)}
            if self.with_labels:
                batch["labels"] = (y.to(self.device) if self.device is not None
                                   else y)
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
             epochs, mode, control, outdir, label, amp, role, log=print) -> dict:
    """One seed of one run set, through the shipped pipeline. Writes as it goes.

    ``role`` is ``"arm"``, ``"gate"`` or ``"micro"`` and goes into the result,
    because the scorer must not have to guess. A gate cell and an arm cell are
    the same shape, same keys, same ``last_stage``, and the only thing in the
    JSON that separated them was the magnitude of the numbers: a 4-minute gate
    at 2000 images folded silently into G1 as a fourth seed and pulled the
    band's mean from 78.60 to 72.23. The caller knows which it ran, so the
    caller says so and nothing downstream infers it from a step count.
    """
    # Checked before anything is built or measured: a cell that would be
    # written with a role nothing downstream accepts is 2 GPU-hours spent on
    # a result the scorer will refuse.
    if role not in CELL_ROLES:
        raise ValueError("role must be one of %s, got %r" % (CELL_ROLES, role))
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
    # Which code this cell tested, derived from the module that was actually
    # imported and compared against the sidecar beside it. A band is not
    # passed on a result that cannot say this.
    import sal
    out = {"label": label, "role": role,
           "strategy": strategy, "seed": seed, "mode": mode,
           "pruning": pruning, "epochs": epochs,
           "augmentation": (AUGMENTATION if getattr(train_data, "flip", False)
                            else "none"),
           "provenance": package_provenance(sal),
           "sal_version": sal.__version__,
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
    # A cached full split is 180 GB and memory-mapped, so each row is a 147 KB
    # read from Lustre. Workers let those reads overlap the compute and block
    # shuffling keeps them sequential: a full shuffle of the file measured 3.6x
    # slower than the same data cached. exp29 sets both the same way.
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)
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
                            outdir=outdir, label="micro_s%d" % seed,
                            role="micro", amp=None)
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

    if args.data_cache:
        P.LOADER_WORKERS, P.LOADER_BLOCK = args.loader_workers, args.loader_block
        P.log("  cache %s, loader workers %d, block %d"
              % (args.data_cache, args.loader_workers, args.loader_block))
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
            NormalizedSplit(val_split, 128, P.normalize,
                            workers=P.LOADER_WORKERS),
            NormalizedSplit(train_split, args.batch_size, P.normalize,
                            workers=P.LOADER_WORKERS, shuffle=True, seed=seed,
                            block=P.LOADER_BLOCK, flip=True, device=device),
            strategy=args.strategy, seed=seed, pruning=args.pruning,
            epochs=args.epochs, mode=args.mode, control=control,
            outdir=outdir, label="%s_%s_s%d" % (args.strategy, args.mode, seed),
            role=("gate" if args.gate else "arm"), amp=amp, log=P.log)
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
    # Numbers only here. The ancestry half of a band ("was this produced by
    # code that is an ancestor of what we would push") needs the git history,
    # and the cluster checkout is not a repository, so the authoritative
    # scoring runs off-cluster against the recorded provenance in these cells.
    verdict = check_all(scored)
    prov = results[-1].get("provenance") if results else None
    # Parenthesised, because `/` binds tighter than `%`: the first version of
    # this line formatted the Path rather than the string and died with a
    # TypeError AFTER every stage had run and written, which failed the gate
    # job and left its arms on a dependency that could never be satisfied.
    (outdir / ("gate_verdict_%s_%s.json" % (args.strategy, args.mode))).write_text(
        json.dumps({"verdict": verdict, "cells": results,
                    "provenance": prov,
                    "scoring_note": "numbers only; ancestry is verified "
                                    "off-cluster by score_v070_gate.py"},
                   indent=1, default=str), encoding="utf-8")
    if prov:
        rec = prov.get("recorded") or {}
        print("  code under test: commit %s, package hash %s, sidecar "
              "matches: %s" % ((rec.get("commit") or "?")[:12],
                               (prov["computed"]["aggregate_sha256"])[:12],
                               prov["matches"]))

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
