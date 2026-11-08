#!/usr/bin/env python
"""1b-corrected — I-JEPA ViT-H/14 on ImageNet-1k with a target encoder that does not move.

exp28 spent ~181 GPU-hours and produced no method comparison, because two
epochs of continued pretraining collapsed **every** arm including the unmasked
control: linear probe 48.60% (pretrained) -> 16.64% baseline, 3.04% DropHead,
2.32% SAL, with the training loss falling the whole way
(``results/lumi_v060/EXP28_JEPA_1B.md``). This is the corrected run.

Why exp28 collapsed
-------------------

``run_jepa_sal.make_step`` computes its target from **the same live weights** it
is training::

    with torch.no_grad(), mask_module.unmasked():
        target = model(pixel_values=pixels).last_hidden_state   # same weights
    predicted = model(pixel_values=visible).last_hidden_state
    loss = mse(predicted, target.detach())

That is a self-referential objective with no fixed point to hold onto. Nothing
in it rewards keeping the representation informative: if the encoder drifts
toward a low-variance output, *both* sides drift together and the MSE falls.
exp28's SAL arm ended at loss 0.0137 — the lowest of the three — and the worst
probe, 2.32%. Real I-JEPA avoids this with a separate EMA target encoder and a
predictor network; ``run_jepa_sal``'s docstring states it has neither, which was
survivable on the short ImageNet-100 runs it was written for and is not
survivable over 1.28M images.

What this script changes
------------------------

===================  ========================================================
**target encoder**   A *second* encoder that the optimizer never touches.
                     ``--target frozen`` (default) keeps the pretrained
                     weights fixed for the whole run, which anchors the
                     objective to a representation known to score 48.6%;
                     ``--target ema`` follows the online encoder at
                     ``--ema-decay`` (0.9999) instead. ``--target self``
                     reproduces exp28 exactly, for the record.
**learning rate**    ``--lr`` 5e-6, against ``run_jepa_sal.LR`` = **1e-5**.
                     (The brief for this script said exp28 ran at 5e-5; it
                     did not — the file says 1e-5, so this is a factor of 2,
                     not 10. Set ``--lr 1e-6`` for an actual tenth.)
**schedule**         Linear warmup over ``--warmup-frac`` (0.10) of total
                     steps, then cosine decay. exp28 had **no scheduler at
                     all** — a constant LR from step 0.
**control gate**     The baseline arm probes itself after **every epoch**. If
                     it has lost more than ``--gate-pp`` (5.0) points against
                     the pretrained reference, it writes a FAIL sentinel and
                     raises. Every arm re-reads that sentinel every
                     ``--gate-every`` steps and aborts on FAIL, so one bad
                     control stops the whole batch instead of burning 150
                     GPU-hours to confirm it.
===================  ========================================================

The gate is the point. exp28's smoke jobs (90s) and its gate (1:22:53) both
passed while the real runs were destroying the model, because neither can see a
collapse that takes 16 hours of training to produce. Gating on the control arm's
own probe is the only check that could have caught it.

Usage::

    # Gate: control arm only, 2000 images, 1 epoch. Exercises the frozen
    # target, the scheduler and the abort path. ~10 min.
    python exp28b_jepa_corrected.py --gate --data-cache ... --output results/jepa1b_corr_gate

    # Full run: three jobs + an eval job in afterok.
    python exp28b_jepa_corrected.py --arm baseline --epochs 2 --batch-size 64 \
        --data-cache ... --output results/jepa1b_corr --seed 42
    python exp28b_jepa_corrected.py --eval --output results/jepa1b_corr \
        --prune-seeds 20 --int4 --aligned --fixed-set-matrix --slice

Evaluation is delegated to ``exp28_jepa_1b.py --eval`` (which itself calls
``run_jepa_sal.benchmark``), so the table is byte-for-byte the one exp28
produced and the two runs are directly comparable.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET = "evanarlian/imagenet_1k_resized_256"
ARMS = ("baseline", "sal", "drophead")
CKPT_FOR = {"baseline": "baseline.pt", "sal": "sal.pt", "drophead": "drophead.pt"}
GATE_FILE = "control_gate.json"


class GateAbort(RuntimeError):
    """The control arm regressed, so this arm is pointless. Stop now."""


# ------------------------------------------------------------------ the gate

def read_gate(outdir: Path):
    """The shared control-gate verdict, or None if the control has not voted."""
    f = outdir / GATE_FILE
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None          # a partial write; treat as "no verdict yet"


def write_gate(outdir: Path, verdict: dict):
    tmp = outdir / (GATE_FILE + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    os.replace(tmp, outdir / GATE_FILE)      # atomic, so no reader sees half


def cancel_siblings(job_ids, J):
    """scancel the other arms. Best-effort: a failed scancel must not mask the gate."""
    for jid in job_ids:
        try:
            subprocess.run(["scancel", str(jid)], check=True, timeout=60)
            J.log(f"  scancel {jid}: sent")
        except Exception as exc:                                  # noqa: BLE001
            J.log(f"  scancel {jid}: FAILED ({exc}) — cancel it by hand")


def probe_now(model, loaders, J, label: str):
    """Linear probe on frozen features, the same call the eval job scores with."""
    from sal import linear_probe

    # No ``torch.no_grad()`` here: linear_probe fits its own linear head and
    # needs gradients for it. It freezes the encoder itself.
    was_training = model.training
    model.eval()
    try:
        acc = linear_probe(model, loaders["probe_train"], loaders["probe_val"])
    finally:
        if was_training:
            model.train()
    J.log(f"  probe [{label}]: {acc * 100:.2f}%")
    return float(acc)


# --------------------------------------------------------- corrected objective

def make_corrected_step(J, target_model, patch_size, use_amp, scheduler,
                        outdir, gate_every, progress=None, restore_masks=None,
                        ema_decay=None):
    """``run_jepa_sal.make_step``, with the target taken from a separate encoder.

    ``target_model`` is never in the optimizer's parameter list. With
    ``ema_decay`` it is updated in place after each step; without, it never
    changes. Either way the online encoder cannot move the target by training,
    which is the property exp28 lacked.

    The closure also drives the LR schedule and polls the control gate, so all
    three arms get both without SALTrainer needing to know about either.
    """
    state = {"restored": restore_masks is None, "n": 0}
    online_params = None

    def step(model, batch, optimizer, mask_module):
        nonlocal online_params
        if not state["restored"]:
            for layer_idx, saved in restore_masks.items():
                mask_module._masks[int(layer_idx)].copy_(
                    saved.to(mask_module._masks[int(layer_idx)].device))
            state["restored"] = True

        # One bad control arm invalidates every other arm, so check often
        # enough to matter and rarely enough to cost nothing.
        state["n"] += 1
        if gate_every and state["n"] % gate_every == 0:
            verdict = read_gate(outdir)
            if verdict and not verdict.get("passed", True):
                raise GateAbort(
                    f"control gate FAILED ({verdict.get('reason', '?')}); "
                    f"aborting at step {state['n']}")

        pixels = batch[0]
        visible = J.mask_patches(pixels, patch_size, J.PATCH_MASK_RATIO)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if use_amp else contextlib.nullcontext())
        with autocast:
            # The target encoder has no masker installed, so there is nothing
            # to suspend: it sees the clean image through unmasked weights by
            # construction, not by a context manager that could be got wrong.
            with torch.no_grad():
                target = target_model(pixel_values=pixels).last_hidden_state
            predicted = model(pixel_values=visible).last_hidden_state
            loss = torch.nn.functional.mse_loss(predicted, target.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        if ema_decay is not None:
            if online_params is None:
                online_params = dict(model.named_parameters())
            with torch.no_grad():
                for name, tp in target_model.named_parameters():
                    op = online_params.get(name)
                    if op is not None:
                        tp.mul_(ema_decay).add_(op.detach(), alpha=1 - ema_decay)

        if progress is not None:
            progress.step(loss=f"{loss.item():.5f}",
                          lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                          pruned=mask_module.stats["pruned_heads"])
        return loss.item()

    return step


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    """Linear warmup then cosine decay. exp28 ran a constant LR with no warmup."""
    warmup = max(1, int(round(total_steps * warmup_frac)))

    def lr_lambda(s: int):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _NoMask:
    """Exactly zero masking. ``SALConfig`` floors num_heads_to_prune at 1, so a
    near-zero prune_fraction is not off; the control has to be genuinely off."""

    def apply_mask(self): pass
    def remove_mask(self): pass
    def unmasked(self): return contextlib.nullcontext()
    stats = {"pruned_heads": 0}


# ------------------------------------------------------------------- training

def train_one(arm, args, J, device, loaders, patch_size, use_amp, outdir,
              reference_probe):
    """Train one arm under the corrected recipe. Returns its metadata."""
    from sal import SALConfig
    from sal.trainer import SALTrainer

    verdict = read_gate(outdir)
    if verdict and not verdict.get("passed", True):
        raise GateAbort(f"control gate already FAILED "
                        f"({verdict.get('reason', '?')}); not starting {arm}")

    model = J.load_model(device, grad_ckpt=False)

    # The target encoder. A separate module, never handed to the optimizer,
    # in eval mode so dropout/norm statistics cannot drift either.
    if args.target == "self":
        target_model = model              # exp28's behaviour, for the record
        J.log("  target: SELF (exp28 reproduction — expect collapse)")
    else:
        target_model = J.load_model(device, grad_ckpt=False)
        target_model.eval()
        for prm in target_model.parameters():
            prm.requires_grad_(False)
        J.log(f"  target: {args.target.upper()}"
              + (f" (decay {args.ema_decay})" if args.target == "ema" else
                 " (pretrained weights, never updated)"))

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    J.log(f"  lr {args.lr} | warmup {args.warmup_frac:.0%} of {total_steps} "
          f"steps | cosine decay | grad clip 1.0")

    t0 = time.time()
    pruned_heads, masker_stats, masks = None, None, {}
    ema = args.ema_decay if args.target == "ema" else None
    epoch_probes = []

    def step_for(optimizer, progress):
        sched = make_scheduler(optimizer, total_steps, args.warmup_frac)
        return make_corrected_step(J, target_model, patch_size, use_amp, sched,
                                   outdir, args.gate_every, progress,
                                   ema_decay=ema), sched

    if arm == "sal":
        J.rule(f"SAL arm ({args.epochs} epochs, mask_ratio={args.mask_ratio})")
        cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio)
        J.log(f"  prune_fraction {cfg.prune_fraction} | "
              f"num_heads_to_prune {cfg.num_heads_to_prune} | "
              f"uniform_per_layer {cfg.uniform_per_layer}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        prog = J.Progress(total_steps, "sal training")
        step, _ = step_for(opt, prog)
        trainer = SALTrainer(model, cfg, opt, loaders["train"],
                             seed=args.seed, train_step=step)
        # SALTrainer owns the epoch loop and the accumulate schedule, so the
        # per-epoch probe is not available here; SAL is not the gated arm and
        # the in-step sentinel poll covers it.
        hist = trainer.train(num_epochs=args.epochs)
        prog.done()
        losses, masker_stats = hist["losses"], hist["masker_stats"]
        pruned_heads = hist.get("pruned_heads")
        J.log(f"  pruned set: {len(pruned_heads) if pruned_heads else 0} heads")
    else:
        if arm == "baseline":
            J.rule(f"baseline arm ({args.epochs} epochs, no head masking) "
                   f"— THIS IS THE GATED ARM")
            masker = _NoMask()
            resampler = None
        else:
            J.rule(f"DropHead arm ({args.epochs} epochs, "
                   f"mask_ratio={args.mask_ratio}, resampled every step)")
            import run_sal_qat_prototype as P
            cfg = SALConfig.auto(model, prune_fraction=args.mask_ratio,
                                 uniform_per_layer=True)
            masker = P.ResampleMasker(model, cfg, seed=args.seed, mode="drophead")
            masker.install()
            resampler = masker

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        losses, gstep = [], 0
        try:
            prog = J.Progress(total_steps, f"{arm} training")
            step, _ = step_for(opt, prog)
            for epoch in range(args.epochs):
                running = 0.0
                for batch in loaders["train"]:
                    batch = [b.to(device) if torch.is_tensor(b) else b
                             for b in batch]
                    if resampler is not None:
                        resampler.step(gstep, total_steps)
                    running += step(model, batch, opt, masker)
                    gstep += 1
                losses.append(running / max(steps_per_epoch, 1))
                J.log(f"\n    epoch {epoch + 1} mean loss {losses[-1]:.6f}")

                # ---- the control gate ----
                if arm == "baseline":
                    acc = probe_now(model, loaders, J, f"baseline epoch {epoch + 1}")
                    drop = (reference_probe - acc) * 100
                    epoch_probes.append({"epoch": epoch + 1, "linear_probe": acc,
                                         "drop_pp": round(drop, 2)})
                    passed = drop <= args.gate_pp
                    reason = (f"baseline probe {acc * 100:.2f}% after epoch "
                              f"{epoch + 1}, {drop:+.2f}pp against the "
                              f"pretrained reference {reference_probe * 100:.2f}% "
                              f"(threshold {args.gate_pp}pp)")
                    write_gate(outdir, {
                        "passed": bool(passed), "reason": reason,
                        "epoch": epoch + 1, "linear_probe": acc,
                        "reference_probe": reference_probe,
                        "drop_pp": round(drop, 2), "threshold_pp": args.gate_pp,
                        "seed": args.seed, "target": args.target, "lr": args.lr,
                        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    J.log(f"  CONTROL GATE: {'PASS' if passed else 'FAIL'} — {reason}")
                    if not passed:
                        if args.sibling_jobs:
                            cancel_siblings(args.sibling_jobs, J)
                        raise GateAbort(reason)
        finally:
            if resampler is not None:
                with contextlib.suppress(Exception):
                    resampler.remove()

    if not all(torch.isfinite(torch.tensor(float(x))) for x in losses):
        raise RuntimeError(f"{arm}: non-finite loss in {losses}. A diverged run "
                           "still yields a probe accuracy, and writing that to "
                           "JSON is worse than crashing.")

    final_probe = probe_now(model, loaders, J, f"{arm} final")

    info = {
        "arm": arm, "experiment": "exp28b_jepa_corrected",
        "epochs": args.epochs, "losses": losses,
        "masker_stats": masker_stats,
        "pruned_heads": (sorted((int(a), int(b)) for a, b in pruned_heads)
                         if pruned_heads else None),
        "mask_ratio": args.mask_ratio, "seed": args.seed,
        "batch_size": args.batch_size, "n_train": args.n_train,
        "model": args.model, "dataset": args.dataset,
        "recipe": {
            "target": args.target,
            "ema_decay": args.ema_decay if args.target == "ema" else None,
            "lr": args.lr, "warmup_frac": args.warmup_frac,
            "schedule": "linear warmup + cosine decay",
            "grad_clip": 1.0, "patch_mask_ratio": J.PATCH_MASK_RATIO,
            "vs_exp28": ("exp28 used lr=1e-5, no warmup, no scheduler, and a "
                         "target computed from the live weights"),
        },
        "reference_probe": reference_probe,
        "epoch_probes": epoch_probes or None,
        "final_probe": final_probe,
        "final_probe_drop_pp": round((reference_probe - final_probe) * 100, 2),
        "train_seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    J.save_checkpoint(outdir / CKPT_FOR[arm], model, None, args.epochs,
                      losses, masks, args, with_optimizer=False)
    (outdir / f"{arm}.json").write_text(json.dumps(info, indent=2),
                                        encoding="utf-8")
    J.log(f"  {arm}: losses {[round(x, 6) for x in losses]} "
          f"| probe {final_probe * 100:.2f}% "
          f"({info['final_probe_drop_pp']:+.2f}pp) "
          f"| {info['train_seconds'] / 3600:.2f} GPU-h")
    return info


# ----------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=list(ARMS), help="train one arm")
    mode.add_argument("--eval", action="store_true",
                      help="delegate to exp28_jepa_1b.py --eval on this output dir")
    mode.add_argument("--gate", action="store_true",
                      help="control arm only, 2000 images, 1 epoch: exercises "
                           "the frozen target, the scheduler and the abort path")

    p.add_argument("--model", default="facebook/ijepa_vith14_1k")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--output", default="results/jepa1b_corr")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1_281_167)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--n-probe-train", type=int, default=None)
    p.add_argument("--n-probe-val", type=int, default=None)
    p.add_argument("--loader-workers", type=int, default=6)
    p.add_argument("--loader-block", type=int, default=512)

    r = p.add_argument_group("the corrected recipe")
    r.add_argument("--target", choices=("frozen", "ema", "self"), default="frozen",
                   help="frozen: pretrained target encoder, never updated "
                        "(default). ema: follows the online encoder at "
                        "--ema-decay. self: exp28's live-weight target.")
    r.add_argument("--ema-decay", type=float, default=0.9999)
    r.add_argument("--lr", type=float, default=5e-6,
                   help="default 5e-6; run_jepa_sal.LR (exp28) is 1e-5")
    r.add_argument("--warmup-frac", type=float, default=0.10,
                   help="fraction of total steps spent warming up (exp28: none)")

    g = p.add_argument_group("the control gate")
    g.add_argument("--gate-pp", type=float, default=5.0,
                   help="abort if the baseline probe falls more than this many "
                        "points below the pretrained reference")
    g.add_argument("--gate-every", type=int, default=200,
                   help="steps between re-reads of the shared gate verdict; "
                        "0 disables polling")
    g.add_argument("--sibling-jobs", nargs="*", default=None,
                   help="SLURM job ids to scancel if the control gate fails")
    g.add_argument("--reference-probe", type=float, default=None,
                   help="skip measuring the pretrained probe and use this "
                        "fraction (e.g. 0.486) as the gate reference")

    # Forwarded to exp28_jepa_1b.py --eval
    p.add_argument("--prune-ratios", type=float, nargs="+", default=[0.33, 0.50])
    p.add_argument("--prune-seeds", type=int, default=20)
    p.add_argument("--int4", action="store_true")
    p.add_argument("--aligned", action="store_true")
    p.add_argument("--fixed-set-matrix", action="store_true")
    p.add_argument("--slice", action="store_true")
    p.add_argument("--latency-runs", type=int, default=None)
    args = p.parse_args(argv)

    if args.eval:
        import exp28_jepa_1b as E
        fwd = ["--eval", "--output", args.output, "--model", args.model,
               "--dataset", args.dataset, "--seed", str(args.seed),
               "--batch-size", str(args.batch_size),
               "--prune-seeds", str(args.prune_seeds),
               "--prune-ratios", *[str(x) for x in args.prune_ratios]]
        if args.data_cache:
            fwd += ["--data-cache", args.data_cache]
        for flag in ("int4", "aligned", "slice"):
            if getattr(args, flag):
                fwd.append("--" + flag)
        if args.fixed_set_matrix:
            fwd.append("--fixed-set-matrix")
        if args.latency_runs:
            fwd += ["--latency-runs", str(args.latency_runs)]
        return E.main(fwd)

    if args.gate:
        args.arm = "baseline"
        args.epochs, args.n_train, args.n_val = 1, 2000, 1000
        args.n_probe_train, args.n_probe_val = 2000, 1000

    import run_jepa_sal as J
    import run_sal_qat_prototype as P

    J.MODEL_ID = args.model
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    J.rule("exp28b — 1b corrected, " + ("GATE" if args.gate else f"train {args.arm}"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    J.log(f"  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    J.log(f"  model {args.model} | dataset {P.dataset_label(args.dataset)}")
    J.log(f"  seed {args.seed} | epochs {args.epochs} | batch {args.batch_size}")

    if not args.data_cache:
        p.error("--data-cache is required: the 193GB uint8 ImageNet-1k train "
                "split does not fit in a small-g job's memory")

    sizes = {
        "n_train": args.n_train, "n_val": args.n_val,
        "n_probe_train": args.n_probe_train or J.N_PROBE_TRAIN,
        "n_probe_val": args.n_probe_val or J.N_PROBE_VAL,
        "n_fi": J.N_FI_PROBE,
    }
    loaders, fi_batches, num_classes = J.build_data_memmap(
        sizes, args.batch_size, args.seed, args.dataset, args.data_cache,
        args.loader_workers, args.loader_block)
    J.log(f"  classes {num_classes}")

    started = time.time()

    # The gate reference. Measured, not assumed: a hardcoded 48.6% would be
    # wrong the moment the probe size, the dataset or the model changes, and
    # the gate would then either never fire or always fire.
    if args.reference_probe is not None:
        reference = float(args.reference_probe)
        J.log(f"  reference probe: {reference * 100:.2f}% (given, not measured)")
    else:
        J.rule("pretrained reference")
        ref_model = J.load_model(device, grad_ckpt=False)
        reference = probe_now(ref_model, loaders, J, "pretrained")
        del ref_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    J.log(f"  gate: abort if the baseline probe falls below "
          f"{(reference - args.gate_pp / 100) * 100:.2f}% "
          f"({args.gate_pp}pp below the reference)")

    probe_model = J.load_model(torch.device("cpu"), grad_ckpt=False)
    patch_size = probe_model.config.patch_size
    del probe_model

    try:
        info = train_one(args.arm, args, J, device, loaders, patch_size,
                         use_amp, outdir, reference)
    except GateAbort as exc:
        J.log(f"\n  ABORTED: {exc}")
        J.log(f"  wall clock {(time.time() - started) / 3600:.2f} h")
        (outdir / f"{args.arm}_aborted.json").write_text(
            json.dumps({"arm": args.arm, "aborted": str(exc),
                        "reference_probe": reference,
                        "seed": args.seed, "target": args.target,
                        "lr": args.lr}, indent=2), encoding="utf-8")
        return 2

    J.log(f"\n  wall clock {(time.time() - started) / 3600:.2f} h")
    if args.gate:
        ok = info["final_probe_drop_pp"] <= args.gate_pp
        J.rule(f"GATE {'PASSED' if ok else 'FAILED'}")
        J.log(f"  control probe {info['final_probe'] * 100:.2f}% against the "
              f"pretrained {reference * 100:.2f}% "
              f"({info['final_probe_drop_pp']:+.2f}pp, threshold {args.gate_pp}pp)")
        J.log("  NOTE: this gate is ~31 optimizer steps. It proves the frozen "
              "target, the scheduler and the abort path run; it CANNOT prove "
              "the recipe survives two epochs of 1.28M images. The per-epoch "
              "control gate in the full run is what does that.")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
