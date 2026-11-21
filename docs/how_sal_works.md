# How SAL Works

> **Scope, first.** This page explains the mechanism of SAL, the training-time
> method. It is **not** the recommended compression path. Where a post-pruning
> fine-tune is available, prune and recover instead: at a fixed head set SAL's
> weights are worth +0.05pp as a starting point for that fine-tune, and the
> fine-tune itself is worth up to 68pp. SAL and DropHead are for the case where
> no fine-tune after pruning is possible, and there **DropHead** is the
> measured choice. See [Getting Started](getting_started.md) for the path and
> the [README](../README.md) for the figures.

## The idea in one sentence

If you train a model while randomly silencing some of its attention heads, the
model learns to spread important work across many heads — so later you can
remove heads for a smaller, faster model without losing much accuracy.

## The mechanism, step by step

A transformer's attention layers are made of many small units called **heads**.
In a normal model, some functions end up concentrated in a few heads. If you
later compress the model by removing heads, those concentrated functions are
lost and accuracy drops sharply.

SAL changes how the model trains:

1. **Heads are temporarily silenced.** During training, SAL picks heads at
   random and zeroes their contribution. They are "switched off" for the rest of
   training. More heads are silenced as training goes on, up to your target
   fraction (**50% by default** since v0.6.0, and drawn uniformly per layer so
   the trained-for set is a valid `slice_heads()` argument).

2. **The model learns to redistribute.** Because it can no longer rely on the
   silenced heads, the model is forced to route their work through the heads
   that remain. Function gets spread out instead of concentrated.

3. **The result is compression-resilient.** A model trained this way has no
   single points of failure in its attention. When you remove heads to deploy a
   smaller model, the remaining heads already know how to carry the load.

Crucially, **which** heads get silenced doesn't need to be clever — random
selection works as well as anything, and it adds essentially zero overhead. The
benefit comes from training under the stress of progressive silencing, not from
picking "the right" heads.

## What the Fragility Index measures

The **Fragility Index (FI)** is a separate diagnostic. It looks at how a model's
attention heads relate to one another and produces a single **fragility score**
between 0 and 1:

- **FI near 0** — heads back each other up; lots of redundancy; robust.
- **FI near 1** — heads stand alone; little redundancy; fragile.

FI also classifies each layer as **IMMUNE**, **BUFFER**, or **CRITICAL** so you
can see where a model is safe to compress and where to be careful. FI is a
thermometer, not a treatment: it tells you how much you can safely compress, it
doesn't do the compressing.

## What is measured, and what it costs

The figures that stand are the ImageNet-1k ones: ViT-B/16 on the full
50 000-image validation split, three seeds, every arm scored against a dense
control trained for the same number of epochs
(`results/lumi_v060/`, and the public data archive,
DOI [10.5281/zenodo.24391200](https://doi.org/10.5281/zenodo.24391200)).
Read them there rather than from a summary table, because each one needs its
control and its seed count beside it.

Three results decide how SAL should be used.

- **SAL's adaptation is real and entirely set-specific.** On the one head set
  it trained for it reads 76.51% against 20.33% for the matched baseline with
  the same set removed, which is +56.18pp. On sets it never saw: 34.52%, where
  random sets off the baseline read 20.38% — a 0.05pp difference. Both the set
  **and** the ratio are fixed at training time (exp25).
- **DropHead, the resampled variant, beats it on the product's own promise**,
  because its resilience holds on *any* head set of that size including SAL's
  own: +63.47pp at 50% of heads against the baseline on the identical set, and
  +54.70pp on SAL's aligned set, 3 of 3 seeds on every cell (exp27). It is
  callable from the package as `mask_mode="drophead"` since v0.7.0.
- **The seed-to-seed spread is the practical difference.** Across six cells
  SAL's spread spans 0.50 to 17.55pp and DropHead's 0.02 to 0.92pp. One SAL
  row reads 60.15 / 51.83 / 26.46% over its three seeds.

**Two claims this page used to carry are withdrawn.** A five-row table of
"SAL advantage at matched compression" across GPT-2 Small, GPT-2 Medium and
ViT-B, and with it the readings that the advantage **grows with model size**
and that SAL cut run-to-run variance by about 73x. They came from single-seed
ImageNet-100-era studies and are superseded: the ImageNet-1k campaign measures
SAL's own variance as the *larger* of the two, and the project's own note on
the +20pp and +2.45pp figures is that the +20pp came from the **smaller** model,
so size, modality, architecture and task difficulty are all confounded and
nothing measured separates them.

You can see the mechanism directly with
[`examples/compare_with_without_sal.py`](../examples/compare_with_without_sal.py),
which trains the same model with and without SAL and compares accuracy after
compressing both.
