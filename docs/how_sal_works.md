# How SAL Works

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
   fraction (33% by default).

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

## Validated results

SAL has been validated across 5 architectures and 4 tasks. At matched
compression, models trained with SAL keep more accuracy than the same models
compressed *after* normal training (post-hoc pruning):

| Model | Task | SAL advantage vs. post-hoc @ matched compression |
|---|---|---|
| GPT-2 Small (124M) | SST-2 | +2.5 pp |
| GPT-2 Small (124M) | QQP | +6.0 pp |
| GPT-2 Small (124M) | MNLI | +7.8 pp |
| GPT-2 Medium (345M) | SST-2 | +7.6 pp |
| ViT-B (86M) | CIFAR-10 | +20.1 pp |

Two further findings:

- **The advantage grows with model size** (e.g. +2.5 pp → +7.6 pp on SST-2 from
  124M to 345M).
- **Far more stable runs** — at 345M parameters, SAL reduced run-to-run accuracy
  variance by roughly 73×.

You can see the mechanism directly with
[`examples/compare_with_without_sal.py`](../examples/compare_with_without_sal.py),
which trains the same model with and without SAL and compares accuracy after
compressing both.
