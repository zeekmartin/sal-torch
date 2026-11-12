# exp34 — DropHead vs magnitude + recovery on BERT-base / GLUE

**COMPLETE — 3 seeds, 4 tasks, 8 pruned cells, 19.06 GPU-hours, all
`COMPLETED 0:0`.** Data in `exp34_s{42,123,456}/exp34_bert_glue.json`.

**`bert-base-uncased` on SST-2, MRPC, QNLI and MNLI**, 3 seeds, GLUE validation
splits (matched, for MNLI), max_len 128, batch 32, lr 2e-5, 3 epochs dense,
recovery 2 epochs (exp29). Script `experiments/exp34_bert_glue.py`.

## Headline

**Magnitude + 2 epochs of recovery beats DropHead on 8 of 8 cells**, by 0.42pp
(SST-2 P33) to 7.03pp (MRPC P33). That is the reverse of the ViT ordering, and
it is not a resilience failure: **DropHead's resilience replicates on BERT on
8/8 cells**, costing 4.6x to 10.9x less accuracy per head removed than
magnitude does one-shot. What sinks it is that **it pays for that resilience in
dense accuracy** — up to −12.58pp on MRPC — and a recovery pass repairs
magnitude's damage more cheaply than anything repairs DropHead's.

So the probe's `< 10pp → skip head training` prediction for BERT is **not** what
these tasks show either: one-shot drops at P50 run −4.17 / −12.42 / −21.68 /
−17.56pp, which is the 10-30pp "either" band and above it, not the skip band.
And in that band DropHead is the wrong choice here.

## Why

Everything this project has measured on head methods at scale is one
architecture family and one modality. exp17 put **BERT-base at the opposite end
of the routing table from ViT-B/16**: a 5.15pp probe drop against 62.21, which
routes to `skip_heads`. Two things follow that are worth measuring rather than
asserting:

1. If the probe is right, **neither head method should buy much on BERT** — the
   `< 10pp → skip head training` row predicts exactly that. Confirming it would
   be the first direct test of the router's *negative* prediction, which is the
   half nobody has checked.
2. If DropHead wins anyway, the `skip` row is wrong for text and the table
   needs revising.

Four tasks rather than one, because SST-2 alone cannot separate "head methods
do nothing on text" from "head methods do nothing on *this* task".

## The dense control gates the run

Each task is fine-tuned dense first and checked against the published
`bert-base-uncased` reference — SST-2 92-93, MRPC ~88, QNLI ~91, MNLI ~84
matched. If the control lands more than `--control-tolerance` (default 2.5pp)
below it, **the run aborts before the head arms**, because their gains are
measured against that control. This is the exp28 lesson wired in from the
start: a broken control makes every downstream number meaningless, and an
experiment that produces a table anyway is worse than one that crashes.

The reference gates the control's sanity. The **control**, not the reference, is
what the head arms are scored against.

## Comparison convention

Every head arm is scored against **the task's own dense fine-tune**, and the
fine-tuning gain is reported separately from the pruning cost
(`CONTROL_CONVENTION.md`). On GLUE there is no "pretrained checkpoint
accuracy" to drift toward — an unfine-tuned `bert-base-uncased` scores at
chance on these tasks — so the trap takes a different form here: **the two head
arms are matched to the control but not to each other.** DropHead pays 3 masked
epochs; magnitude + recovery pays 3 dense + 2 recovery. A table that puts them
side by side without that column is comparing a 3-epoch method to a 5-epoch
one, so every arm carries its epoch count.

## Protocol

Per task, per ratio (33% and 50% of heads):

1. **dense fine-tune**, 3 epochs — the matched control.
2. **DropHead** at that ratio, 3 epochs, then sliced to the **magnitude** set.
3. **magnitude + recovery**: slice the same set, then fine-tune 2 epochs.

Both head arms are scored on the **same head set**, so neither faces a
perturbation the other did not — exp27's protocol.

## Results

### The dense controls all passed the gate

| task | reference | seed 42 | seed 123 | seed 456 | mean | vs reference |
|---|---|---|---|---|---|---|
| SST-2 | 92.5 | 92.55% | 92.78% | 92.78% | **92.70%** | +0.20pp |
| MRPC | 84.0 | 83.82% | 85.05% | 81.86% | **83.58%** | −0.42pp |
| QNLI | 91.0 | 91.63% | 91.49% | 91.23% | **91.45%** | +0.45pp |
| MNLI-m | 84.0 | 84.54% | 84.47% | 84.43% | **84.48%** | +0.48pp |

MRPC also reads **F1 88.54%** against the 88.0 reference on seed 42. Its
accuracy spread across seeds is 3.19pp — by far the widest here, and expected on
3 668 training examples; the other three span 0.11-0.40pp.

### Every arm against its matched control

Means of 3 seeds. `DropHead` is the masked-trained model sliced to the magnitude
head set; `magnitude + 2 ep` is the same set removed from the dense control and
recovered. Epoch counts differ and the column says so.

| task | ratio | DropHead (3 ep) | vs control | magnitude + 2 ep (3+2 ep) | vs control | winner |
|---|---|---|---|---|---|---|
| SST-2 | 33% | 91.86% | −0.84pp | **92.28%** | **−0.42pp** | magnitude +0.42 |
| SST-2 | 50% | 90.83% | −1.87pp | **91.44%** | **−1.26pp** | magnitude +0.61 |
| MRPC | 33% | 75.82% | −7.76pp | **82.84%** | **−0.74pp** | magnitude +7.03 |
| MRPC | 50% | 71.49% | −12.09pp | **77.70%** | **−5.88pp** | magnitude +6.21 |
| QNLI | 33% | 89.23% | −2.22pp | **90.80%** | **−0.65pp** | magnitude +1.57 |
| QNLI | 50% | 85.12% | −6.33pp | **89.65%** | **−1.80pp** | magnitude +4.53 |
| MNLI | 33% | 82.31% | −2.17pp | **83.92%** | **−0.56pp** | magnitude +1.61 |
| MNLI | 50% | 79.58% | −4.90pp | **83.35%** | **−1.13pp** | magnitude +3.77 |

**8 cells, 8 wins for magnitude + recovery**, and the margin grows with the
ratio on 4/4 tasks except MRPC, where it is enormous at both.

### DropHead's resilience is intact — it is the dense cost that sinks it

The table above compares two models that started from different places, so it
does not say whether DropHead's masking worked. Scoring each method's damage
from **its own** starting point does:

| task | ratio | DropHead: sliced − its own dense | magnitude: one-shot − control |
|---|---|---|---|
| SST-2 | 33% | **−0.27pp** | −1.30pp |
| SST-2 | 50% | **−0.88pp** | −4.17pp |
| MRPC | 33% | **−0.74pp** | −8.09pp |
| MRPC | 50% | **+0.49pp** | −12.42pp |
| QNLI | 33% | **−1.53pp** | −9.87pp |
| QNLI | 50% | **−4.67pp** | −21.68pp |
| MNLI | 33% | **−0.82pp** | −5.03pp |
| MNLI | 50% | **−2.16pp** | −17.56pp |

**DropHead does exactly what it claims, on all four text tasks**: removing the
heads costs it 4.6x to 10.9x less than removing them from an ordinarily
fine-tuned model (MRPC P50 is outside that range because DropHead *gains*
there, so the ratio is undefined rather than large). On MRPC at P50 its accuracy even goes *up* when the heads
come out. This is the ViT result replicating on a different modality.

The price is dense accuracy, which it pays in full and never gets back:

| task | DropHead dense, P33 | P50 |
|---|---|---|
| SST-2 | −0.57pp | −0.99pp |
| MRPC | **−7.03pp** | **−12.58pp** |
| QNLI | −0.70pp | −1.66pp |
| MNLI | −1.35pp | −2.75pp |

MRPC is the extreme case and the explanation is its size: masking a third to a
half of the attention heads at every step, on 3 668 training examples, is a
regularizer the task cannot absorb. **DropHead's entire MRPC deficit is already
present before a single head is removed.**

So the two findings are compatible and the conclusion is about *recovery*, not
about resilience:

> **Head masking buys resilience on text as it does on vision. But where a
> recovery pass is affordable, magnitude + 2 epochs wins on 8/8 cells, because
> recovery repairs one-shot damage more cheaply than masked training avoids
> it.** DropHead's case is the deployment that cannot fine-tune after pruning.

### The epoch asymmetry, stated

DropHead pays 3 masked epochs; magnitude + recovery pays 3 dense + 2 recovery.
Magnitude therefore wins while spending **67% more epochs**, and on the small
tasks the extra two epochs are cheap in wall-clock. This is a real caveat on the
comparison and it does not rescue DropHead: on MRPC the gap is 6-7pp against a
dense cost of 7-12.6pp that no epoch budget touches, and on SST-2 the gap
(0.42-0.61pp) is the one place where the asymmetry is a plausible share of it.

## Jobs

| seed | job | state |
|---|---|---|
| 42 | `23671621` | **COMPLETED** 6:25:25 |
| 123 | `23671622` | **COMPLETED** 6:17:39 |
| 456 | `23671623` | **COMPLETED** 6:20:47 |
| — | smoke | `23670106` COMPLETED, `23670448` TIMEOUT at 30:24 (both controls obtained) |

## Caveats, known in advance

- `slice_heads()` is validated on ViT-family and GPT-2 encoders in this
  project; BERT is exercised by the smoke test before any full run.
- No INT4 column: quantization on this model is not what the experiment is
  about, and adding it would double the eval cost for a question exp27 already
  answers on ViT.
- MNLI is scored on `validation_matched` only.
- **MRPC's seed spread is 3.19pp** on the dense control (83.82 / 85.05 / 81.86),
  against 0.11-0.40pp on the other three. Its 6-7pp DropHead gap is far outside
  that, but no MRPC figure here should be quoted to better than ~1pp.
- **One head-selection rule and one masking quota per cell.** Nothing here tests
  whether a different magnitude criterion or a resampling schedule other than
  per-step changes the ordering.
- **The two head arms are not matched to each other in training cost** (3
  masked epochs vs 3 + 2). Both are matched to the control, which is what the
  pruning cost needs, but a direct DropHead-vs-magnitude read has to carry the
  epoch column.
- The loss is checked every step and the run raises on the first non-finite
  value: a diverged run still yields the majority-class rate, and writing that
  to JSON is worse than crashing (exp13).
