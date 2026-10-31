# exp17 — the fragility score does not predict the compression regime. The baseline's own collapse does.

The hypothesis under test was that **FI, read off a pretrained model before any
training, tells apart the two regimes** seen so far: ViT-B/16 (FI 0.066)
collapses under 50% head pruning and needs DropHead's robustness on an arbitrary
set, while I-JEPA ViT-H/14 (FI 0.004) holds and SAL's specialisation is what
pays. High FI would mean fragile; low FI, robust.

**It does not.** Four new architectures put the prediction wrong at both ends,
and the model that breaks it most cleanly does so on the same task, the same
protocol and the same chance level as one that fits.

What does order the table is a quantity that needs no new theory: **how far the
ordinary fine-tune falls when you prune it.** That is measurable directly, and it
predicts the one product decision this series exists to inform.

## The six architectures

Sorted by FI, most fragile first. `base drop` is the standard fine-tune's loss
from clean to a random 50% head set; `sal gain` and `dh gain` are each arm
against that baseline **on the identical set**; `sal−dh` is the two arms against
each other on that set; `control` is the baseline's aligned-minus-random cell,
which separates adaptation from selection.

| model | params | **FI** | base drop | sal gain | dh gain | **sal−dh** | control | n | task |
|---|---|---|---|---|---|---|---|---|---|
| DeiT-small | 21.7M | **0.15625** | −76.30 | +68.20 | +55.47 | **+12.73** | −7.00 | 3 | IN-100 |
| BERT-base | 109.5M | 0.09353 | **−5.15** | +2.94 | +2.87 | +0.08 | +0.98 | 3 | SST-2 |
| ViT-B/16 | 85.9M | 0.06600 | −62.21 | — | — | +2.02 | −5.14 | 10 | IN-100 |
| **DINOv2-large** | 304.6M | **0.00616** | **−94.02** | +90.37 | +59.33 | **+31.04** | −0.10 | 3 | IN-100 |
| I-JEPA ViT-H/14 | 630.8M | 0.00398 | −18.85 | +3.04 | +0.49 | +2.55 | +5.17 | 3 | IN-100 |
| Phi-2 | 2648.6M | **0.00084** | −4.43 | +2.06 | +1.95 | +0.11 | +0.76 | 1 | SST-2 |

ViT-B/16 and I-JEPA are the published points, from `PHASE2F.md` and `PHASE2G.md`;
the other four are new. 20 random draws per cell everywhere, uniform per layer,
`mask_ratio 0.5`. The ViT-B/16 row has no `sal gain`/`dh gain` because its
protocol scored those arms differently; its `sal−dh` is the same same-set
comparison and is directly comparable.

## 1. The prediction fails at both ends

**At the fragile end, BERT-base.** FI 0.094 ranks it as more fragile than
ViT-B/16 (0.066); it loses **5.15pp** where ViT-B/16 loses 62.

**At the robust end, DINOv2-large.** FI 0.0062 ranks it 25x more robust than
DeiT-small; it collapses to **0.96%** from 95.07% clean — chance on a 100-class
task — the worst collapse in the series, 0/3 seeds, CI [−95.87, −92.16].

**DINOv2-large is the decisive one, and BERT-base is weaker than it looks.**
SST-2's majority class is 50.9%, so BERT *cannot* drop more than about 40pp; its
−5.15pp and DINOv2-large's −94.02pp are not on a common scale, and the same
objection covers Phi-2. Both text models sit at a 50.9% floor and both barely
move; both readings are confounded with the floor and neither alone would refute
anything.

DINOv2-large carries no such excuse. **Same dataset, same 100 classes, same 1%
chance level, same protocol and the same 3 seeds as DeiT-small** — and FI orders
the two 25x apart in the direction opposite to what happens.

## 2. Parameter count does not rescue it

The confound flagged before the runs — FI and model size are nearly
rank-identical across these models, so a correlation with FI would also be one
with size — dissolved, but not in FI's favour. Ordered by parameters the drops
run **−76.30, −62.21, −5.15, −94.02, −18.85, −4.43**: no monotonicity at all.
Neither quantity predicts the regime. Size is not a fallback explanation.

## 3. What does order the table: the baseline's own collapse

Set FI aside and sort by `base drop`. The same-set margin of SAL over DropHead
follows it almost exactly:

| base drop | sal − drophead, same set |
|---|---|
| −94.02 (DINOv2-large) | **+31.04** |
| −76.30 (DeiT-small) | **+12.73** |
| −62.21 (ViT-B/16) | +2.02 |
| −18.85 (I-JEPA) | +2.55 |
| −5.15 (BERT-base) | +0.08 |
| −4.43 (Phi-2) | +0.11 |

Six models, two adjacent inversions, spanning two modalities and a 122x range of
parameter count. **The deeper the ordinary model's collapse, the more SAL's
specialisation is worth over DropHead; where nothing collapses, the two arms are
indistinguishable** (+0.08pp and +0.11pp).

This relationship is **not mechanically induced**. `base drop` is built from the
baseline arm alone and `sal−dh` contains no baseline term at all — they share no
measurement, unlike `sal gain`, which does.

The practical reading, which is what this series was for:

| what you measure first | what it implies |
|---|---|
| Prune your fine-tuned model and it **collapses** (tens of points) | SAL, trained on the set you will deploy, is worth a lot — up to +31pp over DropHead here |
| Prune it and it **holds** (a few points) | SAL and DropHead are interchangeable, and neither buys much; compress without either |

The signal costs one evaluation sweep of a model you have already trained — no
second training run, and no FI scan. It is strictly cheaper than the hypothesis
it replaces.

## 4. SAL's cost moves with the same quantity, and sometimes eats the gain

The gains above are not free, and the collapse that makes SAL valuable is also
what makes it expensive:

| model | clean cost, SAL | clean cost, DropHead | SAL-aligned vs **uncompressed baseline** |
|---|---|---|---|
| DINOv2-large | −17.89 | **−63.42** | 91.32% vs 95.07% → **−3.75pp** |
| DeiT-small | **−33.52** | −16.13 | 74.27% vs 89.37% → **−15.10pp** |
| I-JEPA | +0.15 | −1.08 | 72.20% vs 83.44% → −11.24pp |
| Phi-2 | −1.03 | −0.46 | 92.66% vs 94.27% → −1.61pp |
| BERT-base | −0.54 | −0.50 | 90.21% vs 91.44% → −1.23pp |

**The honest headline number is the last column**, not the gain over a collapsed
baseline. "+90pp over the baseline" on DINOv2-large is really "half the heads
removed for 3.75pp", which is the better result and the smaller number.
DeiT-small is the cautionary case: +68.20pp over its baseline, and still 15.10pp
below simply not compressing — at 21.7M the model has nothing spare to reorganise
with.

**Two arms in this table are too damaged to quote.** DropHead on DINOv2-large
reaches only **31.65% clean**, with losses [4.30, 1.88, 0.93] that never
converge; its +59.33pp is a damaged model beating a destroyed one, not a property
of DropHead. The same caution applies more mildly to SAL on DeiT-small.

## 5. The selection component does not generalise either

`PHASE2G.md` found that on I-JEPA part of SAL's advantage was **selection** — the
control scored +5.17pp on SAL's set, so the set was intrinsically easier. Across
six models the control cell has no consistent sign: **−7.00, +0.98, −5.14, −0.10,
+5.17, +0.76**. DINOv2-large in particular is **−0.10pp, CI [−0.45, +0.25], 1/3**
— adaptation with no selection at all, cleanly, on the model where SAL gains
most.

So I-JEPA's selection term is specific to I-JEPA, not a property of
self-supervised vision models or of large models. `PHASE2G.md`'s scoping was
right and should not be widened.

## What is not established

- **Why DINOv2-large collapses.** It is the finding that breaks the hypothesis
  and nothing here explains it. Pretraining objective does not separate it from
  I-JEPA (both self-supervised vision); head count, depth and parameter count do
  not order the table.
- **Anything about the text models beyond "they barely move."** A 50.9% floor
  bounds the measurable drop, so SST-2 cannot test the regime question. A
  many-class language task would.
- **Phi-2 at n=1.** One seed, no interval. Its agreement with BERT-base is
  suggestive and not more.
- **The drop↔margin relationship as a law.** Six points, two inversions, two
  modalities, one pruning ratio, one GPU. It is a usable triage heuristic on this
  evidence, not a calibrated predictor; nothing here gives a threshold.
- **That FI is useless.** This refutes one specific use — predicting the pruning
  regime from a pretrained checkpoint. FI was never a head selector here (every
  set in this experiment is a uniform random draw) and its other documented uses
  are untouched by this result.
- **mask_ratio other than 0.5**, and slicing or latency for any of the four new
  models. Accuracy retention under masking is all that was measured.
