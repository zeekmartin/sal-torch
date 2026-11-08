# Phase 3 — the full ImageNet-1k table

ViT-B/16, **ImageNet-1k (resized 256px, evanarlian/imagenet_1k_resized_256)**,
1 281 167 train / 50 000 val, 1000 classes. Three training seeds (42, 123, 456),
5 epochs, `--mask-ratio 0.5 --uniform-per-layer`. Pruned cells are the mean over
**20 random head selections**, the same selections for every arm.

Two experiments, run on the same data cache and the same evaluation protocol:

- **1a** (`scripts/run_sal_qat_prototype.py`) — baseline / SAL / DropHead, each
  trained then pruned. This is the control for *"does SAL-style training help?"*
  Jobs `23427085`, `23429908`, `23429909`.
- **exp23** (`experiments/exp23_baselines.py`) — post-hoc pruning of the
  pretrained checkpoint, with and without recovery fine-tuning. This is the
  control for *"is SAL better than just pruning well?"* Jobs `23429910`,
  `23429911`, `23429912`.

Substitution note: `ILSVRC/imagenet-1k` is gated, so every number here is on the
open mirror `evanarlian/imagenet_1k_resized_256` — canonical 1000 classes in
canonical order, 256px shortest side. The pretrained checkpoint scores **78.44%**
on this validation set against the 80.3% its card reports, the difference being
the resize and centre crop. All comparisons are internal to this setting.

## 1a — trained arms, mean of three seeds

| arm | FP16 | P33 | P50 | INT8 | INT4 | **P50+INT4** |
|---|---|---|---|---|---|---|
| baseline | **80.44%** | 48.80 ± 7.55 | 20.38 ± 6.29 | 80.44% | 79.01% | 16.06 ± 5.39 |
| SAL | 66.90% | 49.94 ± 8.72 | 34.52 ± 9.49 | 66.89% | 64.82% | 30.63 ± 8.80 |
| DropHead | 77.54% | **77.74 ± 0.29** | **75.33 ± 0.61** | 77.53% | 76.09% | **73.29 ± 0.71** |

The ± is the mean within-seed standard deviation over the 20 head selections,
not the seed-to-seed spread.

## exp23 — post-hoc pruning of the pretrained checkpoint

Dense, no training of any kind: **78.44%**. Each row is one fixed architecture,
so there are no draws; the spread quoted is across the three seeds.

| ratio | random (20 draws) | magnitude | magnitude, global rank | magnitude + recovery | + INT4 |
|---|---|---|---|---|---|
| 33% | 43.43% | 52.21% | 1.09% | **78.85%** (spread 0.23) | 77.09% |
| 50% | 15.39% | 9.56% | 0.38% | **77.27%** (spread 0.17) | **75.06%** |

`magnitude, global rank` is the published method: it ranks heads across the whole
model, which gives an uneven per-layer count, is therefore not a valid
`slice_heads()` argument, and is evaluated by masking. Its quota empties the
early layers — 12, 12, 10, 7, 5, 2 heads removed — which is why it collapses. It
is reported so the cost of the per-layer constraint is visible rather than
assumed away.

# What this says

## 1. The published SAL headline replicates at full scale

SAL over its own baseline at P50+INT4: **+14.57pp** (30.63 against 16.06), on
3/3 seeds. The figure published from ImageNet-100 was +14.5 ± 4.3. The
comparison SAL has always made holds on the full benchmark.

At 33% it is **+1.14pp** (49.94 against 48.80), inside the noise. That confirms
the scope already recorded in CLAUDE.md: SAL is robust at ≥50% pruning and weak
and inconsistent at 33% on this architecture and dataset.

## 2. The control arm beats the mechanism it controls, on every cell

DropHead is non-accumulating random head dropout — the arm that exists to test
whether SAL's gain is ordinary regularisation. It wins everywhere:

| | SAL | DropHead | difference |
|---|---|---|---|
| FP16 | 66.90% | 77.54% | **+10.64pp** |
| P33 | 49.94% | 77.74% | **+27.80pp** |
| P50 | 34.52% | 75.33% | **+40.81pp** |
| P50+INT4 | 30.63% | 73.29% | **+42.66pp** |

So the answer to the question the control was built to ask is that **the gain is
ordinary regularisation, and accumulation is not merely unnecessary but
actively harmful.** Progressive, permanent head removal — the mechanism
described as causal in the design documents — costs 42.7pp at the key cell
against resampling the mask every step.

## 3. DropHead's variance is 15× smaller, and that is a separate finding

Within-seed standard deviation over the 20 head selections at P50:

| arm | seed 42 | seed 123 | seed 456 |
|---|---|---|---|
| SAL | 8.02 | 10.37 | 10.08 |
| DropHead | 0.55 | 0.64 | 0.65 |

DropHead is nearly indifferent to *which* heads are removed. SAL is not: its
accuracy swings ±10pp with the draw, which is what "compression-resilient"
is supposed to mean and is not what SAL delivers.

The same instability shows across seeds in SAL's clean accuracy — **71.53 /
67.57 / 61.61%**, a 9.92pp spread — against DropHead's 77.60 / 77.32 / 77.71%,
a 0.39pp spread. A method whose clean accuracy moves ten points with the seed
cannot be shipped with a 0.3–1.5pp cost claim.

## 4. Against the strong baseline, DropHead is close and answers a harder question

| at P50 + INT4 | |
|---|---|
| SAL | 30.63% |
| **DropHead** | **73.29%** |
| magnitude + recovery (exp23) | **75.06%** |

Post-hoc magnitude pruning with five epochs of recovery beats SAL by 44.43pp.
It beats DropHead by only 1.77pp — and the two are not answering the same
question:

- **recovery** is fine-tuned for **one** head set, the one it will deploy. Its
  number is a single architecture, and it has no number at all for a different
  set.
- **DropHead** is scored on **20 head sets it never saw**, at ±0.71. It needs no
  post-pruning training step at all.

Prune-anywhere resilience at 73.3%, against prune-here-then-repair at 75.1%.
For anyone who does not know the deployment sparsity in advance, or who cannot
run a fine-tuning pass after pruning, the 1.8pp is the price of not needing
either. That is a real product position. It is simply not SAL's.

## 5. Magnitude selection reverses between the two ratios

At 33% the lowest-magnitude heads are the right ones to cut: **52.21%** against
43.43% for a random draw, **+8.78pp**. At 50% the ranking inverts: **9.56%**
against 15.39%, **−5.83pp**. Past some fraction, low output-projection norm
stops identifying the dispensable heads. Any paper that validates magnitude
pruning at one ratio and generalises has not checked this.

# Open

- **exp27 is complete** — see `EXP27_DROPHEAD.md`. 1.17 GPU-hours, inference
  only. It answers the question this document left open in §3 and the answer
  changes the positioning:
  - **DropHead's resilience is not an artefact of random-draw scoring.** On a
    magnitude-chosen head set, pinned on the baseline checkpoint and removed
    from all three, DropHead beats that baseline by **+63.47pp at P50** and
    **+26.45pp at P33** on 3/3 seeds. It beats it by **+54.70pp on SAL's own
    set** too.
  - **At P33 the pruning is free**: 77.69% against DropHead's own dense 77.54%,
    **+0.14pp** (INT4: 76.12 vs 76.09). Against the strong 80.44% reference the
    whole package is **−2.75pp at 33% fewer heads, on any set, with no recovery
    pass** — the best no-recovery result in the campaign. At INT4 it beats SAL
    on SAL's own set by 2.17pp.
  - **SAL's adaptation barely transfers**: +0.89pp at P33 and +8.14pp at P50
    over the baseline on the external set, against DropHead's +26.45 and
    +63.47. And SAL's dense accuracy (66.90%) is below DropHead's P50-pruned
    accuracy (72.54%).
  - **The variance gap is two orders of magnitude.** DropHead's sd across three
    seeds spans 0.02–0.92pp over all six rows; SAL's spans 0.50–17.55pp, with
    one row reading 60.15 / 51.83 / 26.46%. §3 of this document estimated that
    gap at 15x from clean accuracy alone; measured across head sets it is
    larger.
- **exp26 / workstream 1c is complete** — see
  `sal-torch-pro/results/EXP26_MLP.md`. 22 jobs, 97.2 GPU-hours, all
  `COMPLETED 0:0`. Post-hoc magnitude MLP pruning plus five epochs of recovery,
  three seeds, full validation split. Two things in it bear on this document:
  - **The speedup ceiling is 1.74x**, at batch 32, with half the parameters
    removed (head 50% + MLP 50%), costing 6.82pp. `CLAUDE.md`'s 2x joint target
    is not reached by magnitude pruning on this model. **At batch 1 every arm is
    1.00x** — 8.09–8.37 ms against 8.16 ms dense — because a ViT-B/16 there is
    kernel-launch bound. Any latency claim must say batch 32.
  - **The reference matters, and this document's does too.** A recovered arm has
    had five epochs of ImageNet-1k; the pretrained checkpoint has had none. 1a's
    `baseline` arm is the matched control — identical images, epochs, batch size
    and LR schedule, nothing pruned — at **80.44% / 79.01% INT4**. Against it
    MLP-25 costs **−0.63pp** at a 16.4% smaller model and 1.15x, not the +1.37pp
    it reads against the pretrained 78.44%.
  - At matched size (−32.7%), **all-MLP beats head+MLP by 1.55pp** (77.75 vs
    76.20), so the joint arm's extra 0.05x of throughput is not free.
- **exp25 is complete** — see `EXP25_ALIGNED.md`, and read it before quoting
  SAL's rows above. On SAL's **own** head set, scored on all three seeds and the
  full validation split, SAL reads **76.51% at P50** and **73.95% at P50+INT4**,
  against 34.52 and 30.63 on the 20 random sets this table uses. The matched
  control settles what that means: the same 72 heads removed from 1a's baseline
  checkpoint give **20.33%**, against 20.38% for random sets — a 0.05pp
  difference. SAL's set is worth nothing to a model that did not train for it,
  and 42pp to the one that did. Two corrections to this document follow from it:
  SAL's clean-accuracy cost is **−3.93pp** (80.44 → 76.51 at a 16.5% smaller
  model), not the −13.54pp its dense row implies; and §2's +42.66pp for DropHead
  is the right comparison for prune-anywhere resilience but not a test of
  whether accumulation does anything.
- **exp24 is complete** — see `EXP24_SAL_RECOVERY.md`. Answer: neither SAL's nor
  DropHead's weights pre-condition recovery (+0.05 and +0.07pp at P50+INT4 at a
  fixed head set). SAL's *head set* is worth +0.60pp at P50+INT4 on 3/3 seeds and
  nothing at P33. It also found, as an intermediate, that SAL's P50 cell here is
  unrepresentative — now measured directly in exp25.
- Not measured: whether DropHead's advantage survives at 70% pruning, where
  v0.5.1 found SAL had no effect on I-JEPA.
- Not measured: DropHead on GPT-2 Medium or ViT/CIFAR-10, the two settings where
  SAL's 33% row is its best. The control arms have only ever been run here.
