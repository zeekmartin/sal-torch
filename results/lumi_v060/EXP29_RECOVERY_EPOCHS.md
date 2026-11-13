# exp29 — how many epochs of recovery are actually needed?

**ViT-B/16 (`google/vit-base-patch16-224`), ImageNet-1k (`evanarlian/imagenet_1k_resized_256`),
3 seeds (42, 123, 456), full 50 000-image validation split, bfloat16 autocast for training and float16 autocast for scoring, batch 64.**
**81.7 GPU-hours**, 7 jobs, all `COMPLETED 0:0`. Data in `results/lumi_v060/exp29_curve_s{42,123,456}/`,
`exp29_ep10_s42/`, `exp29_ratios_s{42,123,456}/`, gate in `exp29_gate/`.
Script: `experiments/exp29_recovery_epochs.py`.

**Plus exp31, the matched dense control, measured through this same harness**: seed 42, 4.97
GPU-hours (job 23659604, `COMPLETED 0:0`, 4:58:12), in `results/lumi_v060/exp31_s42/`, script
`experiments/exp31_dense_control.py`. It reads **80.556% / 78.956% INT4** and confirms exp25's
80.44% to a tenth of a point; Block B's "vs control" column is quoted against exp25 and restated
against it below.

**Comparison convention** (`CONTROL_CONVENTION.md`): every arm here is scored against a dense control
trained for the **same number of epochs**, and the fine-tuning gain (dense control − pretrained) is
reported separately from the pruning cost (arm − matched control). exp32 has since measured the
**2-epoch** dense control on 3 seeds — **81.554% / 80.273% INT4, fine-tuning gain +3.11pp** — which
reframes Block A's headline and makes Block B's costs comparable at last. See "the 2-epoch peak
belongs to the fine-tuning recipe" below.

Magnitude head pruning plus recovery fine-tuning is the pipeline's cheapest compression path (exp24,
exp26). It has only ever been measured at **5 epochs**, a number nobody chose. This run measures the
curve.

Every curve point is **its own recovery run with a completed warmup+cosine schedule**. A mid-run
checkpoint of a longer run is a different quantity — the LR is still high there — so the points are
not sampled from one trajectory. That is why the run costs 81.7 GPU-hours instead of 12.

## Headline

**Two epochs is the operating point, and more recovery makes the model worse.**

1. **The curve is not monotone. It peaks at 2 epochs and declines.** At 50% of heads removed,
   2 epochs beats 5 epochs by **+0.34 / +0.15 / +0.42pp** (3/3 seeds) and beats 10 epochs by
   **1.12pp** on the seed where 10 was run. Ten epochs (76.51%) is **below one epoch** (76.60%).
   **This is a property of the fine-tuning recipe, not of recovery**: the dense control with nothing
   pruned peaks at 2 epochs too, and by a *larger* margin (+1.11pp from 5→2 against the pruned arm's
   +0.30pp). The measurement stands; the explanation changed. See the reframing below.
2. **One epoch already reaches 98.8-99.1% of the 5-epoch gain, 3/3.** The question as posed — "if
   3 epochs reach 95% of the 5-epoch gain, that is the recommendation" — is answered one point
   earlier than expected: the 95% threshold is cleared at **epoch 1** on all three seeds.
3. **The recommendation is 2 epochs: better accuracy at 39% of the cost.** 1.97h against 5.03h per
   seed, and +0.31pp of mean top-1. Against 10 epochs it is 18% of the cost and +1.06pp.
4. **exp24 replicates exactly.** At 5 epochs this run reads 77.265% mean at 50% and 78.845% at 33%,
   against exp24 arm 1's **77.27%** and **78.85%**. Per-seed deltas at 33% are −0.102 / +0.090 /
   −0.002pp. The two campaigns agree to a tenth of a point.
5. **Head pruning costs accuracy at every ratio, including 25%.** Read against the pretrained
   checkpoint the 25% arm looks like a *gain* (79.41% against 78.44%). Read against the matched
   control — the same five epochs with nothing pruned, exp25's 1a baseline at **80.44%** — it is
   **−1.03pp**. The reference was doing all the work; see "the reference, not the caveat" below.

## Block A — the epoch curve at 50% of heads removed

72 of 144 heads removed, lowest-L2, uniform per-layer quota. The head set is **identical on all three
seeds** (magnitude selection is deterministic given the checkpoint), and so is the damage before
recovery: **9.56%** top-1, 7.60% INT4, on 3/3.

Dense reference: **78.44%** / 77.23% INT4.

| epochs | seed 42 | seed 123 | seed 456 | mean | mean INT4 | % of 5-epoch gain |
|---|---|---|---|---|---|---|
| 0 (sliced) | 9.56% | 9.56% | 9.56% | 9.56% | 7.60% | — |
| 1 | 76.60% | 76.54% | 76.59% | **76.58%** | 74.49% | 98.99 / 98.84 / 99.13 |
| **2** | **77.63%** | **77.48%** | **77.60%** | **77.57%** | **75.55%** | 100.51 / 100.22 / 100.63 |
| 3 | 77.54% | 77.62% | 77.57% | 77.58% | 75.50% | 100.38 / 100.44 / 100.58 |
| 5 | 77.29% | 77.33% | 77.18% | 77.27% | 75.03% | 100.00 (reference) |
| 10 | 76.51% | — | — | 76.51% | 74.56% | 98.85 (seed 42 only) |

INT4 per seed, for the record:

| epochs | seed 42 | seed 123 | seed 456 |
|---|---|---|---|
| 1 | 74.61% | 74.41% | 74.46% |
| **2** | **75.56%** | **75.46%** | **75.63%** |
| 3 | 75.55% | 75.41% | 75.54% |
| 5 | 75.09% | 74.98% | 75.01% |
| 10 | 74.56% | — | — |

### The curve peaks at 2 and goes down

The script's own `first_epoch_at_95pct_of_5ep_gain` returns **1** on all three seeds. But the
interesting number is not where 95% is cleared — it is that **the 5-epoch reference is not the
maximum**. Reading the gain over the sliced model (9.56%):

| epochs | mean gain | vs 2 epochs |
|---|---|---|
| 1 | +67.02pp | −0.99pp |
| **2** | **+68.01pp** | — |
| 3 | +68.02pp | +0.01pp |
| 5 | +67.71pp | −0.30pp |
| 10 | +66.95pp | −1.06pp |

2 and 3 epochs are indistinguishable (+0.01pp mean, and the per-seed signs disagree: +0.09, −0.14,
+0.03). Beyond that it declines, and the decline is consistent: **5 epochs is below 2 on 3/3 seeds,
in FP and in INT4**, and 10 epochs is below 5. The INT4 column falls in step, so this is not a
quantization artefact.

The losses say the same thing from the other side — the 5-epoch run ends at 0.115 and the 10-epoch
run lower still, while accuracy falls. **Lower recovery loss past epoch 2 buys nothing and costs
accuracy.** This is the same decoupling of loss from the metric that exp28 hit on the SSL objective,
in a much milder form, and the same remedy applies: probe, do not watch the loss.

### Cost

Pure training time per point, mean of the three seeds: 1.17h (1 ep), **1.97h (2 ep)**, 3.19h (3 ep),
5.03h (5 ep), 10.89h (10 ep, seed 42). So ~1.0-1.1h per epoch, flat.

**2 epochs costs 39% of the 5-epoch recipe and is 0.31pp better.** That is the recommendation for the
paper and for the pipeline default.

## Block B — the ratio sweep

25% and 33% of heads removed, 5 epochs of recovery, 3 seeds. The 50% row is Block A's 5-epoch point.
Head sets are deterministic and **nested** (the 25% set is a subset of the 33% set, which is a subset
of the 50% set), so these are not three independent architectures.

| ratio | heads | before recovery | after 5 ep | mean | INT4 mean | vs pretrained | **vs control** |
|---|---|---|---|---|---|---|---|
| 25% | 36 | 63.54% | 79.33 / 79.47 / 79.41% | **79.41%** | 77.69% | +0.97pp | **−1.03pp** |
| 33% | 48 | 52.20% | 78.75 / 78.94 / 78.85% | **78.85%** | 77.11% | +0.41pp | **−1.59pp** |
| 50% | 72 | 9.56% | 77.29 / 77.33 / 77.18% | **77.27%** | 75.03% | −1.18pp | **−3.17pp** |
| pretrained | 0 | — | — | 78.44% | 77.23% | — | — |
| **control: dense + same 5 ep** | 0 | — | — | **80.44%** | **79.01%** | +2.00pp | — |
| **exp31 control: dense + same 5 ep, this harness** | 0 | — | — | **80.56%** | **78.96%** | +2.12pp | — |

The two control rows' "vs pretrained" entries are the **fine-tuning gain**: +2.00pp and +2.12pp for
five epochs with nothing pruned. The "vs control" column is the **pruning cost**. Keeping them in
separate columns is the whole point — their sum is what the "vs pretrained" column of the pruned rows
reports, and that sum is not a pruning cost.

Pre-recovery accuracies are identical across seeds to five figures (63.54 / 52.20 / 9.56), as they
must be. The 25% row also beats dense at INT4 (77.69 vs 77.23); the 33% row does not (−0.12pp).

### The exp24 cross-check passes

exp24 arm 1 measured the same two cells at 5 epochs. This run reproduces them:

| cell | exp24 arm 1 | exp29 mean | per-seed delta |
|---|---|---|---|
| 33%, 5 ep | 78.85% | 78.845% | −0.102 / +0.090 / −0.002pp |
| 50%, 5 ep | 77.27% | 77.265% | — |
| 33%, 5 ep, INT4 | 77.09% | 77.107% | — |
| 50%, 5 ep, INT4 | 75.06% | 75.03% | — |

Four cells, all within 0.11pp, on an independent run three weeks later with a different job layout.
**The 25% head cell is new** — the premise that it was already in exp26 was wrong: exp26's 25% and
33% are MLP *neuron* ratios, and its only head arm is 50%.

### The reference, not the caveat

**Head pruning costs accuracy at 25%, and the apparent gain was a reference artefact.**

The 25% arm sits 0.97pp above the pretrained checkpoint. That comparison is not the one that answers
"does pruning cost accuracy", because a recovered arm has had five epochs of ImageNet-1k and the
pretrained checkpoint has had none. The matched control is **exp25's 1a `baseline` arm** — the same
1 281 167 images, 5 epochs, batch 64, lr 5e-5 backbone / 1e-3 head, wd 0.05, linear warmup 5% +
cosine, with nothing pruned — at **80.44% / 79.01% INT4** (80.48 / 80.45 / 80.38, sd 0.05;
`EXP25_ALIGNED.md`). exp26 already uses it for exactly this purpose and states the rule: 80.44
answers whether pruning costs accuracy, 78.44 answers whether a pruned-and-recovered model beats the
checkpoint you downloaded, which is a different and much easier question.

Against it, every ratio costs accuracy, monotonically: **−1.03pp at 25%, −1.59pp at 33%, −3.17pp at
50%**. The 50% figure reproduces exp26's head50 arm (−3.17pp) exactly, which is the cross-check that
the restatement is sound and not an arithmetic slip.

So the honest statement is the ordinary one:

> **Magnitude head pruning plus five epochs of recovery costs 1.03pp at 25% of heads, 1.59pp at 33%
> and 3.17pp at 50%, against the same recipe with nothing pruned.** Nothing here shows pruning
> improving a model.

It also sharpens exp26's conclusion rather than softening it. Head-25 removes ~8.2% of parameters for
−1.03pp; MLP-25 removes 16.4% for −0.63pp (exp26). **Heads cost roughly three times more per unit of
size removed than MLP neurons** — the same direction as exp26's 5x at matched size, measured now at a
third head ratio.

### exp31 — the control, measured by this harness

`exp31_dense_control.py` re-runs that control on seed 42 through *this* harness, so exp29's reference
is measured by the same code path rather than quoted across campaigns. Its pass condition was
agreement with 80.44% inside exp25's 0.05pp seed spread; a result far from it would not revise exp25
but would mean the two harnesses differ somewhere.

**It reads 80.556% top-1 / 78.956% INT4** (seed 42, 5 epochs, nothing pruned or masked, epoch losses
0.606 → 0.062, 4.85h of training). That is **+0.116pp above exp25's three-seed mean** (80.44%) and
**+0.076pp above exp25's own seed 42** (80.48%).

So the two harnesses agree, but **not quite inside the 0.05pp window the pass condition named** — the
gap is 0.08pp against the matched seed. One seed cannot say whether that is harness drift or ordinary
run-to-run variation on a quantity whose measured spread is 0.05pp; it is small either way, and
nothing in this document turns on a tenth of a point. The honest reading is *agreement to ~0.1pp,
with the stated window slightly missed*.

Restating Block B's "vs control" column against the locally measured 80.556% instead of the quoted
80.44% moves every entry by the same +0.116pp:

| ratio | mean | vs pretrained (78.44%) | vs exp25 control (80.44%) | vs exp31 control (80.56%) |
|---|---|---|---|---|
| 25% | 79.41% | +0.97pp | −1.03pp | **−1.15pp** |
| 33% | 78.85% | +0.41pp | −1.59pp | **−1.71pp** |
| 50% | 77.27% | −1.18pp | −3.17pp | **−3.29pp** |

The conclusion is unchanged and slightly strengthened: **head pruning costs accuracy at every ratio,
1.0-1.2pp at 25% and 3.2-3.3pp at 50%.** The exp25-referenced column stays the headline, because it
is the three-seed figure and the one exp26 uses; the exp31 column is one seed through this code path.
Note the INT4 side runs the other way — exp31's 78.956% is **0.05pp below** exp25's 79.01% — so the
offset is not a uniform shift of both metrics.

**A second point applies to this block specifically**: it was run at 5 epochs, and Block A shows that
5 epochs is **past the peak**. This block therefore predicted that the ratio costs above were an
*upper* bound, that each arm would read 0.2-0.4pp better at 2 epochs, and that the control would move
too so the net was not predictable from here. **exp32 has now measured all three, and the prediction
was wrong in both parts.** The arms gained more than predicted (+0.85 / +0.78 / +0.38pp at 25 / 33 /
50%), the control gained more still (+1.11pp), and so the costs **rose** rather than falling. The
sweep's internal ordering (25 > 33 > 50) is unaffected, and the exp24 cross-check required 5 epochs to
be comparable, so 5 was right for the cross-check. The redone sweep is `EXP32_RATIO_AT_2EP.md`; the
reframing is the next section.

### Reframed: the 2-epoch peak belongs to the fine-tuning recipe, not to recovery

exp32 re-ran the matched control at 2 epochs on 3 seeds, and the **dense** arm — nothing pruned,
nothing masked — peaks there as well:

| dense control | top-1 | INT4 | seeds | fine-tuning gain |
|---|---|---|---|---|
| 0 ep — pretrained | 78.44% | 77.23% | deterministic | — |
| **2 ep** | **81.554%** | **80.273%** | 3 — 81.676 / 81.524 / 81.462, sd 0.11 | **+3.11pp** |
| 5 ep — exp31, this harness | 80.556% | 78.956% | 1 | +2.12pp |
| 5 ep — exp25 1a | 80.44% | 79.01% | 3, sd 0.05 | +2.00pp |

Shortening the schedule from 5 epochs to 2 is worth **+1.00 to +1.11pp on the dense arm**, against
**+0.30pp** on the P50 pruned arm. So the decline past epoch 2 is not something recovery does to a
damaged model that has run out of things to repair: **it is what this fine-tuning recipe does to this
model on this dataset**, pruned or not, and the pruned arms show a *weaker* version of it than the
dense arm does.

Block A's measurement is unchanged — 2 epochs beats 5 on 3/3 seeds, in FP and in INT4 — and so is the
recommendation. What changes is the claim attached to it. "Recovery saturates at 2 epochs and then
overfits the pruned model" is **not supported**; the dense control overfits harder. The supported
statement is **"this recipe peaks at 2 epochs whether or not anything was pruned"** — which also
means the knee must not be assumed to transfer to another dataset, another LR schedule, or a model
that is not already heavily fine-tuned for its task. The caveat below that recovery hyperparameters
are fixed now carries more weight, not less.

#### Measured against its matched control, pruning costs *more* at 2 epochs

Scoring each arm against the control trained for the same number of epochs reverses what exp32 was
expected to find:

| ratio | at 5 ep | vs 5-ep control (80.44%) | at 2 ep | vs 2-ep control (81.554%) | cost change |
|---|---|---|---|---|---|
| 25% | 79.41% | −1.03pp | 80.258% | **−1.30pp** | 0.27pp worse |
| 33% | 78.85% | −1.59pp | 79.626% | **−1.93pp** | 0.34pp worse |
| 50% | 77.27% | −3.17pp | **77.642%** | **−3.91pp** | 0.74pp worse |
| 25%, INT4 | 77.69% | −1.32pp | 78.744% | **−1.53pp** | 0.21pp worse |
| 33%, INT4 | 77.11% | −1.90pp | 78.003% | **−2.27pp** | 0.37pp worse |
| 50%, INT4 | 75.03% | −3.98pp | **75.427%** | **−4.85pp** | 0.87pp worse |

Three seeds on every cell on both sides (`EXP32_RATIO_AT_2EP.md`). exp32's 50% cell reads 77.642%
against this document's own 2-epoch Block A point of 77.570% — a 0.07pp replication on an independent
run.

Both sides gain from the shorter schedule and the control gains more, so exp32's question — "if the
control gains what the pruned arms gain, the costs are unchanged and only the price drops" — has the
answer **the control gains more, so the price drops and the cost rises**. Two epochs remains the
right default: it is cheaper and absolutely better on every arm, dense and pruned. But the gap
pruning has to make up is wider than the 5-epoch tables suggested, and **any claim of the form
"pruning costs ~1pp at 25%" must now state the epoch count on both sides.**

## Caveats

- **INT4 is simulated.** `quantize_weights(model, 4)`, a per-channel quantize/dequantize round-trip
  in fp32, not a 4-bit kernel: bitsandbytes NF4 is CUDA-only and absent on the ROCm image. The column
  is the cost of 4-bit weight rounding and says **nothing** about speed or memory.
- **No latency claim.** Nothing here was timed at batch 32, and exp26's result stands: at batch 1
  every arm is 1.00x. Any speedup claim must state the batch size.
- **The 10-epoch point is one seed.** It is the only cell in this document without three, so the
  "10 epochs is worse than 1" statement rests on a single measurement — a large one (−1.12pp against
  2 epochs, where the three-seed spread at 2 epochs is 0.15pp), but a single one.
- **One architecture, one dataset, one selection rule.** Magnitude selection only; nothing here tests
  whether the knee moves with a different head-selection rule, or on a model that is not already
  heavily fine-tuned for its task.
- **Recovery hyperparameters are fixed** (AdamW, 10x LR on the classifier, linear warmup + cosine
  decay over the point's own total steps, grad clip 1.0). A lower LR might flatten the decline past
  epoch 2 rather than remove it; not measured. The decline is therefore a property of *this* recipe,
  not a law.
- The three seeds vary only the recovery run's data order and masking draw. Pruning and quantization
  are deterministic given the checkpoint, so every "before recovery" figure is one measurement
  repeated, not three.

## What this changes

- **The pipeline's recovery default should be 2 epochs, not 5.** Better on 3/3 seeds, in FP and INT4,
  at 39% of the cost.
- **`CompressionPipeline` should probe after each recovery epoch and keep the best checkpoint**, not
  the last. The best epoch is 2 or 3 here and the last is worse; a pipeline that returns the final
  weights ships a measurably inferior model. This is the epoch-1 control gate (`ROADMAP.md` v0.7.0,
  "training harness") doing a second job for free.
- The 33%-vs-50% ordering is now measured at three head ratios on three seeds and is monotone, which
  supports exp26's conclusion that heads are the expensive thing to remove.
- **The published ratio costs go up, not down.** Against matched 2-epoch controls the sweep reads
  −1.30 / −1.93 / −3.91pp rather than −1.03 / −1.59 / −3.17pp. Any document quoting the 5-epoch
  figures must say so, because the 2-epoch recipe is now the default and the pair must match.
- **The 2-epoch knee is a recipe property and should not be shipped as a law.** `CompressionPipeline`
  should probe and keep the best epoch rather than hard-coding 2, which is what the bullet above
  already asks for; the reframing is the reason it matters.
