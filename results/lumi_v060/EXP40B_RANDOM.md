# exp40 / exp40b — random head selection against magnitude, at a matched control

**ViT-B/16, ImageNet-1k** (`evanarlian/imagenet_1k_resized_256`, full 50 000-image
validation split), **3 seeds** (42, 123, 456), 2 epochs of recovery after pruning,
uniform per-layer quota on both sides. 6 jobs, **17.5 GPU-hours**
(exp40 2:00:10 / 2:02:04 / 1:58:18 at P50; exp40b 3:39:24 / 3:40:43 / 4:11:07 at
P25 and P33). Training and recovery under bfloat16 autocast; scoring under
float16 autocast; INT4 is `quantize_weights(model, 4)`, fp32 rounding then
fp16-autocast scoring — a simulated round-trip, not a 4-bit kernel.

Sources: `exp40_random_s{42,123,456}/exp29_recovery.json` (P50),
`exp40b_random_s{42,123,456}/exp29_recovery.json` (P25, P33). The magnitude
column is `EXP32_RATIO_AT_2EP.md`, unchanged.

**`CONTROL_CONVENTION.md` governs every cost here.** The pruning cost is
arm − dense control **at the same epoch count**. The 2-epoch dense control is
**81.554% / 80.273% INT4** (3 seeds, nothing pruned). The pretrained checkpoint
is not a control and a delta against it is not a pruning cost.

---

## The result: at 50% of heads, random beats magnitude by 1.17pp

| ratio | heads | seed 42 | seed 123 | seed 456 | mean | INT4 mean |
|---|---|---|---|---|---|---|
| 25% | 36 | 80.210% | 80.406% | 80.462% | **80.359%** | 78.788% |
| 33% | 48 | 79.572% | 79.864% | 80.120% | **79.852%** | 78.278% |
| 50% | 72 | 78.838% | 78.668% | 78.942% | **78.816%** | 77.022% |

Against the 2-epoch control, and against magnitude selection measured the same
way:

| ratio | random cost | magnitude cost | random − magnitude | separated on 3/3 seeds? |
|---|---|---|---|---|
| 25% | **−1.195pp** | −1.30pp | +0.101pp | no — the ranges overlap |
| 33% | **−1.702pp** | −1.93pp | +0.226pp | no — the ranges overlap |
| 50% | **−2.738pp** | −3.91pp | **+1.174pp** | **yes** |
| 25%, INT4 | −1.485pp | −1.53pp | +0.044pp | no |
| 33%, INT4 | −1.995pp | −2.27pp | +0.275pp | no |
| 50%, INT4 | **−3.251pp** | −4.85pp | **+1.595pp** | **yes** |

The P50 row is the one that carries a claim. Random's **worst** seed (78.668%)
is 0.858pp above magnitude's **best** (77.81%), so the two sets of three do not
overlap at all — six runs, a clean separation, and 1.17pp is four times the
0.274pp spread of random's own seeds. At INT4 the gap widens to 1.595pp, which
also means **random selection is the more INT4-robust of the two at this ratio**,
not merely the more accurate one.

**P25 and P33 are ties and must be reported as ties.** At P33 random's worst seed
(79.572%) sits below magnitude's best (79.70%); at P25, 80.210% below 80.40%.
A 0.10–0.23pp difference inside a 0.14–0.55pp seed range is not a result, and
nothing here says random is better at those ratios. What the three rows together
say is that **the advantage appears as the ratio grows**, and only the 50% row is
measured outside noise.

### The comparison is not perfectly matched, in random's disfavour

For magnitude, selection is deterministic given the checkpoint: the seed moves
only the data order, so its three seeds sample **three data orders**. For random,
the seed also draws the head set, so its three seeds sample **three head sets
and three data orders**. Random's spread therefore contains a source of
variance magnitude's does not — and random still wins at P50 on 3/3. The
asymmetry works against the result being reported, so it is not a reason to
discount it; it is a reason the two spreads are not directly comparable to each
other.

---

## One-shot, before any recovery: the same ordering, far noisier

| ratio | seed 42 | seed 123 | seed 456 | spread |
|---|---|---|---|---|
| 25% | 57.844% | 52.090% | 66.786% | 14.70pp |
| 33% | 30.822% | 45.152% | 48.758% | 17.94pp |
| 50% | 15.426% | 28.086% | 13.424% | 14.66pp |

Magnitude at P50 one-shot reads **9.56%** (`EXP33_SPARSE_VS_STRUCTURED.md`), below
every random seed. So random is ahead of magnitude both before and after
recovery at that ratio.

The spread is the thing to notice. **Which heads are removed moves one-shot
accuracy by 15–18pp and the recovered accuracy by 0.25–0.55pp.** Two epochs of
recovery absorb almost all of the selection difference: the head set matters
enormously to the pruned model and very little to the model that is then
fine-tuned. That is the same shape as exp24's finding — the recovery fine-tune
is worth up to 68pp while the training-time method is worth +0.05pp — now
measured for *selection* rather than for training method. **If no post-pruning
fine-tune is available, selection is a 15pp decision; if one is, it is a
0.3pp decision.**

---

## What this changes

**The public recommendation of magnitude selection for heads is withdrawn.**
Random is the recommendation at P50, where the difference is measured, and a
tie at P25 and P33, where it is not. The shipped code default is already
random (cardinal rule 1), so this aligns the documentation with the code
rather than changing the code.

Note what this is and is not evidence for. `PHASE2E` measured **FI-weighted**
head selection losing to uniform random by 4.8–5.0pp during *training-time*
masking. This page measures **magnitude** selection losing to uniform random in
*post-hoc* pruning followed by recovery. They are different procedures on
different objects and neither implies the other; they happen to point the same
way, which is why cardinal rule 1 reads as it does.

### Scope, stated with the result

- **One architecture, one dataset.** ViT-B/16 on ImageNet-1k. exp34 already
  showed the method ordering inverts on BERT/GLUE, so nothing here should be
  assumed to transfer to text.
- **One recovery budget.** Two epochs, which `EXP29_RECOVERY_EPOCHS.md` shows is
  the peak of this recipe on this dataset. The selection gap at 5 epochs, or at
  zero, is not measured — except for the one-shot column above, where it is
  far larger.
- **Heads only.** The FFN equivalent is `EXP44_RANDOM_FFN.md`, and it does not
  come out the same way: no criterion dominates there.
- **Uniform per-layer quota on both sides.** A non-uniform draw is a different
  experiment and `slice_heads()` would not accept its output.
- **The INT4 column says nothing about speed or memory.** It is the accuracy
  cost of 4-bit weight rounding, scored in fp16 autocast.
