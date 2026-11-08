# exp24 — does SAL pre-condition recovery fine-tuning?

The PRIORITY 0 experiment. `experiments/exp24_sal_recovery.py`, nine SLURM jobs
`23451916`-`23451927`, all `COMPLETED 0:0`, 117.6 GPU-hours. Gate `23451579`
(5:27) exercised the full path including real INT4 and the table writer.

ViT-B/16, **ImageNet-1k (resized 256px, evanarlian/imagenet_1k_resized_256)**,
1 281 167 train / 50 000 val, 1000 classes. Three seeds (42, 123, 456), 5 epochs
of recovery fine-tuning with the *same* `make_optimizer` / `make_step` the SAL
arms use — AdamW, 10x LR on the classifier, linear warmup then cosine decay,
bfloat16 autocast, grad clip 1.0. Same memmap cache, same 50k validation.

## The four arms

| arm | start from | head set removed | status |
|---|---|---|---|
| 1 `magnitude` | pretrained checkpoint | lowest output-projection norm, uniform per layer | **reused from exp23**, not retrained |
| 2 `sal_aligned` | 1a SAL checkpoint | SAL's own trained set (`training.sal.pruned_heads`) | run here |
| 3 `sal_magnitude` | 1a SAL checkpoint | magnitude, same as arm 1 | run here |
| 4 `drophead_magnitude` | 1a DropHead checkpoint | magnitude, same as arm 1 | run here |

Three protocol decisions, approved before launch and recorded in the script's
docstring and in each JSON's `caveats` block:

- **No random draws.** Every arm is one fixed architecture fine-tuned for that
  architecture, so there is nothing to average over. The ± figures in
  `PHASE3_IMAGENET1K.md` have no counterpart here. This is the one asymmetry
  with the DropHead-without-recovery row, and it favours the recovery arms.
- **Aligned at 33% is a subset** of SAL's 50% set, the 48 lowest-magnitude of
  its 72 heads. `aligned_is_subset: true` is recorded at P33 and `false` at P50,
  where the quota matches SAL's set exactly. `aligned_heads()` raises rather than
  pad the set with heads SAL was never trained without.
- **DropHead has no aligned set.** `pruned_heads` is `None` for the
  non-accumulating arm by design — it resamples every step and never settles on
  a set. Arm 4 therefore exists only in its magnitude variant.

## Results — mean of three seeds

| arm | P33 | P33+INT4 | P50 | **P50+INT4** |
|---|---|---|---|---|
| 1 magnitude + recovery | **78.85%** | **77.09%** | 77.27% | 75.06% |
| 2 SAL → aligned + recovery | 78.69% | 76.79% | **77.79%** | **75.66%** |
| 3 SAL → magnitude + recovery | 78.62% | 76.76% | 77.15% | 75.12% |
| 4 DropHead → magnitude + recovery | 78.39% | 76.70% | 77.06% | 75.13% |

Differences against arm 1:

| arm | P33 | P33+INT4 | P50 | **P50+INT4** | seeds in that direction |
|---|---|---|---|---|---|
| 2 `sal_aligned` | −0.17 | −0.30 | **+0.51** | **+0.60** | 3/3 at P50 and P50+INT4 |
| 3 `sal_magnitude` | −0.24 | −0.33 | −0.13 | +0.05 | — |
| 4 `drophead_magnitude` | −0.47 | −0.39 | −0.21 | +0.07 | — |

Per seed at P50+INT4, arm 2 minus arm 1: **+0.28, +0.44, +1.08pp**.

## Per-seed detail

`pre` is accuracy after slicing and before any recovery; `dense` is the
checkpoint's own unpruned accuracy.

| arm | seed | dense | pre P33 | P33 | P33+I4 | pre P50 | P50 | P50+I4 |
|---|---|---|---|---|---|---|---|---|
| sal_magnitude | 42 | 71.53 | 60.15 | 78.93 | 77.08 | 27.83 | 77.14 | 75.12 |
| sal_magnitude | 123 | 67.57 | 51.83 | 78.52 | 76.72 | 21.29 | 77.12 | 75.23 |
| sal_magnitude | 456 | 61.61 | 26.46 | 78.39 | 76.47 | 13.35 | 77.18 | 74.99 |
| sal_aligned | 42 | 71.53 | 72.70 | 78.54 | 76.54 | **76.59** | 77.83 | 75.41 |
| sal_aligned | 123 | 67.57 | 71.51 | 78.63 | 76.87 | **75.98** | 77.63 | 75.53 |
| sal_aligned | 456 | 61.61 | 74.06 | 78.90 | 76.97 | **76.97** | 77.90 | 76.05 |
| drophead_magnitude | 42 | 77.60 | 77.31 | 78.28 | 76.67 | 72.48 | 77.11 | 74.98 |
| drophead_magnitude | 123 | 77.32 | 77.18 | 78.37 | 76.61 | 72.45 | 77.04 | 75.12 |
| drophead_magnitude | 456 | 77.71 | 77.20 | 78.51 | 76.81 | 72.49 | 77.03 | 75.28 |

# What this says

## 1. SAL's pre-conditioning contributes nothing. The decomposition is clean.

The design isolates the two things SAL could be contributing — what it taught
the weights, and which heads it picked — because arms 2 and 3 share the same
checkpoints and differ only in the set removed:

| comparison | what it isolates | P50 | P50+INT4 |
|---|---|---|---|
| arm 3 − arm 1 | SAL's **weights**, same head set | −0.13 | +0.05 |
| arm 2 − arm 3 | SAL's **head set**, same weights | +0.64 | +0.54 |
| arm 4 − arm 1 | DropHead's **weights**, same head set | −0.21 | +0.07 |

Holding the head set fixed, replacing the pretrained weights with SAL's buys
**+0.05pp** at the key cell. The entire effect of arm 2 is head selection. Five
epochs of recovery on 1.28M images overwrite whatever SAL wrote into the
weights; the one thing that survives is the list of heads.

DropHead's weights do the same nothing (+0.07), which matters: it means this is
not a statement about SAL specifically but about recovery fine-tuning. **Recovery
is a destination, not a continuation.**

## 2. Recovery is indifferent to where it starts — to a degree that is the result

The three SAL checkpoints are not close to each other. At P50 their post-slice
accuracies are **27.83 / 21.29 / 13.35%**, and their dense accuracies span
9.92pp (71.53 / 67.57 / 61.61). After five epochs they land at **77.14 / 77.12 /
77.18%** — a 0.06pp spread, and within 0.12pp of a baseline that started from
78.44% dense.

Seed 456 is the demonstration: the worst checkpoint of the three, +63.84pp
regained, finishing **best** of the three. Any per-arm "recovery gain" computed
from the post-slice accuracy is therefore a measure of how bad the starting
point was, not of how much the pre-training helped. Reported as a gain it would
make SAL look strongest exactly where it is weakest.

## 3. Pruning SAL's aligned set *improves* the SAL model, before any fine-tuning

The `pre P50` column for arm 2 is the surprise of the experiment: **76.59 /
75.98 / 76.97%**, against dense accuracies of 71.53 / 67.57 / 61.61%. Removing
the 72 heads SAL trained itself to live without is worth **+5.1 / +8.4 /
+15.4pp** over keeping them.

So SAL's depressed clean accuracy is not a generic training cost — it is
specifically the cost of carrying heads it has learned to route around, which
are then actively harmful while present. It also reframes the 1a table: SAL's
P50 column there (34.52 ± 9.49) is the mean over 20 *random* sets; on its own
set it scores ~76.5% with no fine-tuning at all. The 1a protocol never
evaluated SAL at the one architecture it was trained for.

This is the strongest result in SAL's favour that this campaign has produced,
and it is narrow: it is a 76.5% model where the pretrained checkpoint gives
78.44% dense and 77.27% at P50 after recovery. Prune-exactly-here resilience,
at a price, with no gain over simply pruning and repairing.

## 4. Arm 2's advantage is real, consistent, and too small to carry the method

+0.51pp at P50 and +0.60pp at P50+INT4, on 3/3 seeds, every per-seed difference
positive (+0.28 / +0.44 / +1.08). Not noise. But against it:

- It is **absent at P33** — −0.17 and −0.30, so the sign flips with the ratio.
  That matches the scope note already in CLAUDE.md, and it means the gain is
  not a property of the method but of the 50% operating point.
- The 33% row is a *subset* of the aligned set, i.e. an architecture SAL never
  trained for, and it reads 72.76% pre-recovery against 76.51% for the full set.
  Removing **more** heads is better than removing some of the right ones. There
  is no sub-50% aligned arm that is a fair test.
- Buying it costs ~6 GPU-hours of SAL training per seed, a checkpoint 7-17pp
  worse in dense accuracy, and commits the deployment to one fixed head set
  chosen at training time.
- Against `magnitude + recovery` it is +0.60pp; against DropHead's 73.29% at 20
  unseen sets it is +2.4pp for a single set that must be known in advance.

## 5. The answer to the brief

> If SAL + recovery > recovery alone, SAL is a useful pre-conditioner. If not,
> SAL adds nothing to the compression pipeline.

**SAL is not a useful pre-conditioner.** Arms 3 and 4 settle that: at a fixed
head set, neither SAL's nor DropHead's weights beat the pretrained ones after
recovery, by +0.05 and +0.07pp. What SAL retains is a *head-selection* signal
worth +0.60pp at P50+INT4, available only at the ratio it trained for, and only
if you accept a fixed deployment architecture.

Put beside `PHASE3_IMAGENET1K.md`: without recovery SAL loses to its own control
arm by 42.66pp at this cell; with recovery it wins by 0.60pp over a baseline
that needs no SAL training at all. Neither reading supports the published
framing of progressive accumulating head removal as the causal mechanism.

# Caveats

- Every arm is one fixed architecture with no draws, so there is no
  within-arm variance figure. The three-seed spread is all that is available and
  it reflects recovery training, not head-selection sensitivity.
- Arm 1 is exp23's `magnitude_recovery` row, reused rather than retrained as the
  brief instructed. It is the only arm whose numbers predate this experiment.
- exp23's draw seeds are fixed at `9000 + s`, so its `random` / `magnitude` /
  `magnitude_global` rows are byte-identical across the three seeds. **"3 seeds"
  applies only to the recovery arm** — which is the arm used here, so arm 1 does
  replicate independently.
- The INT4 path is the simulated per-channel round-trip, `quantize_weights(model, 4)`,
  not bitsandbytes NF4.
- One setting only: ViT-B/16 on ImageNet-1k. Nothing here measures the 70% ratio,
  GPT-2 Medium, or CIFAR-10 — the two settings where SAL's 33% row is its best.
