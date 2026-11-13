# The comparison convention: matched dense controls

Every pruned, masked or sparsified arm in this campaign is scored against **a
dense control trained for the same number of epochs on the same data with the
same schedule**, never against the pretrained checkpoint.

The pretrained checkpoint is not a control. An arm that was fine-tuned has had
ImageNet-1k epochs the checkpoint never had, so a comparison against it adds
the fine-tuning gain to the pruning cost and reports the sum. That sum can
easily come out positive, at which point a write-up says "pruning improved the
model" about a model that pruning made worse.

So every table reports the two effects **separately**:

- **fine-tuning gain** = dense control − pretrained. What the epochs bought.
- **pruning cost** = arm − dense control at the *same* epoch count. What the
  compression cost.

An arm's accuracy against the pretrained checkpoint may still be quoted — it
answers "does this beat the weights I downloaded?", which is a real question —
but it is never the headline and never labelled a pruning cost.

## The control ladder, ViT-B/16 on ImageNet-1k

`google/vit-base-patch16-224` on `evanarlian/imagenet_1k_resized_256`, full
50 000-image validation split, batch 64, lr 5e-5 backbone / 1e-3 head, wd 0.05,
linear warmup 5% + cosine, bfloat16 autocast for training (scoring is fp16 autocast; see the precision note in `EXP32_RATIO_AT_2EP.md`). **Nothing pruned, nothing
masked.**

| dense control | top-1 | INT4 | seeds | fine-tuning gain |
|---|---|---|---|---|
| 0 ep — pretrained checkpoint | 78.44% | 77.23% | deterministic | — |
| **2 ep** | **81.554%** | **80.273%** | 3 — 81.676 / 81.524 / 81.462, sd 0.11 | **+3.11pp** (INT4 +3.04) |
| 5 ep — exp31, this harness | 80.556% | 78.956% | 1 | +2.12pp |
| 5 ep — exp25 arm 1a | 80.44% | 79.01% | 3 — sd 0.05 | +2.00pp (INT4 +1.78) |

Sources: `exp32_ctrl_s{42,123,456}/exp31_dense_control.json` (2 ep),
`exp31_s42/` (5 ep), `EXP25_ALIGNED.md` (5 ep, 3 seeds).

**Use the row that matches the arm's epoch count.** A 2-epoch recovery arm goes
against 81.554%; a 5-epoch one against 80.44%. Mixing them is the error this
page exists to prevent, and it is worth 1.1pp — larger than most of the effects
being measured.

## Compare within a seed where the arm is single-seed

exp28 measured the same untrained checkpoint at 41.04-48.60% across three eval
jobs, a 7.56pp spread from evaluation shuffling alone. Where an arm has one
seed, score it against **that seed's own control**, not the three-seed mean.

## The fine-tuning gain is not constant in the epoch count

It is **+3.11pp at 2 epochs and +2.00pp at 5** — the dense arm, with nothing
pruned, is 1.11pp *better* after 2 epochs than after 5. This is a property of
the recipe on this dataset, and it is the reason the convention matters: an
effect that size will swamp a pruning comparison that reads the control off the
wrong row. See `EXP29_RECOVERY_EPOCHS.md`, "the 2-epoch peak belongs to the
fine-tuning recipe".

## BERT / GLUE

exp34 follows the same rule with the per-task dense fine-tune as the control
(`EXP34_BERT_GLUE.md`). There the published GLUE reference only **gates** the
control's sanity; the control is what the arms are scored against. Note the
epoch asymmetry recorded in that document: DropHead arms pay 3 masked epochs
and magnitude+recovery arms pay 3 + 2, so those two are not matched to each
other even though both are matched to the control.
