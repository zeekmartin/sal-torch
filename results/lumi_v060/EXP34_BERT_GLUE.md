# exp34 — DropHead vs magnitude + recovery on BERT-base / GLUE

**STUB — no results yet.**

**`bert-base-uncased` on SST-2, MRPC, QNLI and MNLI**, 3 seeds, GLUE validation
splits (matched, for MNLI), max_len 128, batch 32, lr 2e-5, 3 epochs dense,
recovery 2 epochs (exp29). Script `experiments/exp34_bert_glue.py`.

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

## Protocol

Per task, per ratio (33% and 50% of heads):

1. **dense fine-tune**, 3 epochs — the matched control.
2. **DropHead** at that ratio, 3 epochs, then sliced to the **magnitude** set.
3. **magnitude + recovery**: slice the same set, then fine-tune 2 epochs.

Both head arms are scored on the **same head set**, so neither faces a
perturbation the other did not — exp27's protocol.

## Jobs

| seed | job | state |
|---|---|---|
| | | _pending submission_ |

## Caveats, known in advance

- `slice_heads()` is validated on ViT-family and GPT-2 encoders in this
  project; BERT is exercised by the smoke test before any full run.
- No INT4 column: quantization on this model is not what the experiment is
  about, and adding it would double the eval cost for a question exp27 already
  answers on ViT.
- MNLI is scored on `validation_matched` only.
- The loss is checked every step and the run raises on the first non-finite
  value: a diverged run still yields the majority-class rate, and writing that
  to JSON is worse than crashing (exp13).
