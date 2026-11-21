# Superseded, not deleted: the v0.7.0 gate's first batch

Measured 2026-11-20 on commit 7efca3ab29b0, package hash 9e72ad35717ddd14.
Moved here rather than overwritten, because the second batch writes the same
output directories and measured data is never destroyed. These cells ARE the
evidence for the diagnosis in sal-torch-pro/docs/V070_GATE_DIAGNOSIS.md.

What they measured, and why they are superseded:

* the six ViT arms recovered **without the archived horizontal flip** and with
  the classifier at 10x instead of the archive's 20x. Training loss 0.049
  lower in epoch one and 0.039 in epoch two on 6 of 6 runs, accuracy 0.15 to
  0.26pp lower than exp40 on 3 of 3 seeds. G1-paper and G1-default failed;
  G2 passed under the same two divergences, which is why it is being re-run
  too.
* the three OPT paper arms recovered with `weight_decay` at the shipped 0.05
  where exp36 pins 0.0.
* `gate_v070_vit_gate` is the 4-minute gate cell, 2000 train and 1000 val
  images, 64 optimizer steps. It is NOT a measurement of any band, and
  scoring it as one is what made the scorer refuse a cell that does not say
  what it is.

The OPT **default** trio is not here: its recipe did not change, its band
passed, and its commit is still an ancestor of what would be pushed, so those
cells stay live where the scorer reads them.

Still true of every cell here: the head sets are the archived ones (one-shot
matches exp40 and exp36 exactly), the provenance verifies, and the sparsity
drift is 0.0.
