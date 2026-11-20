# exp37 — does a fragility scan detect compression, and what does it detect?

**TIER A, THREE PROBES, VALID PROBE GEOMETRY — 13 arms, 1 seed, 3 independent
probe draws, 0.08 GPU-hours (job 23713582, `COMPLETED`, 4:31).** Data in
`results/lumi_v060/exp37_g512/exp37_fi_compressed.json`.

**This is the first valid exp37 measurement.** The two earlier runs (jobs
23701223 single-probe and 23702553 three-probe) both built the FI probe as a
tuple of **uint8 tensors in [0, 255]** and are withdrawn in full; their
directories `results/lumi_v060/exp37/` and `exp37_3probes/` are kept as the
documented error and **nothing in them may be quoted in either direction**.
The mechanism is recorded under "what this replaces" below.

**ViT-B/16 / ImageNet-1k** (`evanarlian/imagenet_1k_resized_256`). Three
disjoint FI probes of **512 images each (32 batches of 16), scanned at
`num_samples=500`**, built by the shared harness
(`scripts/run_sal_qat_prototype.py:build_probe`, which validates that the
probe is float and normalised and refuses it otherwise). Pruner calibration
uses 128 **different** images, so no pruner is scanned on what it was fitted
to. Edge density 0.10 (the shipped default), seed 42. Script
`experiments/exp37_fi_compressed.py`. FI is the fraction of head-graph edges
with zero triangle support (`sal.fi.compute_fi`).

Tier B (recovered checkpoints) is still pending exp35's saved weights.

---

## Headline

**One compression of the thirteen produces an FI signal that survives a
re-drawn probe: Wanda unstructured at attention-50%, +15 edges on every
draw, fifteen times its own spread.** Every other node-preserving arm —
every SparseGPT arm, both 2:4 arms, and the INT4 round-trip — fails to
reproduce.

Three things make this page different from a list of deltas:

1. **The earlier run's single survivor was the wrong arm.** It reported
   SparseGPT attention-50% as the one detected compression. On a valid probe
   SparseGPT attention-50% reads +7 / +18 / −1 edges — it changes sign — and
   **Wanda attention-50%, which that run reported as sign-stable and
   *negative*, is the arm that holds.** The two swapped places. The cause is
   the normalisation fix, not the probe geometry.
2. **A delta between two graphs of different size is not a quantity.**
   Structured pruning removes nodes, so `head50`'s FI lives on a 256-edge
   graph where one edge is worth 0.003906, against 0.000971 on the dense
   1030-edge graph. The project has been comparing those numbers directly.
   This page does not.
3. **A three-draw spread underestimates the floor.** Three separate
   three-draw estimates of the dense FI spread on this one checkpoint at this
   one geometry read **9, 15 and 16 edges**; pooled, the six draws this page
   may cite span **15 edges**. The floor is not yet pinned, and the number
   you get depends on how many times you drew.

## 1. The floor, and why it is not the test

The same pretrained checkpoint, unmodified, under this run's three probes:

| probe | dense FI | edges of 1030 |
|---|---|---|
| 0 | 0.067961 | 70 |
| 1 | 0.066019 | 68 |
| 2 | 0.074757 | 77 |

**Spread 0.008738 — 9 edges, 12.6% of the mean.** Nothing about the model
changed between those readings.

Pooled with exp42's three draws of the same quantity on the same checkpoint at
the same geometry (`EXP42_FI_THRESHOLDS.md`):

| run | draws | spread (edges) |
|---|---|---|
| exp37 | 0.067961 / 0.066019 / 0.074757 | 0.008738 (9) |
| exp42 | 0.078641 / 0.064078 / 0.076699 | 0.014563 (15) |
| **pooled** | **6 draws, mean 0.071359** | **0.014563 (15)** |

The two experiments run different code paths into the same FI (exp45 showed
they return identical values on identical input), so these are six draws of
one quantity. **The three-draw estimate was 40% low.** Treat 15 edges as the
current floor on absolute FI and expect it to grow again with more draws.

### But the floor is not what the arm test uses

Two questions are easy to merge and need different noise models.

**(A) Did the compression change FI on a given probe?** The scan is
deterministic on a fixed probe, and every arm is compared against dense *on
the same probe*. The null is therefore exactly zero and any nonzero delta is
real. This is true and nearly useless.

**(B) Is the delta reproducible when the probe is re-drawn?** This is the
operational question, because a diagnosis is taken on one probe and acted on
later. Pairing has already cancelled the probe effect common to both scans,
so the noise model is **the delta's own cross-probe spread**, not the spread
of dense FI. A delta is actionable when the smallest absolute delta across
the three probes exceeds that spread and the sign does not flip.

Criterion (B) is self-referential, which is its virtue: it does not depend on
whether the floor is 9 edges or 19, so the verdicts below do not move when the
floor estimate does.

## 2. The arms that keep the graph intact

144 nodes and 1030 edges in all seven rows, at 0.10 density by construction.
Deltas are paired within a probe and quoted in edges.

| arm | ΔFI p0 / p1 / p2 | min abs | own spread | sign | min/spread | verdict |
|---|---|---|---|---|---|---|
| **Wanda unstructured, attn 50%** | **+15 / +16 / +15** | **15** | **1** | stable | **15.0x** | **ACTIONABLE** |
| SparseGPT 2:4, attn | +8 / +22 / +14 | 8 | 14 | stable | 0.6x | not reproducible |
| SparseGPT unstructured, whole 16.7% | +10 / +9 / +3 | 3 | 7 | stable | 0.4x | not reproducible |
| Wanda 2:4, attn | +5 / +2 / −4 | 2 | 9 | **flips** | 0.2x | not reproducible |
| Wanda unstructured, whole 16.7% | +2 / +15 / +18 | 2 | 16 | stable | 0.1x | not reproducible |
| SparseGPT unstructured, attn 50% | +7 / +18 / −1 | 1 | 19 | **flips** | 0.1x | not reproducible |
| INT4 round-trip, all weights | 0 / −8 / −15 | 0 | 15 | stable | 0.0x | not reproducible |

**Wanda attention-50% is the only compression this scan detects**, and it
detects it cleanly: +0.014563 / +0.015534 / +0.014563, a one-edge spread on a
fifteen-edge effect. That ratio, not the magnitude, is what makes it usable.

**Two arms change sign with the probe.** For SparseGPT 2:4 and SparseGPT
unstructured attention-50%, whether the compressed model reads more or less
fragile than the dense one is decided by which 512 images were drawn. No claim
of direction can be made for either.

**INT4 moves FI down.** 0, −8 and −15 edges — sign-stable in the sense that
no draw reads positive, but with a 15-edge spread on a 15-edge effect, so the
magnitude is not established. The direction now **agrees with the
certification harness**, which reads the same operation on the same checkpoint
at −8 edges. That agreement is the point: the disagreement between the two
harnesses is what exposed the unnormalised probe, and with the probe fixed it
is gone.

## 3. The arms that remove nodes

| arm | nodes | edges | 1 edge = | FI p0 / p1 / p2 | own spread (edges) |
|---|---|---|---|---|---|
| `head25_magnitude` | 108 | 578 | 0.001730 | 0.084775 / 0.095156 / 0.064014 | 0.031142 (18) |
| `head33_magnitude` | 96 | 456 | 0.002193 | 0.114035 / 0.087719 / 0.103070 | 0.026316 (12) |
| `head50_magnitude` | 72 | 256 | 0.003906 | 0.316406 / 0.300781 / 0.300781 | 0.015625 (4) |
| `head50_random_s0` | 72 | 256 | 0.003906 | 0.218750 / 0.199219 / 0.195312 | 0.023438 (6) |
| `head50_random_s1` | 72 | 256 | 0.003906 | 0.234375 / 0.265625 / 0.296875 | 0.062500 (16) |

**None of these may be differenced against dense.** The script prints such a
delta and this page does not use it: FI is a *fraction* of a graph's edges,
and the graphs differ in size by a factor of four. The same absolute FI
difference means four times as many edges at 72 nodes as at 144, and the
certificate's 2-edge tolerance translates as:

| graph | nodes | edges | 1 edge in FI | the 2-edge tolerance becomes |
|---|---|---|---|---|
| dense | 144 | 1030 | 0.000971 | 2.00 edges |
| `head25` | 108 | 578 | 0.001730 | 1.12 edges |
| `head33` | 96 | 456 | 0.002193 | 0.89 edges |
| `head50` | 72 | 256 | 0.003906 | **0.50 edges** |

On a 50%-pruned model the shipped tolerance is **half an edge**, which no
integer edge count can satisfy. A tolerance expressed in FI units is a
different tolerance on every architecture it is applied to.

### The one homogeneous comparison: magnitude against random at the same ratio

Identical node count, identical edge budget, so one edge is worth the same on
both sides. Paired by probe, in edges of 256:

| comparison | Δ p0 / p1 / p2 | min abs | own spread | min/spread | verdict |
|---|---|---|---|---|---|
| magnitude vs `random_s0` | +25 / +26 / +27 | 25 | 2 | 12.5x | ACTIONABLE |
| magnitude vs `random_s1` | +21 / +9 / +1 | 1 | 20 | 0.1x | not reproducible |
| `random_s0` vs `random_s1` | −4 / −17 / −26 | 4 | 22 | 0.2x | not reproducible |

**Magnitude-selected head removal leaves a measurably more fragile graph than
one random draw and is indistinguishable from the other.** With two random
seeds the claim is not established, and the reason is visible in the third
row: the two random draws differ from *each other* by up to 26 edges, so
head-set variance at P50 is larger than most of the effects on this page.

It is worth recording that the direction, where it is measurable, agrees with
`EXP40B_RANDOM.md`: random head selection both recovers better (+1.17pp at
P50) and leaves the lower FI. Two independent measurements pointing the same
way is suggestive and, at two random draws, no more than that.

### `head25` and `head33` are untestable here

Their deltas against dense are cross-graph and so not quantities; and unlike
`head50` they have **no random-selection control at their node count**, so
there is nothing homogeneous to compare them to. Their absolute readings are
in the table above and that is all this run supports. A control arm at 108 and
96 nodes would cost about 0.03 GPU-hours and would make them testable.

## What this replaces

Two conclusions are withdrawn, and the second was itself a withdrawal.

> **INT4 moves the graph.** A quantization round-trip changes no parameter
> count, no mask and no architecture, and it moves FI by +12.5% — more than
> four of the six sparsity arms. A fragility scan sees weight rounding.

Withdrawn: that was the single-probe run, measuring every arm against a dense
baseline that was itself the outlier. INT4's valid reading is 0 to −15 edges,
not reproducible, and **negative** where that claim was positive.

> **One arm of seven survives.** SparseGPT's attention-50% moves FI by at
> least 18 edges on every draw… It is the only compression in this table the
> scan detects.

Withdrawn: that was the three-probe run, which was correct about the method
and wrong about the data, because it also scanned uint8. The valid answer is
the same shape with a different arm — **Wanda** attention-50%, not SparseGPT,
whose valid reading changes sign.

### The defect, recorded

Both earlier runs passed a raw `(images, labels)` tuple of uint8 tensors in
[0, 255] to the scan. Two silent failures composed:
`sal.fi.extract_activation_graph` does not normalise, and `nn.Conv2d`
**promotes uint8 instead of raising**, so the scan ran two orders of magnitude
outside the ViT's [-1, 1] range and returned plausible numbers (dense FI
0.132039 — roughly double the normalised reading); and `sal.fi._iter_data`
treats any iterable as a sequence of batches, so the tuple delivered all 500
images as one batch, the label tensor being spared only because `num_samples`
was already reached.

Both are now fixed and neither can recur silently:

- the probe is built by one shared function, which **raises** on integer
  dtype, on non-finite values and on a value range that does not straddle
  zero (`build_probe` / `validate_probe`);
- `FIScanner` and `FIMonitor` warn on such input by default and raise under
  `strict_probe=True` (v0.7.0);
- this script reproduces the dense FI reference before it measures any arm.

The PHASE1 scans are unaffected — the prototype's own FI path normalises.

**A footnote on the gate.** This run passed against the pre-recalibration band
[0.065049, 0.082524], and probe 1 cleared its lower bound by **one edge**.
That band was a min/max over nine draws, which is not a tolerance interval;
exp42 then failed it on a legitimate draw. It is now [0.050, 0.100] — wide
enough never to adjudicate sampling noise, narrow enough to catch the failures
it exists for (unnormalised uint8 at 0.132039, a 64-image probe at 0.151456),
with a test pinning those four numbers.

## Scope, stated with the result

- **One checkpoint, one seed, three probes, no accuracy.** This asks what the
  diagnostic sees, not what the compression costs. Costs are in
  `EXP32_RATIO_AT_2EP.md`, `EXP33_SPARSITY.md` and
  `EXP35_FFN_AT_2EP.md`, in the public data archive, DOI [10.5281/zenodo.24391200](https://doi.org/10.5281/zenodo.24391200).
- **Detection is not usefulness.** Wanda attention-50% is detected. Nothing
  here shows that the detection predicts anything about that model's accuracy,
  and exp17 measured FI failing as a router across six architectures.
- **Three draws decide nothing marginal.** Every "not reproducible" verdict
  means *not established at three draws*, not *no effect*. The two arms that
  change sign are the ones where three draws are plainly too few.
- **The floor is still moving.** 9, 15 and 16 edges from three separate
  three-draw estimates; 15 from the six pooled draws. Any future single-probe
  FI reading should be assumed to carry at least a 15-edge uncertainty.
- **One edge is not one edge across architectures.** Every FI figure must
  state its node and edge count alongside `num_samples`, or it cannot be
  compared to another.
