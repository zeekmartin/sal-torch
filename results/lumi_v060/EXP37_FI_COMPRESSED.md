# exp37 — does a fragility scan detect compression, and what does it detect?

**TIER A, THREE PROBES — 13 arms, 1 seed, 3 independent probe draws, 0.08
GPU-hours (job 23702553, `COMPLETED 0:0`).** Data in
`results/lumi_v060/exp37_3probes/exp37_fi_compressed.json`. The superseded
single-probe run is job 23701223, kept in `results/lumi_v060/exp37/` and
reproduced below as the documented error. Tier B (recovered checkpoints from
wave 2) is pending exp35.

**ViT-B/16 / ImageNet-1k**, three FI probes of 500 validation images each
(probe seeds 0, 1, 2), pruner calibration of 128 **different** images, edge
density 0.10 (the shipped default), seed 42. Script
`experiments/exp37_fi_compressed.py`. FI is the fraction of head-graph edges
with zero triangle support (`sal.fi.compute_fi`).

## Headline

**The probe is a measurement instrument with a resolution, and until this run
the project had never measured it.** On the 144-node graph that resolution is
coarser than six of the seven effects the single-probe run reported.

Two findings survive, and they are the two the controls were built to separate:

1. **Structured pruning produces a large, probe-robust FI shift that is a
   property of the graph's size and not of which heads were removed.** It is a
   real signal and it is not worth having: a parameter count reports the same
   thing for free.
2. **Of the seven arms that leave the graph intact, exactly one moves FI by
   more than its own probe-to-probe spread**: SparseGPT unstructured at
   attention-50%. The other six — including the INT4 round-trip the
   single-probe run led with — are inside the instrument's noise.

## The noise floor, measured

The **same pretrained checkpoint, unmodified**, scanned under three independent
probe draws:

| probe seed | FI | edges with zero triangle support |
|---|---|---|
| 0 | 0.148544 | 153 of 1030 |
| 1 | 0.159223 | 164 of 1030 |
| 2 | 0.142718 | 147 of 1030 |

**Spread 0.016505 — 17 edges, 11.0% of the mean.** Nothing changed about the
model between those three readings. Any FI difference smaller than that is a
statement about which 500 images were drawn.

The single-probe run's dense reading was **0.132039, i.e. 136 edges** — below
all three draws above. Over the four draws together the dense FI spans
**136 to 164 edges, 0.0272 of FI**. The single-probe run therefore measured
every arm against a floor that was too low, which is the exact mechanism by
which it reported seven positive shifts: the baseline, not the arms, was the
outlier. The 17-edge figure is quoted as the floor below because it comes from
three draws made the same way; 28 edges is the more conservative number and the
one to use when deciding whether a *future* single-probe reading means anything.

## The arms that keep the graph intact

Unstructured pruning, 2:4 and INT4 keep every node and every edge — 144 and
1030 in all seven rows, at the same 0.10 density by construction. ΔFI is
computed **within a probe** (arm minus dense on the same images), so probe
variance cancels to the extent that the two models respond to a probe the same
way; the columns below show how far that holds.

The test in the `delta_exceeds_probe_spread` field is: the **smallest**
absolute ΔFI across the three probes must exceed the ΔFI's **own spread**
across them. A delta that cannot clear its own instability is not a
measurement of the arm.

| arm | scope | ΔFI range over 3 probes | in edges | sign | verdict |
|---|---|---|---|---|---|
| **SparseGPT unstructured** | **attn 50%** | **+0.01748 … +0.02913** | **+18 … +30** | stable | **measured** |
| SparseGPT unstructured | whole 16.7% | +0.00097 … +0.00874 | +1 … +9 | stable | under noise |
| SparseGPT 2:4 | attn | −0.00777 … +0.02330 | −8 … +24 | **flips** | not measured |
| Wanda unstructured | whole 16.7% | −0.00680 … +0.00874 | −7 … +9 | **flips** | not measured |
| INT4 round-trip | all weights | −0.00194 … +0.00485 | −2 … +5 | **flips** | not measured |
| Wanda unstructured | attn 50% | −0.02524 … −0.01262 | −26 … −13 | stable | under noise |
| Wanda 2:4 | attn | −0.02524 … −0.00097 | −26 … −1 | stable | under noise |

**One arm of seven survives.** SparseGPT's attention-50% moves FI by at least
18 edges on every draw, which is both larger than its own 12-edge spread and
comparable to the dense floor. It is the only compression in this table the
scan detects.

**Three arms change sign with the probe.** For SparseGPT 2:4, Wanda whole and
INT4, whether the compressed model reads *more* or *less* fragile than the
dense one is decided by which 500 images were drawn. No claim of any direction
can be made for them.

**Two arms are sign-stable and negative.** Wanda's unstructured and 2:4
attention arms read consistently *below* dense — unstructured attention pruning
makes the graph look less fragile, not more. The single-probe run reported both
as positive. The magnitudes (13-26 edges) sit at the edge of the floor, so the
direction is suggestive and the size is not established; three more draws would
settle it, and nothing in the project currently needs the answer.

### What this withdraws

> **INT4 moves the graph.** A quantization round-trip changes no parameter
> count, no mask and no architecture, and it moves FI by +12.5% — more than
> four of the six sparsity arms. A fragility scan sees weight rounding.

**Withdrawn.** That +12.5% was +0.016505 of FI — **17 edges, the dense noise
floor to the edge**. Under three probes the INT4 round-trip moves FI by −2 to
+5 edges and does not keep its sign. There is no measured basis for the claim
that a fragility scan sees weight rounding, and a reader should assume the
opposite until something measures it: the one quantization arm in this project
is indistinguishable from redrawing the probe.

> **The ordering does not track the accuracy cost** … they spread over a factor
> of **nineteen** in ΔFI, from +1.5% to +28.7%.

**Withdrawn as stated.** The factor of nineteen was a spread across six
readings that are now known to be noise. The underlying point — that FI does
not rank what compression costs — stands on exp17 and on exp33's accuracy
table, not on this one, and is restated below without the ordering.

## The structured arms: a real signal, and it is the node count

Structured pruning removes graph **nodes**. 72 heads gone takes the graph from
144 nodes / 1030 edges to 72 / 256, and FI is computed over what is left.

| arm | nodes | edges | FI per probe (0 / 1 / 2) | ΔFI range |
|---|---|---|---|---|
| dense pretrained | 144 | 1030 | 0.148544 / 0.159223 / 0.142718 | — |
| head25 magnitude | 108 | 578 | 0.212803 / 0.230104 / 0.205882 | +0.0632 … +0.0709 |
| head33 magnitude | 96 | 456 | 0.276316 / 0.304825 / 0.267544 | +0.1248 … +0.1456 |
| **head50 magnitude** | 72 | 256 | 0.371094 / 0.382812 / 0.378906 | +0.2226 … +0.2362 |
| *head50 random, seed 0* | 72 | 256 | *0.390625 / 0.371094 / 0.375000* | *+0.2119 … +0.2421* |
| *head50 random, seed 1* | 72 | 256 | *0.367188 / 0.371094 / 0.355469* | *+0.2119 … +0.2186* |

**Every structured row clears its own spread comfortably**, and the effect is
monotone in the number of nodes removed. This is the one part of the
single-probe table that three probes confirm unchanged.

**And it is blind to the selection.** At 50%, magnitude's three readings are
0.371094 / 0.382812 / 0.378906; the two random controls at the same size span
0.355469 to 0.390625. Magnitude falls inside that band on probe 0 and sits
above both controls on probes 1 and 2 — by 0.0117 and 0.0039, both **smaller
than the 0.0165 dense noise floor**. So the largest gap attributable to *which*
heads were removed is about 12 edges, against the 229 edges attributable to
*how many* were removed. The selection term, if it exists at all, is a
twentieth of the size term and under the instrument's resolution.

**Do not report a structured arm's FI shift as "certification detected the
compression".** Read it as a property of a 72-node graph.

## What FI is, after this run

**A detector with a coarse resolution, for one kind of compression.** It sees
structured pruning loudly, because structured pruning changes the graph it is
computed over. It sees one of six weight-level compressions, and the
single-probe run's apparent sensitivity to the other five was the probe moving,
not the model.

**It does not rank what compression costs.** exp33 measured these same
unstructured cells as accuracy-equivalent after recovery — all eight within
±0.22pp of a control whose own seeds span 0.21pp — and 2:4 as costing about
0.5pp. The one arm whose FI shift is real, SparseGPT attn-50%, is not the one
that costs the most accuracy. This restates exp17's finding at a different
level: FI is a structural diagnostic, not a predictor of what compression will
cost.

## The error this run corrects, kept in full

The single-probe table is reproduced here rather than deleted, because the
shape of the mistake is more instructive than the numbers were. One probe,
one dense reading of 0.132039:

| arm | scope | FI | ΔFI vs the single dense draw | relative |
|---|---|---|---|---|
| Wanda unstructured | attn 50% | 0.133981 | +0.001942 | +1.5% |
| SparseGPT unstructured | whole 16.7% | 0.139806 | +0.007767 | +5.9% |
| Wanda 2:4 | attn | 0.143689 | +0.011650 | +8.8% |
| INT4 round-trip | all weights | 0.148544 | +0.016505 | +12.5% |
| Wanda unstructured | whole 16.7% | 0.151456 | +0.019417 | +14.7% |
| SparseGPT unstructured | attn 50% | 0.157282 | +0.025243 | +19.1% |
| SparseGPT 2:4 | attn | 0.169903 | +0.037864 | +28.7% |

Every entry is positive, the ordering looks like a dose-response curve, and the
write-up read it as one. Three of the seven reverse under a second probe. The
single reading of 0.148544 attributed to INT4 above is, numerically, the dense
checkpoint's reading on probe 0 of the three-probe run — the arm and the
unmodified model are the same number, drawn differently.

**The run that produced that table was not underpowered by accident.** Its own
"limitation that governs all of it" section named the problem exactly:

> One seed, one probe, and no noise floor for the full-graph arms. … the
> smallest of them — Wanda attn-50% at +0.001942, two edges out of 1030 — is
> exactly the size of thing a different probe draw could produce on its own.

It then published the table anyway and qualified it, rather than withholding it
until the floor was measured. **The limitation was correct, it was stated next
to the result as doc rule 3 requires, and it was still not enough**: a stated
caveat does not stop a table from being read, quoted and built on. The lesson
for the campaign is that a measurement whose noise floor is unknown is not a
result with a caveat — it is not yet a result, and the floor costs 0.08
GPU-hours to measure.

## Limits of this run

**One seed and three probes.** The probes close the question the single-probe
run could not answer; the single training seed is untouched, and every number
here is from one pretrained checkpoint.

Three probes bound the floor loosely. 0.016505 is the spread of three draws,
not a confidence interval, and a fourth draw can fall outside it — the
single-probe run's 0.132039 did.

Other limits, unchanged: one architecture, one edge density (0.10, the shipped
default; FI is defined relative to it), one calibration set, and the pruners are
the vendored SparseGPT and Wanda from `experiments/exp33_sparse_core.py`.

**exp42** carries this one level down, to the per-layer deltas that drive
`classify_layers` and its `critical_thr` default, under the same three probe
draws.

## What tier B adds

Tier A is **one-shot**: every arm is compressed from the pretrained checkpoint
and scanned immediately. The models a user actually ships have had a recovery
pass, and recovery may move the graph back toward the dense one — or further
away. Tier B scans the wave-2 checkpoints once exp35 writes them
(`--save-checkpoint`), which is the first time this project will have compressed
*and recovered* weights to scan at all: wave 1 saved none. **Tier B runs three
probes from the start**, and any arm it reports will be held to the floor
measured above.
