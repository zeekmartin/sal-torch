# exp42 — the layer-classification thresholds, and the FI probe-noise floor

**ViT-B/16, `evanarlian/imagenet_1k_resized_256`, AMD MI250X / ROCm 6.2.**
Two runs, six probe draws in all, 0.11 GPU-hours:

| run | geometry | job | data |
|---|---|---|---|
| **exp42-g512** | **512 images, 32 batches of 16**, `num_samples=500` | 23713872, 2:29 | [`exp42_g512/`](exp42_g512/exp42_fi_thresholds.json) |
| exp42-norm | 500 images, 31 batches of 16 + 1 of 4 | 23709984, 4:01 | [`exp42_norm/`](exp42_norm/exp42_fi_thresholds.json) |

Both are normalised and both are valid; the second is kept because its three
draws are three more measurements of the floor, and because **the two runs
disagree about the shipped default**, which is the most useful thing on this
page. Probe seeds 0/1/2, edge density 0.10, the pretrained checkpoint, nothing
trained and nothing compressed. The probe is built by the shared harness
(`scripts/run_sal_qat_prototype.py:build_probe`), which refuses an integer or
unnormalised probe.

---

## Headline

1. **The floor is bigger than any single run says.** Four independent
   three-draw estimates of the dense FI spread on this one checkpoint read
   **9, 11, 15 and 16 edges**. Pooled, the nine draws this page may cite span
   **18 edges of 1030**. A three-draw spread is not an estimate of a floor.
2. **The shipped default `critical_thr = 0.25` is the best threshold in the
   swept range, and it is not stable.** Over six draws it leaves exactly
   **one layer of twelve undecided** (layer 3, CRITICAL on 4 of 6), against
   three to five undecided layers at every other informative threshold.
3. **The previous claim that five thresholds were "usable on every probe" is
   withdrawn.** That test compared the *number* of CRITICAL layers. The
   number agrees at every threshold; the *layers* agree at two. A count is
   not a diagnosis.
4. **`classify_layers_by_rank` is stable at k=1 and k=5 and unstable at
   k=3** — a structural fact about this model, not a defect, and the
   actionable recommendation on this page.

## 1. The probe-noise floor

The same unchanged pretrained checkpoint under every valid draw the project
has made of this quantity:

| run | geometry | draws | spread | edges | % of mean |
|---|---|---|---|---|---|
| exp42-g512 | 512 (32×16) | 0.078641 / 0.064078 / 0.076699 | 0.014563 | 15 | 19.9% |
| exp42-norm | 500 (ragged) | 0.079612 / 0.070874 / 0.081553 | 0.010680 | 11 | 13.8% |
| exp37-g512 | 512 (32×16) | 0.067961 / 0.066019 / 0.074757 | 0.008738 | 9 | 12.6% |
| **pooled** | | **9 draws, mean 0.073355** | **0.017476** | **18** | **23.8%** |

exp37 runs the same `sal.fi` path on a different draw sequence, and exp45
showed the project's two FI code paths return identical values on identical
input, so these are nine draws of one quantity.

> **Any FI difference smaller than 0.0175 on this model at this probe size has
> been shown to occur with no cause.** Expect that number to grow again with
> more draws: each of the four three-draw subsets understated it, by between
> 12% and 53%.

This is a floor, not an error bar — a min/max over draws is not a tolerance
interval, which is a lesson this page paid for. The reference gate built on
the three-draw min/max rejected a *legitimate* reading (job 23713581 failed on
0.064078, one edge below a bound derived from nine earlier draws). The gate's
band is now [0.050, 0.100]: wide enough never to adjudicate sampling noise,
narrow enough to catch the failures it exists for — an unnormalised uint8
probe reads 0.132039 and a 64-image probe 0.151456 — with a test pinning those
four numbers.

The graph's own resolution is one edge, **1/1030 = 0.00097**. The probe noise
is eighteen times that, so what limits an FI comparison is which images were
drawn, not how finely the graph can express a change.

### Every FI figure must state its sample count

Same model, same normalised construction, same edge density, two probe sizes:

| images | FI | source |
|---|---|---|
| 64 | 0.151456 | [`exp42_norm/…_64img.json`](exp42_norm/exp42_fi_thresholds_64img.json) |
| 500–512 | 0.064078 – 0.081553 | the nine draws above |

**A factor of roughly two, on the same model** — larger than any compression
effect measured in this campaign (`EXP37_FI_COMPRESSED.md`).

> **An FI figure without its `num_samples` is not interpretable**, exactly as
> a latency figure without its batch size is not.

`FIScanner` records `num_samples` on its result and `ScanResult.describe()`
prints it, so the figure and its probe size cannot be separated downstream.

## 2. The classification thresholds

`classify_layers` labels a layer IMMUNE / BUFFER / CRITICAL from the relative
FI change when that layer's heads are removed. Sweeping `critical_thr` over
the stored per-layer deltas, **over all six draws**, and asking which layers
are CRITICAL on *every* draw rather than how many:

| `critical_thr` | always CRITICAL | never | undecided | set stable |
|---|---|---|---|---|
| 0.05 | 0,1,2,3,4,5,7,8,9,10,11 | 1 | none | yes — but 11 of 12, saturated |
| 0.10 | 0,1,2,3,4,9,10,11 | 1 | L5 (5/6), L7 (5/6), L8 (4/6) | no |
| 0.15 | 0,1,2,3,4,9 | 1 | L5, L7, L8, L10, L11 | no |
| 0.20 | 0,1,2,3,4 | 3 | L5, L8, L9, L11 | no |
| **0.25** | **0,1,2,4** | **7** | **L3 (4/6)** | **no — one layer** |
| 0.30 | 0,1,2,4 | 7 | L3 (4/6) | no — one layer |
| 0.35 | 0,2 | 7 | L1, L3, L4 (4/6 each) | no |

**0.25 and 0.30 are the only informative thresholds where a single layer is
probe-dependent.** 0.05 is perfectly stable and useless: it calls eleven of
twelve layers CRITICAL, because 5% of an FI of 0.073 on a 1030-edge graph is
**3.8 edges**, and a bar under four edges high is one nearly every layer
clears.

So the shipped default is vindicated — by a criterion the earlier run did not
apply — and it still carries a one-layer uncertainty. **A single-probe
classification of this model at the default returns `{0,1,2,4}` plus layer 3
about two-thirds of the time.**

### What the two runs disagree about, and why it matters

| | 512 geometry | 500 ragged geometry |
|---|---|---|
| n CRITICAL at 0.25 | 5 / 5 / 5 | 4 / 5 / 4 |
| same layers at 0.25 | **yes**, `{0,1,2,3,4}` | **no**, `{0,1,2,4}` twice |

Taken alone, the 512 run says 0.25 is probe-stable. Taken alone, the 500 run
says its count is not even constant. Both are right about their own three
draws. **The reconciliation is layer 3**, whose relative delta over the six
draws reads 0.2398 / 0.2489 / 0.3553 / 0.3999 / 0.5509 / 0.6044 — it straddles
0.25, so a threshold there classifies it by the luck of the draw. Neither run
could see that with three draws.

This is the same lesson as the floor, one level up: **three draws can show a
spread and cannot bound one**, and two runs that each had three draws reached
opposite conclusions about the same default.

### `immune_thr = 0.01` is effectively unreachable

1% of 0.073 is 0.8 edges, below the graph's own resolution. Across the six
draws it labels one layer IMMUNE on one draw and none on the other five. The
IMMUNE class does not exist on this architecture at this probe size. It is
left at 0.01 because lowering it is meaningless and raising it would change
the meaning of the label; a caller who needs an IMMUNE class here must pass
their own threshold.

## 3. What is stable: the block, not the order inside it

Rank of each layer by relative FI delta (1 = most sensitive), six draws:

| layer | ranks over 6 draws | swing |
|---|---|---|
| **2** | 1 / 1 / 1 / 1 / 1 / 1 | **0** |
| 0 | 3 / 3 / 3 / 2 / 4 / 4 | 2 |
| 1 | 5 / 4 / 5 / 4 / 3 / 2 | 3 |
| 3 | 4 / 2 / 4 / 5 / 2 / 5 | 3 |
| 4 | 2 / 5 / 2 / 3 / 5 / 3 | 3 |
| 9 | 6 / 6 / 8 / 7 / 7 / 8 | 2 |
| 7 | 9 / 10 / 11 / 9 / 9 / 9 | 2 |
| 10 | 7 / 9 / 10 / 8 / 10 / 10 | 3 |
| 5 | 11 / 8 / 9 / 10 / 8 / 6 | 5 |
| 8 | 10 / 11 / 7 / 11 / 6 / 7 | 5 |
| 11 | 8 / 7 / 6 / 6 / 11 / 11 | 5 |
| **6** | 12 / 12 / 12 / 12 / 12 / 12 | **0** |

| k | top-k identical on all 6 draws | intersection | union |
|---|---|---|---|
| 1 | **yes** | `{2}` | `{2}` |
| 3 | **no** | `{2}` — one of three | `{0,1,2,3,4}` |
| 5 | **yes** | `{0,1,2,3,4}` | `{0,1,2,3,4}` |

**The five most sensitive layers are the same five on every draw. Their order
is not.** Layers 0, 1, 3 and 4 permute freely among ranks 2–5 while
collectively staying above every other layer, so `k=3` cuts through a block
that has no stable internal ordering and returns `{2,0,4}` on one draw and
`{2,3,1}` on another — one layer in common out of three.

That makes the recommendation concrete, and it is not the one the previous
revision of this page gave:

> **Use `classify_layers_by_rank(deltas, k=5)` on this model**, which returns
> `{0,1,2,3,4}` on 6 of 6 draws. `k=1` is also exact. **`k=3` is the worst
> available choice — and it is the function's default.**

A rank API is still the better of the two, because it does not require
guessing a number — but it is not probe-free, and whether a given `k` is
stable is a property of the model's delta distribution, not of the API. A
caller who needs a layer set that survives re-probing should intersect several
draws; on this model three draws suffice to recover `{0,1,2,3,4}`.

## 4. Scope

- **One architecture**: 12 layers × 12 heads, FI 0.064–0.083, 1030 edges.
  The deltas are not scale-free — they depend on the model, the probe and the
  edge density — so neither the threshold result nor the 0.0175 floor
  transfers to a different graph without being re-measured.
- **Two probe sizes, 500 and 512**, treated as one because the 2.4%
  difference is far inside the noise; the 64-image reading above shows the
  figure does move with size, by a factor of two.
- **Six draws**, which is enough to show that three were not enough and not
  enough to characterise the distribution. Every "undecided" count above is
  `n of 6`, so a layer at 4/6 is a layer about which this run says little
  beyond *it is near the bar*.
- **Nothing is trained or compressed here.** The thresholds are a property of
  the classification arithmetic on a pretrained checkpoint. Whether a
  CRITICAL label predicts anything about pruning that layer is a separate
  question, and exp17 measured FI failing as a router across six
  architectures.
- Per-layer deltas for all six draws are in the two JSONs, so any further
  threshold question is answerable with no GPU.

## 5. What this changes in the package

1. `classify_layers`' `critical_thr = 0.25` default stands, now backed by the
   six-draw test, and its docstring should state the one-layer uncertainty
   rather than implying a stable partition.
2. **`classify_layers_by_rank`'s default is `k=3`, the least reproducible
   value of `k` on this model.** `k=5` returns the same set on 6 of 6 draws
   and `k=1` on 6 of 6; `k=3` shares one layer of three between draws. The
   docstring already warns that a rank does not buy probe independence and
   cites the three-draw version of exactly this, so the finding is a
   sharpening, not a surprise — but the default should be revisited and the
   docstring should carry the top-5 result. A v0.7.0 item, not changed here.
   The earlier claim that the ordering "holds at the extremes" is right and
   incomplete: what holds is the top-5 *set*, with a stable first and last.
3. `FIScanner` records `num_samples` alongside the score, so no FI figure can
   be quoted without its probe size.
4. A reference gate must be a tolerance interval, not a min/max over past
   draws. The band is [0.050, 0.100] and the four numbers that calibrate it
   are pinned by a test.
