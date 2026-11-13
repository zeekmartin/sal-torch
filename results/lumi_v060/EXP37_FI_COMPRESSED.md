# exp37 — does a fragility scan detect compression, and what does it detect?

**TIER A COMPLETE — 13 arms, 1 seed, 1 probe, 0.07 GPU-hours (job 23701223,
`COMPLETED 0:0`).** Data in `results/lumi_v060/exp37/exp37_fi_compressed.json`.
Tier B (recovered checkpoints from wave 2) is pending exp35.

**ViT-B/16 / ImageNet-1k**, FI probe of 500 validation images, pruner
calibration of 128 **different** images, edge density 0.10 (the shipped
default), seed 42. Script `experiments/exp37_fi_compressed.py`. FI is the
fraction of head-graph edges with zero triangle support (`sal.fi.compute_fi`).

## Headline

**Two different questions were being asked under one name, and the random
control separated them.**

For **structured** pruning the scan does not detect the compression — it detects
that the graph got smaller. For **unstructured, 2:4 and INT4** it detects a real
change on a graph of identical size, which is the non-trivial result and the one
worth having.

## The structured arms: the signal is the node count

Structured pruning removes graph **nodes**. 72 heads gone takes the graph from
144 nodes / 1030 edges to 72 / 256, and FI is computed over what is left.

| arm | nodes | edges | FI | relative change |
|---|---|---|---|---|
| dense pretrained | 144 | 1030 | 0.132039 | — |
| head25 magnitude | 108 | 578 | 0.197232 | +49.4% |
| head33 magnitude | 96 | 456 | 0.265351 | +101.0% |
| **head50 magnitude** | 72 | 256 | **0.371094** | **+181.0%** |
| *head50 random, seed 0* | 72 | 256 | *0.398438* | *+201.8%* |
| *head50 random, seed 1* | 72 | 256 | *0.351562* | *+166.3%* |

**The magnitude-chosen set sits between the two random sets.** The two random
controls span 0.0469 of FI for the same operation at the same size, and
magnitude's 0.371094 falls inside that range. So at this resolution FI says
nothing about *which* heads were removed — only that 72 of them are gone, which
a parameter count already reports for free.

That is what the controls were built for, and the answer they give is negative.
**Do not report a structured arm's FI shift as "certification detected the
compression".** Read it as a property of a 72-node graph.

## The sparsity and quantization arms: a real signal on an identical graph

Unstructured pruning, 2:4 and INT4 keep every node and every edge — 144 and
1030 in all seven rows, at the same 0.10 density by construction. Any FI change
is therefore a change in the *structure* of the graph, not in its size.

| arm | scope | FI | ΔFI | relative | edges moved |
|---|---|---|---|---|---|
| Wanda unstructured | attn 50% | 0.133981 | +0.001942 | +1.5% | +2 |
| SparseGPT unstructured | whole 16.7% | 0.139806 | +0.007767 | +5.9% | +8 |
| Wanda 2:4 | attn | 0.143689 | +0.011650 | +8.8% | +12 |
| **INT4 round-trip** | all weights | 0.148544 | +0.016505 | **+12.5%** | +17 |
| Wanda unstructured | whole 16.7% | 0.151456 | +0.019417 | +14.7% | +20 |
| SparseGPT unstructured | attn 50% | 0.157282 | +0.025243 | +19.1% | +26 |
| **SparseGPT 2:4** | attn | 0.169903 | +0.037864 | **+28.7%** | +39 |

Two things fall out.

**INT4 moves the graph.** A quantization round-trip changes no parameter count,
no mask and no architecture, and it moves FI by +12.5% — more than four of the
six sparsity arms. A fragility scan sees weight rounding. Nothing in the project
predicted that and nothing else measures it.

**The ordering does not track the accuracy cost.** exp33 measured these same
unstructured cells as accuracy-equivalent after recovery — all eight within
±0.22pp of a control whose own seeds span 0.21pp — and 2:4 as costing about
0.5pp. Here they spread over a factor of **nineteen** in ΔFI, from +1.5% to
+28.7%, and the ordering is not the accuracy ordering: SparseGPT's attn-50%
moves FI 13x more than Wanda's attn-50% while the two differ by hundredths of a
point in accuracy.

**So FI detects compression and does not rank its cost.** That is exp17's
finding at a different level: FI is a structural diagnostic, not a predictor of
what compression will cost. Here it is a *detector* — which is the job
certification actually needs — and a detector is allowed to be insensitive to
cost.

## The limitation that governs all of it

**One seed, one probe, and no noise floor for the full-graph arms.**

The two random head sets give a noise estimate for the 72-node graph: 0.0469 of
FI, which is 20% of the structured signal itself. There is **no equivalent
estimate for the 144-node arms**, and the smallest of them — Wanda attn-50% at
+0.001942, two edges out of 1030 — is exactly the size of thing a different
probe draw could produce on its own.

So the table above establishes the *large* entries (SparseGPT 2:4 at +39 edges,
SparseGPT attn at +26, Wanda whole at +20, INT4 at +17) and leaves the two
smallest unsupported. **exp42 measures that floor directly** — the same
pretrained checkpoint under three independent probe draws — and tier A should be
re-read against it before any entry under about ten edges is quoted.

Other limits: one architecture, one edge density (0.10, the shipped default; FI
is defined relative to it), one calibration set, and the pruners are the
vendored SparseGPT and Wanda from `experiments/exp33_sparse_core.py`.

## What tier B adds

Tier A is **one-shot**: every arm is compressed from the pretrained checkpoint
and scanned immediately. The models a user actually ships have had a recovery
pass, and recovery may move the graph back toward the dense one — or further
away. Tier B scans the wave-2 checkpoints once exp35 writes them
(`--save-checkpoint`), which is the first time this project will have compressed
*and recovered* weights to scan at all: wave 1 saved none.
