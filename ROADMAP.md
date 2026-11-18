# Roadmap

`sal-torch` makes transformers **survive compression** — by measuring what a
given model stands to lose, routing it to the method that suits it, and training
it to reorganize around the structure you remove.

This document is where the project is going. Dates are release dates, not
promises about the future ones.

---

## Shipped

### v0.1.0 — Core SAL + Fragility Index · 2026-06-25

The mechanism and the diagnostic.

- **SAL training-time head masking** (`SALConfig`, `HeadMasker`, `SALCallback`,
  `SALTrainer`) — heads are progressively silenced during a training window and
  *stay* silenced, forcing the model to redistribute their function into the
  survivors. Three lines to add to a HuggingFace `Trainer`, or bring your own
  PyTorch loop.
- **Fragility Index** (`compute_fi`, `FIScanner`, `FIMonitor`) — a `[0,1]`
  structural score for how much redundant pathway a model's attention graph
  has. Low FI = robust, high FI = fragile. Purely diagnostic; usable with or
  without SAL.
- Architecture auto-detection for 12 transformer families.

### v0.2.0 — Know your model before you touch it · 2026-06-26

- **`PlasticityScanner`** — a three-axis absorption map (routing flexibility,
  inter-layer redundancy, intra-layer redundancy) that labels every layer
  `ELASTIC` / `SATURATED` / `HUB`, then turns it into concrete
  *prune-these* / *never-touch-these* head lists.
- **`sal.compare()`** — benchmark SAL against magnitude and random post-hoc
  pruning at a matched compression level. Plug in your own method.
- **Visual PDF reports** for every scanner.

### v0.3.0 — Continual learning without replay buffers · 2026-06-27

- **`StructuralGuard`** — fine-tune on a new task without overwriting the old
  one. The structural map picks the critical attention heads and freezes them
  via gradient masking; the redundant heads stay free to absorb the new domain.
  No EWC, no replay buffer, no distillation. Head-level granularity, so some
  heads in a layer can be frozen while others keep learning.
- **`DriftMonitor`** — measure structural forgetting after *any* fine-tuning,
  guarded or not: a forgetting score, per-layer retention, and which layers
  changed class.

### v0.4.0 — the Robustness Suite · 2026-07-29

**Shipped. Validated across four single-seed runs (below), then re-validated
across five seeds in v0.5.0 — which confirmed most of it and corrected one
claim.**

We asked practitioners how they actually compress models. 33 people answered:

| How you compress | Share |
|---|---|
| Quantization (INT8 / INT4) | **39%** |
| Magnitude pruning | 24% |
| Distillation | 21% |
| Not compressing yet | 15% |

SAL was built and validated against **head pruning**. But most of you quantize.
So the question v0.4.0 has to answer is blunt:

> **Does a SAL-trained model survive quantization better than a standard one?**

If yes, structural resilience is a general property, not a pruning trick. If no,
we will say so in this document and scope SAL honestly.

**Answer: yes — but only if you fully fine-tune.** The full evidence, including
the two runs where SAL lost, is below.

**Scope:**

- ✅ **`RobustnessTest`** — run one model through a battery of degradations
  (INT8, INT4, head pruning at several rates, inference-time neuron dropout) and
  report baseline / after / delta / survived per method, plus an aggregate
  robustness score.
- ✅ **`robustness_compare()`** — the head-to-head: a SAL-trained model versus a
  standard one, across every method, with a winner per row.
- ✅ **Visual robustness reports** — bar and radar charts, JSON and PDF.
- ✅ **Honest publication of the result**, whichever way it lands.

INT4 uses bitsandbytes NF4 when `sal-torch[quant]` is installed, and falls back
to a simulated per-channel INT4 round-trip otherwise, so the row never silently
vanishes from a report.

### The evidence, in the order we got it

Four runs. Two of them SAL lost. They are all here because the losses are what
explain the wins.

Each run trains one model twice from identical weights — plain vs. SAL at
`prune_fraction=0.33` — then evaluates both **dense** under the battery. SAL
wins per category, scored by absolute accuracy:

| # | model / task | training | clean cost | quant | prune | combined |
|---|---|---|---|---|---|---|
| 1 | DistilBERT / SST-2 | full FT | +1.17pp | 1/2 | **2/2** | not tested |
| 2 | GPT-2 Medium / SST-2 | **LoRA r=16** | **-3.12pp** | 0/2 | 1/2 | 1/2 |
| 3 | Phi-2 2.7B / MMLU | **LoRA r=16** | -1.17pp | 0/2 | 1/2 | 0/2 |
| 4 | GPT-2 Medium / SST-2 | **full FT** | +0.39pp | **2/2** | **2/2** | **2/2** |

Run 1 also tested inference-time neuron dropout at 10% and 20%; the standard
model won both, by 0.5pp and 0.6pp, inside that run's noise floor. Runs 2-4 do
not test dropout, so it gets no column.

**Run 1 — DistilBERT.** Head pruning: a large, real win. At 50% of heads removed
the standard model gives up 14.3% of its accuracy and the SAL model gives up
6.7%, less than half. Quantization: nothing measurable — every margin under one
accuracy point, and INT4 made the *standard* model look better than fp32, which
is the giveaway that you are reading sampling noise. We published that as "SAL
is pruning-specific" and went looking for a scale where quantization hurts.

**Runs 2 and 3 — LoRA.** Both negative. On GPT-2 Medium, SAL lost four of six
compressed variants and paid 3.1 points of clean accuracy for the privilege. On
Phi-2 2.7B every margin landed within one or two eval examples of zero, and the
premise failed outright: NF4 cost Phi-2 *nothing* (0.4570 dense → 0.4570 int4 at
a third the size), so there was no damage for SAL to mitigate. What did hurt
Phi-2 was head pruning, -7.8pp at 33%, and SAL did not help there either.

At that point the plausible readings were "SAL does not scale" or "LoRA starves
it", and they were confounded — model size and training method had varied
together across every run.

**Run 4 — the control.** Same model, task, data, seed and battery as run 2. The
only change is that every weight moves instead of rank-16 adapters:

```
variant           baseline       SAL     delta  base_size  sal_size    winner
-----------------------------------------------------------------------------
dense               0.8848    0.8887   +0.0039     1419.3    1419.3       SAL
int8                0.8828    0.8887   +0.0059      513.3     513.3       SAL
int4                0.8750    0.8926   +0.0176      361.9     361.9       SAL
prune33             0.8359    0.8555   +0.0195     1419.3    1419.3       SAL
prune50             0.8145    0.8379   +0.0234     1419.3    1419.3       SAL
prune33+int8        0.8398    0.8555   +0.0156      513.3     513.3       SAL
prune33+int4        0.8340    0.8613   +0.0273      361.9     361.9       SAL
```

Seven for seven, on absolute accuracy and on retention, at no cost to the clean
model. `int4` and both pruning and both combined rows clear the 0.195pp noise
floor; `dense` and `int8` individually do not, but all seven agree in sign.

### What it means

**SAL requires full fine-tuning.** The mechanism is the model reorganizing
around silenced heads. LoRA freezes exactly the weights that would do the
reorganizing, so the perturbation lands and the adaptation cannot. Rank-16
adapters on `c_attn` cannot absorb 126 silenced heads. Under LoRA you get the
cost of SAL and none of the benefit — that is not a subtle effect, it is
-3.12pp of clean accuracy and four lost variants.

**Under full fine-tuning, the benefit extends past pruning to quantization.**
Which answers the question this release was built to ask, for the 39%.

### The Pareto result — as it looked on one seed

`SAL/int4` scores **0.8926 at 361.9MB**. The *uncompressed* standard model
scores 0.8848 at 1419.3MB.

Higher accuracy, one quarter the size, and the only point on the
accuracy-vs-size frontier.

**This is the claim the five-seed run corrected.** Across seeds `SAL/int4`
averages 0.9023 against the uncompressed standard model's 0.9005 — a 0.18pp gap
inside a ±1pp spread. The deployment recommendation survives, but as *equal
accuracy at a quarter of the size* rather than higher accuracy. See the v0.5.0
section below.

### What was still open after run 4

- **Full fine-tuning is necessary, not automatically sufficient.** Run 1 was
  full fine-tuning too and showed no quantization effect — but its INT4 cost the
  baseline nothing, so there was no headroom. The pattern across all four runs is
  that SAL recovers quantization damage in proportion to how much damage there
  is.
- **Scale.** Run 3 was LoRA-only, so at 2.7B the two explanations are still
  confounded. Phi-2 under full fine-tuning separates them; it needs roughly 44GB
  for weights, gradients and optimizer state.
- ~~**Single seed everywhere.**~~ **Closed in v0.5.0** — five seeds on the full
  eval split, below.

Reproduce: `scripts/modal_v040_test.py` (run 1) and
`scripts/modal_robustness_scale.py` (runs 2-4, via `SAL_TIERS`). Raw numbers in
`scripts/robustness_scale_results.json`.

For the 24% on magnitude pruning: `sal.compare()` already covers you today.
For the 21% on distillation: see v0.7.0.
For the 15% not compressing yet: the docs are getting a real getting-started
path, because "should I compress at all?" is a legitimate answer.

### v0.5.0 — `CompressionPipeline`, and five seeds instead of one · 2026-08-12

The v0.4.0 evidence pointed at one specific recipe: **fully fine-tune with SAL,
then compress.** Getting there by hand means wiring `SALConfig` → `HeadMasker` →
your training loop → head selection → a quantization backend → `RobustnessTest`,
and getting the training method wrong silently costs 3 points of accuracy. This
release makes that path the default — and then runs it five times to find out
how much of the v0.4.0 result was the seed.

Shipped:

- ✅ **`slice_heads()`** — the missing piece. Everything before v0.5.0 *masked*
  heads, so the model behaved as if smaller while staying exactly as large.
  Slicing removes them from the weights: a model that actually shrinks, runs
  without hooks, and does not need sal-torch installed.
- ✅ **`quantize()` / `quantize_info()`** — one call over bitsandbytes and
  `torch.ao`, with the output head never quantized and GPT-2's `Conv1D`
  converted first (miss that and nothing gets quantized, silently).
- ✅ **`CompressionPipeline`** — scan → sal_train → compress → validate →
  export, measured at every stage. **Refuses LoRA/QLoRA** with an explanation,
  warns under 100M parameters, and `export()` reloads what it wrote rather than
  assuming the round trip held.
- ✅ End-to-end validation on GPT-2 Medium — the pipeline runs, and the export
  reloads at exactly the accuracy that was measured.
- ✅ **Five-seed validation on the full eval split** — the outstanding item from
  v0.4.0, closed. See "Five seeds" below.

**The validation run, honestly.** GPT-2 Medium / SST-2 on one T4, both arms
through the identical pipeline (`scripts/modal_v050_test.py`):

```
                       original      trained    sliced      int4     final size
SAL                      0.5020       0.8965    0.8496    0.8398    347MB (4.1x)
standard (no SAL)        0.5020       0.8926    0.8770    0.8633    347MB (4.1x)
```

The pipeline does what it claims: 1419MB → 347MB, every stage measured, and the
exported artifact reloads at 0.8398 — the number it was given, to four decimals.

**SAL lost this one.** Dense accuracy was a tie (0.8965 vs 0.8926, two eval
examples apart), so the 2.3-point gap opened under compression — the reverse of
the v0.4.0 result on the same model and task.

The difference we could name was **head selection**. `compress()` removed the
lowest-magnitude heads at the time; the v0.4.0 battery removed *random* ones,
and random removal is exactly what SAL trains against. So the hypothesis was
that magnitude ranking is a worse guide on a SAL-trained model specifically —
one deliberately made resilient to losing any given head. That turned out to be
half right, and the default has since changed to `random`.

**Measured** (`scripts/modal_selection_experiment.py`, 2 arms × 3 strategies, the
same 120 heads removed in every cell, masked then INT4 so only *which* heads
varies):

```
arm           dense    magnitude       random    fi_guided
----------------------------------------------------------
standard     0.8926       0.8594       0.8867       0.7520
SAL          0.8965       0.8398       0.8906       0.7617

retention vs each arm's own dense model
standard    100.00%       96.28%       99.34%       84.25%
SAL         100.00%       93.68%       99.35%       84.97%
```

Three things fall out of it:

1. **The default was the problem.** `magnitude` is the worst viable strategy for
   *both* arms — it costs the standard model 2.7 points and the SAL model 5.7.
   `random` is better for everyone, and that finding has nothing to do with SAL.
   The v0.5.0 regression was substantially an artifact of the default.
2. **On this seed, SAL still did not win.** Under `random` the two arms were
   0.0039 apart — two eval examples, exactly the noise floor. Fixing the
   selection removed the regression; it did not, on one seed, restore the win.
   The five-seed run below is what actually answered that.
3. **`fi_guided` is bad here, on both arms.** ~84% retention against random's
   ~99%. Concentrating removal in the layers a fragility scan calls cheap does
   far more damage than spreading it evenly — at least at a 33% budget on a
   model where only two layers classify as non-CRITICAL, forcing most of the
   budget into CRITICAL layers anyway (spill of 64–80 of 120 heads).

**Also fixed by this run:** `sal_train` was not clipping gradients while
`SALTrainer` clips by default. Silencing a third of the heads makes gradients
spikier, and the unclipped run scored 0.7676 against the clipped run's 0.8965 —
a 13-point artifact that would have been reported as "SAL does not work".

### Five seeds — the item this project has owed since v0.4.0

Every SAL-vs-standard number above is one seed. That has been the standing
caveat since v0.4.0, and it is not a small one: two GPT-2 baselines on identical
data differ by about a point, one conclusion here was already overturned by a
missing gradient clip, and the pipeline run above reversed v0.4.0 outright on
the same model and task. So the v0.4.0 protocol was run five times.

GPT-2 Medium / SST-2, **full fine-tuning**, seeds 42/123/456/789/1337, scored on
the **complete 872-example validation split** rather than a 512 subset. Each
seed trains the same checkpoint twice and runs both arms through the same
battery, with both arms losing the same heads
(`scripts/modal_multiseed_validation.py`, one T4 per seed):

```
variant        standard            SAL                 delta    ahead  consistent
----------------------------------------------------------------------------------
dense          0.9005 +- 0.0104    0.9062 +- 0.0080   +0.0057    4/5      yes
int8           0.8888 +- 0.0102    0.8961 +- 0.0033   +0.0073    4/5      yes
int4           0.9002 +- 0.0111    0.9023 +- 0.0094   +0.0021    3/5      NO
prune33        0.8571 +- 0.0240    0.8817 +- 0.0121   +0.0245    5/5      yes
prune50        0.8087 +- 0.0455    0.8294 +- 0.0368   +0.0206    4/5      yes
prune33+int8   0.8284 +- 0.0090    0.8472 +- 0.0229   +0.0188    4/5      yes
prune33+int4   0.8567 +- 0.0208    0.8725 +- 0.0132   +0.0158    5/5      yes
```

"Consistent" is a sign test — SAL ahead on 4 or more of the 5 seeds. It asks
whether the *direction* holds, which is precisely what a single seed cannot say.

**What replicates.** SAL wins 5 of the 6 compressed variants, at no cost to
clean accuracy (+0.57pp — it is not buying resilience with accuracy). The effect
is largest and most reliable where SAL was designed to work: **head pruning**,
+2.45pp at 33% on *every* seed and +2.06pp at 50%. Both combined recipes hold.
That is the core v0.4.0 claim, confirmed.

**What does not.** *INT4 on its own does not replicate.* Three seeds of five and
+0.21pp — inside the run-to-run spread, i.e. a coin flip. The v0.4.0 headline
that SAL won *all seven* variants including INT4 was one seed. Across five,
quantization-only gains are small, and INT4 is the weakest row in the table.
Where SAL helps at INT4 is alongside pruning (`prune33+int4`, +1.58pp on 5/5),
which is the pruning effect carrying the row.

**And the Pareto claim softens.** `SAL/int4` averages 0.9023 at 361.9MB against
the uncompressed standard model's 0.9005 at 1419.3MB. A 0.18pp gap inside a ±1pp
spread is not "higher accuracy at a quarter of the size" — it is **equal
accuracy at a quarter of the size**. Still the deployment recommendation. Not a
free accuracy gain, and the README has been corrected to say so.

**One observation, not a claim.** The SAL arm has the smaller standard deviation
on 6 of the 7 variants, most sharply on `prune33` (0.0121 vs 0.0240). Five seeds
is far too few to call that a property, but compression outcomes were more
*predictable* here, not merely better on average. Worth a proper test later.

Two protocol notes, both recorded in `scripts/multiseed_results.json`. INT8 here
is `torch.ao` dynamic quantization on CPU where v0.4.0 used bitsandbytes
LLM.int8() on CUDA, so those rows are a different measurement rather than a
replication; INT4 is bitsandbytes NF4 in both. And the eval-time head choice is
seeded independently of the training seed, so SAL is never scored on the exact
heads it trained against.

**What is still open after this.** The five seeds cover one model on one task.
DistilBERT and Phi-2 are still single-seed, the LoRA result is single-seed, and
nothing above 355M has been fully fine-tuned at all. Multi-seed is now the bar
for new claims, not something to retrofit again later.

---

### v0.5.1 — SAL for self-supervised models · 2026-09-17

`SALTrainer` was wired to cross-entropy, which meant SAL only applied to models
with a task head. It now takes a `train_step=` callback, so the objective lives
wherever the caller wants it and SAL keeps only the prune schedule. Shipped with
the metrics that replace accuracy for encoders (linear probe, kNN, CKA,
latency), `ijepa`/`dinov2` architecture support, and `slice_heads()` working on
ViT-family encoders.

**The falsification test was run.** I-JEPA ViT-H/14 on ImageNet-100, SAL-trained
vs a control trained identically minus head masking, three seeds. The direction
is not noise: **SAL wins 35 of 36 paired comparisons** at 33% and 50% pruning.
The size is small — +0.4 to +1.9pp — so the honest reading is a reliable but
modest effect, not a large one.

What else the runs showed:

- **70% pruning is outside SAL's range.** SAL ≈ control within ±1.6pp, both
  when trained at 30% masking and at 70%.
- **DINOv2 ViT-L/14** (one seed): 13/18 wins, larger where the pruned model
  survives (+3.0pp probe, +6.9pp kNN at 33% sliced), and both arms collapse at
  50%. Consistent with "SAL recovers damage in proportion to how much there is",
  still a post-hoc reading.
- **Slicing is real but modest.** 50% of I-JEPA's heads: 631M → 526M, 1.10× GPU
  and 1.20× CPU. Attention is a third of a ViT block; heads alone cannot reach 2×.

Two things the scoping got wrong, recorded so they are not rediscovered:
Meta never released an I-JEPA ViT-B/16 (H/14 and g/16 are the catalogue), and a
"predict your own clean representation" objective has a loss of exactly zero for
any arm that is not being perturbed.

---

### v0.6.0 / v0.6.1 — two methods, two regimes, and the control arm that won · 2026-11

Run on LUMI-G (AMD MI250X / ROCm 6.2) under a 5,000 GPU-hour EuroHPC
allocation. The campaign that was meant to confirm SAL at scale instead
reorganised the product around what the measurements said. Full trail in
`results/lumi_v060/`.

**The setting that settles things.** ViT-B/16 on **ImageNet-1k**
(`evanarlian/imagenet_1k_resized_256`, the open mirror of the gated set —
canonical 1000 classes, 256px shortest side), **three seeds, the full 50 000-image
validation split**. Every earlier ImageNet-100 figure is superseded where the
cells overlap.

**DropHead — resampling the mask every step — beat SAL as a head-training
method.** It entered as SAL's control arm, to test whether progressive
accumulation was doing anything. It was, and the sign was negative:

| at 50% of heads + INT4, vs the *same* baseline | |
|---|---|
| SAL, on sets it did not train for | +14.6pp |
| **DropHead, on any set** | **+57.2pp** |
| DropHead over SAL | **+42.7pp** |

- **Pruning 33% of DropHead's heads costs +0.14pp against its own dense row**
  (77.69% vs 77.54%, 76.12 vs 76.09 at INT4) — **but that row is not the
  control**. Against the dense checkpoint trained for the same five epochs with
  nothing pruned (80.44%) the cell reads **−2.75pp**, because DropHead's
  training costs **2.90pp** on its own. Earlier revisions called this "free",
  which compared an arm to itself; that wording is withdrawn. At 50% it costs
  5.0pp against its own dense row. On a fixed magnitude-chosen set DropHead is
  **+63.5pp** over a standard fine-tune, and **+54.7pp** on SAL's own set — a
  set it has no more seen than any other.
- **DropHead is also stable**: seed-to-seed sd of 0.02-0.92pp across six
  measured rows, against SAL's 0.50-17.55pp. One SAL row reads
  60.15 / 51.83 / 26.46% on three seeds of the same configuration.

**SAL keeps a narrower, real claim: one head set, fixed at training time.** On
its own trained-for set SAL reads **76.51%** — against **34.52%** on sets it
never saw, and against **20.33%** for the baseline checkpoint on the identical
set. So the adaptation is genuine (+56pp) and entirely set-specific (+0.05pp of
it is the set being intrinsically easier). The honest cost is **−3.93pp** against
the matched baseline, at a 16.5% smaller model — not the drop its dense row
suggests. The ratio must be fixed too: at 33%, a subset of its own set, it falls
to 72.76%.

**SAL is not a pre-conditioner.** At a fixed head set, neither SAL's nor
DropHead's weights beat the pretrained ones after a recovery fine-tune:
**+0.05pp** and **+0.07pp**. What survives is a head-*selection* signal worth
+0.60pp, and nothing at 33%.

**The probe routes compression. The fragility score does not.** Tested across six
architectures (DeiT-small, BERT-base, ViT-B/16, DINOv2-large, I-JEPA ViT-H/14,
phi-2), FI read off a pretrained checkpoint fails to predict the pruning regime
at both ends: BERT-base has a higher FI than ViT-B/16 and loses 5pp where
ViT-B/16 loses 62; DINOv2-large has an FI 25× lower than DeiT-small and collapses
to chance on the same task and protocol. Parameter count does not order it
either. What does is **how far an ordinary fine-tune falls when you prune it** —
which `CompressionPipeline.probe()` measures directly in two evaluation sweeps
and no training:

| probe drop at 50% of heads | route | what it means |
|---|---|---|
| **> 30pp** | `sal_aligned` | head training pays for itself. The route string names the *regime*, not a method. |
| **10-30pp** | `drophead` | head training pays for itself less. |
| **< 10pp** | `skip_heads` | head training buys nothing measurable — **the one row the evidence contradicts**, see exp34 below. |

**The method is not chosen by the route; it is chosen by whether you can
fine-tune after pruning.** If you can, magnitude selection + a 2-epoch recovery
is the strongest option measured and the training-time method is close to
irrelevant (+0.05pp for SAL's weights, +0.07pp for DropHead's, at a fixed head
set). If you cannot, DropHead is the general method and SAL the specialist.
The earlier form of this table carried "SAL, if you control the deployed head
set" in the top row, which v0.7.0 step 3 withdraws three sections below; the
contradiction is resolved here in favour of the withdrawal.

**FI stays in the product for diagnosis and certification, not as a router** —
and not as a head selector either: fragility-weighted selection loses to uniform
random on 0/10 seeds at both mask ratios (−5.02pp and −4.77pp).

**FFN pruning is where the compression actually is.** Measured against a matched
control that saw the same five epochs with nothing pruned, at comparable size
removed: **25% of FFN neurons costs 0.63pp** where **50% of heads costs 3.17pp**
— heads are roughly five times more expensive per unit of size. At an identical
32.7% of parameters removed, all-FFN beats head+FFN by 1.55pp.

**On speedup, the honest numbers.** The **highest measured speedup is 1.74×**, at
49.1% of parameters removed (50% heads + 50% FFN) and a cost of 6.82pp, **at
batch 32**. **At batch 1 every configuration is 1.00×** — 8.09-8.37 ms against
8.16 ms dense — because a ViT-B/16 at batch 1 is kernel-launch bound. The 2×
target carried by earlier versions of this roadmap **is not reachable by
magnitude pruning on this model and has been withdrawn.** Any latency figure
here is a batch-32 figure.

**What did not work, recorded so it is not rediscovered.** The first attempt to
scale the campaign to I-JEPA ViT-H/14 (631M) on ImageNet-1k failed as an
experiment: two epochs of continued self-supervised pretraining collapsed
**every** arm including the unmasked control (linear probe 46.07% pretrained
→ 15.32% baseline, 3.20% DropHead, 2.69% SAL, three-seed means), with the
training loss falling the whole way. **196.6 GPU-hours, no method comparison
obtained.** The cause turned out to be neither the scale nor the
hyperparameters but the objective itself: the step computed its target from the
same live weights it was training, so an encoder drifting toward a low-variance
output drags both sides of the loss down together. On that objective **loss
progress is decoupled from representation quality**, which is why nothing being
watched looked wrong. The lesson is now a standing rule: **gate on the control
arm, not on completion** — see "Training harness" under v0.7.0.

**What the corrected run then measured.** With a **frozen target encoder** (LR
5e-6, 10% warmup, cosine decay) the control arm holds across two full
ImageNet-1k epochs on **3/3 seeds**, and the comparison the collapsed run could
not make goes through. On a 631M self-supervised encoder, **DropHead beats SAL
on every cell and every seed**: **+29.02pp** over the control at 50% head
pruning against SAL's +7.89pp, **+11.88pp** at 33% where SAL reads −0.22pp and
changes sign with the seed, and **+0.67pp on SAL's own trained-for head
set** — the one cell where SAL was still expected to win. SAL also gives up
**12.35pp** of dense accuracy (3/3 seeds) and its best cell lands 5.11pp below
the control's dense row. As on ViT-B/16, **pruning 33% of heads is free for
DropHead** (−0.04pp) and 50% costs 1.14pp. 223.6 GPU-hours; 420.2 for the 1B
campaign as a whole. This is why step 3 of v0.7.0 no longer carries an exception
for SAL.

---

## Planned

### v0.7.0 — diagnose, route, compress, recover

v0.6.x established that there is no single best method, and that the choice is
measurable before any training happens. v0.7.0 makes that the shape of the
product: one pipeline that measures a model, picks the method, compresses, and
repairs.

**The pipeline, in order:**

1. **Diagnose** — `FIScanner` for structural fragility, reported and
   certifiable. A diagnostic, not a decision.
2. **Route** — `CompressionPipeline.probe()` returns `sal_aligned` /
   `drophead` / `skip_heads` from measured accuracy loss. Already shipped in
   v0.6.1; v0.7.0 adds routing for the FFN ratio, which currently has no
   equivalent.
3. **Compress** — **the choice is set by whether a post-pruning fine-tune is
   available, not by the route.** If it is, prune by magnitude and recover;
   head-training methods are worth +0.05pp (SAL) and +0.07pp (DropHead) as
   starting points for that fine-tune, i.e. nothing, and on BERT-base across
   four GLUE tasks magnitude + 2 epochs beat DropHead on 8/8 cells (exp34). If
   it is not, **DropHead, with no exception among head-training methods**: the
   "SAL when the deployed head set is fixed at training time" carve-out is
   **withdrawn**, since exp28b measured exactly that cell on a 631M
   self-supervised encoder and DropHead matched or beat SAL on 3/3 seeds while
   leaving the head set free until deployment. SAL stays in the package,
   documented and supported, scoped to "no recovery possible and the deployed
   set is fixed before training". FFN neuron pruning carries the bulk of the
   size reduction.
4. **Recover** — a short fine-tune after pruning, which is worth more than any
   training-time masking when the deployment architecture is known. **Two
   epochs**, not five: exp29 measured the curve and it peaks at 2, with 5 below
   2 on 3/3 seeds in FP and INT4, and 10 epochs below 1. Two epochs is 39% of
   the cost of the old default and 0.31pp better. The pipeline should also
   **probe after each recovery epoch and keep the best checkpoint**, since the
   last one is measurably worse than the best.

**Also in v0.7.0:**

- **`recover()` as a pipeline stage — item 1.** `CompressionPipeline` has no
  recovery stage today, so every accuracy this project publishes for post-hoc
  pruning includes a step the product does not perform. Default 2 epochs,
  mask-preserving by default, and matched-control reporting that separates
  `fine_tuning_gain_pp` from `pruning_cost_pp` and emits the second only when
  the epoch counts match. Specified in `SPEC_V070_RECOVERY.md`.

  **`recover()` must take its data from the target domain, and must check
  that the repair is larger than the domain shift it introduces.** This is
  not a refinement, it is a condition under which the stage is worth
  running at all. On OPT-1.3B, 32M tokens of C4 make the *unpruned* model
  worse on raw-WikiText-2 by **0.6398 perplexity** (pretrained 14.6240,
  dense control 15.3260 / 15.3536 / 15.1120 over three seeds). The
  recovery is a domain shift that costs, not a gain that offsets, which is
  why a cost taken against the pretrained checkpoint instead of the
  matched control is inflated on every arm: for unstructured-50 it reads
  1.6338 instead of 0.9939, **65% too high**. Source:
  `sal-torch-pro/results/EXP36_OPT_SPARSE.md`.

  At 1.3B the stage still pays, because the damage it repairs (about 3.2
  perplexity one-shot for the 2:4 and FFN arms) is several times that
  0.64. **At 6.7B it stops paying for the mildest arms, and the
  recommendation has to say so.** The same 32M C4 tokens cost the 6.7B
  dense control **+1.0748** perplexity (10.8601 before, 11.9907 / 12.0425
  / 11.7714 after, three seeds), while removing 50% of the attention
  weights without a structure constraint costs only **+1.1987** one-shot.
  Repair and damage are then the same size, and the recovered
  `wanda_unstructured_50` arm reads **worse** perplexity than its own
  one-shot model: 12.2514 against 12.0588 as a three-seed mean, worse on
  3 of 3 seeds (+0.3114, +0.0493, +0.2172). Nothing is wrong with the
  recovery; it is being asked to repair less than it breaks.

  So the API should make the corpus an explicit, required decision rather
  than a default, report the control's own movement on the evaluation set
  beside the arm's, and say plainly that a recovery whose matched control
  degrades is measuring a domain shift. A two-line guard covers the case:
  if the dense control moves against the evaluation metric by as much as
  the arm's one-shot damage, the honest output is the one-shot model.
  Source: `results/w5_opt67_{ctrl,wu50}_s{42,123,456}/exp36_opt_sparse.json`
  in the pro repository, three seeds each. These rows are measured and
  verified but **not yet in a canonical results table**, so treat the
  1.3B figures above as the quotable ones until that table is regenerated.

- **Selection defaults differ by component, and both are measured.**
  `recover()` and the compression stage ahead of it should default to
  **random selection for attention heads** and **magnitude for FFN
  neurons**. These are not the same answer and the project has stopped
  assuming they would be. On heads, random beats magnitude by **1.17pp**
  at 50% with no overlap between the two seed ranges. On FFN neurons at
  25%, magnitude costs **-0.869pp** against the matched 2-epoch control
  (range -0.988 to -0.664 over **7 seeds**) where random costs
  **-1.397pp** (range -1.668 to -1.212 over 7 seeds): the ranges are
  disjoint, so the 0.528pp is a result and not seed noise. Note the
  inversion that makes this worth stating: random is **1.91pp better
  one-shot** and 0.528pp worse after recovery, so a one-shot benchmark
  picks the wrong FFN default. Magnitude's one-shot figure is identical
  across seeds because the selection is deterministic; only the recovery
  differs. Sources: `sal-torch-pro/results/CANONICAL_TABLES.md` and
  `sal-torch-pro/results/EXP35_FFN_AT_2EP.md`.
- **`mask_mode="drophead"` on `HeadMasker` — item 2.** DropHead is recommended
  throughout this document and is **not callable from the package**:
  `HeadMasker` accumulates and holds, which is SAL, and the per-step resampled
  variant exists only in `scripts/run_sal_qat_prototype.py`. v0.7.0 adds it as a
  supported mode, with tests reproducing the prototype's behaviour at identical
  seeds and masks.
- **A multi-probe layer classifier — item 3.** exp42 measured
  `classify_layers_by_rank` over six independent probe draws on an unchanged
  ViT-B/16: `k=1` and `k=5` return the same set on 6 of 6 draws and `k=3` —
  the current default — shares **one layer of three** between draws. What is
  stable is the top-5 *block*; the ordering inside it is not, so `k=3` cuts
  through a block that has no reproducible internal order.
  **The default is not changed to 5**: `k=5` is stable on this model, and
  which `k` is stable is a property of the model's delta distribution, not of
  the API. The fix is to stop asking for one draw. Add a variant that takes
  several probe draws and returns layers partitioned into **stable /
  unstable / undecided** rather than a single ranked list, so the caller sees
  the reproducibility instead of inferring it. Same for `classify_layers`,
  whose shipped `critical_thr = 0.25` leaves exactly one layer of twelve
  undecided on this model (layer 3, CRITICAL on 4 of 6 draws) — the best
  threshold in the swept range, and still not a stable partition.
  Source: `results/lumi_v060/EXP42_FI_THRESHOLDS.md`.
- **Say what a CRITICAL label is for — item 4.** `classify_layers`'
  documentation must state that the label is **descriptive, not pruning
  advice**: it says a layer's removal moves FI, not that the layer should be
  kept or dropped. exp17 tested FI as a compression router across six
  architectures and it fails at both ends (`PHASE2H.md`), and FI-weighted
  head *selection* loses to uniform random on 0/10 seeds (`PHASE2E.md`).
  A user who reads CRITICAL as "do not prune this" is reading a routing
  claim the project has measured and rejected. Docs change only; the
  classification itself is unaffected.
- **An FFN-ratio router**, so step 2 covers the component that carries most of
  the compression rather than only the heads.
- **Make DropHead's masking cheaper than a forward hook.** Measured on
  BERT-base (exp34's smoke, MI250X, seq 128, batch 32): a plain fine-tune runs
  at **453 samples/s** and the same fine-tune with DropHead's per-step
  resampled masking runs at **284 samples/s** — the hooks cost **~37% of
  training throughput**. DropHead is the default method, so that overhead is
  paid on every recommended run. Two things to do, in order: **measure the
  same overhead on a ViT**, where the attention block is a smaller share of
  the step and the cost should differ; then **try applying the mask inside the
  attention computation as a vectorised op** rather than through a pre-hook on
  the output projection, and keep whichever is faster at identical masks. The
  hook is what makes the mechanism architecture-agnostic, so a faster path has
  to be shown equivalent before it replaces it, not instead of it.
- **The training harness safeguard** — the control gate, promoted from
  experiment script to library. Specced below.
- **A corrected self-supervised recipe** — **done, validated** (exp28b): a
  frozen target encoder, LR 5e-6, 10% warmup, cosine decay. The control arm holds
  across two full ImageNet-1k epochs on 3/3 seeds, against the ~32pp collapse of
  the uncorrected recipe. What remains for v0.7.0 is exposing it as a supported
  path rather than an experiment file.
- **Topology-guided distillation** — distillation throws away the teacher's
  structure and hopes the student rediscovers it. If we know which of the
  teacher's heads are structurally critical and which are redundant, we can tell
  the student what to preserve.
- **Mixture-of-Experts support** (`ExpertMasker`), and additional plasticity axes
  beyond the three shipped today.

#### Training harness — the control gate

A long training run can destroy a model while every signal you are watching looks
healthy. exp28 spent **196.6 GPU-hours** on three arms whose unmasked control arm
lost 27–33pp of representation quality: the losses fell the whole way, the
90-second smoke jobs passed, and a 1:22:53 gate job passed. The failure was only
visible by measuring the *control* checkpoint's features against where it
started.

exp28b added that measurement as an ad-hoc block in
`experiments/exp28b_jepa_corrected.py` and it ran clean on 3/3 seeds. v0.7.0
promotes it to a supported safeguard in the library. **Specification only; not
implemented yet.**

**What it does.** Before training, measure the starting checkpoint on a cheap
task-appropriate metric and keep it as the reference. After each epoch, measure
the control arm again. If it has regressed by more than a threshold against that
reference, abort loudly — and let the sibling arms of the same experiment abort
with it, since they are measuring against a control that no longer means
anything.

**The five properties that made it work, each of which is a requirement:**

1. **The reference is measured, not declared.** A hardcoded "48.6%" is wrong the
   moment the probe size, the dataset or the model changes. exp28b measures the
   untrained checkpoint in the same job, with the same probe, on the same data.
2. **It gates the control arm, not the perturbed ones.** A masked arm is
   *supposed* to score lower; only the control's own regression is unambiguous
   evidence that the recipe is broken.
3. **The verdict is a file, and siblings poll it.** The control arm writes a
   pass/fail sentinel atomically; every arm re-reads it on a fixed step interval
   and raises on fail. This is what lets one job's finding stop the other
   arms — they are separate SLURM jobs with no shared memory. Optionally it
   `scancel`s the named siblings outright.
4. **Compare within a measurement job, never across.** exp28b's probe reads
   41.04–48.60% on the *same untrained checkpoint* in three different jobs
   (7.56pp). A gate that compared a fresh epoch-1 probe against a reference
   measured in some earlier job would fire at random. The reference and the check
   must share the job, the probe size and the data shuffle.
5. **The threshold is generous.** exp28b used 5pp, and the real failure was
   27–33pp. The gate is there to catch catastrophe cheaply, not to adjudicate
   small differences; a tight threshold turns it into a source of false aborts.

**Shape of the API** (indicative, not settled):

```python
from sal.harness import ControlGate

gate = ControlGate(
    reference=lambda m: linear_probe(m, probe_train, probe_val),
    threshold_pp=5.0,
    sentinel="runs/exp/control_gate.json",
    role="control",          # or "sibling" — siblings only read
    poll_every_steps=200,
    siblings=["23567665", "23567667"],   # optional: scancel on abort
)
gate.establish(model)        # measures and stores the reference
...
gate.check(model, epoch=1)   # control arm: measure, write sentinel, raise on fail
gate.poll()                  # every arm, inside the step loop: raise if failed
```

**Cost and benefit.** One probe per epoch per run — ~10 GPU-hours against the
196.6 exp28 spent, i.e. about 5%. It belongs next to the existing "fail loudly on
a non-finite loss" rule, one level up: a diverged loss is caught in seconds, a
quietly collapsing control arm currently is not caught at all.

#### The v0.7.0 release gate

**No v0.7.0 tag until the shipped pipeline reproduces the published figures.**
Every accuracy on this page and in the README was produced by
`scripts/run_sal_qat_prototype.py`, not by `CompressionPipeline`. That is
acceptable for a research trail and not acceptable for a product claim, and it
is how the missing recovery stage went unnoticed through two releases.

The gate: run the shipped pipeline end to end — `probe()` → magnitude selection
→ `recover()` → `validate()` — on ViT-B/16 / ImageNet-1k, and check it lands
within seed tolerance of the published cell (77.64% at 50% of heads after two
epochs, three seeds spanning 77.44–77.81%). The control it is scored against is
the 2-epoch dense control, verified before the run rather than quoted.

If the pipeline cannot reproduce the number, the number is not a product claim
and the README says so instead.

### v0.8.0 — Edge deployment exports

Making compressed models deployable on real edge hardware without requiring
sal-torch on the target device.

- **TensorRT export** (`sal.export.to_tensorrt()`) — for NVIDIA Jetson
  (Orin NX, AGX Orin, Thor): a `.engine` file that runs natively on the Jetson's
  GPU.
- **OpenVINO export** (`sal.export.to_openvino()`) — for Intel edge hardware
  (Movidius, Core Ultra NPU). Same principle: a deploy-ready artifact.
- **`inference_benchmark()`** on `CompressionPipeline` — latency and throughput
  on the target runtime (PyTorch, ONNX Runtime, TensorRT, OpenVINO) at every
  compression stage, **reported per batch size**, since v0.6.x showed the
  batch-1 and batch-32 pictures are entirely different.
- **Batch optimization** — resolve the padding overhead identified in v0.4.0
  (batched inference slower than single-sample on short texts). Fixed shapes +
  dynamic batching + sequence bucketing.

### sal-torch-pro — the commercial extension

**Already measured and shipping:**

- **FFN neuron pruning** (`sal_pro.mlp`) — **validated on ImageNet-1k**, three
  seeds, full validation split (exp18, exp26; 97.2 GPU-hours). This is the
  component that carries the compression: 25% of neurons costs 0.63pp against
  3.17pp for 50% of heads at comparable size. **Speedups, with the two sources
  separated**: slicing alone at batch 32 is **1.15× / 1.19× / 1.36×** at
  25 / 33 / 50% of neurons (exp26, ImageNet-1k, MI250X eager). Adding
  `torch.compile --max-autotune` on top reaches 1.28× / 1.33× / 1.50×
  (exp16, ImageNet-100) — **those are pruning × compiler, not pruning**, and
  an earlier revision of this roadmap quoted them as if the pruning had earned
  them. The dense model compiles to 1.12× on its own, which is the reference
  any pruning speedup has to be read against.
- **`torch.compile` and ONNX export** (`sal_pro.deploy`) — **ONNX export
  validated** (exp22), compiled inference benchmarked (exp16). The compiled dense
  model is 1.12× at batch 32, which is the reference any pruning speedup has to
  be read against.
- **QAT**, compliance reports, and fleet monitoring.

**Planned:**

- **`edit_impact_bound()`** — given a proposed weight edit, bound how far it can
  move the model's behaviour before the edit is applied. Intended as the
  pre-flight check for compression and patching decisions.
- **`compare_weights()`** — a structural diff between two checkpoints, reporting
  what changed where rather than a scalar distance. The natural companion to
  `DriftMonitor`, which measures the same thing over a training run.
- **`DeployPipeline`** — scan → route → compress → recover → optimize → export →
  benchmark. The customer provides a model ID and a target device; the pipeline
  produces a `.onnx`, `.engine`, or `.xml` file with a benchmark report.
- **`FleetMonitor`** — track which compressed model version is deployed on which
  device, with structural verification of the compressed artifact.
- **Pro-only features**, license-gated: `DeployPipeline`,
  `FleetMonitor`, multi-target export, batch optimization.

Scope note: `edit_impact_bound()` and `compare_weights()` are named here as
deliverables; **their precise semantics are deliberately left unfixed** until a
real use case asks for them. Both are shaped by what the caller needs to decide
— what counts as "behaviour", which bound is useful, what granularity a diff
should report — and specifying that in the abstract tends to produce an API that
fits nobody. They move from planned to specified when a concrete request arrives.

### v1.0.0 — Stable and production-ready

- **API frozen** with semantic-versioning guarantees.
- **Published benchmarks** across model families, compression methods, and
  deployment targets (PyTorch, ONNX Runtime, TensorRT, OpenVINO), with
  reproduction scripts — no claim in the README without a script behind it.
- **Compliance reporting** for regulated deployments: a signed, auditable record
  of what was compressed, by how much, what the structural impact was, and
  which deployment artifact was produced.
- Broad architecture coverage validated on real models, not just detected.

---

## How to contribute

This is a commercial package under BSL 1.1 — free for research and evaluation,
licensed for commercial production. Development happens in the open.

- **Request a feature or report a bug** →
  [GitHub Issues](https://github.com/zeekmartin/sal-torch/issues)
- **Tell us your architecture doesn't work** → open an issue with the model ID.
  Architecture support is demand-driven; the registry grows from these reports.
- **Tell us how you compress** → the v0.4.0 scope above came directly from a
  community poll. Roadmap priorities follow what people actually run.
- **Benchmark disagreements welcome.** If SAL underperforms on your workload, an
  issue with a reproduction is the most useful thing you can send us.

The full evidence trail is published above — the two runs where SAL lost, the
LoRA configuration we now advise against, and the v0.4.0 INT4 claim the
five-seed run failed to reproduce. Results land here first, favourable or not,
and corrections stay next to what they correct.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
