# Roadmap

`sal-torch` makes transformers **survive compression** — by training them to
reorganize around missing structure, and by telling you which structure you can
afford to lose.

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

---

## Next: v0.4.0 — the Robustness Suite

**In development — API shipped on `main`, validated across four runs (below).**

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

### The Pareto result

`SAL/int4` scores **0.8926 at 361.9MB**. The *uncompressed* standard model
scores 0.8848 at 1419.3MB.

Higher accuracy, one quarter the size. It is the only point on the
accuracy-vs-size frontier; no variant of the standard model reaches it at any
size. This is the first result in the project that is a deployment
recommendation rather than a measurement.

### What is still open

- **Full fine-tuning is necessary, not automatically sufficient.** Run 1 was
  full fine-tuning too and showed no quantization effect — but its INT4 cost the
  baseline nothing, so there was no headroom. The pattern across all four runs is
  that SAL recovers quantization damage in proportion to how much damage there
  is.
- **Scale.** Run 3 was LoRA-only, so at 2.7B the two explanations are still
  confounded. Phi-2 under full fine-tuning separates them; it needs roughly 44GB
  for weights, gradients and optimizer state.
- **Single seed everywhere.** Two GPT-2 baselines on identical data differ by
  0.58pp — that is the run-to-run floor, and no margin below it means anything.
- **More seeds and full eval splits** before any of this is a benchmark rather
  than a signal.

Reproduce: `scripts/modal_v040_test.py` (run 1) and
`scripts/modal_robustness_scale.py` (runs 2-4, via `SAL_TIERS`). Raw numbers in
`scripts/robustness_scale_results.json`.

For the 24% on magnitude pruning: `sal.compare()` already covers you today.
For the 21% on distillation: see v0.6.0.
For the 15% not compressing yet: the docs are getting a real getting-started
path, because "should I compress at all?" is a legitimate answer.

---

## Planned

### v0.5.0 — `CompressionPipeline` · in development

The v0.4.0 evidence points at one specific recipe: **fully fine-tune with SAL,
then quantize to INT4.** On GPT-2 Medium that beat the uncompressed standard
model at a quarter of the size. Getting there by hand means wiring `SALConfig` →
`HeadMasker` → your training loop → head selection → a quantization backend →
`RobustnessTest`, and getting the training method wrong silently costs 3 points
of accuracy.

Shipped on `main`:

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
- ⏳ End-to-end validation on GPT-2 Medium, reproducing the Pareto result
  through the pipeline rather than by hand.

Also queued for this window: **more seeds and full eval splits** on the four
runs above, so the v0.4.0 section can stop calling itself a signal.

### v0.6.0 — Topology-guided distillation and wider architectures

Distillation currently throws away the teacher's structure and hopes the student
rediscovers it. If we know which of the teacher's heads are structurally
critical and which are redundant, we can tell the student what to preserve.
Targeted at the 21%.

Alongside it: Mixture-of-Experts support (`ExpertMasker` — the same
accumulate-and-hold idea applied to expert routing), and additional plasticity
axes beyond the three shipped today.

### v1.0.0 — Stable and production-ready

- **API frozen** with semantic-versioning guarantees.
- **Published benchmarks** across model families and compression methods, with
  reproduction scripts — no claim in the README without a script behind it.
- **Compliance reporting** for regulated deployments: a signed, auditable record
  of what was compressed, by how much, and what the structural impact was.
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

The v0.4.0 quantization result is published above — including the two runs where
SAL lost, and the LoRA configuration we now advise against. Results land here
first, favourable or not.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
