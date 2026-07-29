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

**In development — API shipped on `main`, validation in progress.**

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

**Scope:**

- ✅ **`RobustnessTest`** — run one model through a battery of degradations
  (INT8, INT4, head pruning at several rates, inference-time neuron dropout) and
  report baseline / after / delta / survived per method, plus an aggregate
  robustness score.
- ✅ **`robustness_compare()`** — the head-to-head: a SAL-trained model versus a
  standard one, across every method, with a winner per row.
- ✅ **Visual robustness reports** — bar and radar charts, JSON and PDF.
- ⏳ **Honest publication of the result**, whichever way it lands.

INT4 uses bitsandbytes NF4 when `sal-torch[quant]` is installed, and falls back
to a simulated per-channel INT4 round-trip otherwise, so the row never silently
vanishes from a report.

For the 24% on magnitude pruning: `sal.compare()` already covers you today.
For the 21% on distillation: see v0.5.0.
For the 15% not compressing yet: the docs are getting a real getting-started
path, because "should I compress at all?" is a legitimate answer.

---

## Planned

### v0.5.0 — Topology-guided distillation

Distillation currently throws away the teacher's structure and hopes the student
rediscovers it. If we know which of the teacher's heads are structurally
critical and which are redundant, we can tell the student what to preserve.
Targeted at the 21%.

### v0.6.0 — Wider architectures

Mixture-of-Experts support (`ExpertMasker` — the same accumulate-and-hold idea
applied to expert routing), plus additional plasticity axes beyond the three
shipped today.

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

Watch the repo to follow the v0.4.0 quantization result — it lands here first.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
