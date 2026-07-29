# sal-torch

![CI](https://github.com/zeekmartin/sal-torch/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/sal-torch) ![Python](https://img.shields.io/pypi/pyversions/sal-torch) ![Downloads](https://pepy.tech/badge/sal-torch) ![License](https://img.shields.io/badge/license-BSL%201.1-blue)


**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

## Install

```bash
pip install sal-torch            # core
pip install sal-torch[hf]        # + HuggingFace Trainer
pip install sal-torch[reports]   # + PDF/visual reports
pip install sal-torch[crypto]    # + commercial license verification
pip install sal-torch[all]       # everything
```

```python
from sal import SALConfig, SALCallback

config = SALConfig.auto(model)
trainer = Trainer(model=model, callbacks=[SALCallback(config)])
trainer.train()
```

Three lines. Any transformer. Compression-resilient.

## Know your model before you touch it

### FIScanner — how fragile is this model?

The **Fragility Index** is a structural diagnostic, scored in `[0, 1]`, that
measures how much redundant pathway a model's attention graph has. Heads are
compared by their activation signatures; an edge between two heads is *fragile*
when they share no common neighbour, i.e. the function it carries has no backup.
FI is the fraction of such edges.

- **Low FI** → heavily triangulated graph, lots of redundancy → robust.
- **High FI** → many unsupported edges → fragile under compression.

FI is purely diagnostic — it measures, it never perturbs. You can use it with or
without SAL training.

```python
from sal import FIScanner

result = FIScanner(model, probe_dataset).scan()

print(result.fi_score)         # 0.0 - 1.0; lower is more robust
print(result.summary)          # "FI=0.1842 | 3 immune, 2 buffer, 1 critical"
print(result.critical_layers)  # layers whose removal moves FI the most
print(result.immune_layers)    # layers you can compress with little effect

result.save("fragility.json")
result.save("fragility.pdf")   # per-head heatmap (needs sal-torch[reports])
```

Track it *during* training with `FIMonitor`, or call the primitives directly:

```python
from sal import FIMonitor, compute_fi, extract_activation_graph

trainer = Trainer(model=model, callbacks=[FIMonitor(probe_dataset, interval=500)])

adjacency = extract_activation_graph(model, probe_dataset)
fi = compute_fi(adjacency)
```

### PlasticityScanner — where can a model absorb compression?

FI tells you how fragile a model *is*. `PlasticityScanner` tells you how much
room it has to *reorganize*, so you know where it is safe to compress. It scores
three complementary axes per layer — routing flexibility (attention entropy),
inter-layer redundancy (linear CKA), and intra-layer redundancy (an MI proxy) —
and folds them into an **absorption map** that labels each layer `ELASTIC`
(safe), `SATURATED` (bottleneck), or `HUB` (compensates when others are pruned).

```python
from sal import PlasticityScanner

pmap = PlasticityScanner(model, probe_dataset).scan()
print(pmap.summary)              # "3 elastic, 1 saturated, 2 hub | mean routing=0.61 ..."

rec = pmap.recommend(target_compression=0.33)
rec.safe_to_prune                # [(layer, head), ...] — prune these first
rec.never_touch                  # heads in hub layers — leave alone
rec.expected_impact              # heuristic accuracy delta

pmap.save("plasticity.json")     # raw scores
pmap.save("plasticity.pdf")      # visual report (needs sal-torch[reports])
```

### sal.compare() — SAL vs. other pruning methods

Benchmark SAL against post-hoc baselines at a matched compression level and see
which keeps the most accuracy (or lowest loss) after heads are removed.

```python
from sal import compare

result = compare(model, train_dataset, eval_dataset,
                 methods=["sal", "magnitude", "random_posthoc"],
                 compression=0.33, sal_epochs=3, metric="accuracy")
print(result.table)              # method | score | pruned_heads | time
print(result.winner)
result.save("comparison.pdf")    # bar chart + table

# plug in your own method
compare.register_method("my_pruner", lambda model, ds, eval_ds, ctx: my_score)
```

## Does it survive real compression?

We polled practitioners on how they actually compress models. Of 33 responses,
**39% quantize** (INT8/INT4) — more than pruning and distillation. SAL was built
against head pruning, so the honest question is whether the resilience it trains
in generalizes to the compression people actually ship.

### RobustnessTest — one model, every degradation

```python
from sal import RobustnessTest

test = RobustnessTest(model, eval_dataset, metric="accuracy")
report = test.run(methods=["int8", "int4", "head_pruning_33", "head_pruning_50",
                           "neuron_dropout_10", "neuron_dropout_20"])

print(report.table)
# real output — DistilBERT fine-tuned on SST-2, no SAL, 512 eval examples
# method              baseline     after     delta     std  survived
# ------------------------------------------------------------------
# int8                  0.8594    0.8359   -0.0234       -        OK
# int4                  0.8594    0.8672   +0.0078       -        OK
# head_pruning_33       0.8594    0.7656   -0.0938       -      FAIL
# head_pruning_50       0.8594    0.7363   -0.1230       -      FAIL
# neuron_dropout_10     0.8594    0.8392   -0.0202   0.015        OK
# neuron_dropout_20     0.8594    0.8320   -0.0273   0.022        OK

print(report.robustness_score)   # aggregate 0-1: mean quality retained
print(report.survival_rate)      # fraction of methods survived

report.save("robustness.json")
report.save("robustness.pdf")    # bars + retention radar (needs sal-torch[reports])
```

A method counts as **survived** when relative degradation stays within
`survival_threshold` (default 5% of the clean baseline). Methods:

| Method | What it does |
|---|---|
| `int8` | Dynamic INT8 over every `nn.Linear` (`torch.ao.quantization`, CPU) |
| `int4` | 4-bit weight-only — bitsandbytes NF4 when available, otherwise a simulated per-channel INT4 round-trip (the backend used is recorded on each result) |
| `head_pruning_<pct>` | Silences `<pct>`% of attention heads using the shipped `HeadMasker` |
| `neuron_dropout_<pct>` | Zeroes `<pct>`% of FFN neurons at inference — dead units / noisy hardware. Repeated over several fault patterns; mean ± std reported |

`pip install sal-torch[quant]` adds bitsandbytes for real NF4. Without it, `int4`
falls back to simulation rather than disappearing from your report — pass
`allow_simulated_quant=False` if you would rather see the row skipped.

### robustness_compare() — SAL-trained vs. standard

```python
from sal import robustness_compare

result = robustness_compare(
    sal_model=sal_trained_model,
    baseline_model=standard_model,
    eval_dataset=eval_dataset,
    methods=["int8", "int4", "head_pruning_33"],
    metric="accuracy",
)

print(result.table)
print(result.summary)
result.save("robustness_comparison.pdf")
```

Each model is scored against **its own** clean baseline, so the comparison
measures resilience rather than which model was better to begin with. The row
winner is whichever model loses proportionally less.

### What we measured

DistilBERT fine-tuned on SST-2 twice from identical weights — once plain, once
with SAL at `prune_fraction=0.33` — then both evaluated dense under every method
(`scripts/modal_v040_test.py`, one T4, single seed, 512 eval examples):

```
method              base_after  base_deg   sal_after   sal_deg    winner
------------------------------------------------------------------------
int8                    0.8359     2.73%      0.8516     2.24%       SAL
int4                    0.8672    -0.91%      0.8652     0.67%  baseline
head_pruning_33         0.7656    10.91%      0.8047     7.62%       SAL
head_pruning_50         0.7363    14.32%      0.8125     6.73%       SAL
neuron_dropout_10       0.8392     2.35%      0.8464     2.84%  baseline
neuron_dropout_20       0.8320     3.18%      0.8385     3.74%  baseline
------------------------------------------------------------------------
robustness_score  baseline=0.9442  SAL=0.9603
```

Read this honestly:

- **Head pruning: a large, real advantage.** At 50% of heads removed the standard
  model loses 14.3% of its accuracy and the SAL-trained one loses 6.7% — less
  than half. This is the regime SAL trains for, and it delivers.
- **Quantization and neuron dropout: no clear effect.** Every margin there is
  under 1 accuracy point. At 512 eval examples one example is 0.195%, so those
  rows are inside the noise floor. INT4 even *improved* the standard model,
  which is a tell that you are reading sampling noise, not resilience.

So on current evidence **SAL's resilience is specific to structural pruning; we
cannot claim it transfers to quantization.** The suite exists so that claim gets
re-tested at larger scale rather than assumed — see [ROADMAP.md](ROADMAP.md).

## Continual learning without replay buffers

### StructuralGuard — protect what matters when you fine-tune

When you fine-tune a trained model on a new task, it quietly overwrites the
structure that carried the old one. `StructuralGuard` reads the model's
structural map and **freezes the critical attention heads** (hub layers,
structural bottlenecks, and the functionally unique heads) while leaving the
redundant heads free to absorb the new task. No EWC, no replay buffer, no
distillation — the topology itself decides what to protect.

```python
from sal import StructuralGuard

# After training on task A, build a guard from the model's structure.
guard = StructuralGuard.from_model(model, probe_dataset, protection_level=0.5)

print(guard.protected_heads)   # [(layer, head), ...] frozen during fine-tuning
print(guard.trainable_heads)   # [(layer, head), ...] free to absorb task B
print(guard.protection_map)    # {layer: [protected head indices]}

guard.protect(model)           # zero gradients for protected heads (backward hooks)
trainer.train()                # fine-tune on task B with ANY training loop
guard.release()

drift = guard.measure_drift(model, probe_dataset=probe_dataset)
print(drift.forgetting_score)      # 0 = nothing forgot, 1 = total reorganization
print(drift.protected_integrity)   # ~1.0 if the protected heads held

guard.save("model_guard.json")     # serialize; reload before task C, D, ...
guard = StructuralGuard.load("model_guard.json")
```

Protection is at the **head level** — some heads in a layer can be frozen while
others in the same layer keep learning. `protection_level` (0.0–1.0) sets the
fraction of the most critical heads to protect.

HuggingFace `Trainer`? Use the callback — it applies protection on
`train_begin`, measures drift on `train_end`:

```python
from sal import StructuralGuardCallback

guard = StructuralGuard.from_model(model, probe_dataset)
callback = StructuralGuardCallback(guard)
trainer = Trainer(model=model, callbacks=[callback])
trainer.train()
print(callback.drift_report.summary)
```

### DriftMonitor — measure structural forgetting after any fine-tuning

`DriftMonitor` quantifies how much a model's structure moved, guarded or not.
Snapshot before and after, then compare.

```python
from sal import DriftMonitor

monitor = DriftMonitor(model, probe_dataset)
monitor.snapshot("before_task_b")
trainer.train()
monitor.snapshot("after_task_b")

drift = monitor.compare("before_task_b", "after_task_b")
print(drift.summary)
print(drift.layer_drift)             # per-layer activation retention (1 = identical)
print(drift.classification_changes)  # layers whose fragility class flipped
drift.save("drift_report.json")
drift.save("drift_report.pdf")       # visual before/after comparison
```

Snapshots are keyed, so you can track drift across many sequential tasks and
compare any pair.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — 3-line SAL training on DistilBERT
- [`examples/standalone_fi.py`](examples/standalone_fi.py) — Fragility Index scan, no training
- [`examples/full_control.py`](examples/full_control.py) — manual config + standalone trainer
- [`examples/compare_with_without_sal.py`](examples/compare_with_without_sal.py) — SAL vs. baseline under compression

New here? Start with [docs/getting_started.md](docs/getting_started.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what's shipped, what's next, and how to request
features. v0.4.0 is a **robustness suite** answering whether SAL-trained models
also survive quantization — 39% of practitioners we polled compress that way.

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
