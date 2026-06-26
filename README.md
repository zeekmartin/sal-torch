# sal-torch

![CI](https://github.com/zeekmartin/sal-torch/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/sal-torch) ![Python](https://img.shields.io/pypi/pyversions/sal-torch) ![Downloads](https://pepy.tech/badge/sal-torch) ![License](https://img.shields.io/badge/license-BSL%201.1-blue)


**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

## Install

```bash
pip install sal-torch            # core
pip install sal-torch[hf]        # + HuggingFace Trainer
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

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — 3-line SAL training on DistilBERT
- [`examples/standalone_fi.py`](examples/standalone_fi.py) — Fragility Index scan, no training
- [`examples/full_control.py`](examples/full_control.py) — manual config + standalone trainer
- [`examples/compare_with_without_sal.py`](examples/compare_with_without_sal.py) — SAL vs. baseline under compression

New here? Start with [docs/getting_started.md](docs/getting_started.md).

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
