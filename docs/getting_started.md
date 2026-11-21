# Getting Started

`sal-torch` is a **measured compression tool**: diagnose, route, prune,
recover, then verify against a matched control. It also ships a structural
**Fragility Index** (FI) diagnostic, and a training-time head-masking method
(SAL, and DropHead since v0.7.0). All three are independent — use any one on
its own.

> **Read this before picking a path.** If you can fine-tune after pruning,
> **do that, and skip head training entirely**: prune at random, recover for
> two epochs, and verify against a control trained for the same two epochs.
> At a fixed head set, SAL's weights are worth **+0.05pp** and DropHead's
> **+0.07pp** as a starting point for that fine-tune, while the fine-tune
> itself is worth up to **68pp** (exp24, ViT-B/16 on ImageNet-1k, 3 seeds,
> `results/lumi_v060/EXP24_SAL_RECOVERY.md`). Head training is the branch you
> take when **no post-pruning fine-tune is possible**, and there the method is
> DropHead. The shortest version of the recommended path:
>
> ```python
> from sal import CompressionPipeline
>
> pipe = CompressionPipeline(model, eval_dataset, metric="accuracy")
> pipe.probe()                                     # measure, do not guess
> pipe.compress(pruning=0.50, slice_heads=True)    # random selection, the default
> pipe.recover(train_dataset, epochs=2, domain="same as eval")
> print(pipe.validate().table)
> ```
>
> The SAL training run below is the other branch. It is supported and it is
> not the default recommendation.

## Install

```bash
pip install sal-torch            # core (SAL + FI)
pip install sal-torch[hf]        # + HuggingFace Trainer integration
pip install sal-torch[all]       # everything (HF, reports, license, dev)
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

## Your first SAL training run

Add two objects to a normal HuggingFace training loop — `SALConfig.auto`
inspects the model, `SALCallback` does the head masking:

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sal import SALConfig, SALCallback

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

config = SALConfig.auto(model, prune_fraction=0.33)   # silence 33% of heads
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="out", num_train_epochs=1),
    train_dataset=train_dataset,
    callbacks=[SALCallback(config)],
)
trainer.train()
```

During training, heads are progressively silenced and held off. The model learns
to redistribute their work, so it stays accurate when those heads are later
removed for deployment.

Expected console output (abridged):

```
SAL HeadMasker installed: 6 layers, 72 heads
{'loss': 0.55, ...}
{'loss': 0.09, ...}
```

A full runnable version is in [`examples/quickstart.py`](../examples/quickstart.py).

## Your first FI scan

FI needs no training — point it at any model and a few batches of text:

```python
from transformers import AutoModelForSequenceClassification
from sal import FIScanner

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

result = FIScanner(model, probe_dataset=probe_batches).scan()
print(result.fi_score)        # e.g. 0.34   (0 = robust, 1 = fragile)
print(result.summary)         # FI=0.34 | 1 immune, 1 buffer, 4 critical
print(result.critical_layers) # [0, 1, 2, 3]
result.save("fi_report.json")
```

`probe_batches` is just a list of dicts like
`{"input_ids": ..., "attention_mask": ...}`. Expected output:

```
0.3359
FI=0.3359 | 1 immune, 1 buffer, 4 critical
[0, 1, 2, 3]
```

A full runnable version is in [`examples/standalone_fi.py`](../examples/standalone_fi.py).

## Next steps

- [How SAL works](how_sal_works.md) — the mechanism, in plain language.
- [Architecture support](architecture_support.md) — supported models and how to add one.
- [Licensing](licensing.md) — Community vs. Professional vs. Enterprise.
- More examples: [`examples/full_control.py`](../examples/full_control.py),
  [`examples/compare_with_without_sal.py`](../examples/compare_with_without_sal.py).
