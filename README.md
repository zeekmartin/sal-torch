# sal-torch

**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

```python
from sal import SALConfig, SALCallback

config = SALConfig.auto(model)
trainer = Trainer(model=model, callbacks=[SALCallback(config)])
trainer.train()
```

Three lines. Any transformer. Compression-resilient.

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
