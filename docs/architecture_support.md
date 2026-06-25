# Architecture Support

`SALConfig.auto(model)` detects the architecture from the model's
HuggingFace config and locates its attention output projections automatically.
Both SAL (masking) and FI (diagnostic) use the same detection, so they always
operate on the same heads.

## Supported architectures

`layers` / `heads` are auto-detected per checkpoint; the values below are for a
common variant. **Tested** = covered by integration tests on real models (CPU +
GPU). **Supported** = auto-detected via the registry; not yet integration-tested.

| `model_type` | Example model | Layers | Heads | Status |
|---|---|---|---|---|
| `distilbert` | distilbert-base-uncased | 6 | 12 | ✅ Tested |
| `gpt2` | gpt2 | 12 | 12 | ✅ Tested |
| `bert` | bert-base-uncased | 12 | 12 | ✅ Tested |
| `vit` | google/vit-base-patch16-224 | 12 | 12 | ✅ Tested |
| `roberta` | roberta-base | 12 | 12 | Supported |
| `llama` | Llama / TinyLlama | 32 | 32 | Supported |
| `mistral` | Mistral-7B | 32 | 32 | Supported |
| `qwen2` | Qwen2 | — | — | Supported |
| `phi` | Phi-2 | — | — | Supported |
| `phi3` | Phi-3 | — | — | Supported |
| `gemma` | Gemma | — | — | Supported |
| `gemma2` | Gemma 2 | — | — | Supported |

```python
from sal import SALConfig
config = SALConfig.auto(model)
print(config.num_layers, config.num_heads_per_layer)
```

If an architecture isn't in the registry, `SALConfig.auto` raises a clear
`SALArchitectureError` rather than guessing.

## Adding an unsupported architecture manually

You don't need registry support — just describe the model with an explicit
`SALConfig`. You need three things:

- `num_layers`, `num_heads_per_layer` — usually on `model.config`.
- `attention_pattern` — a dotted path to each layer's attention module, with
  `{}` where the layer index goes, written **relative to `model.base_model`**
  (the part under the task head). SAL pulls the output projection
  (`o_proj` / `out_proj` / `c_proj` / `out_lin` / `.output.dense`) out of it.

```python
from sal import SALConfig, SALCallback

config = SALConfig(
    num_layers=model.config.num_hidden_layers,
    num_heads_per_layer=model.config.num_attention_heads,
    attention_pattern="encoder.layer.{}.attention",   # e.g. a BERT-like model
    prune_fraction=0.33,
)
trainer = Trainer(model=model, callbacks=[SALCallback(config)], ...)
```

Patterns used by the built-in registry (all relative to `model.base_model`):

| Family | `attention_pattern` |
|---|---|
| LLaMA / Mistral / Phi / Gemma / Qwen | `layers.{}.self_attn` |
| GPT-2 | `h.{}.attn` |
| BERT / RoBERTa | `encoder.layer.{}.attention` |
| DistilBERT | `transformer.layer.{}.attention` |
| ViT | `layers.{}.attention` |

To verify your pattern, install the masker and check the hook count matches your
layer count:

```python
from sal.masker import HeadMasker
m = HeadMasker(model, config)
m.install()
print(len(m._hooks), "==", config.num_layers)
m.remove()
```

If detection or hooking fails for an architecture you need, email
[contact@cognitive-engineering.dev](mailto:contact@cognitive-engineering.dev).
