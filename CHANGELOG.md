# Changelog

## 0.5.0.dev0 — CompressionPipeline (in development)

Turns the v0.4.0 recipe into one object, and makes the savings real.

- **`slice_heads()`** — physically removes attention heads from the weights.
  Masking makes a head *behave* as if it were gone; slicing makes the model
  smaller. Narrows Q/K/V by rows and the output projection by the matching
  columns, updates head bookkeeping, and returns a model that runs with no
  hooks and without sal-torch installed. Handles separate and fused Q/K/V and
  both `nn.Linear` and GPT-2 `Conv1D`. Refuses uneven per-layer removal and
  grouped-query attention rather than producing a subtly wrong model.
- **`quantize()` / `quantize_info()`** — one call over bitsandbytes (LLM.int8(),
  NF4) and `torch.ao` (dynamic INT8), with backend auto-selection, GPT-2
  `Conv1D` conversion, and the output head never quantized.
- **`CompressionPipeline`** — scan → sal_train → compress → validate → export,
  measuring size and accuracy at every stage. **Refuses LoRA/QLoRA models**
  with an explanation, warns below 100M parameters, and supports an
  `accuracy_floor` that stops the run rather than returning a small broken
  model. `export()` reloads what it wrote and reports whether the round trip
  actually held.
- Compression waterfall and per-stage quality charts in `sal.visualize`.
- **`compress()` gains `strategy=`, defaulting to `"random"`** (`random` /
  `magnitude` / `fi_guided`). Measured on GPT-2 Medium at a matched 120-head
  budget, `random` retains 99.3% of accuracy against `magnitude`'s 96.3%
  (standard) and 93.7% (SAL-trained) — better for both arms, so the default is
  not SAL-specific. `fi_guided` retains ~84% and is not recommended at this
  budget. The earlier `magnitude` default was responsible for most of the
  v0.5.0 compression regression.
- `sal_train()` now clips gradients (1.0 by default), matching `SALTrainer`.
  Silencing a third of the heads makes gradients spikier than ordinary
  fine-tuning; unclipped, GPT-2 Medium scored 0.7676 where clipped it scores
  0.8965.
- Validated end to end on GPT-2 Medium / SST-2: 1419MB → 347MB (4.1x), export
  reloads at exactly the measured accuracy. **A no-SAL control finished ahead
  under compression** (0.8633 vs 0.8398) — see `ROADMAP.md`; head-selection
  strategy is the leading suspect and is being measured next.

## 0.4.0 (2026-07-29) — Robustness suite, full fine-tuning validation

Models that prove their resilience.

- **`RobustnessTest` / `RobustnessReport`** — run a model through INT8, INT4,
  head pruning, and inference-time FFN neuron dropout, and report baseline /
  after / delta / survived per method plus an aggregate `robustness_score`.
  INT4 uses bitsandbytes NF4 when available and falls back to a simulated
  per-channel INT4 round-trip otherwise.
- **`robustness_compare()`** — head-to-head resilience of a SAL-trained model
  against a standard one, scored against each model's own clean baseline.
- Robustness bar chart, retention radar, and comparison PDF in `sal.visualize`.
- New `[quant]` extra (bitsandbytes) for real 4-bit quantization.
- Validated over four runs (single seed each; see `ROADMAP.md` for the full
  trail, losses included): **SAL requires full fine-tuning.** Fully fine-tuned,
  a SAL-trained GPT-2 Medium wins all seven compression variants including
  INT4, and `SAL/int4` beats the *uncompressed* standard model at a quarter of
  the size. Under LoRA r=16 the identical setup loses four of six variants and
  costs 3.1 points of clean accuracy — the adapters are too small to absorb
  what the head masking removes. LoRA/QLoRA is now documented as not
  recommended.
- `HeadMasker` masks follow the model's device and dtype, so SAL works on
  half-precision models. It previously raised on the first forward pass of any
  bf16/fp16 model.
- `StructuralGuard.release()` no longer takes a `model` argument — the gradient
  hooks live on the parameter tensors, so releasing them needs no model
  reference. Call `guard.release()` instead of `guard.release(model)`.
- `SALTrainer` and `ScanResult` are now exported from the `sal` namespace.
- Dropped unused dependencies: `scipy` (core) and `peft` (the `hf` extra).
- 110 unit tests pass on CPU (was 83).

**Headline result.** Fully fine-tuned on GPT-2 Medium / SST-2, SAL wins all
seven compression variants — INT8, INT4, head pruning at 33% and 50%, and both
combined recipes — at no cost to clean accuracy (+0.39pp). `SAL/int4` scores
0.8926 at 361.9MB against the *uncompressed* standard model's 0.8848 at
1419.3MB: higher accuracy at a quarter of the size, and the only point on the
accuracy-vs-size frontier.

**Known limitation.** Under LoRA/QLoRA the identical setup loses. SAL works by
letting the model reorganize around silenced heads, and adapters freeze the
weights that would do the reorganizing. Not recommended — see the "When to use
SAL" table in the README.

## 0.3.0 (2026-06-27)

Continual learning without replay buffers.

- **`StructuralGuard`** — protects critical attention heads during fine-tuning on
  a new task by zeroing their gradients with backward hooks. Protection is at
  head-level granularity (some heads in a layer can be frozen while others keep
  learning), driven by the plasticity absorption map plus a per-head redundancy
  score. Composes with SAL and serializes to JSON.
- **`StructuralGuardCallback`** — applies the guard over a HuggingFace `Trainer`
  run and stores the resulting drift report.
- **`DriftMonitor` / `DriftReport` / `StructuralSnapshot`** — measure structural
  forgetting after any fine-tuning: `forgetting_score`, FI delta, per-layer CKA
  retention, `protected_integrity`, and layer classification changes. Snapshots
  are keyed, so drift can be tracked across many sequential tasks.
- Guard and drift PDF reports in `sal.visualize`.
- `arch_support.get_qkv_projections()` for head-level Q/K/V/O weight slicing
  (separate and fused layouts).
- 83 CPU unit tests. Validated guarded-vs-unguarded on a Modal T4
  (DistilBERT SST-2 → MNLI).

## 0.2.0 (2026-06-26)

Know your model before you touch it.

- **`PlasticityScanner` / `PlasticityMap` / `Recommendation`** — three-axis
  absorption map (routing entropy, inter-layer CKA, intra-layer MI proxy) that
  labels each layer `ELASTIC`, `SATURATED`, or `HUB`, and turns it into concrete
  prune / never-touch head lists.
- **`sal.compare()`** — benchmark SAL against post-hoc `magnitude` and
  `random_posthoc` baselines at a matched compression level, with a plugin
  registry for custom methods.
- **Visual reports** (`sal-torch[reports]`) — one-page PDFs for FI scans,
  plasticity maps, and method comparisons.
- Ed25519 offline license tooling and embedded production public key.

## 0.1.0-dev (2026-06-25)

- Initial scaffold
- Core SAL: HeadMasker, SALConfig, SALCallback, SALTrainer
- FI: activation graph extraction, Fragility Index, layer classification
- FIScanner: one-shot structural analysis
- Architecture auto-detection: 12 architectures
- License system: Ed25519 offline verification
