# Changelog

## 0.4.0.dev0 — Robustness suite (in development)

- **`RobustnessTest` / `RobustnessReport`** — run a model through INT8, INT4,
  head pruning, and inference-time FFN neuron dropout, and report baseline /
  after / delta / survived per method plus an aggregate `robustness_score`.
  INT4 uses bitsandbytes NF4 when available and falls back to a simulated
  per-channel INT4 round-trip otherwise.
- **`robustness_compare()`** — head-to-head resilience of a SAL-trained model
  against a standard one, scored against each model's own clean baseline.
- Robustness bar chart, retention radar, and comparison PDF in `sal.visualize`.
- New `[quant]` extra (bitsandbytes) for real 4-bit quantization.
- First validation (DistilBERT/SST-2 on a T4, single seed): SAL roughly halves
  the damage from head pruning at 33% and 50%, but shows **no measurable
  advantage under quantization** — every INT8/INT4/dropout margin sits inside
  the noise floor of the eval set. Documented in `ROADMAP.md` and `README.md`.
- `StructuralGuard.release()` no longer takes a `model` argument — the gradient
  hooks live on the parameter tensors, so releasing them needs no model
  reference. Call `guard.release()` instead of `guard.release(model)`.
- `SALTrainer` and `ScanResult` are now exported from the `sal` namespace.
- Dropped unused dependencies: `scipy` (core) and `peft` (the `hf` extra).

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
