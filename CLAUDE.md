# CLAUDE.md — sal-torch

Context for future sessions working on this package.

## What this is

`sal-torch` is a **commercial** PyTorch package (BSL 1.1 license) that makes any
transformer **compression-resilient** through training-time head masking. It is
a **product**, not a research repo: clean API, clear errors, no research
artifacts.

Two independent, composable components:

- **SAL (Structurally Adaptive Learning)** — training-time random head masking
  that forces functional redistribution. *Perturbs.*
- **FI (Fragility Index)** — a post-hoc structural fragility diagnostic.
  *Measures.*

SAL and FI are **separate**. A user can use either without the other.

## Cardinal rules (do not violate)

1. **RANDOM is the default method.** Zero overhead, best-or-equal accuracy at
   validated scales (≤124M). Structured selection (FI/λ₂/liquefaction) all
   *underperform* random in ablations and are not shipped as defaults.
2. **SAL and FI are separate.** SAL perturbs, FI measures. Keep them decoupled.
3. **Do NOT expose spectral-framework internals** in code, docstrings, or error
   messages: no λ₂ / algebraic connectivity / Fiedler value, no TCGE, no
   simplicial hierarchy, no Laplacian theory. FI is a **"fragility score"**,
   full stop.
4. **All unit tests must pass on CPU** with the tiny model fixture
   (4 layers × 8 heads × 64 hidden). No GPU required for the unit suite.
5. The **scaffold API is the target**; the reference research code
   (`H:\Structural Awareness Loss\`) is the **source of truth for the
   mechanism**. Keep the clean API, use the validated implementation details.
6. Treat `pyproject.toml`, `LICENSE`, `README.md` as off-limits unless the task
   explicitly says otherwise. Local only — do not push to any remote.

## The validated mechanism (source of truth: reference research code)

These details were reconciled against the canonical reference implementation
(`sal_v5_extra.py`, the `random` branch — the validated default).

### Head masking (`sal/masker.py`)

- **Forward PRE-hook on the attention output projection** (`o_proj` /
  `out_proj` / `c_proj` / `dense`). The hook zeros per-head slices of the
  projection's **INPUT** — the concatenated per-head attention outputs, the
  only place where feature dims map cleanly onto heads. **Never** zero the
  projection *output* (post-projection features are fully mixed; zeroing them
  does not zero heads — this was a bug in the original scaffold).
- **Pruned heads accumulate.** During the prune window, randomly chosen heads
  are deactivated progressively and **stay** deactivated. This progressive
  structural damage is the causal mechanism — the model self-reorganizes to
  operate without the removed heads.
- After the window, the pruned set is **held** through end of training (not
  restored), so adaptation is baked into the weights.
- `schedule` controls how the pruned count grows: `"random"` (default,
  window-proportional ramp), `"progressive"` (one head per `prune_interval`),
  `"burst"` (full target at window open). All accumulate; all use random
  selection.

### Fragility Index (`sal/fi.py`)

- **FI = fraction of edges with zero triangle support** = `#{edges (i,j) with
  (A·A)[i,j] == 0} / #edges`. Range [0,1]. Low = triangulated/robust, high =
  fragile. This is the validated FI — **NOT** `1 − λ₂/λ_max` (the original
  scaffold used the Laplacian Fiedler value; that is a separate, underperforming
  research method and is forbidden by cardinal rule 3).
- Graph build: capture the **input** to each output projection via a pre-hook;
  per-head signature = mask-weighted seq-mean per sentence, flattened over
  `[sentences, head_dim]`, float64; Pearson similarity between heads; binary
  adjacency at a **fixed edge density (default 0.10 → 90th-percentile |sim|
  threshold)**.
- Layer classification (IMMUNE / BUFFER / CRITICAL) = relative FI change when a
  layer's heads are removed (defaults: <1% immune, ≥5% critical). This is a
  product API layered on top of the validated FI.

## Package layout

```
sal/
  __init__.py      version, set_license(), license_info()
  config.py        SALConfig (+ .auto(model)) — clean API target
  masker.py        HeadMasker — pre-hook, accumulation, hold-after-window
  callback.py      SALCallback (HF Trainer integration)
  trainer.py       SALTrainer (standalone PyTorch loop)
  fi.py            compute_fi (triangle fragility), extract_activation_graph, classify_layers
  scanner.py       FIScanner, FIMonitor
  arch_support.py  detect_architecture() — registry of supported archs
  license.py       Ed25519 offline license (signature verify still a stub)
  report.py        compliance report (stub — Phase 5)
tests/             CPU-only unit tests + conftest tiny model fixture
```

Planned-but-not-yet-implemented (per design doc): `plasticity.py`,
`guard.py`, `ExpertMasker`, PDF reports.

## Development phase status

- **Phase 1 (Reconciliation + Core SAL): DONE.** masker.py and fi.py aligned to
  the validated mechanism; 26 unit tests pass on CPU.
- **Phase 2 (Architecture Support): DONE.** Module/projection finding centralized
  in `arch_support.py` (relative to `model.base_model`, so masker + FI hook the
  same modules). Validated on real models — DistilBERT, GPT-2, ViT, BERT — both
  on CPU and on a Modal T4 GPU. SAL training pipeline (SALConfig.auto -> SALCallback
  -> HF Trainer -> FIScanner) verified end-to-end.
- Next: Phase 3 (FIScanner/FIMonitor hardening), Phase 4 (PlasticityScanner),
  Phase 5 (license signing + reports).

## Integration tests

CPU unit suite stays at **26 passed**; integration tests are marked
`@pytest.mark.integration` and **skipped by default** (need network + model
downloads):
```
python -m pytest                                   # 26 unit tests (CPU), integration skipped
python -m pytest --run-integration                 # + real-model arch + training tests
modal run scripts/modal_integration_test.py        # same logic on a Modal T4 GPU
```
The Modal image needs `pytest` + `accelerate>=1.1.0` (Trainer dep). On
Windows, run Modal with `PYTHONUTF8=1` so the CLI can print its build glyphs.
Transformers 5.x silently ignores `head_mask` — confirming why SAL hooks the
projection directly.

## Reference docs

- `SAL_TORCH_DESIGN.md` — architecture, target API, package structure, roadmap.
  NOTE: §D5 mentions λ₂; that is superseded by cardinal rule 3 + the reference —
  FI is the triangle-fragility score, not λ₂.
- `SAL_ROADMAP_V3.md` — ablation results, validated claims (random > structured
  selection; gains come from progressive training-time sparsification).
- `H:\Structural Awareness Loss\` — private research reference. Canonical
  mechanism: `sal_v5_extra.py` (`random` branch). Not shipped.

## Testing

```
python -m pytest -q       # CPU only, ~30s, must be all-green
```
Fixtures in `tests/conftest.py`: `tiny_model` (custom 4×8×64 GPT-2-like),
`tiny_config`, `probe_data`.
