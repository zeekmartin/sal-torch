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
  plasticity.py    PlasticityScanner (routing/CKA/MI), PlasticityMap, Recommendation
  compare.py       sal.compare() — SAL vs magnitude/random post-hoc baselines
  slicing.py       slice_heads() — physically remove heads (uniform per layer, MHA only)
  quantize.py      quantize(), quantize_info() — bitsandbytes NF4/LLM.int8, torch.ao INT8
  pipeline.py      CompressionPipeline — scan → sal_train → compress → validate → export
  evaluation.py    linear_probe, knn_accuracy, cka_similarity, measure_latency, compression_report
  visualization.py feature maps, compression-impact charts, attention maps (matplotlib)
  robustness.py    RobustnessTest, RobustnessReport, robustness_compare — compression survival
  guard.py         StructuralGuard, StructuralGuardCallback — head-level continual learning
  drift.py         DriftMonitor, DriftReport, StructuralSnapshot — structural forgetting
  visualize.py     PDF reports (FI heatmap, plasticity, comparison, guard, drift)
  arch_support.py  detect_architecture(), output/QKV projection finding
  license.py       Ed25519 offline license verification (PyNaCl, embedded pubkey)
  report.py        compliance report (stub — Phase 5)
tests/             CPU-only unit tests + conftest tiny model fixture
```

Planned-but-not-yet-implemented (per design doc): `plasticity.py` axes beyond
the three shipped, `ExpertMasker`.

## Development phase status

- **Phase 1 (Reconciliation + Core SAL): DONE.** masker.py and fi.py aligned to
  the validated mechanism; 26 unit tests pass on CPU.
- **Phase 2 (Architecture Support): DONE.** Module/projection finding centralized
  in `arch_support.py` (relative to `model.base_model`, so masker + FI hook the
  same modules). Validated on real models — DistilBERT, GPT-2, ViT, BERT — both
  on CPU and on a Modal T4 GPU. SAL training pipeline (SALConfig.auto -> SALCallback
  -> HF Trainer -> FIScanner) verified end-to-end.
- **Phase 4 (Plasticity + compare + reports): DONE (v0.2.0).** PlasticityScanner
  (3-axis routing/CKA/MI absorption map), sal.compare(), visual PDF reports.
- **v0.3.0 (Continual learning): DONE.** `StructuralGuard` protects critical
  attention heads (HUB + SATURATED layers + low-redundancy ELASTIC heads) during
  fine-tuning by zeroing their gradients with backward hooks — head-level
  granularity, composes with SAL, serializable. `DriftMonitor` measures
  structural forgetting (forgetting_score, FI delta, per-layer CKA retention,
  protected_integrity, classification changes). Protection ranking is driven by
  PlasticityScanner (routing/CKA/MI) + per-head redundancy — **no spectral
  internals**. 83 unit tests pass on CPU; validated guarded-vs-unguarded on Modal
  T4 (DistilBERT SST-2 → MNLI). `arch_support.get_qkv_projections()` added for
  head-level Q/K/V/O weight slicing.
- **v0.4.0 (Robustness suite): RELEASED (2026-07-29).** `RobustnessTest` scores a model
  under INT8, INT4, head pruning, and inference-time FFN neuron dropout;
  `robustness_compare()` puts a SAL-trained model head-to-head with a standard
  one, each scored against its own clean baseline. INT4 prefers bitsandbytes NF4
  (`[quant]` extra) and falls back to a simulated per-channel INT4 round-trip,
  recording which backend ran. Head pruning reuses `HeadMasker`; neuron dropout
  targets FFN *expansion* projections (out_features > in_features) with a fixed
  fault pattern per trial. 108 unit tests pass on CPU. Motivated by a community
  poll (33 votes): 39% quantize, 24% magnitude-prune, 21% distill.
- **v0.5.0 (CompressionPipeline): RELEASED (2026-08-12).** `slice_heads()`,
  `quantize()`, `CompressionPipeline`; five-seed GPT-2 Medium validation (SAL
  +2.45pp at 33% head pruning on 5/5 seeds; INT4 alone does not replicate).
- **v0.5.1 (Self-supervised models): RELEASED (2026-09-17).** Loss-agnostic
  `SALTrainer(train_step=...)`, `sal.evaluation` / `sal.visualization`,
  `ijepa`/`dinov2` support, `slice_heads()` fixed for ViT-family encoders,
  `scripts/run_jepa_sal.py` (`--control --slice --prune-ratios --model`).
  I-JEPA ViT-H/14: SAL wins 35/36 over 3 seeds at 33/50% (+0.4 to +1.9pp); no
  effect at 70%. DINOv2 ViT-L/14 (1 seed): 13/18. Slicing 50% of heads: 1.1x GPU /
  1.2x CPU (attention is ~1/3 of a ViT block). 271 unit tests pass on CPU.
- **v0.6.0 (in progress): validated on LUMI (AMD MI250X / ROCm 6.2).** Phase 1
  reproduced the published structural scans on ROCm exactly (I-JEPA ViT-H/14
  FI=0.003975 vs 0.0040 on NVIDIA, same 1/4/27 layer split). exp04 then
  re-measured SAL+QAT on ViT-B/16 / ImageNet-100 at **15 seeds and 20 random
  head selections per cell**, which changed three things. Full trail in
  `results/lumi_v060/PHASE1.md`, `PHASE2.md`, `PHASE2A.md`, `PHASE2B.md`.
  - **The headline is +14.5pp ± 4.3** at 50% pruning + INT4 (95% CI
    [+12.4, +16.7], 15/15 seeds), **not the +22pp** published from 3 seeds, and
    QAT adds **+2.4pp ± 0.9** (15/15). Those three seeds were the most
    favourable of five.
  - **`--prune-seeds 3` is too few for any published cell.** Head-selection
    sampling, not the training seed, was the dominant noise term: sd 11.07 → 4.28
    on the same checkpoints. It is also the cheapest to remove (evaluation only,
    no retraining). Two readings in `PHASE2.md` — "unstable baseline" and "SAL
    actively harmful on seed 2024" — were artefacts of 3 selections and are
    retracted in `PHASE2A.md`. The baseline in fact spans 1.6pp.
  - **Dose-response is monotone**: gain at P50+I4 rises +6.2 → +20.8pp as
    `prune_fraction` goes 0.1 → 0.5, costing 0.3-1.5pp of clean accuracy. Best
    when the training ratio matches the deployment ratio, so the **0.33 default
    under-delivers against a 50% target** (default not yet changed — product
    decision).
  - **Scope**: robust at ≥50% pruning; **weak and inconsistent at 33% on
    ViT-B/16/ImageNet-100**, though 33% is SAL's best row on GPT-2 Medium and
    ViT/CIFAR-10. Task difficulty is the obvious candidate and is confounded
    with architecture and dataset; nothing measured separates them.
  - Control arms `drophead` and `nonaccum` are implemented in
    `scripts/run_sal_qat_prototype.py` (opt-in via `--variants`, default
    unchanged) but **not yet run**. They test whether the gain is ordinary
    regularisation and whether accumulation is load-bearing.
  - MLP pruning and the compile/ONNX paths live in **sal-torch-pro**
    (`sal_pro.mlp`, `sal_pro.deploy`) — mechanism unit-tested, accuracy and
    speedup not yet measured.
- LUMI notes: compute nodes have no internet, so pre-fetch into
  `HF_HOME=/scratch/project_465003656/sal-torch/hf_cache` from a login node and
  run with `HF_HUB_OFFLINE=1`. Pin `OMP_NUM_THREADS` to `SLURM_CPUS_PER_TASK`.
  The first training step costs ~60s of MIOpen kernel compilation.
  I-JEPA needs transformers >= 4.47, so it runs in a second env `.venv-tf47`
  (`SAL_VENV=.venv-tf47`); the default `.venv` stays at 4.46.0.
- **Set `MIOPEN_DISABLE_CACHE=1`** (done in `experiments/run_exp.sh`). Without
  it, runs intermittently die with `RuntimeError: miopenStatusInternalError` on
  the **first `conv2d`** — on a ViT that is the patch embedding, before any
  sal-torch code runs. Only convolutions are hit, because GEMMs go through
  rocBLAS. MIOpen's kernel database lives at `/tmp/gfx90a6e.ukdb`, which is
  shared state whose validity varies by node and over time (nid005029 failed,
  then ran the same code fine an hour later). **Do not "fix" this by pointing
  `MIOPEN_USER_DB_PATH` at a fresh per-job directory** — measured on MI250X,
  that fails *every* time on `/tmp`, `/dev/shm` and Lustre alike even though all
  are writable, because MIOpen cannot build a usable database from an empty
  directory. See `logs/*-mioloc.out` for the comparison.
- **Load GPT-2 with `attn_implementation="eager"` on ROCm.** Under the default
  SDPA path, GPT-2 on MI250X returns *finite logits* but corrupted gradients:
  measured grad norms of 7.7e13, 3.6e18 and `inf` inside the first 40 steps,
  against 15-140 under eager on the same batches. `clip_grad_norm_` then scales
  by `1/inf = 0`, so nothing is learned and the loss eventually goes NaN. The
  first exp13 GPT-2 runs reported `losses [nan, nan]` on all three arms and
  49.08% accuracy on every cell — 428/872, the SST-2 majority class. Because
  only the backward pass is affected, a forward-only check passes: masking was
  separately verified to move GPT-2 logits by 4.2 with every head off. ViT and
  DINOv2 train correctly under SDPA; this is GPT-2-specific so far. See
  `logs/*-diag-gpt2nan.out`.
- **Fail loudly on a non-finite loss.** A diverged run still yields an accuracy
  (the majority-class rate) and writing that to JSON is worse than crashing;
  `exp13` now raises on the first non-finite loss.
- **Write measurements before anything that presents them.** Twenty expF jobs
  died inside `summary_table()` after ~35 min of training each and left nothing
  behind, because the table was written before the JSON — 11.7 GPU-hours for
  nothing. The prototype now saves the JSON first and wraps table/figures in a
  `try` that only warns. Related trap: `skip_int4` is tied to `--smoke`, so a
  smoke run never populates the `prune50_int4` key cell and never exercises the
  code that reads it. Gate a batch with a *small non-smoke* run instead
  (`--epochs 1 --n-train 2000 --prune-seeds 2`), which costs ~4 min and covers
  the real table path.
- Smoke-test one job before submitting a batch, and submit the batch with
  `sbatch --dependency=afterok:<smoke-id>`. This caught the MIOpen failure with
  20 jobs held at `DependencyNeverSatisfied` and zero GPU-hours burned. Cancel
  held jobs by explicit job id, never `scancel --state=PENDING -u <user>`.
- Next: finish v0.6.0 — control arms (drophead/nonaccum), MLP prune sweep and
  joint head+MLP for the 2x real speedup target; v0.7.0 topology-guided
  distillation; Phase 5 (license signing + compliance reports).
  See `ROADMAP.md` (public).

## Integration tests

The CPU unit suite must stay all-green (**271 passed** as of v0.5.1, plus the
8 integration tests skipped); integration tests are marked
`@pytest.mark.integration` and **skipped by default** (need network + model
downloads):
```
python -m pytest                                   # CPU unit tests, integration skipped
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
python -m pytest -q       # CPU only, ~6-8 min, must be all-green
```
Fixtures in `tests/conftest.py`: `tiny_model` (custom 4×8×64 GPT-2-like),
`tiny_config`, `probe_data`. Run the suite with `CUDA_VISIBLE_DEVICES=` on a GPU
box: `tests/test_run_jepa_script.py` fakes a CPU model and fails under CUDA
autocast otherwise.
