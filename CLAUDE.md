# CLAUDE.md — sal-torch

Context for future sessions working on this package.

## What this is

`sal-torch` is a **commercial** PyTorch package (BSL 1.1 license) that makes any
transformer **compression-resilient** through training-time head masking. It is
a **product**, not a research repo: clean API, clear errors, no research
artifacts.

Three independent, composable components:

- **SAL (Structurally Adaptive Learning)** — training-time random head masking,
  accumulate-and-hold. *Perturbs.* **Specific**: its resilience applies to the
  one head set it trained for and nowhere else (exp25).
- **DropHead** — the same quota, mask resampled every step. *Perturbs.*
  **General**: its resilience holds on any head set of that size, including
  SAL's own (exp27). Started life as SAL's control arm and outperforms it on the
  product's actual promise.
- **FI (Fragility Index)** — a post-hoc structural fragility diagnostic.
  *Measures.* Diagnosis and certification only — **not** a router and not a head
  selector (exp17, PHASE2E).

They are **separate**. A user can use any one without the others. Which head
method to use is decided by the **probe**, not by FI: see the routing table in
`README.md` (>30pp drop → SAL, 10-30pp → either, <10pp → skip head training).

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
7. **Quote only numbers that are in `results/lumi_v060/*.md` or
   `sal-torch-pro/results/*.md`.** The ImageNet-1k series (exp23-exp28,
   3 seeds, full 50 000-image validation split) **supersedes** every
   ImageNet-100 figure for the same cell. Several headline claims published
   before it are retracted — see "What the ImageNet-1k series retracted" below.
   Do not reinstate a retracted number from an older doc, a commit message or
   the README.

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
- **v0.6.0 / v0.6.1: validated on LUMI (AMD MI250X / ROCm 6.2).** Phase 1
  reproduced the published structural scans on ROCm exactly (I-JEPA ViT-H/14
  FI=0.003975 vs 0.0040 on NVIDIA, same 1/4/27 layer split). exp04 then
  re-measured SAL+QAT on ViT-B/16 / ImageNet-100 at **15 seeds and 20 random
  head selections per cell**. Full trail in `results/lumi_v060/PHASE1.md`
  through `PHASE2H.md`.
  - **`--prune-seeds 3` is too few for any published cell.** Head-selection
    sampling, not the training seed, was the dominant noise term: sd 11.07 → 4.28
    on the same checkpoints. It is also the cheapest to remove (evaluation only,
    no retraining). Two readings in `PHASE2.md` — "unstable baseline" and "SAL
    actively harmful on seed 2024" — were artefacts of 3 selections and are
    retracted in `PHASE2A.md`. The baseline in fact spans 1.6pp.
  - **Dose-response is monotone** on ImageNet-100: gain at P50+I4 rises
    +6.2 → +20.8pp as `prune_fraction` goes 0.1 → 0.5. Best when the training
    ratio matches the deployment ratio. The clean-accuracy cost quoted with it
    (0.3-1.5pp) **does not survive three seeds — see the retractions below.**
  - **Scope**: robust at ≥50% pruning; weak and inconsistent at 33% on
    ViT-B/16/ImageNet-100, though 33% is SAL's best row on GPT-2 Medium and
    ViT/CIFAR-10. Task difficulty is the obvious candidate and is confounded
    with architecture and dataset; nothing measured separates them.
  - **The probe routes compression; FI does not.** exp17 (`PHASE2H.md`) tested
    whether FI read off a pretrained checkpoint predicts the pruning regime,
    across **six architectures** (DeiT-small, BERT-base, ViT-B/16,
    DINOv2-large, I-JEPA ViT-H/14, phi-2). **It fails at both ends**:
    BERT-base has a *higher* FI than ViT-B/16 and loses 5.15pp where ViT-B/16
    loses 62.21; DINOv2-large has an FI 25x *lower* than DeiT-small and
    collapses to 0.96% — chance on a 100-class task — on the same dataset,
    protocol, class count and seed count. Parameter count does not rescue it
    either (drops run −76, −62, −5, −94, −19, −4 by size). What does order the
    table is **the baseline's own drop under pruning**, measured directly by a
    probe: >30pp → SAL, 10-30pp → either, <10pp → skip head training. FI stays
    in the product for diagnosis and certification, **not** as a router.
  - **FI-weighted head *selection* loses to uniform random on 0/10 seeds**, at
    both mask ratios: −5.02pp (CI [−5.64, −4.40]) at 0.3 and −4.77pp
    (CI [−5.15, −4.39]) at 0.5 (`PHASE2E.md`). This is cardinal rule 1,
    measured at scale.

### What the ImageNet-1k series retracted (exp23-exp28)

**ViT-B/16, `evanarlian/imagenet_1k_resized_256`** (the gated `ILSVRC/imagenet-1k`
is unavailable; this is the open mirror — canonical 1000 classes in canonical
order, 256px shortest side; the pretrained checkpoint reads 78.44% on it against
the 80.3% on its card, the difference being resize + centre crop). **Three seeds
(42, 123, 456), full 50 000-image validation split, autocast bfloat16.** This is
the most rigorous evidence the project has and it overrides ImageNet-100 for
every shared cell. Docs: `PHASE3_IMAGENET1K.md`, `EXP24_SAL_RECOVERY.md`,
`EXP25_ALIGNED.md`, `EXP27_DROPHEAD.md`, `EXP28_JEPA_1B.md`, and
`sal-torch-pro/results/EXP26_MLP.md`.

- **SAL's headline replicates — and loses to its own control.** SAL over its
  own baseline at P50+INT4 is **+14.57pp** (30.63 vs 16.06), 3/3 seeds, which
  matches the +14.5 ± 4.3 published from ImageNet-100. On the **same cell
  DropHead reads 73.29% against the baseline's 16.06% — +57.23pp**, and beats
  SAL by **+42.66pp**. SAL's dense accuracy (66.90%) is below DropHead's
  *P50-pruned* accuracy (72.54%). **The +22pp figure published from 3 seeds is
  retracted** (`PHASE2A.md`); so is any framing of accumulate-and-hold as the
  causal mechanism — the non-accumulating control is simply better.
- **SAL's clean-accuracy cost is not 0.3-1.5pp.** Across three seeds SAL's
  clean accuracy is **71.53 / 67.57 / 61.61%** — a 9.92pp spread — against
  DropHead's 77.60 / 77.32 / 77.71% (0.39pp). Across all six rows of exp27 SAL's
  seed-to-seed sd spans **0.50-17.55pp** and DropHead's **0.02-0.92pp**, a 15x+
  gap; one SAL row reads 60.15 / 51.83 / 26.46%. **A method whose accuracy moves
  thirty points with the training seed cannot carry a sub-2pp cost claim.**
  Note also that **SAL's dense row is not its operating point**: deployed on its
  own set it reads 76.51%, i.e. **−3.93pp** against the matched baseline's
  80.44% at a 16.5% smaller model (exp25). Quote that, not the dense row.
- **exp25 — SAL's adaptation is real and entirely set-specific.** On its own
  trained-for set: **76.51%** (sd 0.50). On sets it never saw: **34.52%**. The
  same set removed from the matched baseline checkpoint: **20.33%**, against
  20.38% for random sets — a **0.05pp** difference. So SAL's head set is worth
  ~nothing to a model that did not train for it and **+56.18pp** to the one that
  did. Both the set *and* the ratio must be fixed at training time: at P33 (a
  subset of its own set, so an architecture it never trained for) SAL reads
  72.76%, below magnitude+recovery's 78.85%.
- **exp24 — SAL is NOT a pre-conditioner.** The PRIORITY 0 experiment, 117.6
  GPU-hours. At a *fixed* head set, neither SAL's nor DropHead's weights beat the
  pretrained ones after five epochs of recovery: **+0.05pp** and **+0.07pp** at
  P50+INT4. What survives is a head-*selection* signal worth **+0.60pp** at
  P50+INT4 (3/3 seeds) — and **nothing at P33**, where the sign flips (−0.17,
  −0.30). Buying it costs ~6 GPU-hours of SAL training per seed and commits the
  deployment to one head set chosen at training time.
- **exp27 — DropHead validated on ImageNet-1k, and it is general.** Against the
  baseline checkpoint on the **identical** head set: **+63.47pp** at P50
  (+63.43 INT4), **+26.45pp** at P33, and **+54.70pp** on *SAL's own* aligned
  set — 3/3 seeds, every cell. **Pruning 33% of heads is free for DropHead**:
  77.69% against its own dense 77.54% (**+0.14pp**), and 76.12 vs 76.09 at INT4.
  At P50 it costs 5.01pp. SAL on an external fixed set: +8.14pp (P50),
  +0.89pp (P33). So **SAL's adaptation is worth 56pp on one set and 0-8pp
  anywhere else; DropHead's is worth 26-63pp everywhere.**
- **exp26 (sal-torch-pro) — MLP pruning is far cheaper than head pruning, and
  2x is not reachable.** 22 jobs, 97.2 GPU-hours. Measured against the *matched*
  control (1a's baseline, 80.44% / 79.01% INT4 — same 5 epochs, nothing pruned),
  at comparable size: **MLP-25 costs −0.63pp at −16.4% params; head-50 costs
  −3.17pp at −16.5%** — heads are **5x more expensive per unit of size removed**.
  At matched −32.7%, all-MLP beats head+MLP by **1.55pp** (77.75 vs 76.20).
  **Max measured speedup is 1.74x at −6.82pp** (head50+mlp50, **batch 32**).
  **At batch 1 every arm is 1.00x** (8.09-8.37ms vs 8.16ms dense) because a
  ViT-B/16 there is kernel-launch bound — *any* latency claim must state batch 32.
  **The 2x joint target is not achievable by magnitude pruning on this model.**
- **exp28 — I-JEPA ViT-H/14 at 1B: a recipe failure, not a SAL result.**
  ~181 GPU-hours. Two epochs of continued I-JEPA pretraining on ImageNet-1k
  collapse **every** arm including the unmasked control: linear probe
  48.60% (pretrained) → **16.64% baseline, 3.04% DropHead, 2.32% SAL**. Losses
  *fall* while features are destroyed (SAL ends lowest at 0.0137 and most
  collapsed). The probe is exonerated — exp28 reads 48.60% on the original and
  the standalone noise probe reads 49.09% ± 0.58 — and so is `slice_heads()`
  (1e-06 against masking on all 12 sliced cells). **With a broken control the run
  measures nothing about either method**; the LR/schedule/EMA recipe must be
  adapted for SSL objectives first. Do not cite any `results/jepa1b*` number as
  a method comparison. See `EXP28_JEPA_1B.md`.
- **INT4 caveat, applies to every number above.** INT4 is
  `quantize_weights(model, 4)`, a **simulated per-channel quantize/dequantize
  round-trip in fp32**, not a 4-bit kernel. bitsandbytes NF4 is CUDA-only and
  absent on the ROCm image, so native INT4 was never measured on LUMI. The
  column is the accuracy cost of 4-bit weight rounding and says **nothing** about
  speed or memory.
- MLP pruning and the compile/ONNX paths live in **sal-torch-pro**
  (`sal_pro.mlp`, `sal_pro.deploy`): MLP pruning is now measured (exp26,
  exp18) and ONNX export is validated (exp22).
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
- **v0.6.0 defaults changed**: `prune_fraction` 0.33 -> 0.50 and `uniform_per_layer=True`
  (exp14, `results/lumi_v060/PHASE2F.md`). The uniform draw makes SAL's trained set a
  valid `slice_heads()` argument, so the retained accuracy becomes a 16.5% smaller
  model; it costs nothing measured. `num_heads_to_prune` snaps to a multiple of the
  layer count under that flag, since an equal quota is otherwise unreachable.
- Smoke-test one job before submitting a batch, and submit the batch with
  `sbatch --dependency=afterok:<smoke-id>`. This caught the MIOpen failure with
  20 jobs held at `DependencyNeverSatisfied` and zero GPU-hours burned. Cancel
  held jobs by explicit job id, never `scancel --state=PENDING -u <user>`.
- **Gate on the control arm, not on completion.** exp28's smoke jobs (90s) and
  gate (1:22:53) both passed while the real runs were destroying the model,
  because neither can see a representational collapse that takes 16 hours of
  training to produce. A cheap probe of the *control* checkpoint after ~1000
  steps would have saved ~157 GPU-hours. This is the "fail loudly on a
  non-finite loss" lesson one level up: **fail loudly when the control arm
  regresses against its own starting point.**
- Next: v0.7.0 on the new narrative — **diagnose (FI) → route (probe) → DropHead
  by default, SAL when the deployed head set is fixed → MLP prune → recover**;
  fix the SSL training recipe before re-running anything at 1B; Phase 5
  (license signing + compliance reports). The `nonaccum` control arm is still
  unrun, though exp27 largely answers what it was for. See `ROADMAP.md`
  (public).

## Integration tests

The CPU unit suite must stay all-green (**319 passed** as of v0.6.1, plus the
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
