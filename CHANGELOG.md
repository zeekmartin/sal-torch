# Changelog

## [0.5.1] — 2026-09-17

SAL stops being supervised-only, and the self-supervised benchmark behind it has
now been run: I-JEPA ViT-H/14 at three seeds, DINOv2 ViT-L/14 at one.

### Added
- **Loss-agnostic `SALTrainer`** — see `train_step=` below. Backward
  compatible: `train_step=None` keeps the existing cross-entropy loop.
- **`run_jepa_sal.py` flags**: `--slice` (score physically sliced models next to
  the masked ones, each checked against its masked twin), `--prune-ratios`,
  `--model` (any ViT-family encoder, e.g. `facebook/dinov2-large`), alongside the
  existing `--smoke`, `--control`, `--seed`, `--resume`.
- **`scripts/aggregate_jepa_multiseed.py`** — mean ± sample std and paired
  SAL-vs-control wins across seeds.
- **`SALTrainer(train_step=...)`** — a custom training step callback, signature
  `(model, batch, optimizer, mask_module) -> loss`. The loop keeps the prune
  schedule (it calls `masker.step()` before every callback and leaves masking
  on); the callback owns forward, loss, backward, clipping, optimizer and
  scheduler. `gradient_accumulation_steps` and `max_grad_norm` are deliberately
  not applied in custom mode — a custom objective may accumulate differently.
- **`HeadMasker.apply_mask()` / `remove_mask()` / `unmasked()`** — suspend and
  resume, as distinct from the existing `activate()` / `deactivate()`.
  `deactivate()` refills the masks with ones and discards the accumulated pruned
  set, which is exactly wrong for a callback that needs one unperturbed forward
  pass mid-step.
- **`sal.evaluation`** — `linear_probe`, `knn_accuracy`, `cka_similarity`,
  `representation_similarity`, `measure_latency`, `count_params`,
  `extract_features`, `compression_report`. torch + numpy only.
- **`sal.visualization`** — attention and per-head feature maps, original vs
  compressed side by side, and grouped bars across benchmark arms. matplotlib
  only.
- **Architecture support for `ijepa` and `dinov2`.** Module finding already
  worked through the fallback patterns, but `SALConfig.auto()` goes through
  `detect_architecture()`, which refused both.
- **`examples/jepa_sal.py`** and **`scripts/modal_jepa_sal.py`** — the I-JEPA
  compression benchmark. `--smoke` runs the example end to end on CPU.
- **`scripts/run_jepa_sal.py`** — the same experiment with no cloud SDK, for any
  GPU box over SSH. Auto-detects VRAM to pick a batch size, turns on gradient
  checkpointing below 48GB, checkpoints every epoch and resumes from one, and
  drops every artefact (metrics, scans, figures, safetensors weights) into one
  directory to `scp` back. `--control` trains the no-SAL arm.
- **`scripts/setup_runpod.sh`** — prepares a fresh machine and verifies GPU,
  sal-torch, the I-JEPA checkpoint and the dataset *before* the paid run starts.
  It does not install torch: cloud images ship a build matched to their CUDA
  driver, and replacing it is how a working machine stops working.

### Fixed
- **`SALTrainer` reported nonsense in `masker_stats`.** It read `.stats` after
  `masker.remove()`, which clears the mask tensors — so every run reported zero
  active heads, i.e. "100% of heads pruned", whatever `prune_fraction` was.
  Present in the supervised path since v0.5.0. Stats are now snapshotted before
  the hooks come off.
- **`get_qkv_projections()` missed DINOv2's Q/K/V**, which sit one level deeper
  than either searched path, so head-level weight slicing could not see them.
- **`slice_heads()` on ViT-family encoders** (I-JEPA, DINOv2 under transformers
  4.x). Head bookkeeping under `.attention` was not retargeted, and
  `verify_input` failed on models that return `last_hidden_state` instead of
  `logits`. Sliced I-JEPA outputs now match the masked model exactly.

### Benchmark results — I-JEPA ViT-H/14 (631M), ImageNet-100
Five epochs per arm on one A100; SAL trained at `mask_ratio=0.3` against a
control trained identically without head masking. Results in `data/results/`.

**Three seeds (42/123/456): SAL wins 35 of 36 paired comparisons.**

| setting | SAL | control | delta | SAL ahead |
|---|---|---|---|---|
| random-33% probe | 76.8% | 76.4% | +0.4pp | 3/3 |
| random-33% kNN | 71.2% | 70.0% | +1.2pp | 3/3 |
| random-50% probe | 64.3% | 62.9% | +1.4pp | 3/3 |
| random-50% kNN | 47.5% | 45.6% | +1.9pp | 3/3 |
| magnitude-50% probe | 65.8% | 64.8% | +0.9pp | 2/3 |

The gains are small — several are sub-1pp — and consistent in direction rather
than large. The cross-seed std is often wider than the gap, because a seed
changes the data subset and pruning draw for both arms alike; the paired win
count is the relevant test. No cost on the unpruned model (probe 83.2% vs
83.1%). Summary: `jepa_sal_multiseed_summary.json`.

**`slice_heads()` real speedup** (seed 42, batch 1, 224px, A100 / 12 CPU threads):

| heads removed | params | GPU | CPU |
|---|---|---|---|
| 0% | 631M | 28.7ms | 582ms |
| 31% (5/16 per layer) | 565M | 27.8ms (1.03×) | 518ms (1.12×) |
| 50% (8/16) | 526M | 26.1ms (1.10×) | 487ms (1.20×) |
| 69% (11/16) | 486M | 25.0ms (1.15×) | ~460ms (~1.3×)¹ |

¹ Separate run, against its own 594ms baseline.

Modest by construction: head removal shrinks only Q/K/V/O, attention is about a
third of a ViT-H block's parameters, and the residual width is unchanged.
Magnitude-sliced quality equals magnitude-masked quality exactly.

**70% pruning: no effect.** SAL ≈ control within ±1.6pp with mixed signs, both
when SAL trained at `mask_ratio=0.3` and at `0.7`. The measured range for SAL on
I-JEPA is 33–50% head pruning.

**Cross-architecture — DINOv2 ViT-L/14 (304M), seed 42 only:** SAL wins 13 of
18. DINOv2 is far more fragile to head removal than I-JEPA — at 50% both arms
collapse and SAL does not help. Where the pruned model survives, the gains are
larger: 33% per-layer-random sliced, +3.0pp probe (84.0% vs 81.0%) and +6.9pp
kNN (75.4% vs 68.5%). Magnitude selection is a poor choice on DINOv2 (33%:
probe 36%). Single seed; not a validated claim.

### Notes on the benchmark, before anyone quotes it
- **There is no I-JEPA ViT-B/16.** Meta released I-JEPA at ViT-H/14 and ViT-g/16
  only. The scripts default to `facebook/ijepa_vith14_1k` (632M), which needs an
  A100 rather than the A10G originally scoped.
- **The training objective is I-JEPA-shaped, not I-JEPA** — no predictor
  network, no EMA target encoder. It predicts the clean-image representation
  from a patch-masked image. The first draft used the clean image on *both*
  sides, which makes the loss identically zero whenever head masking is off: the
  no-SAL control arm would have run its optimizer on a constant while logging as
  though it were training.
- **Selection differs between masked and sliced `random` rows.** The masker's
  `random` samples across the whole model; slicing needs the same count per
  layer, so sliced rows sample within each layer (`random-uniform`).
- **Post-hoc random heads are not the heads SAL trained without.** Their overlap
  with the training-time pruned set is at chance level, so SAL is not scored on
  heads it already learned to do without.
- **Seed coverage is uneven.** I-JEPA at 33/50% has three seeds; the slicing,
  70% and DINOv2 results are one seed each.


## 0.5.0 (2026-08-12) — CompressionPipeline, and five seeds instead of one

Turns the v0.4.0 recipe into one object, makes the savings real, and finally
puts a number on how much of the v0.4.0 result was the seed.

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
  reloads at exactly the measured accuracy.
- **Five-seed validation** (`scripts/modal_multiseed_validation.py`, results in
  `scripts/multiseed_results.json`). GPT-2 Medium / SST-2, full fine-tuning,
  seeds 42/123/456/789/1337, scored on the complete 872-example validation
  split instead of a 512 subset. Every prior SAL-vs-standard number in this
  project was a single seed; this replaces them.

  ```
  variant        standard            SAL                 delta    ahead  consistent
  ----------------------------------------------------------------------------------
  dense          0.9005 +- 0.0104    0.9062 +- 0.0080   +0.0057    4/5      yes
  int8           0.8888 +- 0.0102    0.8961 +- 0.0033   +0.0073    4/5      yes
  int4           0.9002 +- 0.0111    0.9023 +- 0.0094   +0.0021    3/5      NO
  prune33        0.8571 +- 0.0240    0.8817 +- 0.0121   +0.0245    5/5      yes
  prune50        0.8087 +- 0.0455    0.8294 +- 0.0368   +0.0206    4/5      yes
  prune33+int8   0.8284 +- 0.0090    0.8472 +- 0.0229   +0.0188    4/5      yes
  prune33+int4   0.8567 +- 0.0208    0.8725 +- 0.0132   +0.0158    5/5      yes
  ```

  **Holds:** SAL wins 5 of 6 compressed variants at no cost to clean accuracy
  (+0.57pp), and the effect is largest and most reliable under **head pruning**
  — +2.45pp at 33% on every seed, +2.06pp at 50%. Both combined recipes hold.

  **Does not hold:** *`int4` alone does not replicate* — 3/5 seeds and +0.21pp,
  inside the run-to-run spread. The v0.4.0 "wins all seven variants including
  INT4" was one seed. The Pareto claim softens with it: `SAL/int4` averages
  0.9023 at 361.9MB against the uncompressed standard model's 0.9005 at
  1419.3MB, which is **equal accuracy at a quarter of the size**, not higher.
  Both README and ROADMAP have been corrected.

  Caveat on the `int8` rows: this run uses `torch.ao` dynamic INT8 (CPU) where
  v0.4.0 used bitsandbytes LLM.int8() on CUDA, so those rows are a different
  measurement rather than a replication. INT4 is bitsandbytes NF4 in both. The
  eval-time head choice is seeded independently of the training seed, so SAL is
  not scored on the heads it trained against; both arms of a seed lose the same
  heads.

  Offered as an observation, not a claim: the SAL arm has the smaller standard
  deviation on 6 of the 7 variants.
- 157 unit tests pass on CPU (was 110), plus 8 integration tests skipped by
  default.

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

> **Partly superseded by v0.5.0.** The five-seed run reproduces the pruning and
> combined wins and the zero clean-accuracy cost, but **not** the INT4-alone
> win (3/5 seeds, +0.21pp) and not the Pareto claim as stated — across seeds
> `SAL/int4` matches rather than beats the uncompressed standard model. The
> paragraph above is left as written because it is what one seed showed.

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
