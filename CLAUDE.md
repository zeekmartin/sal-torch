# CLAUDE.md — sal-torch

Context for future sessions working on this package.

## What this is

`sal-torch` is a **commercial** PyTorch package (BSL 1.1 license). It is a
**product**, not a research repo: clean API, clear errors, no research
artifacts.

**Positioning, decided 2026-11-13 after the pre-report audit.** It is a
**measured compression tool** — diagnose → prune → recover → verify — **not a
training-time resilience method**. SAL and DropHead remain supported methods,
**scoped to "no post-pruning fine-tune is possible"**. The reason is measured:
at a fixed head set, SAL's weights are worth **+0.05pp** and DropHead's
**+0.07pp** as a starting point for a recovery fine-tune, while the recovery
fine-tune itself is worth up to **68pp**. When you can fine-tune after pruning,
the training-time method is close to irrelevant.

### Three rules for every public document

These bind `README.md`, `ROADMAP.md`, this file, and every user-facing string
in `sal/*.py` — a docstring or an error message is a published claim.

1. **Describe only what the released code does.** No unimplemented pipeline
   step presented as available. Two live violations, both now flagged in the
   docs and both v0.7.0 items: `CompressionPipeline` has **no recovery stage**,
   and **DropHead is not callable from the package** (`HeadMasker` accumulates
   and holds, which is SAL; the resampled variant lives only in
   `scripts/run_sal_qat_prototype.py`).
2. **Every figure traceable to a JSON, with the epoch-matched control, the
   hardware, the batch size and the seed count stated.** No "free" or "beats"
   inside seed noise.
3. **Limitations sit next to the results they qualify**, not in a separate
   caveats section further down.

Three independent, composable components:

- **SAL (Structurally Adaptive Learning)** — training-time random head masking,
  accumulate-and-hold. *Perturbs.* **Specific**: its resilience applies to the
  one head set it trained for and nowhere else (exp25). **It has no remaining
  measured advantage over DropHead, on any cell, including that one set**
  (exp27, exp28b). Shipped and supported; not the recommended default anywhere.
- **DropHead** — the same quota, mask resampled every step. *Perturbs.*
  **General**: its resilience holds on any head set of that size, including
  SAL's own (exp27, exp28b). Started life as SAL's control arm and outperforms
  it on the product's actual promise. **This is the default *head-training*
  method — and head training is itself the fallback**, used when no
  post-pruning fine-tune is possible. **It is not implemented in `sal/`**: the
  resampled mask exists only in `scripts/run_sal_qat_prototype.py`
  (`--mask-mode drophead`). `mask_mode="drophead"` on `HeadMasker` is v0.7.0
  item 2.
- **FI (Fragility Index)** — a post-hoc structural fragility diagnostic.
  *Measures.* Diagnosis and certification only — **not** a router and not a head
  selector (exp17, PHASE2E).

They are **separate**. A user can use any one without the others.

**What decides the method is not the route — it is whether a post-pruning
fine-tune is available.** If it is: prune by magnitude, recover 2 epochs, and
skip head training entirely (exp24, exp29, exp32, exp34). If it is not: the
probe decides whether head training pays for itself at all (>30pp drop → yes,
10-30pp → less, <10pp → the router says no), and **DropHead is what you train
with** (exp27, exp28b). The `<10pp → skip` row is the one exp34 contradicts —
see wave 1 below.

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
5. The **scaffold API is the target**; the **private research reference** is
   the **source of truth for the mechanism**. Keep the clean API, use the
   validated implementation details. That reference is not shipped and **its
   location is not recorded in this file**, which is on the public remote — see
   cardinal rule 6 and the local notes outside any repository.
6. Treat `pyproject.toml`, `LICENSE`, `README.md` as off-limits unless the task
   explicitly says otherwise. **The `sal-torch` remote is PUBLIC**
   (`github.com/zeekmartin/sal-torch`), so a push needs a **pre-push scan**
   first: secrets and tokens, references to private repos, and pro internals
   beyond what is already documented as a pro feature. `sal-torch-pro` and the
   other private sibling repos are **private**; pushing them is allowed.
   `agent-certify` is public (Apache-2.0, on PyPI). **This file is itself on
   the public remote**, so it must not name an unreleased product or a private
   path either — that is why the sibling repos are not listed here by name.
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
    across **four architectures** — DeiT-small and DINOv2-large on
    ImageNet-100, BERT-base and phi-2 on SST-2, `n_eval` 872 in each case,
    so **not** the ImageNet-1k protocol the rest of the campaign uses.
    (PHASE2H's table shows six rows; the ViT-B/16 and I-JEPA ones are, in
    that page's own words, "the published points, from `PHASE2F.md` and
    `PHASE2G.md`" — imported from other runs on ImageNet-100, which cardinal
    rule 7 supersedes. Do not build a claim on a comparison between an exp17
    row and an imported one.)
    **It fails at both ends, inside exp17**: DINOv2-large has an FI **25x
    lower** than DeiT-small and loses *more* (−94.02 against −76.30pp), while
    BERT-base sits between them in FI (0.0935) and loses least of the three
    (−5.15pp). Ranked by FI the four read phi-2 < DINOv2-large < BERT-base <
    DeiT-small; ranked by what they actually lose, DINOv2-large < DeiT-small
    < BERT-base < phi-2 — **4 of 6 pairs inverted**. Parameter count does not
    rescue it either. What does order the table is **the baseline's own drop
    under pruning**, measured directly by a probe: >30pp → SAL, 10-30pp →
    either, <10pp → skip head training. FI stays in the product for diagnosis
    and certification, **not** as a router. The four-architecture table is
    generated into the canonical tables, outside this repository.
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
  ~196 GPU-hours, **3/3 seeds scored**. Two epochs of continued I-JEPA
  pretraining on ImageNet-1k collapse **every** arm including the unmasked
  control: linear probe **46.07% pretrained → 15.32% baseline, 3.20% DropHead,
  2.69% SAL** (three-seed means; each control drops 27–33pp against its *own*
  `original`). Losses *fall* while features are destroyed. **The n=1 reading
  "SAL ends lowest and most collapsed" is withdrawn**: across three seeds SAL's
  epoch-2 loss is 0.0478 ± 0.0318, *above* the baseline's 0.0312, and the
  baseline has both the lowest loss and the best features — loss progress on this
  objective is simply **decoupled** from representation quality. `slice_heads()`
  is exonerated (1e-06 against masking on all 12 sliced cells, all seeds). **The
  probe's ±0.58pp figure is also corrected**: that was the spread within one
  shuffle; across the three eval jobs the same untrained checkpoint reads
  41.04–48.60% (7.56pp), so compare within a seed, never across. **With a broken
  control the run measures nothing about either method.** Do not cite any
  `results/jepa1b*` number as a method comparison. See `EXP28_JEPA_1B.md`.
- **exp28b — the cause was the objective, and with it fixed DropHead wins at
  631M too.** 223.6 GPU-hours, **3/3 seeds**, two full ImageNet-1k epochs each;
  the 1B campaign (exp28 + exp28b) comes to **420.2 GPU-hours**. `make_step`
  computed its target from the **same live weights it was training**, an
  objective with no fixed point. exp28b changed only that — a **frozen target
  encoder**, LR 5e-6, 10% warmup, cosine decay — and **the control arm now
  survives on 3/3 seeds** (−0.64, +0.72, +7.56pp against each seed's own
  pretrained reference, gate sentinels in `results/lumi_v060/jepa1b_corr*/`).
  With a sound control the method comparison runs, and **DropHead beats SAL on
  every cell and every seed**: +29.02pp over the control at 50% head pruning
  against SAL's +7.89pp, +11.88pp at 33% where SAL reads −0.22pp and flips sign
  with the seed, and **+0.67pp on SAL's own trained-for head set (3/3)** — the
  last cell where SAL could have been the right choice. SAL also pays **12.35pp
  of dense accuracy** (3/3) and its best cell sits 5.11pp below the control's
  dense row. Pruning 33% of heads is free for DropHead here too (−0.04pp) and
  50% costs 1.14pp. Note `sal@aligned` is **+7.24pp above SAL's own dense row**:
  as in exp25, SAL's dense row is not its operating point. See
  `EXP28B_JEPA_CORRECTED.md`.
- **The SSL recipe lesson: continued self-supervised pretraining needs a frozen
  target encoder.** A target computed from the live weights has no fixed point,
  so the encoder can drift toward a low-variance output, drag both sides of the
  MSE down together and **destroy the representation while the loss falls**. That
  is exp28 in one sentence. `scripts/run_jepa_sal.py:make_step` **is fixed as of
  v0.7.0 batch 1**: it takes its target from `load_target_encoder()`, a frozen
  pretrained copy, and the parameter is required so no call site can fall back
  to the live-weight form. That script still runs `LR` 1e-5 with no schedule,
  though, where the measured recipe is 5e-6 with 10% warmup and cosine decay —
  so `experiments/exp28b_jepa_corrected.py` remains the reference for any
  continued-pretraining run. Corollary: on this objective **loss progress is
  decoupled from representation quality**, so a loss curve is not a gate; probe
  the features.
- **INT4 caveat, applies to every number above.** INT4 is
  `quantize_weights(model, 4)`, a **simulated per-channel quantize/dequantize
  round-trip**, not a 4-bit kernel. The rounding is done in fp32; the rounded
  model is then **scored under float16 autocast**, like every other accuracy in
  the campaign (`run_sal_qat_prototype.py:accuracy`). Training and recovery run
  under bfloat16 autocast -- the two are different and the write-ups said
  "autocast bfloat16" for both until the pre-report audit. bitsandbytes NF4 is CUDA-only and
  absent on the ROCm image, so native INT4 was never measured on LUMI. The
  column is the accuracy cost of 4-bit weight rounding and says **nothing** about
  speed or memory.
- **INT4 robustness after continued pretraining (exp28b).** On the **dense**
  rows of the I-JEPA table, 4-bit weight rounding costs **under 0.5pp for all
  three arms** — control −0.16, DropHead −0.43, SAL −0.49pp, 3/3 seeds each.
  On the **pruned** rows it grows to 1.4–3.9pp, so head pruning and quantization
  are not independent and the cheap cell is dense-INT4, not pruned-INT4.
  **Do not quote the pretrained row as the "before" figure**: the same untrained
  checkpoint reads −9.20, −1.44 and −0.32pp of INT4 cost on the three seeds,
  which is probe noise, not a quantization effect. There is no measured
  "continued pretraining reduces the INT4 cost from −9.2pp" claim.
### Wave 1 (exp29-exp34) — the matched-control campaign, 104 GPU-h

**`results/lumi_v060/CONTROL_CONVENTION.md` governs every cost on this page.**
Fine-tuning gain = dense control − pretrained. **Pruning cost = arm − dense
control trained for the SAME number of epochs.** The pretrained checkpoint is
**not** a control, and a delta against it is the *sum* of the two effects, never
a pruning cost. Where an arm is single-seed, score it within its own seed.

The control ladder (ViT-B/16, ImageNet-1k, nothing pruned):

| epochs | accuracy | INT4 | seeds |
|---|---|---|---|
| 0 (pretrained) | 78.44% | 77.228% | — |
| **1** | **81.727%** | **80.567%** | 3 (81.738 / 81.744 / 81.700) |
| **2** | **81.554%** | **80.273%** | 3 (81.676 / 81.524 / 81.462) |
| 2 (re-measured) | 81.623% | 80.336% | 3 (81.636 / 81.600 / 81.632) |
| 3 | 81.158% | 79.821% | 3 (81.074 / 81.202 / 81.198) |
| 5 | 80.556% | 78.956% | 1 (exp31) |
| 5 | 80.44% | 79.01% | 3 (exp25 arm 1a) |
| 5 (re-measured) | 80.456% | 78.947% | 3 (80.636 / 80.404 / 80.328) |

**The dense control peaks at ONE epoch, not two.** The 1-epoch rows read
81.738 / 81.744 / 81.700% and **every** 2-epoch seed is below all three, on
the published set (81.462 / 81.524 / 81.676) and on wave 3's re-measured set
(81.600 / 81.632 / 81.636) alike, so the two ranges are disjoint rather than
overlapping inside seed noise. The INT4 column orders the same way. The
1-epoch and 3-epoch rungs were measured in wave 3 and sat uncopied on scratch
until 2026-11-17; this page previously claimed the control also peaked at 2.

**This changes no cost.** `CONTROL_CONVENTION.md` compares an arm against the
control trained for the SAME number of epochs, so every cost measured at 2
epochs stands and nothing is recomputed. What the peak changes is the
*framing*: a 2-epoch cost is measured against a control that is already past
its own best, which is one reason the matched cost shrinks as recovery
lengthens. The re-measured rows are **repeats**, not replacements: the
published exp32 and exp25 values stay authoritative and the delta is the
record, per the duplicate-control rule.

Use `experiments/matched_control.py` rather than copying these numbers: it loads
the control from exp32's JSONs, verifies it against the published mean before
deriving anything, and **emits `pruning_cost_pp` only when the epoch counts
match**. `tests/test_matched_control.py` pins that behaviour.

- **exp29 — two epochs of recovery, not five.** The *pruned* curve peaks at 2
  and declines: 5 is below 2 on 3/3 seeds in FP and INT4, and 10 is below 1.
  Two epochs is 39% of the cost. **The peak is a property of this recipe on
  this dataset, not of recovery** — the dense control with nothing pruned
  declines too, and faster. **Corrected 2026-11-17**: this said the control
  "also peaks at 2, and by a larger margin". The control peaks at **1** (see
  the ladder above, disjoint ranges), so the two curves do not share a knee
  and the pruned arm's peak at 2 is its own. Do not assume either knee
  transfers.
- **exp31/exp32 — the ratio sweep at 2 epochs, and the bug that motivated the
  audit.** exp32 measures 33/40/50% of heads + 2 epochs against the verified
  2-epoch control, 3 seeds: **−1.93 / −2.27 / −3.91pp** (FP) and −4.85pp at
  50%+INT4. **Both readings of "two beats five" are true and must be stated
  together**: two epochs gives a *better model* (77.64% vs 77.27%) and a
  *larger pruning cost* (−3.91 vs −3.17), because the control is itself better
  at two epochs. The committed `exp32_ctrl_*` JSONs carry `vs_this_control_pp`
  values computed from 5-epoch arms against the 2-epoch control — wrong by
  0.97-1.12pp. **Measured data is never mutated**: the fix is
  `results/lumi_v060/CORRECTIONS.json` plus `experiments/results_corrections.py`,
  which applies corrections at load time.
- **exp33 — unstructured sparsity is free; the structure is what costs.** 8
  configs × 3 seeds unstructured (SparseGPT + Wanda) plus 2:4, with
  mask-preserving recovery (gradient hook + final re-apply, `sparsity_drift`
  0.0 on 24/24 runs). At matched parameter count every cell lands within
  ±0.22pp of a control whose own seeds span 0.21pp. **One-shot, removing 50% of
  the attention weights without a structure constraint costs −1.23pp; removing
  the same parameters as whole heads costs −68.88pp.** The accuracy ordering is
  the exact inverse of the speedup ordering (unstructured ≈0 / 2:4 −0.5 /
  structured −3.91pp). SparseGPT's error compensation only earns its keep as
  the constraint tightens (loses to Wanda by 0.39pp at attn-P33, wins +2.35pp
  at 2:4); Wanda prunes 3-18x faster. **INT4 robustness is worse for `whole`
  scope on 4/4 pairings — the opposite of the FP32 ordering.**
- **exp34 — the router's `<10pp → skip` row is wrong on text.** BERT-base,
  4 GLUE tasks, 3 seeds. **Magnitude + 2 epochs of recovery beats DropHead on
  8/8 cells**, by 0.42 to 7.03pp — the reverse of the ViT ordering. DropHead's
  resilience *does* replicate (8/8, 4.6-10.9x less accuracy lost per head
  one-shot); what sinks it is that it pays for that resilience in dense
  accuracy, up to −12.58pp on MRPC. And the probe's prediction fails: one-shot
  drops at P50 are **−4.17 / −12.42 / −21.68 / −17.56pp**, i.e. mostly outside
  the skip band. DropHead's hooks also cost training throughput, and **the
  cost scales with the mask quota**: **24-30% at 33% of heads and 33-39% at
  50%**, over the eight cells' own `dense_seconds` and `drophead_seconds`.
  A single "~37%" is the 50% figure on three of the four tasks, not a number
  for the experiment; the per-cell table is generated into the canonical
  tables, outside this repository.

**Precision, stated exactly**: training and recovery run under **bfloat16**
autocast; scoring runs under **float16** autocast
(`run_sal_qat_prototype.py:accuracy`). INT4 is fp32 rounding, then scored under
fp16 autocast like every other row. The write-ups said "autocast bfloat16" for
both until the pre-report audit.

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
- Next: v0.7.0 on the repositioned narrative — **diagnose (FI) → route (probe)
  → prune (magnitude by default) → recover (2 epochs) → verify against a
  matched control**. Head training is the branch taken only when no
  post-pruning fine-tune is possible, and there **DropHead with no exception**:
  the "SAL when the deployed head set is fixed" carve-out is **gone**, since
  exp28b measured that cell at 631M and DropHead matches or beats SAL there on
  3/3 seeds without committing to a head set at training time.
  **v0.7.0 item 1** is `recover()` (`SPEC_V070_RECOVERY.md`); **item 2** is
  `mask_mode="drophead"` on `HeadMasker`. **Release gate: no v0.7.0 tag until
  the shipped `CompressionPipeline` runs probe → magnitude → recover →
  validate end-to-end on ViT-B/16 / ImageNet-1k and reproduces 77.64% at 50% of
  heads within the published three-seed spread (77.44-77.81%).** Every number
  the project publishes today came from `scripts/run_sal_qat_prototype.py`, not
  from the pipeline, which is how the missing recovery stage survived two
  releases. The SSL recipe is **fixed and validated**
  (exp28b, frozen target encoder). Remaining: the epoch-1 control gate as a
  shipped safeguard and an FFN-ratio router (both specced in `ROADMAP.md`), and
  Phase 5 (license signing + compliance reports). The `nonaccum` control arm is
  still unrun, and exp27 plus exp28b now largely answer what it was for. See
  `ROADMAP.md` (public).

## Integration tests

The CPU unit suite must stay all-green (**459 passed, 13 skipped** as of wave 4;
it read 418 at wave 2, 330 at v0.7.0 batch 1 and 319 at v0.6.1); integration
tests are marked
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
- **The v3 ablation notes** — ablation results and validated claims (random
  beats structured selection; the gains come from progressive training-time
  sparsification). **Removed from this repository on 2026-11-14** and kept
  only in the local working copy: it discussed selection criteria in terms
  cardinal rule 3 forbids in a public file. It is in this repository's git
  history, which was deliberately not rewritten, and it is `.gitignore`d so
  it cannot be re-added by accident. The claims it carried that still matter
  are in `ROADMAP.md` and `results/lumi_v060/`.
- The **private research reference** — canonical mechanism, `random` branch.
  Not shipped, and deliberately not located here: this file is on the public
  remote, so a local path to unshipped research does not belong in it. The
  location lives in the local notes outside any repository.

## Testing

```
python -m pytest -q       # CPU only, ~6-8 min, must be all-green
```
Fixtures in `tests/conftest.py`: `tiny_model` (custom 4×8×64 GPT-2-like),
`tiny_config`, `probe_data`. Run the suite with `CUDA_VISIBLE_DEVICES=` on a GPU
box: `tests/test_run_jepa_script.py` fakes a CPU model and fails under CUDA
autocast otherwise.
