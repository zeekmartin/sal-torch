# sal-torch — Technical Design Document

## Version: 0.1.0 (initial)
## Date: June 25, 2026
## Author: Cognitive Engineering (cognitive-engineering.dev)
## License: BSL 1.1 (Business Source License)

---

## 1. What sal-torch Is

A PyTorch-native package that makes any transformer model structurally resilient to compression through training-time head deactivation. Two independent components:

- **SAL (Structurally Adaptive Learning)**: training-time random head masking that forces functional redistribution. Not a loss function — structured perturbation + adaptation.
- **FI (Fragility Index)**: post-hoc diagnostic that measures structural fragility of a model's attention graph. Classifies layers as IMMUNE / BUFFER / CRITICAL.

SAL perturbs. FI measures. They are independent — a user can use SAL without FI, or FI without SAL.

---

## 2. Target API

### 2.1 Three-Line Integration (Segment B)

```python
from sal import SALConfig, SALCallback

config = SALConfig.auto(model)  # auto-detects architecture
trainer = HFTrainer(model=model, callbacks=[SALCallback(config)])
trainer.train()
```

### 2.2 Full Control (Segment A)

```python
from sal import SALConfig, SALCallback, FIMonitor

config = SALConfig(
    prune_fraction=0.33,
    prune_start_ratio=0.10,
    prune_end_ratio=0.80,
    schedule="random",           # "random" | "progressive" | "burst"
    fi_interval=500,             # compute FI every N steps (0 = disabled)
)

fi_monitor = FIMonitor(
    probe_dataset=probe_ds,      # small dataset for activation extraction
    classify_layers=True,        # IMMUNE / BUFFER / CRITICAL
    export_report=True,          # exportable compliance report
)

trainer = HFTrainer(
    model=model,
    callbacks=[SALCallback(config), fi_monitor],
)
trainer.train()

# Post-training diagnostics
report = fi_monitor.report()
report.save("structural_report.json")
report.save("structural_report.pdf")  # for compliance teams
```

### 2.3 Standalone FI Diagnostic (no training)

```python
from sal import FIScanner

scanner = FIScanner(model, probe_dataset=probe_ds)
results = scanner.scan()

print(results.fi_score)           # e.g. 0.073
print(results.layer_map)          # {0: "IMMUNE", 1: "BUFFER", ...}
print(results.critical_layers)    # [4, 7, 11]

results.save("model_xray.json")
```

### 2.4 License Verification

```python
import sal

# Community: no license needed, works out of the box
# Pro/Enterprise: set once
sal.set_license("path/to/license.lic")

# Or via environment variable
# export SAL_LICENSE_FILE=/path/to/license.lic

# License info
print(sal.license_info())
# → {"tier": "professional", "org": "Acme Corp", "expires": "2027-06-25"}
```

### 2.5 Plasticity Scanner (Segment A & Enterprise)

```python
from sal import PlasticityScanner

scanner = PlasticityScanner(model, probe_dataset=probe_ds)
pmap = scanner.scan()

# Three axes of structural plasticity
print(pmap.routing)          # per-layer routing flexibility score
print(pmap.cka_similarity)   # CKA between layers (representation redundancy)
print(pmap.mutual_info)      # MI between head activations (functional overlap)

# Where does the model have room to absorb compression?
print(pmap.absorption_map)   # {layer_0: "saturated", layer_3: "elastic", ...}
print(pmap.hub_layers)       # layers that compensate when others fail (typically last)

# Compression recommendations
recs = pmap.recommend(target_compression=0.33)
print(recs.safe_to_prune)    # layers/heads that can go without risk
print(recs.never_touch)      # layers that are structural hubs
print(recs.expected_impact)  # predicted accuracy delta

# Export
pmap.save("plasticity_report.json")
pmap.save("plasticity_report.pdf")   # visual report with heatmaps
```

PlasticityScanner answers "WHERE can this model absorb damage?" before any pruning happens. FIScanner tells you how fragile the model IS. PlasticityScanner tells you how much capacity it HAS to reorganize. Together they give the complete structural picture: fragility (current state) + plasticity (adaptation potential).

### 2.6 Continual Learning Guard (v0.3.0+)

```python
from sal import StructuralGuard

# After initial SAL training, freeze the structural map
guard = StructuralGuard(model, probe_dataset=probe_ds)

# When fine-tuning on a new task:
guard.protect()  # freezes structurally critical components
# Train on new task — only elastic components learn
trainer.train()

# Verify structural integrity after new task
drift = guard.measure_drift()
print(drift.forgetting_score)    # how much old task capability was lost
print(drift.structural_delta)    # topology change vs baseline
```

---

## 3. Package Structure

```
sal-torch/
├── pyproject.toml
├── LICENSE                        # BSL 1.1 text
├── LICENSE_CHANGE_DATE            # BSL → Apache 2.0 conversion date
├── README.md
├── CHANGELOG.md
│
├── sal/
│   ├── __init__.py                # version, set_license(), license_info()
│   │
│   ├── config.py                  # SALConfig dataclass
│   │   └── SALConfig              # prune_fraction, schedule, intervals
│   │   └── SALConfig.auto(model)  # auto-detect architecture → config
│   │
│   ├── masker.py                  # Core masking mechanism
│   │   └── HeadMasker             # forward hooks, binary masks, re-randomization
│   │   └── ExpertMasker           # MoE router probability zeroing (experimental)
│   │
│   ├── callback.py                # HuggingFace Trainer integration
│   │   └── SALCallback            # on_step_begin: apply masks, on_step_end: re-randomize
│   │
│   ├── trainer.py                 # Standalone SAL training loop (no HF dependency)
│   │   └── SALTrainer             # for users not using HF Trainer
│   │
│   ├── fi.py                      # Fragility Index computation
│   │   └── extract_activation_graph(model, dataset) → Graph
│   │   └── compute_fi(graph) → float
│   │   └── classify_layers(graph) → dict[int, LayerClass]
│   │
│   ├── scanner.py                 # High-level FI diagnostic
│   │   └── FIScanner              # wraps fi.py into a user-friendly API
│   │   └── FIMonitor              # HF Callback for live FI tracking during training
│   │
│   ├── plasticity.py              # Plasticity mapping (3 axes)
│   │   └── PlasticityScanner      # routing / CKA / MI analysis
│   │   └── PlasticityMap          # absorption map, hub detection, recommendations
│   │
│   ├── guard.py                   # Continual learning structural guard (v0.3.0+)
│   │   └── StructuralGuard        # freeze critical, allow elastic to learn
│   │   └── DriftReport            # forgetting score, structural delta
│   │
│   ├── report.py                  # Compliance report generation
│   │   └── StructuralReport       # JSON + PDF export
│   │   └── ComplianceReport       # regulatory-friendly format
│   │
│   ├── arch_support.py            # Architecture auto-detection
│   │   └── detect_architecture(model) → ArchInfo
│   │   └── get_attention_layers(model) → list[Module]
│   │   └── get_head_count(model) → int
│   │   # Supported: LLaMA, Mistral, GPT-2, BERT, RoBERTa,
│   │   #            DistilBERT, ViT, Phi-2, Phi-3, Gemma, Qwen2
│   │
│   ├── license.py                 # License verification (Ed25519 offline)
│   │   └── verify_license(path) → LicenseInfo
│   │   └── LicenseInfo            # tier, org, expiry, features
│   │   └── PUBLIC_KEY             # embedded Ed25519 public key
│   │
│   └── _keys/
│       └── sal_public.pem         # Ed25519 public key (shipped with package)
│
├── tests/
│   ├── test_masker.py
│   ├── test_config.py
│   ├── test_callback.py
│   ├── test_fi.py
│   ├── test_scanner.py
│   ├── test_plasticity.py
│   ├── test_arch_support.py
│   ├── test_license.py
│   └── conftest.py                # tiny model fixtures (4L×8H×64h)
│
├── examples/
│   ├── quickstart.py              # 3-line integration
│   ├── full_control.py            # segment A usage
│   ├── standalone_fi.py           # diagnostic only
│   ├── plasticity_scan.py         # pre-compression plasticity analysis
│   ├── bert_finetuning.py         # BERT + SAL
│   ├── llama_finetuning.py        # LLaMA + SAL + QLoRA
│   └── compliance_report.py       # generate regulatory report
│
├── docs/
│   ├── getting_started.md
│   ├── how_sal_works.md           # mechanism explanation (no proprietary details)
│   ├── architecture_support.md
│   ├── licensing.md               # tiers, pricing, how to get a key
│   └── api_reference.md
│
└── tools/
    └── generate_license.py        # PRIVATE — not shipped. Ed25519 key signing tool.
```

---

## 4. Core Technical Decisions

### D1: HeadMasker Implementation

Forward **pre-hooks** on the attention output projection (`o_proj` / `out_proj` /
`c_proj` / `dense`). A binary mask of shape `[num_heads]` per layer zeros the
per-head slices of the projection's **input** — the concatenated per-head
attention outputs. This is the only point in the block where feature dimensions
map cleanly onto heads; masking the projection *output* would zero arbitrary
mixed features, not heads.

Pruning is **accumulative and progressive**. During the prune window, randomly
chosen heads are deactivated over time and **stay** deactivated; the count grows
toward `prune_fraction` per the chosen `schedule`. After the window the pruned
set is **held through the end of training** (not restored), so the model adapts
to operate without the removed heads and the adaptation is baked into the
weights. This progressive structural damage — not the selection criterion — is
the validated causal mechanism, which is why random selection is the default.

### D2: Random Selection Only (for now)

SAL ablation results show random = topology-guided at 124M params. At 345M+, topology-guided reduces variance 73×. However, topology-guided requires FI computation mid-training, which is expensive.

Decision: ship with `schedule="random"` as default and recommended. Topology-guided as experimental/advanced option for segment A users who want it and accept the compute cost. This keeps the package simple and the compute overhead minimal.

### D3: Architecture Auto-Detection

`SALConfig.auto(model)` inspects `model.config` (HuggingFace) to determine:
- Number of layers
- Number of attention heads per layer
- Attention module path (e.g., `model.layers[i].self_attn` vs `model.encoder.layer[i].attention`)
- Head dimension

Fallback: if architecture is unknown, raise a clear error with instructions for manual config. Do NOT silently guess.

Detection strategy:
```python
def detect_architecture(model):
    config = getattr(model, 'config', None)
    if config is None:
        raise SALArchitectureError("Model has no .config attribute. Use SALConfig(...) manually.")
    
    arch_type = config.model_type  # "llama", "gpt2", "bert", etc.
    registry = {
        "llama": LlamaArch,
        "mistral": MistralArch,
        "gpt2": GPT2Arch,
        "bert": BertArch,
        "roberta": RobertaArch,
        "distilbert": DistilBertArch,
        "vit": ViTArch,
        "phi": PhiArch,
        "phi3": Phi3Arch,
        "gemma": GemmaArch,
        "gemma2": Gemma2Arch,
        "qwen2": Qwen2Arch,
    }
    
    if arch_type not in registry:
        raise SALArchitectureError(
            f"Architecture '{arch_type}' not yet supported. "
            f"Supported: {list(registry.keys())}. "
            f"Use SALConfig(...) with manual head/layer specification."
        )
    
    return registry[arch_type](config)
```

### D4: HuggingFace Trainer Integration

SALCallback implements `TrainerCallback`:
- `on_train_begin`: install hooks, validate architecture
- `on_step_begin`: apply current mask
- `on_step_end`: check if re-randomization is due, apply if yes
- `on_train_end`: remove hooks, log final stats

Also provide SALTrainer (standalone) for users NOT using HF Trainer — uses a standard PyTorch training loop with the same masker.

### D5: FI Computation

FI is a **triangle-support fragility score** for the model's activation graph.

Activation graph extraction:
1. Register forward pre-hooks on each attention output projection
2. Run a small probe dataset through the model (~500 samples)
3. Capture the per-head activation signature at the projection input (the one
   place where feature dimensions map cleanly onto heads)
4. Build the graph: nodes = heads, edges = head-pairs whose signatures are
   similar (Pearson similarity, kept at a fixed edge density)
5. Score fragility by triangle support: an edge is **fragile** if the two heads
   it connects share no common neighbour (no triangle reinforces it)

**FI = fraction of edges with zero triangle support**, in `[0, 1]`:
- **FI = 0** — every edge is reinforced by at least one triangle. Fully
  triangulated, redundant pathways everywhere → robust.
- **FI = 1** — no edge has triangle support → fragile.

Layer classification thresholds (relative FI change when a layer's heads are removed):
- IMMUNE: removing this layer changes FI by < 1%
- BUFFER: removing this layer changes FI by 1-5%
- CRITICAL: removing this layer changes FI by > 5%

These thresholds are configurable but defaults are validated across architectures.

### D6: PlasticityScanner — Three-Axis Structural Analysis

PlasticityScanner measures a model's capacity to absorb structural damage, NOT its current fragility (that's FI). Three complementary axes:

**Axis 1 — Routing flexibility:** For each layer, how many alternative paths exist for information to flow if heads are removed? Measured by attention routing entropy. High entropy = many paths = elastic. Low entropy = bottleneck = saturated.

**Axis 2 — CKA similarity:** Centered Kernel Alignment between adjacent layers. If two layers produce very similar representations, one is redundant and can absorb work from a pruned neighbor. High CKA between layers = safe compression zone.

**Axis 3 — Mutual Information:** MI between head activations within a layer. If heads have high MI, they share function and can compensate for each other. Low MI = each head is unique = pruning any one loses irreplaceable function.

The absorption map combines all three axes:
- **Elastic**: high routing entropy + high inter-layer CKA + high intra-layer MI → safe to compress
- **Saturated**: low on any axis → structural hub, do not touch
- **Hub**: layer that absorbs function from other pruned layers (validated: typically last layer)

This is the feature that transforms SAL from "training tool" to "structural intelligence platform." The scan runs in minutes (single forward pass of probe dataset), and the recommendations are actionable: "prune layers 2,5,8 first, expect <1pp impact."

### D7: License Enforcement

At `import sal`:
1. Check for `SAL_LICENSE_FILE` env var or `sal.set_license()` call
2. If found: verify Ed25519 signature, parse embedded JSON (tier, org, expiry, features)
3. If not found: Community mode (all features, production use restricted by license terms only)

The license file contains:
```json
{
    "tier": "professional",
    "organization": "Acme Corp",
    "issued": "2026-06-25",
    "expires": "2027-06-25",
    "features": ["sal", "fi", "scanner", "report"],
    "max_models": null,
    "signature": "<Ed25519 signature of above fields>"
}
```

What license enforcement does NOT do:
- No phone-home
- No feature gating (Community has everything)
- No model size limits
- No telemetry

What it does:
- Log a warning at import if no license and running in detectable production env (optional)
- Add license info to compliance reports
- `sal.license_info()` returns current tier for programmatic checks

The enforcement is deliberately light. The real enforcement is legal (BSL terms). The license file exists primarily for compliance reporting and organizational tracking.

---

## 5. Dependencies

### Required
- `torch >= 2.1`
- `numpy`
- `scipy` (numerical utilities)
- `PyNaCl` or `ed25519` (for license verification)

### Optional
- `transformers >= 4.38` (for HF Trainer integration and auto-detection)
- `peft >= 0.8` (for QLoRA + SAL composition)
- `matplotlib` (for visualization in reports)
- `reportlab` or `fpdf2` (for PDF compliance reports)

### Dev
- `pytest`
- `pytest-cov`

---

## 6. Testing Strategy

### Unit Tests (must pass without GPU)
- `test_masker.py`: mask application, re-randomization, hook installation/removal
- `test_config.py`: SALConfig creation, auto-detection, validation
- `test_fi.py`: graph construction, triangle-support FI computation, layer classification
- `test_plasticity.py`: 3-axis analysis, absorption map, hub detection, recommendations
- `test_license.py`: Ed25519 verification, expiry checking, tier parsing
- All tests use a tiny model fixture (4 layers, 8 heads, hidden_dim=64)

### Integration Tests (require GPU, run separately)
- SAL + HF Trainer full training loop (10 steps, tiny model)
- SAL + QLoRA composition
- FIScanner on real model (DistilBERT, smallest)
- Cross-architecture: verify hooks work on each supported architecture

### Validation Tests (manual, pre-release)
- Reproduce published results: GPT-2 124M, +7pp vs post-hoc
- Reproduce 345M variance results: 73× reduction
- Cross-architecture FI consistency

---

## 7. Compliance Report Format

The PDF compliance report (Enterprise tier, or audit add-on) contains:

1. **Model Identity**: architecture, parameter count, training config hash
2. **Pre-SAL Structural Baseline**: FI score, layer map, critical layers
3. **SAL Training Configuration**: prune_fraction, schedule, epochs
4. **Post-SAL Structural State**: FI score, layer map, delta vs baseline
5. **Compression Resilience Score**: predicted performance retention under N% pruning
6. **Layer-by-Layer Analysis**: IMMUNE/BUFFER/CRITICAL classification with FI contribution
7. **Recommendations**: which layers to prune first, expected impact
8. **Metadata**: package version, timestamp, license info

This report is designed to be attachable to a model card or regulatory filing.

---

## 8. What Is NOT in the Package

- SAL internal research code (H:\Structural Awareness Loss)
- Topostability / TopoIntegrity full framework
- GraphCoherence visualization
- Internal research framework and its mathematical proofs — not exposed
- Octopus architecture
- Any proprietary Cognitive Engineering IP beyond SAL mechanism + FI computation

The package exposes the RESULTS of the research (masking mechanism, FI metric, layer classification) without exposing the underlying proprietary research framework that produced them.

---

## 9. Build & Distribution

- **Build**: pyproject.toml, setuptools or hatchling
- **Distribution**: PyPI (public, BSL license declared)
- **Versioning**: SemVer (0.1.0 → 0.x for pre-1.0, 1.0.0 when API is stable)
- **CI**: GitHub Actions — lint, type-check, unit tests on CPU
- **Docs**: MkDocs or Sphinx, hosted on docs.cognitive-engineering.dev/sal

---

## 10. Development Phases

### Phase 0 — Scaffold (1-2 days) — ✅ COMPLETE
Repo setup, pyproject.toml, CI, license text, tiny model fixture, empty module files with docstrings.

### Phase 1 — Core SAL (3-5 days) — ✅ COMPLETE
HeadMasker, SALConfig, SALCallback, SALTrainer reconciled to the validated
mechanism (pre-hook on projection input, accumulative pruning, held set; FI =
triangle-support fragility). 26 unit tests pass on CPU.

### Phase 2 — Architecture Support (2-3 days) — ◀ IN PROGRESS
arch_support.py with auto-detection for top 10 architectures. Integration tests
on real models (DistilBERT, GPT-2, ViT, BERT) and an end-to-end SAL training
plumbing test, runnable locally or on GPU via Modal.

### Phase 3 — FI & Scanner (3-5 days)
Activation graph extraction, FI computation, layer classification, FIScanner, FIMonitor callback.

### Phase 4 — Plasticity Scanner (3-4 days)
Three-axis analysis (routing, CKA, MI), absorption map, hub detection, compression recommendations.

### Phase 5 — License & Reports (2-3 days)
Ed25519 verification, license.py, compliance report generation (JSON + PDF), plasticity heatmaps.

### Phase 6 — Documentation & Examples (2-3 days)
Getting started, API reference, examples, architecture support matrix.

### Phase 7 — Validation (3-5 days)
Reproduce published results on GPU. Cross-architecture validation. Performance benchmarks.

**Total estimated: 4-5 weeks with Claude Code**

---

## 11. Product Roadmap

### v0.1.0 — Foundation (July 2026)
**"Make any model compression-resilient in 3 lines of code"**

- SAL core: HeadMasker, SALConfig, SALCallback, SALTrainer
- FIScanner: structural fragility diagnostic (standalone, no training required)
- FIMonitor: live FI tracking during training
- Architecture auto-detection (10+ architectures)
- License system (Ed25519 offline)
- Compliance report export (JSON + PDF)
- Examples: BERT, LLaMA+QLoRA, ViT

### v0.2.0 — Intelligence (Q4 2026)
**"Know your model before you touch it"**

- PlasticityScanner: 3-axis structural mapping (routing / CKA / MI)
- Absorption map: which layers can take damage, which are hubs
- Compression recommendations: safe-to-prune vs never-touch
- `sal.compare()`: benchmark SAL vs SparseGPT / Wanda / magnitude on your model
- DeepSpeed integration
- Visual reports with heatmaps (PDF)

### v0.3.0 — Resilience (Q1 2027)
**"Models that survive anything"**

- StructuralGuard: continual learning protection (freeze critical, retrain elastic)
- Drift monitoring: measure catastrophic forgetting structurally
- Robustness suite: INT8/INT4 quantization survival, neuron dropout, fault injection
- Robustness certification report (defense / aerospace / medical)
- Dynamic inference routing: activate subnetworks by query complexity

### v0.4.0 — Transfer (Q2-Q3 2027)
**"Teach structure, not just outputs"**

- Topology-guided distillation: transfer structural organization teacher → student
- Cross-architecture structural comparison
- Multi-modal support (vision + language)
- Architecture discovery (experimental): FI as fitness function for topology evolution

### v1.0.0 — Platform (Q4 2027)
**"The structural intelligence layer for any AI pipeline"**

- Stable API guarantee
- JAX/Flax support
- HuggingFace Hub integration (structural metadata on model cards)
- Federated SAL: topology-guided model merging
- Enterprise dashboard (optional SaaS add-on)

---

## 12. Open Questions (to resolve during development)

1. **BSL change date**: when does BSL convert to Apache 2.0? Industry standard is 3-4 years.
2. **PyPI name**: `sal-torch` or `sal` or `structural-adaptive-learning`? Check availability.
3. **MoE support**: ship ExpertMasker in 0.1.0 or defer to 0.2.0?
4. **PDF report library**: reportlab (more control) vs fpdf2 (simpler)?
5. **Minimum Python version**: 3.9+ or 3.10+?
6. **Topology-guided selection**: include as experimental in 0.1.0 or defer?
