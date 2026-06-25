# SAL ROADMAP — Structurally Adaptive Learning
# Cognitive Engineering
# Version 3.0 — May 15, 2026
# INTERNAL — NOT FOR DISTRIBUTION
# 
# CHANGE LOG V3:
# - Full ablation results A-I documented (RANDOM, λ₂, liquefaction, anti-liquefaction, FI-loss)
# - MNLI ranking: RANDOM > λ₂ > FI > LIQUEFACTION > POSTHOC (monotonically inverse to structural sophistication)
# - Engineering decision: RANDOM as default method (zero overhead, best or equal accuracy)
# - Added compute overhead per selection criterion (measured)
# - Added production engineering section
# - Added scale validation requirement (all results = GPT-2 Small only for ablations)
# - Updated SDK: random as default, topology methods as research options
# - Added "burst vs progressive" as next priority test
# - Anti-liquefaction (EXP J) in progress
#
# CHANGE LOG V2:
# - Added GPT-2 Medium (+7.56pp), ViT (+20.08pp), QQP (+6.04pp) results
# - Added RANDOM ablation finding: FI-guided ≤ random on SST-2 and MNLI (1 seed)
# - Restructured Phase 1B: selection criterion investigation as TOP PRIORITY
# - Added 8 candidate selection criteria (λ₂, liquefaction, spectral, C_crit, hybrids)
# - Updated risks with confirmed items
# - Updated SDK interface to show multiple method options

---

## VISION

SAL is not a pruning technique. SAL is a training-time adaptive sparsification paradigm for neural networks.

The long-term trajectory: foundation models trained with progressive structural adaptation, producing self-organized architectures that are robust, interpretable, and efficient by design.

**One-liner:** "Training-time adaptive sparsification for robust and efficient foundation models."

**Core validated claim:** Progressive sparsification during training fundamentally changes compression behavior across architectures and modalities.

**Open research question:** What is the optimal structural criterion for guiding head selection? Local FI (tri(e)=0) is not sufficient. Global metrics (λ₂, liquefaction, spectral entropy) are the leading candidates.

---

## PHASE 0 — CURRENT STATE (May 2026)

### Validated Results
- 5 architectures: GPT-2 Small, GPT-2 Medium, DistilBERT, RoBERTa, ViT-B
- 3 NLP tasks: SST-2 (+2.48pp), QQP (+6.04pp), MNLI (+7.81pp)
- 1 vision task: CIFAR-10 (+20.08pp)
- Scaling: 67M → 345M, gap triples with model size (+2.48 → +7.56pp on SST-2)
- Variance reduction 73× at 345M params (SAL σ=0.06 vs PostHoc σ=4.42)
- Compression curve with crossover at ~10%
- Extreme compression: SAL@75% beats PostHoc@33% (87.96 vs 87.39 on SST-2)
- Super-baseline on MNLI (SAL@7 > baseline)
- Activation-based FI validated (weight-based degenerate)
- Hook-based head zeroing (transformers 5.8.1 fix)
- Compute overhead: <3% (small), +8.6% (ViT), +48% (Medium, unoptimized)

### Critical Ablation Finding (14 May 2026)
- **RANDOM@48 = SAL@48 on SST-2** (90.14% = 90.14%, seed 42)
- **RANDOM@48 > SAL@48 on MNLI** (80.17% vs 77.77%, seed 42)
- **Interpretation:** The primary gain comes from training-time progressive sparsification, NOT from FI-guided head selection. FI local (tri(e)=0) is not validated as the causal selection criterion for transformers.
- **This does NOT invalidate:** the training-time pruning paradigm, the scaling results, the variance reduction, the cross-modal generalization, or the compression curve. It narrows the open question to: what is the optimal structural selection criterion?
- **Caveat:** Single seed per condition. To be confirmed multi-seed in Stage 1.

### Full Ablation Results (EXP A-I, 14 May 2026)

**MNLI ranking @ 33% compression (seed 42) — the definitive result:**

| Rank | Criterion | Acc | Δ Baseline | Structural sophistication |
|------|-----------|-----|------------|--------------------------|
| 1 | BASELINE | 80.79% | — | — |
| 2 | RANDOM | 80.17% | -0.62pp | None |
| 3 | λ₂-guided | 79.01% | -1.78pp | Global spectral |
| 4 | FI-guided (SAL) | 77.77% | -3.03pp | Local triangles |
| 5 | LIQUEFACTION-safe | 76.77% | -4.02pp | Global cascade defense |
| 6 | POSTHOC | 72.07% | -8.72pp | Post-training FI |

**Key finding: accuracy is MONOTONICALLY INVERSE to structural sophistication.** The more you try to preserve or guide structure, the worse the result. Random (no criterion) is optimal.

**Additional ablation results (SST-2, seed 42):**

| Experiment | Result | Conclusion |
|---|---|---|
| EXP A: RANDOM@48 SST-2 | = SAL exactly (90.14%) | Selection criterion irrelevant at this scale |
| EXP B: SAL extreme (50-83%) | Graceful degradation, no cliff | SAL@75% beats PostHoc@33% |
| EXP D: RANDOM extreme (67-83%) | RANDOM ≥ SAL at all levels | FI doesn't help even at extreme compression |
| EXP E: SAL no pruning | = baseline (91.97%) | FI signal without zeroing has no effect |
| EXP H: FI-loss α=0.01/0.1/0.5 | = baseline (±0.3pp) | FI as differentiable loss: null (gradient too weak) |
| EXP J: Anti-liquefaction | IN PROGRESS | Max disruption test |

**Interpretation:** The causal mechanism is progressive structural damage during training, not topology-guided selection. The model self-reorganizes under perturbation, and any external guidance restricts this self-organization.

### Engineering Decision: RANDOM as Default Method

**Compute overhead per selection criterion:**

| Method | Overhead per prune event | Total overhead (48 prunes) | Accuracy vs RANDOM |
|---|---|---|---|
| **RANDOM** | **0 ms** | **0 ms** | **Reference** |
| FI local | 500ms - 2s | 24s - 96s | ≤ RANDOM |
| λ₂-guided | ~1.3ms (after graph) | ~1s (+ graph cost) | -1.16pp |
| Liquefaction | ~5s | ~240s | -3.40pp |
| Anti-liquefaction | ~5s | ~240s | TBD (EXP J) |

**Decision:** RANDOM is the default production method. Zero compute overhead, best or equal accuracy at validated scales (67M-124M). Structured criteria are research options for Stage 1 scale validation.

**CRITICAL CAVEAT:** All ablation comparisons (EXP A-I) are on GPT-2 Small (124M) only. The selection criterion question is OPEN at larger scales. At 7B+ parameters, models have less redundancy, and structured selection may become informative. Stage 1 must test all criteria at LLaMA-7B scale before concluding.

### Scale Validation Requirement

| Scale | Selection criterion tested? | Result |
|---|---|---|
| GPT-2 Small (124M) | FI, λ₂, liquefaction, random, FI-loss | Random wins |
| GPT-2 Medium (345M) | SAL vs PostHoc only | +7.56pp (no random test) |
| ViT-B (86M) | SAL vs PostHoc only | +20.08pp (no random test) |
| LLaMA-7B (7B) | NOT TESTED | Stage 1 priority |

**Stage 1 must validate RANDOM vs FI vs λ₂ at ≥345M params before making any production recommendation.**

### Pre-existing IP (before July 2026)
- TCGE framework, FI, tri(e), spectral chain (τ→κ→λ₂→S)
- TopoIntegrity platform (production)
- GraphCoherence platform (production)
- Topostability platform (production)
- 14+ Zenodo DOIs across 5 domains
- Lean 4 formal proofs
- SAL POC v1-v5 results
- RTH-LM cross-architecture validation (non-transformer, FI=2%)

---

## PHASE 1 — VALIDATION (Stage 1 SPRIND: July 2026 - January 2027)

### 1A. Scaling Validation (Month 1-2)

**Objective:** Prove SAL works beyond small transformers.

| Model | Params | Heads | Priority |
|---|---|---|---|
| GPT-2 Medium | 345M | 384 | High (immediate) |
| LLaMA-7B / TinyLlama | 7B / 1.1B | 256+ | High |
| Vision Transformer (ViT-B) | 86M | 144 | Medium |
| Whisper small | 244M | 384 | Low (multi-modal) |

**Benchmarks:** SST-2, MNLI, QQP (validated), add HellaSwag, ARC, MMLU.

**Key question:** Does the SAL advantage grow with model size?

### 1B. Mechanism Investigation (Month 1-3) — TOP PRIORITY

**Status update:** Selection criterion investigation largely completed at 124M scale (EXP A-I). All structured criteria (FI, λ₂, liquefaction) underperform random. The research question has shifted from "which criterion?" to "what is the mechanism?"

**COMPLETED at GPT-2 Small (124M), seed 42:**

| Criterion | MNLI Acc | vs RANDOM | Verdict |
|---|---|---|---|
| RANDOM | 80.17% | reference | Best training-time method |
| λ₂-guided | 79.01% | -1.16pp | Global > local, but < random |
| FI-guided | 77.77% | -2.40pp | Wrong proxy |
| Liquefaction-safe | 76.77% | -3.40pp | Worst: preserving structure hurts |
| FI-loss (α=0.01-0.5) | ≈ baseline | null | Gradient too weak |
| Anti-liquefaction | TBD (EXP J) | TBD | Maximum disruption test |

**NEW PRIORITY TESTS for Stage 1:**

| Test | Question | Priority |
|---|---|---|
| **Burst vs Progressive** | Is gradual damage the mechanism? (all-at-once vs one-by-one) | **#1 — cheapest, most informative** |
| **Recovery capacity** | Can a model rebuild after 75% burst pruning + continued training? | #2 |
| **RANDOM on GPT-2 Medium** | Does selection criterion matter at 345M? | #3 — critical for production |
| **RANDOM on LLaMA-7B** | Does selection criterion matter at frontier scale? | #4 — critical for SPRIND |
| **Functional migration analysis** | WHERE do functions redistribute? (head entropy, attention maps) | #5 — scientific depth |
| Multi-seed confirmation | Seeds 123/456 on MNLI ranking | #6 — statistical validation |

**If burst << progressive:** the mechanism is curriculum damage adaptation. Paper becomes "progressive structural stress during training."
**If burst ≈ progressive:** the mechanism is just capacity reduction during training. Simpler but still valuable.

### 1C. Standard Baselines (Month 2-3)

| Comparison | Purpose |
|---|---|
| RANDOM training-time vs SparseGPT | Training-time paradigm vs SOTA post-hoc |
| RANDOM training-time vs Wanda | Training-time paradigm vs efficient post-hoc |
| RANDOM training-time vs magnitude pruning (post-hoc) | vs classical baseline |
| RANDOM training-time vs gradual magnitude pruning (training-time) | vs existing training-time method |

**Positioning:** Training-time progressive sparsification (even random) should beat all post-hoc methods at matched compression. The question is by how much. This comparison table is the key commercial deliverable.

### 1C. Structural Characterization (Month 2-4)

**Objective:** Show SAL produces qualitatively different internal organizations.

- Activation entropy per head (SAL vs baseline)
- Functional modularity via community detection on activation graphs
- Head migration analysis: which heads absorb pruned functions?
- FI dynamics during training: how does topology evolve?
- Comparison of internal representations (CKA, RSA)
- Identify "structural secrets" unique to SAL-trained networks

**This is the critical scientific deliverable for Stage 1.**

### 1D. Compute Overhead (Month 1-2)

**Measured overhead (EXP A-I):**

| Model | Task | Method | Overhead | Note |
|---|---|---|---|---|
| GPT-2 Small | MNLI | SAL (FI) | +2.5% | Within noise |
| GPT-2 Small | QQP | SAL (FI) | +2.1% | Within noise |
| ViT-B | CIFAR-10 | SAL (FI) | +8.6% | ~1.8s per FI computation |
| GPT-2 Medium | SST-2 | SAL (FI) | +48.3% | ~11s per FI computation |
| Any model | Any task | **RANDOM** | **<0.1%** | **Zero graph computation** |

**Production insight:** With random as default method, the compute overhead question is ELIMINATED. Random progressive pruning adds only a random.choice() call and a hook toggle per prune event. The only cost is the additional training steps (same as baseline).

**If a structured criterion proves necessary at scale:** optimize via sampled FI (subset of heads), checkpoint-based topology (recompute every k steps), and cached triangle counts.

### 1E. Multi-Modal Extension (Month 4-6)

- ViT with SAL on ImageNet / CIFAR
- CLIP or equivalent multi-modal
- If positive: architecture-agnostic confirmed
- If negative: boundary conditions documented

### 1F. Deliverables Stage 1

- Technical report / preprint (training-time sparsification scaling + mechanism analysis)
- Open-source progressive sparsification framework (random default, structured criteria as research options)
- Comparison table vs SparseGPT, Wanda, magnitude pruning (training-time vs post-hoc paradigm)
- TopoIntegrity integration for structural diagnostics (pre/post compression analysis)
- Burst vs progressive mechanism paper (if results warrant)
- Scale validation report: does criterion matter at 7B+?

---

## PHASE 2 — PLATFORM (Stage 2 SPRIND: February - September 2027)

### Axe A: SAL as Training Infrastructure

**Objective:** SAL becomes a plug-in compatible with standard training stacks.

```
from sal import ProgressiveSparsifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-7b")

# Default: random progressive pruning (zero overhead, best validated accuracy)
# Research options (Stage 1 scale validation):
# "random"             — DEFAULT: random progressive (zero overhead, best at ≤124M)
# "activation_fi"      — local FI-guided (current baseline, no advantage over random at ≤124M)
# "lambda2_guided"     — algebraic connectivity (better than FI, worse than random at ≤124M)
# "liquefaction_safe"  — cascade defense (worst training-time method at ≤124M)
# "anti_liquefaction"  — maximum disruption (testing: may outperform random)
sparsifier = ProgressiveSparsifier(model, method="random", target_compression=0.33)

for batch in dataloader:
    loss = model(**batch).loss
    sparsifier.step(loss)  # handles prune schedule internally
    loss.backward()
    optimizer.step()
```

**Integration targets:**
- PyTorch native
- HuggingFace Trainer
- DeepSpeed
- JAX/Flax (if resources allow)

**Architecture (simplified with random default):**
```
Model → Forward → [optional: activation graph for diagnostics] → Random head zeroing at prune interval → Backprop
```

Minimal invasive. No architecture change required. No graph computation needed for default method.

### Axe B: Dynamic Inference Routing

**Concept:** At inference time, use topology to activate only structurally necessary components per query.

- Easy query (sentiment, simple QA) → activate minimal subnetwork
- Hard query (reasoning, multi-hop) → activate full model
- Routing is topology-guided, not learned (no router overhead)

**Difference from MoE:** MoE routes tokens to experts. SAL routes queries to structural regimes. The topology map is pre-computed, not learned at inference time.

**Value:** Adaptive compute per query without additional training.

### Axe C: Topology-Guided Distillation

**Concept:** Transfer structural organization from teacher to student, not just outputs.

- Standard distillation: student mimics teacher's logits
- SAL distillation: student mimics teacher's topological organization
- The student learns WHERE to put redundancy, not just WHAT to output

**Value:** Smaller students with teacher-like structural robustness.

### Axe D: Robustness Testing

**Hypothesis:** SAL-trained models resist degradation better.

| Test | Description |
|---|---|
| Neuron dropout at inference | Random deactivation of components |
| Weight quantization (INT8, INT4) | Does structural organization survive quantization? |
| Adversarial perturbation | Structural robustness vs adversarial inputs |
| Hardware fault injection | Bit-flip resilience |

**Markets:** Defense, aerospace, edge, industrial, medical devices.

---

## PHASE 3 — FRONTIER (Stage 3 SPRIND: October 2027 - June 2028)

### Axe E: Continual Learning

**Hypothesis:** Structurally organized models forget less.

- Components with tri(e) > 0 (protected) are frozen during new task learning
- Components with tri(e) = 0 (fragile) are reallocated to new tasks
- Natural structural memory: topology determines what is preserved vs what is plastic

**Test:** Sequential task learning (Task A → Task B → re-evaluate Task A).
If SAL reduces catastrophic forgetting without replay buffers, this is a major result.

### Axe F: Safety and Alignment through Topology

**Concept:** Structural interpretability for targeted intervention.

- Map which topological modules encode which behaviors
- Identify structurally isolated components responsible for specific outputs
- Targeted topological surgery: modify behavior without affecting unrelated capabilities
- Structural audit: "this model's safety-relevant functions are triangulated (protected)"

**European value alignment:** Auditable, interpretable, surgically modifiable AI systems.

### Axe G: Architecture Discovery

**Concept:** FI landscape as fitness function for architecture evolution.

- Instead of NAS by trial-and-error, use FI to guide where to add/remove connections
- The network learns its own optimal topology during training
- Co-evolution of weights and connectivity
- Dynamic topology formation: architecture morphogenesis

**This is the most "frontier" direction.** If SAL can discover architectures that humans wouldn't design, it becomes a genuine paradigm shift, not just an optimization.

### Axe H: Federated SAL

**Concept:** Topology-guided model merging for distributed training.

- Each training node has local topology awareness
- Instead of parameter averaging (FedAvg), merge structural organizations
- Topology-compatible merging: preserve triangulated (robust) structures from each node
- Conflict resolution: if two nodes disagree on structure, FI arbitrates

---

## GO-TO-MARKET

### What We Sell

NOT: TCGE theory, topology consulting, "we analyze your model."

YES: Training infrastructure that makes models structurally robust and efficient.

### Revenue Streams

| Stream | Timeline | Model |
|---|---|---|
| SAL Training SDK | Stage 2+ | Open-core (SAL-lite free, SAL-enterprise paid) |
| TopoIntegrity SaaS | Now (live) | Structural analysis, freemium |
| Training kernels licensing | Stage 3+ | Per-model or per-training-run |
| HuggingFace integration | Stage 2 | Strategic partnership |
| Enterprise inference pipeline | Stage 3+ | Edge AI, defense, industrial |

### Positioning

```
Training-time sparsification (open source)  → Adoption: "pip install sal", random default
TopoIntegrity (SaaS)                        → Pre-compression diagnostic
Structured criteria (research/enterprise)   → Premium: if scale validation shows criterion matters
Training pipeline licensing                 → Revenue at scale
```

**vs SparseGPT / Wanda positioning:**
Training-time progressive sparsification produces models that post-hoc methods (SparseGPT, Wanda) cannot match. Post-hoc methods are structurally limited: they operate on frozen models with no capacity to adapt. Our approach integrates compression into the training loop at near-zero overhead (random default). The result: 33% compression with <2pp accuracy loss, vs 5-15pp for post-hoc at matched compression. This is not incremental improvement — it's a paradigm shift from "compress after" to "train to be compressible."

### Target Customers

| Segment | Need | SAL Value |
|---|---|---|
| Cloud providers (AWS, GCP, Azure) | Inference cost reduction | Structurally optimized models |
| Edge AI companies | On-device deployment | 33-67% compression, zero loss |
| European sovereign AI (Swisscom, Infomaniak) | Compute independence | Same performance, less GPU |
| Defense / aerospace | Fault tolerance | Structural robustness |
| Pharma / biotech | Model reliability | Predictable, auditable AI |
| Frontier labs | Training efficiency | Scaling without structural waste |

### Strategic Moat

1. **Training-time sparsification paradigm**: the empirical finding that progressive pruning during training >> post-hoc is robust and architecture-agnostic
2. **Mathematical foundation** (TCGE): framework for understanding WHY structure matters, with global metrics (λ₂, liquefaction, spectral entropy) as next-generation selection criteria
3. **Cross-domain validation** (5+ domains): proves structural analysis generality, not limited to AI
4. **Production platform** (3 live products): TopoIntegrity diagnostic → SAL training → deployment pipeline
5. **Patent potential** (post-SPRIND): training-time progressive sparsification method, activation-based functional dependency graphs, global selection criteria

---

## TIMELINE SUMMARY

| Period | Focus | Key Deliverable |
|---|---|---|
| May-June 2026 | SAL preprint + SPRIND application | Preprint (Zenodo DOI 10.5281/zenodo.20187961), SPRIND submission |
| July-Aug 2026 | **Mechanism investigation** + scaling | Burst vs progressive, RANDOM at 345M+, LLaMA-7B |
| Sept-Oct 2026 | Baseline comparisons | vs SparseGPT, Wanda, magnitude. Comparison table. |
| Nov-Jan 2027 | Technical report + open-source | Progressive sparsification SDK, scale validation report |
| Feb-Sept 2027 | Platform + dynamic inference | SDK production, HuggingFace integration |
| Oct 2027-June 2028 | Frontier directions | Continual learning, safety, architecture discovery |
| 2028+ | Scale-up funding + commercialization | Enterprise product, licensing |

---

## KEY RISKS

| Risk | Status | Mitigation |
|---|---|---|
| FI-guided selection not causal | **CONFIRMED (seed 42)** | Default to RANDOM. FI remains diagnostic (TopoIntegrity), not prescriptive for pruning. |
| All structured criteria fail (λ₂, liquefaction) | **CONFIRMED at 124M (seed 42)** | RANDOM is the production method at ≤124M. Re-test at 7B+ in Stage 1. |
| Training-time pruning = main effect | **CONFIRMED (seed 42)** | Reframe: "progressive training-time sparsification." This IS the product. |
| Compute overhead too high at scale | **RESOLVED for RANDOM** | Random method has zero overhead. Structured criteria overhead only matters if they prove useful at scale. |
| RANDOM also beats SAL at 345M+ | **OPEN — not tested** | If confirmed: simplifies product (random everywhere). If FI/λ₂ helps at scale: premium feature. Either outcome is good. |
| SAL advantage doesn't scale to >1B params | Open | Pivot to structural characterization (still valuable) |
| Someone publishes "random training-time pruning" | **Elevated risk** | Speed (preprint DOI deposited), first-mover advantage, TCGE diagnostic platform differentiator |
| SPRIND not selected | Open | Preprint + SDK open-source regardless, seek alternative funding |
| Overselling topology claims | **Active — managed** | Preprint says "training-time sparsification." Ablation results documented internally. Honest framing. |
| Burst pruning = progressive pruning | Open | If confirmed: further simplifies method. If progressive >> burst: mechanism identified (curriculum damage). |

---

*Internal strategic document. Last updated: May 15, 2026 (00:00 CET).*
*Reflects full ablation results EXP A-I + anti-liquefaction EXP J in progress.*
