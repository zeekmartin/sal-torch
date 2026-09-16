# sal-torch

![CI](https://github.com/zeekmartin/sal-torch/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/sal-torch) ![Python](https://img.shields.io/pypi/pyversions/sal-torch) ![Downloads](https://pepy.tech/badge/sal-torch) ![License](https://img.shields.io/badge/license-BSL%201.1-blue)


**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

## Install

```bash
pip install sal-torch            # core
pip install sal-torch[hf]        # + HuggingFace Trainer
pip install sal-torch[reports]   # + PDF/visual reports
pip install sal-torch[crypto]    # + commercial license verification
pip install sal-torch[all]       # everything
```

```python
from sal import SALConfig, SALCallback

config = SALConfig.auto(model)
trainer = Trainer(model=model, callbacks=[SALCallback(config)])
trainer.train()
```

Three lines. Any transformer. Compression-resilient.

> **Fully fine-tune.** SAL works by letting the model reorganize around silenced
> heads, so it needs the weights that do the reorganizing to be trainable. Under
> LoRA/QLoRA we measured it as actively harmful — see
> [When to use SAL](#when-to-use-sal).

## CompressionPipeline — the validated path, in one object

The measured recipe is: **fully fine-tune with SAL, then compress.** Doing that
by hand means wiring config → masker → training loop → head selection → slicing
→ a quantization backend → an eval harness, and getting the training method
wrong quietly costs about three points of accuracy. `CompressionPipeline` makes
that path the default and measures every stage, so what comes out is a
deployment decision rather than a number.

```python
from sal import CompressionPipeline

pipe = CompressionPipeline(model, eval_dataset, metric="accuracy")

print(pipe.scan().recommendation)      # fragility, absorption map, projected sizes
pipe.sal_train(train_dataset, epochs=3, prune_fraction=0.33)
pipe.compress(pruning=0.33, quantization="int4", slice_heads=True)

report = pipe.validate()
print(report.table)
# measured — GPT-2 Medium / SST-2, one T4 (scripts/modal_v050_test.py)
# stage                    size_mb   ratio    accuracy  seconds
# -------------------------------------------------------------
# original                  1419.3   1.00x      0.5020      0.0
# sal_trained               1419.3   1.00x      0.8965    242.1
# pruned+sliced             1293.4   1.10x      0.8496      0.4
# quantized (int4)           346.5   4.10x      0.8398      3.1

pipe.export("compressed_model/")       # reloaded and checked, not just written
# the exported model reloads at 0.8398 — exactly what was measured
pipe.report().save("compression_report.pdf")
```

`slice_heads=True` is what makes the saving real: masking a head makes it
*behave* as if it were gone, slicing removes it from the weight matrices. The
exported model runs with **no hooks and without sal-torch installed**.

**Which heads you remove matters more than SAL does.** `compress()` takes a
`strategy`, and the choice is worth making deliberately — measured on GPT-2
Medium, removing the same 120 heads three different ways:

```
arm           dense    magnitude       random    fi_guided
standard     0.8926       0.8594       0.8867       0.7520
SAL          0.8965       0.8398       0.8906       0.7617
```

`random` is the best of the three **for both arms** and is the default. The
obvious heuristic, `magnitude`, is 2.7 points worse for a standard model and 5.7
for a SAL-trained one — a small weight norm turns out to be a poor proxy for a
head the model can spare. `fi_guided` — spend the budget where the fragility scan
says it is cheap — is much worse than either, because concentrating removal does
more damage than spreading it.

That grid was one seed, and under `random` its two arms landed 0.0039 apart —
a tie. The five-seed run in [What we measured](#what-we-measured) settles it
properly: with `random` selection, SAL leads head pruning by 2.45pp on 5 of 5
seeds. See [ROADMAP.md](ROADMAP.md) for the full trail.

**It refuses LoRA/QLoRA models.** Not a warning — an error, with the reason and
what to do instead. SAL works by letting the model reorganize around silenced
heads, and adapters freeze the weights that would do the reorganizing; measured,
SAL under LoRA lost four of six compression variants *and* gave up 3.1 points of
clean accuracy. It also warns below 100M parameters, and an optional
`accuracy_floor` stops the run rather than handing back a model that is small
and broken.

### The pieces, if you want them separately

```python
from sal import slice_heads, quantize, quantize_info

print(quantize_info(model))   # sizes per method + which backends work here
small = slice_heads(model, heads_to_remove=[(0, 3), (1, 3), ...])
small = quantize(small, method="int4")   # bitsandbytes NF4, or torch.ao INT8
```

`slice_heads` requires the same number of heads removed from every layer, and
refuses grouped-query attention — architectures store one head count, and
removing query heads without whole KV groups corrupts the mapping. It raises in
both cases rather than returning something quietly wrong.

## Know your model before you touch it

### FIScanner — how fragile is this model?

The **Fragility Index** is a structural diagnostic, scored in `[0, 1]`, that
measures how much redundant pathway a model's attention graph has. Heads are
compared by their activation signatures; an edge between two heads is *fragile*
when they share no common neighbour, i.e. the function it carries has no backup.
FI is the fraction of such edges.

- **Low FI** → heavily triangulated graph, lots of redundancy → robust.
- **High FI** → many unsupported edges → fragile under compression.

FI is purely diagnostic — it measures, it never perturbs. You can use it with or
without SAL training.

```python
from sal import FIScanner

result = FIScanner(model, probe_dataset).scan()

print(result.fi_score)         # 0.0 - 1.0; lower is more robust
print(result.summary)          # "FI=0.1842 | 3 immune, 2 buffer, 1 critical"
print(result.critical_layers)  # layers whose removal moves FI the most
print(result.immune_layers)    # layers you can compress with little effect

result.save("fragility.json")
result.save("fragility.pdf")   # per-head heatmap (needs sal-torch[reports])
```

Track it *during* training with `FIMonitor`, or call the primitives directly:

```python
from sal import FIMonitor, compute_fi, extract_activation_graph

trainer = Trainer(model=model, callbacks=[FIMonitor(probe_dataset, interval=500)])

adjacency = extract_activation_graph(model, probe_dataset)
fi = compute_fi(adjacency)
```

### PlasticityScanner — where can a model absorb compression?

FI tells you how fragile a model *is*. `PlasticityScanner` tells you how much
room it has to *reorganize*, so you know where it is safe to compress. It scores
three complementary axes per layer — routing flexibility (attention entropy),
inter-layer redundancy (linear CKA), and intra-layer redundancy (an MI proxy) —
and folds them into an **absorption map** that labels each layer `ELASTIC`
(safe), `SATURATED` (bottleneck), or `HUB` (compensates when others are pruned).

```python
from sal import PlasticityScanner

pmap = PlasticityScanner(model, probe_dataset).scan()
print(pmap.summary)              # "3 elastic, 1 saturated, 2 hub | mean routing=0.61 ..."

rec = pmap.recommend(target_compression=0.33)
rec.safe_to_prune                # [(layer, head), ...] — prune these first
rec.never_touch                  # heads in hub layers — leave alone
rec.expected_impact              # heuristic accuracy delta

pmap.save("plasticity.json")     # raw scores
pmap.save("plasticity.pdf")      # visual report (needs sal-torch[reports])
```

### Cross-architecture structural profiles

Both scanners are architecture-agnostic, which is only useful if the numbers
they return are architecture-*specific*. Four pretrained checkpoints, no
training, one 256-sample probe each
(`scripts/modal_plasticity_compare.py`, ~2-7s per model on a T4):

| model | probe | layers | heads | FI | buffer | critical | elastic | saturated | hub |
|---|---|---|---|---|---|---|---|---|---|
| `distilbert-base-uncased` | text | 6 | 12 | **0.3086** | 2 | 4 | 1 | 3 | 2 |
| `gpt2` | text | 12 | 12 | 0.0990 | 1 | 11 | 0 | 2 | **10** |
| `bert-base-uncased` | text | 12 | 12 | 0.0903 | 3 | 9 | 3 | 7 | 2 |
| `google/vit-base-patch16-224` | vision | 12 | 12 | 0.0660 | 2 | 10 | 1 | 5 | 6 |

| model | routing | CKA | MI |
|---|---|---|---|
| `distilbert-base-uncased` | 0.3141 | 0.9577 | 0.0430 |
| `gpt2` | 0.1787 | 0.9713 | 0.0610 |
| `bert-base-uncased` | 0.3161 | 0.9712 | 0.0363 |
| `google/vit-base-patch16-224` | **0.7110** | 0.9170 | 0.0291 |

- **DistilBERT is the fragile one — FI 0.3086**, roughly 3× the nearest model
  (3.1× GPT-2, 3.4× BERT, 4.7× ViT) and the only distilled checkpoint here.
  Distillation is exactly the process that would strip redundant pathway. It is
  also the only 6-layer model, so depth is a live confound.
- **BERT has more room to compress than GPT-2** — 3 elastic layers against
  **0**, and 2 hub layers against GPT-2's 10. Nearly every GPT-2 layer is doing
  compensating work, so there is very little the absorption map is willing to
  call safe.
- **GPT-2 and BERT are identical on paper** — 12 × 12, 144 heads — and land at
  opposite ends of that map. Causal and bidirectional attention build different
  structures at the same size.
- **ViT routes far more freely** than any text model (0.711 against 0.18–0.32),
  with the lowest FI and the lowest intra-layer redundancy. GPT-2 is its
  opposite on every axis. ViT's 6 hub layers sit between BERT's 2 and GPT-2's
  10 — a distinct profile, but hub-heaviness is not a vision-specific trait.

The practical point: **these four models want four different compression
strategies**, and the scan that tells you which takes 1.6–6.8s per model on a
T4 with no training and no labels. Prune BERT's elastic layers; think much
harder before touching GPT-2.

Two honest caveats. **No model here has a single IMMUNE layer** — the <1%
relative-FI threshold looks unreachable on real pretrained checkpoints, so treat
IMMUNE as theoretical at the current calibration. And the ViT row is probed with
images while the other three get sentences: the three text models are a
controlled comparison, ViT is a fourth architecture on its own probe.

**FI ranks structure; it does not predict compression survival.** ViT-base has
the *lowest* FI in this table and still gives up 27 accuracy points at 33% head
pruning (see [What we measured](#what-we-measured)). Use `RobustnessTest` for
survival questions and FI for structural ones.

Load with `attn_implementation="eager"`, or the model returns no attention
weights, routing entropy comes back NaN, and hub detection quietly reports zero
hubs — which reads exactly like a finding.

**Self-supervised encoders scan too.** I-JEPA ViT-H/14
(`facebook/ijepa_vith14_1k`, 631M, 32 × 16) loads through `AutoModel` and needs
no special handling — the architecture registry does not list `ijepa`, but the
scanners locate its attention projections anyway. Against ViT-base *and* a
ViT-large depth control on the same probe, most of its profile is a depth trend
rather than an objective effect: its FI (0.0040) is indistinguishable from
ViT-large's (0.0041). Two axes are not. Its heads are about twice as correlated
within a layer as either supervised ViT, and its routing entropy *falls* with
depth where both supervised models *rise* — broad early, narrow late. See
`scripts/modal_jepa_scan.py` and `scripts/jepa_scan_results.json`.

### sal.compare() — SAL vs. other pruning methods

Benchmark SAL against post-hoc baselines at a matched compression level and see
which keeps the most accuracy (or lowest loss) after heads are removed.

```python
from sal import compare

result = compare(model, train_dataset, eval_dataset,
                 methods=["sal", "magnitude", "random_posthoc"],
                 compression=0.33, sal_epochs=3, metric="accuracy")
print(result.table)              # method | score | pruned_heads | time
print(result.winner)
result.save("comparison.pdf")    # bar chart + table

# plug in your own method
compare.register_method("my_pruner", lambda model, ds, eval_ds, ctx: my_score)
```

## Does it survive real compression?

We polled practitioners on how they actually compress models. Of 33 responses,
**39% quantize** (INT8/INT4) — more than pruning and distillation. SAL was built
against head pruning, so the honest question is whether the resilience it trains
in generalizes to the compression people actually ship.

Short answer: **yes, if you fully fine-tune — and the win is a pruning win
before it is a quantization win.** On the primary evidence, five seeds of GPT-2
Medium, a SAL-trained model keeps 2.45pp more accuracy under 33% head pruning on
5 of 5 seeds at no cost to clean accuracy; INT4 alone is a coin flip, and under
LoRA the whole thing loses. A preliminary three-seed vision study points the
same way and much larger — cross-modal validation is under investigation, not
established. The numbers, including the rows SAL did not win, are in
[What we measured](#what-we-measured).

### RobustnessTest — one model, every degradation

```python
from sal import RobustnessTest

test = RobustnessTest(model, eval_dataset, metric="accuracy")
report = test.run(methods=["int8", "int4", "head_pruning_33", "head_pruning_50",
                           "neuron_dropout_10", "neuron_dropout_20"])

print(report.table)
# real output — DistilBERT fine-tuned on SST-2, no SAL, 512 eval examples
# method              baseline     after     delta     std  survived
# ------------------------------------------------------------------
# int8                  0.8594    0.8359   -0.0234       -        OK
# int4                  0.8594    0.8672   +0.0078       -        OK
# head_pruning_33       0.8594    0.7656   -0.0938       -      FAIL
# head_pruning_50       0.8594    0.7363   -0.1230       -      FAIL
# neuron_dropout_10     0.8594    0.8392   -0.0202   0.015        OK
# neuron_dropout_20     0.8594    0.8320   -0.0273   0.022        OK

print(report.robustness_score)   # aggregate 0-1: mean quality retained
print(report.survival_rate)      # fraction of methods survived

report.save("robustness.json")
report.save("robustness.pdf")    # bars + retention radar (needs sal-torch[reports])
```

A method counts as **survived** when relative degradation stays within
`survival_threshold` (default 5% of the clean baseline). Methods:

| Method | What it does |
|---|---|
| `int8` | Dynamic INT8 over every `nn.Linear` (`torch.ao.quantization`, CPU) |
| `int4` | 4-bit weight-only — bitsandbytes NF4 when available, otherwise a simulated per-channel INT4 round-trip (the backend used is recorded on each result) |
| `head_pruning_<pct>` | Silences `<pct>`% of attention heads using the shipped `HeadMasker` |
| `neuron_dropout_<pct>` | Zeroes `<pct>`% of FFN neurons at inference — dead units / noisy hardware. Repeated over several fault patterns; mean ± std reported |

`pip install sal-torch[quant]` adds bitsandbytes for real NF4. Without it, `int4`
falls back to simulation rather than disappearing from your report — pass
`allow_simulated_quant=False` if you would rather see the row skipped.

### robustness_compare() — SAL-trained vs. standard

```python
from sal import robustness_compare

result = robustness_compare(
    sal_model=sal_trained_model,
    baseline_model=standard_model,
    eval_dataset=eval_dataset,
    methods=["int8", "int4", "head_pruning_33"],
    metric="accuracy",
)

print(result.table)
print(result.summary)
result.save("robustness_comparison.pdf")
```

Each model is scored against **its own** clean baseline, so the comparison
measures resilience rather than which model was better to begin with. The row
winner is whichever model loses proportionally less.

### What we measured

The primary evidence is GPT-2 Medium at five seeds. A three-seed vision study
follows it as **secondary, preliminary** evidence — promising, and not yet
enough to call SAL cross-modally validated.

#### Five seeds, full eval split — GPT-2 Medium (primary evidence)

Every number this project published before v0.5.0 was a single seed, which is
not enough: two GPT-2 baselines trained on identical data differ by about a
point, and one single-seed conclusion here has already been overturned. So the
v0.4.0 protocol was re-run five times.

GPT-2 Medium / SST-2, **full fine-tuning**, seeds 42/123/456/789/1337, scored on
the **complete 872-example validation split**. Each seed trains the same
checkpoint twice — plain, and with SAL at `prune_fraction=0.33` — then puts both
through the same battery, with both arms losing the same heads.

| variant | standard | SAL | delta | SAL ahead on | consistent? |
|---|---|---|---|---|---|
| dense | 0.9005 ± 0.0104 | 0.9062 ± 0.0080 | **+0.57pp** | 4/5 | yes |
| int8 | 0.8888 ± 0.0102 | 0.8961 ± 0.0033 | **+0.73pp** | 4/5 | yes |
| int4 | 0.9002 ± 0.0111 | 0.9023 ± 0.0094 | +0.21pp | 3/5 | **no** |
| prune33 | 0.8571 ± 0.0240 | 0.8817 ± 0.0121 | **+2.45pp** | 5/5 | yes |
| prune50 | 0.8087 ± 0.0455 | 0.8294 ± 0.0368 | **+2.06pp** | 4/5 | yes |
| prune33+int8 | 0.8284 ± 0.0090 | 0.8472 ± 0.0229 | **+1.88pp** | 4/5 | yes |
| prune33+int4 | 0.8567 ± 0.0208 | 0.8725 ± 0.0132 | **+1.58pp** | 5/5 | yes |

Mean ± sample std across seeds. "Consistent" means SAL led on **4 or more of the
5 seeds** — a sign test on the direction, which is the question a single seed
cannot answer, not a t-test on the magnitude.

**What holds.** SAL wins 5 of 6 compressed variants, and it costs nothing on the
clean model (+0.57pp — SAL is not buying resilience with accuracy). The effect
is largest and most reliable exactly where SAL was designed to work: **head
pruning**, +2.45pp at 33% on every seed and +2.06pp at 50%. Both combined
recipes hold too.

**What does not.** *`int4` on its own does not replicate.* Three seeds of five
and +0.21pp — smaller than the run-to-run spread, i.e. a coin flip. The v0.4.0
claim that SAL won *all seven* variants including INT4 was one seed; across
five, quantization-only gains are small, and the INT4 row is the weakest in the
table. Where SAL helps under INT4 is in combination with pruning
(`prune33+int4`, +1.58pp on 5/5) — which is the pruning effect carrying the row.

**About the Pareto claim.** v0.4.0 reported `SAL/int4` beating the *uncompressed*
standard model. Across five seeds it averages 0.9023 at 361.9MB against the
standard model's 0.9005 at 1419.3MB. That +0.18pp is inside the spread, so the
honest statement is **equal accuracy at a quarter of the size**, not higher. It
is still the deployment recommendation; it is not a free accuracy gain.

**One observation, offered as such.** The SAL arm has the *smaller* standard
deviation on 6 of the 7 variants — most sharply on `prune33` (0.0121 vs 0.0240).
Five seeds is too few to call this a property, but SAL-trained models were more
predictable under compression here, not merely better on average.

Reproduce with `scripts/modal_multiseed_validation.py` (5× Modal T4, ~10 min);
raw per-seed numbers in `scripts/multiseed_results.json`.

> **One caveat on the `int8` rows.** This run quantizes INT8 with `torch.ao`
> dynamic quantization (CPU), where v0.4.0 used bitsandbytes LLM.int8() on CUDA.
> The INT8 rows are therefore a different measurement, not a replication of the
> earlier ones. INT4 is bitsandbytes NF4 in both.

#### Vision transformers — secondary, preliminary

**Cross-modal validation is under investigation, not established.** What follows
is three seeds on one architecture and one task. It is the beginning of an
answer, not the answer.

ViT-base-patch16-224 on CIFAR-10, full fine-tuning, seeds 42/123/456, 1000 eval
images, same battery structure as above (`scripts/modal_vit_validation.py`):

| variant | standard | SAL | delta | SAL ahead on |
|---|---|---|---|---|
| dense | 0.9673 ± 0.0029 | 0.9570 ± 0.0075 | −1.03pp | 0/3 |
| int8 | 0.9603 ± 0.0042 | 0.9550 ± 0.0056 | −0.53pp | 1/3 |
| prune33 | 0.7013 ± 0.0472 | 0.8450 ± 0.0070 | **+14.37pp** | 3/3 |
| prune50 | 0.3143 ± 0.0550 | 0.4760 ± 0.1120 | **+16.17pp** | 3/3 |
| prune33+int8 | 0.5723 ± 0.0549 | 0.7727 ± 0.0328 | **+20.03pp** | 3/3 |

The pruning gains are large and unanimous. Read them with four things attached:

- **SAL costs clean accuracy here.** −1.03pp on `dense`, losing all three seeds,
  where on GPT-2 it cost nothing. Two architectures, two answers.
- **Quantization-only does nothing**, as on GPT-2. INT8 barely dents ViT at all
  (96.7% → 96.0%), so there was no damage to recover.
- **The two studies are not on a common scale.** CIFAR-10 is 10-way, SST-2 is
  2-way, so raw percentage points do not transfer. Normalized to headroom above
  chance, SAL moves `prune33` retention from 69.3% to 86.9% here, against
  +4.7pp on GPT-2. The gap narrows under the correction but does not close.
- **Three seeds is thin**, and "ahead on 3/3" is unanimity, not significance.

**What we do not yet know is why the gap is this large.** The consistent pattern
across every run since v0.4.0 is that SAL recovers damage in proportion to how
much damage there is — ViT-base collapses under head pruning (96.7% → 70.1% at
33%) where GPT-2 Medium degrades gently, so there is far more to recover. That
reading fits, but it is a post-hoc fit to two data points. Note that it is *not*
a model-size story: the +20pp came from the **smaller** model (ViT-base, 86M)
and the +2.45pp from the larger one (GPT-2 Medium, 355M). Size, modality,
architecture and task-difficulty are all confounded across these two studies,
and nothing here separates them. Do not assume your model lands at either end —
measure it with `RobustnessTest`.

#### The earlier single-seed runs

The four runs below are what motivated the multi-seed study. They stay here
because the two SAL *lost* are what established the LoRA finding. Each trains
one model twice from identical weights, then evaluates both dense under the
battery. Scripts in `scripts/`; results in
`scripts/robustness_scale_results.json`. **Single seed each — read them as
signals, not benchmarks.**

| run | model / task | training | clean cost | quantization | pruning | combined |
|---|---|---|---|---|---|---|
| v0.4.0 | DistilBERT / SST-2 | full FT | +1.17pp | 1/2 | **2/2** | not tested |
| scale | GPT-2 Medium / SST-2 | **LoRA r=16** | **-3.12pp** | 0/2 | 1/2 | 1/2 |
| scale | GPT-2 Medium / SST-2 | **full FT** | +0.39pp | **2/2** | **2/2** | **2/2** |
| scale | Phi-2 2.7B / MMLU | **LoRA r=16** | -1.17pp | 0/2 | 1/2 | 0/2 |

The DistilBERT run also tested inference-time neuron dropout at 10% and 20%,
which the standard model won both times — by 0.5pp and 0.6pp, inside that run's
noise floor. The scale runs do not test dropout, so it has no column here.

The two GPT-2 rows are a controlled comparison: identical model, task, data,
seed and battery. The only thing that changes is whether LoRA is in the way.

#### Under LoRA, the same setup fails

```
variant           baseline       SAL     delta    winner
--------------------------------------------------------
dense               0.8906    0.8594   -0.0312  baseline
int8                0.8828    0.8613   -0.0215  baseline
int4                0.8594    0.8301   -0.0293  baseline
prune33             0.8496    0.8418   -0.0078  baseline
prune50             0.8145    0.8496   +0.0352       SAL
prune33+int8        0.8496    0.8477   -0.0020  baseline
prune33+int4        0.8301    0.8359   +0.0059       SAL
```

Same model, same data, same seed. SAL loses four of six compressed variants and
gives up 3.1 points of clean accuracy to get there. Only the heaviest structural
damage (`prune50`) still favours it.

We are leaving this table in the README because it is the finding that explains
the mechanism: **SAL works by letting the model reorganize around silenced
heads, and LoRA freezes the weights that would do the reorganizing.** Rank-16
adapters on `c_attn` cannot absorb 126 silenced heads. The perturbation lands,
the adaptation cannot.

#### What is not established

- **Quantization-only resilience.** Five seeds put `int4` at +0.21pp on 3/5 —
  no consistent effect. `int8` is consistent but small (+0.73pp), and measured
  on a different backend than v0.4.0's. If you never prune, do not assume SAL
  buys you anything at INT4; measure it on your model.
- **Scale is still open.** Phi-2 2.7B was LoRA-only, so "LoRA starves it" and
  "SAL stops working above ~350M" remain confounded at that size. Phi-2 under
  full fine-tuning is the experiment that separates them.
- **Cross-modal validation.** Under investigation. One vision architecture on
  one task at three seeds is a preliminary result, not a validated claim, and
  the README should not be read as making one.
- **Why the effect size differs so much between the two studies.** ViT gained
  +14 to +20pp under pruning where GPT-2 Medium gained +2.45pp. The pattern fits
  "SAL recovers damage in proportion to how much there is", but that is a
  post-hoc reading of two points. It is *not* a size story — the larger gain
  came from the smaller model — and size, modality, architecture and task
  difficulty are confounded across the two runs.
- **Whether SAL costs clean accuracy.** It cost nothing on GPT-2 (+0.57pp on
  5 seeds) and 1.03pp on ViT (0/3 seeds). Two architectures, two answers.
- **Everything else is single-seed.** DistilBERT and Phi-2 have one run each,
  and nothing above 355M has been fully fine-tuned at all.
- **Everything under LoRA.** The negative LoRA result is itself single-seed.
  It is consistent with the mechanism, and it agrees across two model sizes,
  but it has not had the same treatment.

### When to use SAL

| your setup | recommendation |
|---|---|
| **Full fine-tuning, and you prune heads** | **Yes.** The strongest and best-replicated case: +2.45pp at 33% pruning on 5/5 seeds of GPT-2 Medium, at no cost to clean accuracy. |
| **Vision transformers** | **Promising, preliminary.** Three seeds on ViT-base/CIFAR-10 gave +14 to +20pp under pruning, unanimously — but it cost 1.03pp of clean accuracy, and one architecture on one task is not cross-modal validation. Measure yours. |
| **Full fine-tuning, quantization only** | **Measure first.** INT8 gave +0.73pp on GPT-2 (4/5 seeds) and nothing on ViT; INT4 alone was a coin flip (3/5, +0.21pp). Use `RobustnessTest` on your own model before committing. |
| **LoRA / QLoRA adapters** | **Not recommended.** Measured worse than not using SAL at all, and it costs clean accuracy. The adapters are too small to redistribute what the masking removes. |
| **Models above ~1B** | **Unvalidated.** No full-fine-tuning result at that scale yet. |

If you are on LoRA and want compression resilience, the honest answer today is
that SAL is not the tool; use `RobustnessTest` to measure what your compression
actually costs and `PlasticityScanner` to choose where to cut.

## Continual learning without replay buffers

### StructuralGuard — protect what matters when you fine-tune

When you fine-tune a trained model on a new task, it quietly overwrites the
structure that carried the old one. `StructuralGuard` reads the model's
structural map and **freezes the critical attention heads** (hub layers,
structural bottlenecks, and the functionally unique heads) while leaving the
redundant heads free to absorb the new task. No EWC, no replay buffer, no
distillation — the topology itself decides what to protect.

```python
from sal import StructuralGuard

# After training on task A, build a guard from the model's structure.
guard = StructuralGuard.from_model(model, probe_dataset, protection_level=0.5)

print(guard.protected_heads)   # [(layer, head), ...] frozen during fine-tuning
print(guard.trainable_heads)   # [(layer, head), ...] free to absorb task B
print(guard.protection_map)    # {layer: [protected head indices]}

guard.protect(model)           # zero gradients for protected heads (backward hooks)
trainer.train()                # fine-tune on task B with ANY training loop
guard.release()

drift = guard.measure_drift(model, probe_dataset=probe_dataset)
print(drift.forgetting_score)      # 0 = nothing forgot, 1 = total reorganization
print(drift.protected_integrity)   # ~1.0 if the protected heads held

guard.save("model_guard.json")     # serialize; reload before task C, D, ...
guard = StructuralGuard.load("model_guard.json")
```

Protection is at the **head level** — some heads in a layer can be frozen while
others in the same layer keep learning. `protection_level` (0.0–1.0) sets the
fraction of the most critical heads to protect.

HuggingFace `Trainer`? Use the callback — it applies protection on
`train_begin`, measures drift on `train_end`:

```python
from sal import StructuralGuardCallback

guard = StructuralGuard.from_model(model, probe_dataset)
callback = StructuralGuardCallback(guard)
trainer = Trainer(model=model, callbacks=[callback])
trainer.train()
print(callback.drift_report.summary)
```

### DriftMonitor — measure structural forgetting after any fine-tuning

`DriftMonitor` quantifies how much a model's structure moved, guarded or not.
Snapshot before and after, then compare.

```python
from sal import DriftMonitor

monitor = DriftMonitor(model, probe_dataset)
monitor.snapshot("before_task_b")
trainer.train()
monitor.snapshot("after_task_b")

drift = monitor.compare("before_task_b", "after_task_b")
print(drift.summary)
print(drift.layer_drift)             # per-layer activation retention (1 = identical)
print(drift.classification_changes)  # layers whose fragility class flipped
drift.save("drift_report.json")
drift.save("drift_report.pdf")       # visual before/after comparison
```

Snapshots are keyed, so you can track drift across many sequential tasks and
compare any pair.

## Self-supervised models (v0.5.1)

`SALTrainer` no longer assumes your loss is cross-entropy. Pass `train_step=`
and you own the step; SAL keeps owning the prune schedule.

```python
def my_train_step(model, batch, optimizer, mask_module):
    loss = my_custom_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

trainer = SALTrainer(model, config, optimizer, dataloader,
                     train_step=my_train_step)
trainer.train(num_epochs=5)
```

The masker is already installed and already masking when your callback is
entered, so a callback that ignores `mask_module` entirely is still SAL-trained.
You get it as the fourth argument for the case that needs it — reading an
*unperturbed* target mid-step:

```python
with torch.no_grad(), mask_module.unmasked():
    target = model(pixel_values=pixels).last_hidden_state   # no heads masked
predicted = model(pixel_values=visible).last_hidden_state   # heads masked
```

Use `unmasked()` (or `remove_mask()` / `apply_mask()`) rather than
`deactivate()`. The first pair *suspends* masking; `deactivate()` resets the
pruned set to empty and silently undoes the schedule's accumulated damage.

`train_step=None` keeps the v0.5.0 cross-entropy loop exactly as it was.

### Scoring a model that has no accuracy

A self-supervised encoder has no head, so "did compression hurt?" has to be
asked about the representations. `sal.evaluation`:

```python
from sal import linear_probe, knn_accuracy, cka_similarity, measure_latency

probe = linear_probe(model, train_loader, val_loader)     # frozen encoder + linear head
knn   = knn_accuracy(model, train_loader, val_loader, k=20)
cka   = cka_similarity(original, compressed, loader)      # 1.0 = identical, no labels needed
ms    = measure_latency(compressed, device="cuda")        # median, not mean
```

`cka_similarity` is the one to reach for first: it is label-free and invariant
to an invertible linear map, so a model whose heads have been physically sliced
out is still comparable to the original despite the width change.

`compression_report()` bundles all of it, and reports any metric it could not
compute as `None` *with the reason* under `"skipped"` — an unexplained `None`
in a benchmark table is indistinguishable from a measured zero.

### Seeing what changed

```python
from sal.visualization import compare_feature_maps
compare_feature_maps(original, compressed, image, save_path="before_after.png")
```

Both rows share one colour scale, deliberately. Per-panel normalization rescales
a model whose activations collapsed toward zero back up to full range, so it
renders as identical to the original — hiding the exact failure the figure
exists to show.

Architectures: `ijepa` and `dinov2` join the auto-detected list.

**Status: unvalidated.** The plumbing is tested (247 CPU tests); the benchmark
behind it has not been run. See [`examples/jepa_sal.py`](examples/jepa_sal.py)
and [`scripts/modal_jepa_sal.py`](scripts/modal_jepa_sal.py), and read the
caveats in both — in particular, Meta never released an I-JEPA ViT-B/16, and
the training objective those scripts use is I-JEPA-*shaped*, not I-JEPA.

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — 3-line SAL training on DistilBERT
- [`examples/standalone_fi.py`](examples/standalone_fi.py) — Fragility Index scan, no training
- [`examples/full_control.py`](examples/full_control.py) — manual config + standalone trainer
- [`examples/compare_with_without_sal.py`](examples/compare_with_without_sal.py) — SAL vs. baseline under compression
- [`examples/jepa_sal.py`](examples/jepa_sal.py) — SAL on a self-supervised vision encoder, with a custom training step (`--smoke` runs on CPU)

New here? Start with [docs/getting_started.md](docs/getting_started.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for what's shipped, what's next, and how to request
features — including the full evidence trail behind the robustness claims,
losses included. v0.5.0 shipped `CompressionPipeline`, `slice_heads()`,
`quantize()`, and the five-seed validation above. Next up is topology-guided
distillation (v0.6.0), for the 21% who distill.

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
