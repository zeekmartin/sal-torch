# sal-torch

![CI](https://github.com/zeekmartin/sal-torch/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/sal-torch) ![Python](https://img.shields.io/pypi/pyversions/sal-torch) ![Downloads](https://pepy.tech/badge/sal-torch) ![License](https://img.shields.io/badge/license-BSL%201.1-blue)


**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

> **v0.6.0 changes two defaults.** `SALConfig.prune_fraction` is now **0.50**
> (was 0.33) and head selection is **uniform per layer** by default
> (`uniform_per_layer=True`). The first follows the monotone dose-response: train
> at the ratio you will deploy at. The second is what makes the set of heads SAL
> trained without expressible as a `slice_heads()` argument, so the accuracy it
> retains becomes a smaller model rather than a mask. Pass
> `uniform_per_layer=False` for the pre-v0.6.0 global draw.
>
> v0.6.0 also adds a second method, **DropHead**, which is the one to use when
> you do not control which heads get removed — it beats a standard fine-tune by
> **+49.8pp** at 50% pruning + INT4 on 10/10 seeds, where SAL on an arbitrary
> set does not. See [v0.6.0 — two methods, two deployment
> regimes](#v060--two-methods-two-deployment-regimes).

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

# Step 1. Probe before you train anything. Two eval sweeps, no training, nothing
# modified — it removes half the heads at random and measures what that costs.
print(pipe.probe().recommendation)     # -> sal_aligned / drophead / skip_heads

print(pipe.scan().recommendation)      # fragility, absorption map, projected sizes
# Step 2. Train at the ratio you intend to deploy at — the gain is monotone in it,
# and the best cells are the matched ones. Both default to 0.50 since v0.6.0.
pipe.sal_train(train_dataset, epochs=3, prune_fraction=0.50)
pipe.compress(pruning=0.50, quantization="int4", slice_heads=True)

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

> **FI does not select heads, and no result in this project suggests it should.**
> It is a diagnostic: it tells you how fragile a model is, not which head to
> remove. The `fi_guided` row above is the evidence *against* using it that way,
> and v0.6.0 added a much better-powered replication at training time. A
> fragility-weighted *training* draw (`fi_drophead`: fragile heads dropped at
> 2x the base rate, resilient at 0.5x, same expected rate overall) was measured
> against a plain uniform draw on ViT-B/16 / ImageNet-100, 10 seeds, 20 head
> selections per cell:
>
> | | mask_ratio 0.3 | mask_ratio 0.5 |
> |---|---|---|
> | fragility-weighted − uniform | **−5.02pp**, CI [−5.64, −4.40], **0/10 seeds** | **−4.77pp**, CI [−5.15, −4.39], **0/10 seeds** |
>
> Uniform random wins on every seed at both ratios, with ±0.4pp intervals. Use
> `FIScanner` to decide *whether* to compress and how much; use random selection
> to decide *which* heads. Full trail in `results/lumi_v060/PHASE2E.md`.

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

## Start with the probe, not with a guess

**The single most useful thing you can measure about your model costs two
evaluation sweeps and no training: remove half the heads at random and see how
far it falls.** That number decides which method is worth paying for, and nothing
cheaper predicts it — not the architecture, not the parameter count, and not the
fragility index.

```python
pr = pipe.probe()            # or probe(fraction=0.33) to match your target
print(pr.drop_pp, pr.route)  # 62.21  sal_aligned
print(pr.recommendation)
```

| probe drop at 50% of heads | route | what to do |
|---|---|---|
| **> 30pp** — fragile | `sal_aligned` | **SAL.** Accumulate-and-hold, save `pruned_heads()`, prune exactly that set. This is where SAL's advantage is large. |
| **10–30pp** — intermediate | `drophead` | **Either.** SAL if you choose the deployed head set, **DropHead** if you don't — it is robust on any set of that size. |
| **< 10pp** — robust | `skip_heads` | **Skip head training.** It buys nothing measurable here. Go straight to compression. |

Then, in order:

3. **Prune FFN neurons.** This is where the bulk of the compression is — see
   [MLP neurons vs. attention heads](#mlp-neurons-vs-attention-heads) below.
4. **`torch.compile`.** Measured 1.12x on a dense ViT-B/16 at batch 32, 1.50x on
   a pruned one, and 2.2x at batch 1 where pruning alone buys almost nothing.

### Where the thresholds come from

Six architectures, probed and then trained both ways, scored on an **identical**
head set so neither arm gets a perturbation the other did not face. Sorted by
probe drop:

| model | params | FI | **probe drop** | SAL gain | DropHead gain | **SAL − DropHead** | route |
|---|---|---|---|---|---|---|---|
| DINOv2-large | 304.6M | 0.0062 | **−94.0pp** | +90.4 | +59.3 | **+31.0** | SAL |
| DeiT-small | 21.7M | 0.1563 | −76.3pp | +68.2 | +55.5 | **+12.7** | SAL |
| ViT-B/16 | 85.9M | 0.0660 | −62.2pp | — | — | **+2.0** | SAL |
| I-JEPA ViT-H/14 | 630.8M | 0.0040 | −18.9pp | +3.0 | +0.5 | +2.6 | either |
| BERT-base | 109.5M | 0.0935 | −5.2pp | +2.9 | +2.9 | +0.1 | skip |
| phi-2 | 2648.6M | 0.0008 | −4.4pp | +2.1 | +2.0 | +0.1 | skip |

Gains are against a standard fine-tune on that same set, in points. ViT-B/16 is
the 10-seed protocol from the tables further down; the rest are 3 seeds (phi-2,
1) with 20 random draws per cell, 5 epochs, `mask_ratio 0.5`, on MI250X.

**SAL's lead over DropHead tracks the probe almost monotonically** — +31.0pp at a
94pp drop down to +0.1pp at 4pp, two adjacent inversions across two modalities
and a 122x range of parameter count. The two quantities share no measurement:
the probe comes from the standard fine-tune alone, and the margin contains no
baseline term.

> **The fragility index does *not* order that table, and it was the original
> hypothesis.** BERT-base has a *higher* FI than ViT-B/16 (0.094 vs 0.066) and
> loses 5pp where ViT-B/16 loses 62; DINOv2-large has an FI 25x lower than
> DeiT-small and collapses to chance on the same task, protocol and class count.
> Parameter count does not order it either (−76, −62, −5, −94, −19, −4 by size).
> **FI stays in the product for diagnosis and certification. It is not a router,
> and `scan()` no longer presents it as one.**

Caveats worth carrying: six points make a usable heuristic, not a calibrated
boundary, and nothing here gives a principled threshold. The two text models sit
at a 50.9% majority-class floor, so their small drops are partly a ceiling
effect — DINOv2-large is the case that falsifies FI without that excuse.

### MLP neurons vs. attention heads

Head pruning is the *small* half of the compression. At an **identical parameter
count**, pruning FFN neurons costs about a third of what pruning heads costs:

| ViT-B/16 at 71.7M parameters (−16.5%) | retention vs. the dense model |
|---|---|
| 50% of heads, DropHead | −12.2pp |
| 50% of heads, SAL aligned | −10.2pp |
| **25% of FFN neurons** | **−3.1pp** |

The FFN has redundancy the attention block does not: 3072 neurons per layer give
a random draw far more room than 12 heads do. At 50% of neurons the reduction is
33.0% of all parameters — twice what 50% of heads removes — for −7.2pp, still
less than heads cost at half the reduction. Adding head pruning *on top of*
neuron pruning was measured and is a bad trade: at a matched 57.5M, neurons alone
lose 7.2pp while heads-33%+neurons-33% lose 14.3pp and heads-50%+neurons-25% lose
20.3pp, for 0.04x–0.12x of extra speed.

And the smaller model is faster, measured the same way as the head results above
— ViT-B/16 at batch 32 on one MI250X, median of 50 runs after 10 warmup,
`max-autotune`:

| neurons pruned | params | retention | eager | compiled | **total** | throughput |
|---|---|---|---|---|---|---|
| none (dense) | 85.9M | — | 47.72 ms | 42.63 ms | 1.12x | 751 img/s |
| 25% | 71.7M | −3.1pp | 41.61 ms | 37.42 ms | **1.28x** | 855 img/s |
| 33% | 67.2M | −4.3pp | 40.16 ms | 36.01 ms | **1.33x** | 889 img/s |
| 50% | 57.5M | −7.2pp | 35.46 ms | 31.89 ms | **1.50x** | 1004 img/s |

At batch 1 none of this helps: slicing measures 0.98x and 0.93x for the 25% and
33% cells, and the speedup there is the compiler's — 2.2x with
`reduce-overhead`, at any size. Roughly half the parameters have to go before
slicing moves a batch-1 number at all. The two regimes are the same ones the
head results run into, in [Does the smaller model actually run
faster?](#does-the-smaller-model-actually-run-faster).

**This package masks attention heads.** MLP neuron pruning is in
[sal-torch-pro](#sal-torch-pro), the commercial extension (license required; not
on public PyPI). At equal parameter count (−16.5%), MLP pruning costs −3pp
against −10pp for head pruning.

The numbers are reported here because they change how you should budget the
head-pruning decision above: it is worth doing when the probe says your model is
fragile, and it is not where your compression comes from.

## Know your model before you touch it

### FIScanner — how fragile is this model?

The **Fragility Index** is a structural diagnostic, scored in `[0, 1]`, that
measures how much redundant pathway a model's attention graph has. Heads are
compared by their activation signatures; an edge between two heads is *fragile*
when they share no common neighbour, i.e. the function it carries has no backup.
FI is the fraction of such edges.

- **Low FI** → heavily triangulated graph, lots of redundancy.
- **High FI** → many unsupported edges.

FI is purely diagnostic — it measures, it never perturbs. You can use it with or
without SAL training.

> **Do not use FI to predict how much compression will cost, and do not use it to
> choose a method.** That was the hypothesis and it was tested on six
> architectures and refuted in both directions: a model with a *higher* FI lost
> twelve times less accuracy than one with a lower FI, and the lowest-FI vision
> model in the set collapsed to chance. `probe()` is the router — it measures the
> model's loss under the pruning you intend, which is the quantity that actually
> orders the outcomes. See [Start with the
> probe](#start-with-the-probe-not-with-a-guess).
>
> What FI is for: structural reporting and certification, tracking a model's
> graph over training with `FIMonitor`, and locating layers to leave alone with
> `classify_layers()`. Those uses are untouched by the routing result.

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

The primary evidence is GPT-2 Medium at five seeds, and ViT-B/16 on
ImageNet-100 at **fifteen** seeds. A three-seed ViT/CIFAR-10 study sits between
them as the earlier, thinner vision result. Two modalities are measured; what
is still missing is a reading of *why* the effect size differs so much between
them, so "cross-modally validated" remains an overstatement.

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

#### ViT-B/16 on ImageNet-100 — fifteen seeds, and where SAL stops helping

The most finely measured study in the project: ViT-B/16 on ImageNet-100 (100-way),
5 epochs, **15 seeds**, every compressed cell averaged over **20 random head
selections** rather than the 3 used above
(`scripts/run_sal_qat_prototype.py`, full trail in
`results/lumi_v060/PHASE2{,A,B}.md`). Head pruning at 50% plus INT4 weights:

| arm | P50+I4 top-1 | vs baseline |
|---|---|---|
| standard fine-tune | 31.0% ± 1.4 | — |
| SAL | 43.2% ± 3.8 | **+12.1pp** |
| SAL + QAT | 45.6% ± 4.3 | **+14.5pp ± 4.3**, 95% CI [+12.4, +16.7], 15/15 seeds |

Three things came out of it that the thinner studies could not show.

- **Higher `prune_fraction` during training increases compression resilience
  monotonically.** Sweeping it over {0.1, 0.2, 0.3, 0.4, 0.5}, the gain at
  half-pruned + INT4 rises steadily: **+6.2 → +9.0 → +12.7 → +16.6 → +20.8pp**.
  The best cells are the ones where the *training* ratio matches the
  *deployment* ratio, and the cost is also monotone and small — 0.3 to 1.5pp of
  clean accuracy. A monotone response across five levels is much stronger
  evidence that the mechanism is real than any single cell. Practical rule:
  **train at the ratio you intend to deploy at.** This is why the shipped
  default changed from 0.33 to **0.50** in v0.6.0 — at 0.33 most of the effect
  goes unused against a 50% target.
- **At 50% pruning SAL is robust; at 33% on this setup it is weak and
  inconsistent.** The 33% mean is small (+2 to +7pp depending on mask ratio)
  and one recurring seed sits *below* baseline at four of the five mask ratios,
  at 20 head selections — not noise. The reading that fits: SAL trains a model
  to operate under heavy structural damage, so when the real damage is *milder*
  than what it trained for, the adaptation can be a net loss.
  **This does not generalise into a rule about 33%.** On GPT-2 Medium (above)
  33% is SAL's single most reliable row, +2.45pp on 5/5 seeds, and on
  ViT/CIFAR-10 it is +14.37pp on 3/3. Three setups, three answers at the same
  prune ratio — so 33% is where the *model and task* decide, and the only
  honest instruction is to measure yours with `RobustnessTest`.
- **Three head selections per cell is too few to publish from.** The same
  fifteen checkpoints scored at 3 selections gave a standard deviation of
  11.07pp; at 20 it is 4.28pp, with the mean essentially unchanged. The
  baseline looked like it spanned 23.6–36.5% at 3 selections and in fact spans
  1.6pp. Head-selection sampling, not the training seed, was the dominant noise
  term — and it is the cheapest to remove, since it costs evaluation passes and
  no retraining.

#### v0.6.0 — two methods, two deployment regimes

The v0.6.0 campaign on LUMI (AMD MI250X, ~85 GPU-hours) added the control arms
that had never been run, and they changed the picture. Everything below is
ViT-B/16 / ImageNet-100, 5 epochs, **10 seeds**, **20 shared head selections per
cell**, full trail in `results/lumi_v060/PHASE2{C,D,E,F}.md`.

**DropHead** — per-step resampled head dropout, the full ratio from step 0, no
accumulation — is the strong general method. At 50% head pruning + INT4:

| arm | P50+I4, mask_ratio 0.3 | vs baseline | P50+I4, mask_ratio 0.5 | vs baseline |
|---|---|---|---|---|
| standard fine-tune | 31.06% | — | 30.62% | — |
| **DropHead** | **72.26%** | **+41.20pp**, CI [+40.44, +41.96], 10/10 | **80.39%** | **+49.77pp**, CI [+48.68, +50.86], 10/10 |
| fragility-weighted draw | 67.24% | +36.18pp, 10/10 | 75.62% | +45.00pp, 10/10 |
| ramped ratio | 63.38% | +32.31pp, 10/10 | 72.35% | +41.72pp, 10/10 |

Both attempts to improve the draw **lose to plain uniform random on 0/10 seeds**
(see the FI note above). What matters is total exposure to the perturbation, not
which heads get hit — a ramp simply delivers less of it.

**SAL** — accumulate heads during a window and hold them to the end — behaves
differently: it specialises. It is far worse than DropHead on an *arbitrary* head
set, and better than it on the *one set it trained without*. Scored at
mask_ratio 0.5 with `uniform_per_layer=True`, every cell a perturbation of
identical size:

| arm | clean | **its own trained set** | a random set | sliced, trained set | sliced, random set |
|---|---|---|---|---|---|
| standard fine-tune | 93.77% | 31.56% | 36.70% | 31.55% | 34.92% |
| **SAL** | 91.32% | **83.62%** | 54.56% | **83.61%** | 53.33% |
| DropHead | 92.36% | 81.60% | 82.16% | 81.59% | 82.25% |

The sliced model is **85.9M → 71.7M parameters, 16.5% smaller**, and masked and
sliced accuracy agree to **±0.01pp over 10 seeds** — so the retained accuracy is
a genuinely smaller model, not a masking artifact.

| comparison | result |
|---|---|
| SAL on its own set vs a random set | **+29.06pp**, CI [+26.78, +31.33], 10/10 |
| **standard fine-tune** on that same set vs a random set | −5.14pp, CI [−12.79, **+2.51**], 2/10 |
| SAL vs DropHead, **identical** head set | **+2.02pp**, CI [+1.54, +2.50], 10/10 |
| **sliced** SAL on its set vs **sliced** DropHead on a random set | **+1.36pp**, CI [+0.53, +2.18], 8/10 |
| the same, **with INT4 on top** | +0.49pp, CI [−0.39, **+1.37**], 6/10 — **not established** |

The second row is the control that matters and it is why this is adaptation
rather than selection: a model that never trained with masking does **not** find
SAL's head set easy, and the point estimate is negative at both ratios. The set
is not made of unimportant heads; the trained model's relationship to it is.

The last row is the honest limit. SAL's edge over DropHead is established in
FP16 and after slicing, but **once INT4 is stacked on top the interval spans
zero**. Against the standard fine-tune both methods are worth roughly the same
there (+50.78pp for SAL-aligned, +50.30pp for DropHead-random, both 10/10).

**So: which one you want depends on whether you control the head set.**

| your deployment | use | measured |
|---|---|---|
| You choose which heads to remove, and can keep a list | **SAL** with `uniform_per_layer=True` (the default), saving `pruned_heads()` with the checkpoint and slicing exactly that set | 16.5% smaller, **+1.4pp** over the best alternative at the same size, 8–10/10 seeds |
| The pruning is out of your hands, or varies per deployment | **DropHead** (`mask_mode="drophead"` in the prototype script) | **+49.8pp** over a standard fine-tune at 50% + INT4, 10/10 seeds |

SAL costs ~1pp of clean accuracy (91.32% vs DropHead's 92.36%), so its net
advantage at deployment is ~1.4pp, not ~2pp. And SAL scored on a set it did not
train for is **27.6pp behind** DropHead. That gap is the price of specialisation
and nothing in v0.6.0 reduced it.

#### Does the smaller model actually run faster?

Up to v0.6.0 this README reported a size reduction and declined to claim a
latency one. It is now measured: ViT-B/16, AMD Instinct MI250X, torch
2.5.1+rocm6.2, median of 50 runs after 10 warmup, `slice_heads()` for the sliced
arms and stock `torch.compile` for the compiled ones.

**Batch 32 — batched or offline inference**

| arm | params | eager | from slicing | best compiled | from compiling | **total** | throughput |
|---|---|---|---|---|---|---|---|
| dense | 85.9M | 46.93 ms | 1.00x | 42.03 ms (`max-autotune`) | 1.12x | 1.12x | 761 img/s |
| sliced-33 | 76.4M | 41.67 ms | 1.13x | 37.22 ms (`max-autotune`) | 1.12x | 1.26x | 860 img/s |
| **sliced-50** | **71.7M** | 39.03 ms | **1.20x** | **34.19 ms** (`max-autotune`) | 1.14x | **1.37x** | **936 img/s** |

Slicing and compiling **compound almost exactly**: 1.20 × 1.14 = 1.37, and
throughput rises 761 → 936 img/s, **+23%**. Worth noting on its own —
**sliced-50 in plain eager (39.03 ms) already beats the dense model under
`max-autotune` (42.03 ms)**, so removing half the heads buys more than compiling
the dense model does, and costs no compilation time.

**Batch 1 — single requests**

| arm | eager | from slicing | best compiled | from compiling | total |
|---|---|---|---|---|---|
| dense | 8.31 ms | 1.00x | 3.86 ms (`reduce-overhead`) | **2.16x** | 2.16x |
| sliced-33 | 8.64 ms | 0.96x | 3.80 ms (`reduce-overhead`) | 2.27x | 2.19x |
| sliced-50 | 8.62 ms | 0.96x | 3.73 ms (`max-autotune`) | 2.31x | 2.23x |

**Slicing buys nothing at batch 1 — it is marginally slower.** At this shape the
GPU is bound by kernel launch overhead, not by arithmetic, so taking work out of
each kernel does not shorten the wall clock. Compilation is what helps, and it
helps a lot: **2.16x** on the dense model, because `reduce-overhead` attacks
exactly that launch overhead. The sliced models finish only 3% ahead of the
compiled dense one.

| you serve | do this | expect |
|---|---|---|
| **Batched / offline inference** | slice **and** compile with `max-autotune` | **1.37x**, +23% throughput |
| **Single requests** | **compile** with `reduce-overhead`; slice for the memory, not the latency | **2.16x** from compiling, ~0 from slicing |

`max-autotune` costs 86-194 s of compilation per shape and `reduce-overhead`
7-12 s, paid once per process. One caution from running this: `torch.compile`
caches a limited number of variants per process (8 by default), and past that
limit it falls back to eager **silently** — a sweep over several models and batch
sizes needs `torch._dynamo.reset()` between them, or the later rows quietly
report eager timings as though they were compiled.

#### Cross-architecture — does DropHead's advantage travel?

Three architectures and two training objectives, same masking protocol
(`experiments/exp13_crossarch_drophead.py`, 3 seeds each for the two new ones):

| model / task | arm | clean | prune 33% | prune 50% |
|---|---|---|---|---|
| **ViT-B/16** / ImageNet-100 (supervised, 10 seeds) | baseline | 93.8% | 69.0% | 35.3% |
| | DropHead | 92.7% | **88.4%** | **82.2%** |
| **DINOv2-base** / ImageNet-100 (self-supervised, 3 seeds) | baseline | 92.7% | 17.2% | 1.6% |
| | DropHead | 89.5% | **82.1%** (+64.9pp, 3/3) | **60.1%** (+58.6pp, 3/3) |
| **GPT-2 Medium** / SST-2 (causal LM, 3 seeds) | baseline | 92.7% | 89.4% | 84.1% |
| | DropHead | 92.1% | **90.7%** (+1.3pp, 3/3) | **88.4%** (+4.3pp, 3/3) |

The ordering holds everywhere, but the margins differ by an order of magnitude —
and the baseline row says why. **GPT-2 on SST-2 keeps 84.1% at 50% head pruning**
where DINOv2 collapses to 1.6%. There is almost no damage to repair, so there is
almost nothing to win. This is the clearest statement yet of the scope rule:
these methods pay off in proportion to how much compression actually hurts your
model. A binary sentiment task on a 24-layer model is a poor benchmark for them,
whichever way it points.

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

- **That SAL's accumulating schedule beats plain DropHead in general.** It does
  not. On an arbitrary head set DropHead is ahead by **27.6pp** (82.16% vs
  54.56% at mask_ratio 0.5, 10 seeds). SAL's advantage exists only on the
  specific set it trained without, which means you must record that set and
  prune exactly it. If you cannot, use DropHead.
- **That SAL's edge survives INT4.** Sliced and in FP16 it is +1.36pp, CI
  [+0.53, +2.18], 8/10 seeds. With INT4 weights stacked on top the same
  comparison is +0.49pp with the interval spanning zero, 6/10. The deployment
  claim holds for slicing; it is unproven for slicing *plus* 4-bit weights.
- **That slicing helps at batch size 1.** It does not — it is marginally
  *slower* (0.96x), because that shape is bound by kernel launch overhead rather
  than arithmetic. The speedup at batch 32 is measured and real (1.20x eager,
  1.37x compiled); see [Does the smaller model actually run
  faster?](#does-the-smaller-model-actually-run-faster). Where the two regimes
  cross over is somewhere between 1 and 32 and has not been located, and the
  whole sweep is one GPU (MI250X), one architecture and one image size.
- **That head slicing alone reaches a 2x speedup.** 1.37x is what 50% of heads
  buys with compilation on top. Attention is roughly a third of a transformer
  block, so the remaining headroom is in the FFN, and joint head+MLP pruning is
  not in this package.
- **That the aligned/sliced result travels.** It has been measured on **one
  architecture** (ViT-B/16 / ImageNet-100). The cross-architecture table covers
  DropHead on DINOv2 and GPT-2, not the aligned-pruning path.
- **Quantization-only resilience.** Five seeds put `int4` at +0.21pp on 3/5 —
  no consistent effect. `int8` is consistent but small (+0.73pp), and measured
  on a different backend than v0.4.0's. If you never prune, do not assume SAL
  buys you anything at INT4; measure it on your model.
- **Scale is still open.** Phi-2 2.7B was LoRA-only, so "LoRA starves it" and
  "SAL stops working above ~350M" remain confounded at that size. Phi-2 under
  full fine-tuning is the experiment that separates them.
- **Cross-modal validation.** Two modalities are now measured, the vision side
  at fifteen seeds. What is not established is *why* the effect size differs by
  an order of magnitude between them, so the README still should not be read as
  claiming SAL is cross-modally validated.
- **What decides whether 33% pruning helps.** Three setups give three answers at
  the same prune ratio — GPT-2 Medium +2.45pp on 5/5, ViT/CIFAR-10 +14.37pp on
  3/3, ViT-B/16/ImageNet-100 weak and one seed negative. Task difficulty
  (2-way / 10-way / 100-way) is the obvious candidate and is confounded with
  architecture and dataset. Nothing measured so far separates them, so there is
  no rule yet for predicting which side your model lands on.
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

**Run `probe()` first.** The old advice here was "use DropHead by default"; six
architectures later, the default is to spend two eval sweeps measuring your own
model instead of inheriting someone else's regime. The table below is what to do
once you have that number.

| your setup | recommendation |
|---|---|
| **Any model, before anything else** | **`pipe.probe()`.** Two eval sweeps, no training. Drop > 30pp → SAL on the set you will deploy; 10–30pp → either, DropHead if the set is not yours; < 10pp → skip head training. See [Start with the probe](#start-with-the-probe-not-with-a-guess). |
| **The probe says > 30pp, and you know which heads you will remove** | **SAL, with `uniform_per_layer=True` (the default).** Save `pruned_heads()` with the checkpoint and slice that set. 16.5% fewer parameters and +1.4pp over the best alternative at the same size, 8–10/10 seeds. Without the saved set the advantage is unrecoverable. SAL's lead over DropHead on the identical set reached +31.0pp on the most fragile model measured. |
| **The probe says > 10pp and the pruning is out of your hands** | **DropHead, not SAL.** Per-step resampled head dropout: +49.8pp over a standard fine-tune at 50% pruning + INT4 on ViT-B/16, 10/10 seeds, and it holds on DINOv2 and GPT-2. SAL on an arbitrary set is 27.6pp *behind* it. |
| **The probe says < 10pp** | **Skip head training.** On BERT-base and phi-2 both methods landed within 0.1pp of a standard fine-tune, and on I-JEPA DropHead cost 1.1pp of clean accuracy for no gain. Compress and quantize directly. |
| **Where your compression actually comes from** | **FFN neurons, not heads** — a third of the accuracy cost at equal size. Not in this package; see [MLP neurons vs. attention heads](#mlp-neurons-vs-attention-heads). |
| **Full fine-tuning, and you prune heads** | **Yes.** The strongest and best-replicated case: +2.45pp at 33% pruning on 5/5 seeds of GPT-2 Medium, at no cost to clean accuracy. |
| **Vision transformers, pruning at ≥50%** | **Yes.** The best-measured case in the project: ViT-B/16/ImageNet-100, 15 seeds, 20 head selections per cell, **+14.5pp ± 4.3** at 50% pruning + INT4 (95% CI [+12.4, +16.7], 15/15 seeds). Costs 0.3–1.5pp of clean accuracy, rising with `prune_fraction`. |
| **Vision transformers, pruning at 33%** | **Measure first.** On ViT-B/16/ImageNet-100 the benefit is weak and inconsistent (one recurring seed below baseline), while on ViT-base/CIFAR-10 it was +14 to +20pp on 3/3 and on GPT-2 Medium +2.45pp on 5/5. Same ratio, three different answers — this is where your model and task decide. |
| **Full fine-tuning, quantization only** | **Measure first.** INT8 gave +0.73pp on GPT-2 (4/5 seeds) and nothing on ViT; INT4 alone was a coin flip (3/5, +0.21pp). Use `RobustnessTest` on your own model before committing. |
| **Self-supervised vision encoders** | **Yes, at 33–50% head pruning, with modest expectations.** I-JEPA ViT-H/14: +0.4 to +1.9pp, 35/36 wins over 3 seeds. No effect at 70%. DINOv2 is single-seed. |
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

### Self-supervised vision results (v0.5.1)

ImageNet-100, five epochs per arm on one A100. The SAL arm trains with
`mask_ratio=0.3`; the control trains identically without head masking. Both are
then pruned post hoc and scored on frozen representations.

**I-JEPA ViT-H/14 (631M)** — SAL wins **35 of 36** paired comparisons across
three seeds:

| pruning | metric | SAL | control | delta | SAL ahead |
|---|---|---|---|---|---|
| 33% random | linear probe | 76.8% | 76.4% | +0.4pp | 3/3 |
| 33% random | kNN | 71.2% | 70.0% | +1.2pp | 3/3 |
| 50% random | linear probe | 64.3% | 62.9% | +1.4pp | 3/3 |
| 50% random | kNN | 47.5% | 45.6% | +1.9pp | 3/3 |

Consistent, and small: read these as "SAL does not hurt and reliably helps a
little", not as the +2.45pp GPT-2 or +14pp ViT-base effect. No cost on the
unpruned model (83.2% vs 83.1% probe). At **70%** pruning SAL and control are
indistinguishable, whether SAL trains at 30% or 70% masking.

**DINOv2 ViT-L/14 (304M)** — one seed, SAL wins 13 of 18. DINOv2 is much more
fragile to head removal: at 50% both arms collapse. Where it survives, the gap is
larger:

| pruning | metric | SAL | control | delta |
|---|---|---|---|---|
| 33% sliced (random per layer) | linear probe | 84.0% | 81.0% | +3.0pp |
| 33% sliced (random per layer) | kNN | 75.4% | 68.5% | +6.9pp |

**What slicing buys.** `slice_heads()` on I-JEPA at 50% of heads: 631M → 526M
parameters, 1.10× GPU and 1.20× CPU latency at batch 1. Head removal only
narrows attention, which is a third of a ViT block, so expect ~1.1–1.3× from
heads alone, not 2×.

Caveats: the training objective is I-JEPA-*shaped* masked prediction, applied to
DINOv2 as well (not its self-distillation objective); DINOv2, slicing and 70%
are single-seed. Raw numbers in `data/results/`.

See [`examples/jepa_sal.py`](examples/jepa_sal.py),
[`scripts/run_jepa_sal.py`](scripts/run_jepa_sal.py) (standalone — any GPU box
over SSH, `bash scripts/setup_runpod.sh` to prepare one) and
[`scripts/modal_jepa_sal.py`](scripts/modal_jepa_sal.py) (Modal). Read the
caveats in all of them — in particular, Meta never released an I-JEPA ViT-B/16,
the training objective is I-JEPA-*shaped* rather than I-JEPA, and the default
table compares head *selection* rather than SAL training (`--control` adds the
arm that tests SAL training itself).

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
`quantize()`, and the five-seed validation above. v0.5.1 opened SAL to
self-supervised encoders. Next up is SAL + quantization-aware training and MLP
pruning (v0.6.0), aiming for a real 2× speedup.

## sal-torch-pro

A separate commercial extension, licensed and not on public PyPI. It adds, on top
of everything above:

- **MLP neuron pruning** — `MLPMasker`, `slice_mlp_neurons`. About a third of the
  accuracy cost of head pruning at equal compression (−3pp against −10pp at
  −16.5% of parameters), and it composes with `HeadMasker` for joint head+MLP
  compression.
- **`torch.compile` integration and latency benchmarking** — `benchmark_compile`
  across modes and batch sizes, ONNX export and verification.
- **Quantization-aware training** — `SALQATTrainer` and NF4 QAT, which added
  +2.4pp ± 0.9 over post-hoc INT4 on ViT-B/16 (15/15 seeds).
- **An extended `CompressionPipeline`** that wires those stages in, and
  **compliance reports** and **fleet monitoring** for deployed models.

Everything documented elsewhere in this README — SAL head masking, aligned
pruning with `uniform_per_layer` and `slice_heads()`, `FIScanner`,
`PlasticityScanner`, `RobustnessTest`, `StructuralGuard`, `probe()` and
`CompressionPipeline` — is in this package and needs nothing else.

Contact: david.martin@cognitive-engineering.dev

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
