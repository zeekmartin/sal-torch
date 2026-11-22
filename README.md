# sal-torch

![CI](https://github.com/zeekmartin/sal-torch/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/pypi/v/sal-torch) ![Python](https://img.shields.io/pypi/pyversions/sal-torch) ![Downloads](https://pepy.tech/badge/sal-torch) ![License](https://img.shields.io/badge/license-BSL%201.1-blue)


**Structurally Adaptive Learning for PyTorch**

Training-time sparsification that makes neural networks structurally resilient to compression.

> **What v0.7.0 closed, and what it measured.**
>
> Two gaps were named at the top of this page for two releases. Both are now
> shipped, and the point of v0.7.0 is that the shipped code was then made to
> reproduce the published numbers rather than merely resemble them.
>
> 1. **`CompressionPipeline` runs a recovery pass.** The stages are
>    `probe → scan → compress → recover → validate → export`. Until v0.7.0 every
>    accuracy on this page came from the research scripts and not from the
>    pipeline, which is how a missing recovery stage survived two releases. At
>    50% of heads the stage is the difference between **9.56%** and **78.81%**.
> 2. **DropHead is callable.** `HeadMasker` takes `mask_mode="drophead"`, the
>    per-step resampled quota this page recommends. `mask_mode="sal"` is the
>    accumulate-and-hold variant and remains the default for backward
>    compatibility; it is not the recommended one.
>
> **The release gate.** Before this tag the shipped pipeline was run against
> cells the project had already published, on bands written down *before* the
> runs and never widened afterwards. **Five of six bands reproduced; one did
> not.**
>
> | cell | pre-registered | the shipped pipeline | |
> |---|---|---|---|
> | ViT-B/16, 50% of heads, random, 2 epochs | mean [78.668, 78.942] | **78.814%**, cost -2.74pp | reproduced |
> | ViT-B/16, 50% of heads, magnitude, 2 epochs | mean [77.44, 77.81] | **77.611%**, cost -3.94pp | reproduced |
> | OPT-1.3B, 50% of heads, 32M C4 tokens | mean [21.783, 22.374] | **21.996** perplexity, cost +6.73 | mean reproduced |
> | the same cell, per seed | each seed in [21.683, 22.474] | 21.498 / 21.896 / 22.596 | **two seeds outside** |
>
> That last row is the one that failed, and it is reported rather than
> smoothed. Its tolerance had been **declared** at 0.10 perplexity because no
> decoder cell in the campaign had ever been re-run at a fixed seed. Re-running
> the original script, unchanged, at the failing seed put the same-seed floor
> at **0.3093** perplexity, three times the declared figure: the pipeline reads
> 21.498 and the original script reads 21.574, a gap of 0.076, while both sit
> below the single archived value of 21.883. The band still fails, because the
> failing seed deviates by 0.385 and that is more than 0.3093, so it is
> recorded FAILED and the tag was taken as an explicit owner decision with that
> failure named. **What this means for you:** the decoder perplexity in this
> README is reproducible to about 0.3, not to 0.1, and the vision accuracies to
> about 0.17pp. Treat differences smaller than that as noise.
>
> The v0.6.x campaign measured every arm on **ImageNet-1k**, three seeds, the
> full 50 000-image validation split, and the measured path is:
>
> **`probe()` → head training *or* magnitude selection → prune FFN neurons → recover → `torch.compile`**
>
> - **`probe()` first.** Two evaluation sweeps, no training: it measures how
>   much head damage costs *your* model. Nothing cheaper predicts that — not the
>   architecture, not the parameter count, and not the fragility index. It does
>   **not** pick the method: what picks the method is whether you can fine-tune
>   after pruning.
> - **DropHead**, per-step resampled head dropout, is the general head-training
>   method: it beats a standard fine-tune by **+57.2pp** at 50% head pruning +
>   INT4. Its resilience holds on *any* head set of that size, and its
>   seed-to-seed spread is 0.02–0.92pp. **Removing 33% of its heads leaves it
>   +0.14pp above its own dense row (77.69 vs 77.54) — but that row is not the
>   control.** Against the dense checkpoint trained for the same five epochs
>   with nothing pruned (80.44%), the same cell reads **−2.75pp**, because
>   DropHead's training itself costs **2.90pp** of dense accuracy. "Free"
>   compared an arm to itself; the earlier wording is withdrawn.
> - **SAL** is the specialist, and its carve-out is **withdrawn**: on its own
>   trained-for set it reads **76.51%**, +1.48pp over DropHead on that same set
>   at 85.9M — but **−0.67pp** on the same cell at 631M, and **34.52%** on a set
>   it did not train for. It is supported, not recommended. Use it only where
>   you cannot fine-tune after pruning, the deployed head set is fixed before
>   training, and you have measured it ahead on your own model.
> - **Most of the compression is in the FFN, not the heads.** At two epochs of
>   recovery against the matched 2-epoch control, 25% of FFN neurons costs
>   **−0.83pp** and 50% of heads costs **−3.91pp** — heads are **4.7x more
>   expensive per unit of size removed**. (At five epochs the same comparison
>   read −0.63 and −3.17pp; the multiplier survives the control correction.)
> - **Which heads you remove: random, not magnitude.** +1.17pp at 50% of
>   heads, separated on 3/3 seeds, and a tie at 25% and 33%. The code default
>   is already random. **For FFN neurons no criterion dominates** — magnitude
>   is +0.43pp at 25% of neurons and random +0.30pp at 50%, so the answer
>   depends on the ratio.
> - **2.04x on CPU is the largest speedup measured**, at batch 32, for 50% of
>   heads plus 50% of FFN neurons, and it costs **−9.73pp**. It is a
>   backend-specific number: see [the speedups](#mlp-neurons-vs-attention-heads).
>
> See [the ImageNet-1k results](#v06x--the-imagenet-1k-replication).
>
> **v0.6.0 also changed two defaults.** `SALConfig.prune_fraction` is now
> **0.50** (was 0.33) and head selection is **uniform per layer** by default
> (`uniform_per_layer=True`). The first follows the monotone dose-response: train
> at the ratio you will deploy at. The second is what makes the set of heads SAL
> trained without expressible as a `slice_heads()` argument, so the accuracy it
> retains becomes a smaller model rather than a mask. Pass
> `uniform_per_layer=False` for the pre-v0.6.0 global draw.
>
> **Two caveats that apply to every number in this README.** INT4 figures from
> the LUMI campaign are a **simulated per-channel quantize/dequantize
> round-trip**, not a 4-bit kernel — bitsandbytes NF4 is CUDA-only and absent on
> that ROCm image, so they measure the accuracy cost of 4-bit weight rounding
> and say nothing about speed or memory. The rounding is done in fp32; the
> rounded model is then **scored under `float16` autocast**, like every other
> accuracy in the campaign, while training and recovery run under `bfloat16`
> autocast. Earlier revisions of this page said "autocast bfloat16" for both.
>
> And "ImageNet-1k" here is
> **`evanarlian/imagenet_1k_resized_256`**, the open mirror of the gated set —
> canonical 1000 classes in canonical order, 256px shortest side. The pretrained
> checkpoint reads 78.44% on it against the 80.3% on its card, the difference
> being the resize and centre crop. All comparisons are internal to that setting.

## Known issues

**Exploding recovery gradients with SDPA attention under bfloat16 autocast on
AMD MI250X.** Measured on Qwen2.5-1.5B: the same model, the same batches and
the same 15 steps reach a pre-clip gradient norm of **6.93e14** under SDPA
with bf16 autocast, against **498.7** with `attn_implementation="eager"` at
the same precision and **610.4** with SDPA in fp32. With `max_grad_norm=1.0`
the clip then scales every step by about 1e-14, so **`recover()` finishes,
reports a loss curve that moved, and the model has not learned**: a 10M-token
fine-tune left accuracy at 52.97% against the untrained model's 53.23%.

It is model-dependent and not universal. OPT-1.3B under the same precision is
healthy, with pre-clip maxima of 46.81 to 992.6 across six archived recovery
runs, every loss curve falling, so no published figure in this repository is
affected.

**Until v0.7.1**, load the model with `attn_implementation="eager"` (about
6 GiB more on a 1.5B model) or train without the autocast, and read
`grad_norm_max` on the returned record: anything past about 1e4 is not a
large gradient, it is this. v0.7.1 makes `recover()` raise on the first such
step instead of clipping it away.

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

`CompressionPipeline` wires config → masker → training loop → head selection →
slicing → a quantization backend → an eval harness, and measures every stage, so
what comes out is a deployment decision rather than a number.

> **The example below is a demonstration of the API, not a result.** It is one
> T4 run, one model, one seed, with no control arm and no recovery pass, and it
> is kept because it shows what each call returns. Do not read its accuracies
> as a measurement of anything: the measured path is
> [the release gate](#v070--what-the-shipped-pipeline-reproduces) and
> [the ImageNet-1k results](#v06x--the-imagenet-1k-replication), both with
> three seeds and a matched control.
>
> **The recommended path is not "SAL-train, then compress".** That was the
> v0.5.0 reading and the ImageNet-1k campaign superseded it: at a fixed head
> set, SAL's weights are worth **+0.05pp** after a recovery pass against simply
> starting from the pretrained checkpoint, while the recovery pass itself is
> worth up to **68pp**. If you can fine-tune after pruning, that is where the
> accuracy is and the training method is close to irrelevant, so the path is
> `probe → compress → recover → validate` with `sal_train` left out entirely.

```python
from sal import CompressionPipeline

pipe = CompressionPipeline(model, eval_dataset, metric="accuracy")

# Step 1. Probe before you train anything. Two eval sweeps, no training, nothing
# modified — it removes half the heads at random and measures what that costs.
print(pipe.probe().recommendation)     # -> sal_aligned / drophead / skip_heads

print(pipe.scan().recommendation)      # fragility, absorption map, projected sizes

# Step 2. Prune. Random selection is the default and it is the right one.
pipe.compress(pruning=0.50, quantization="int4", slice_heads=True)

# Step 3. Recover. This is where the accuracy is: 9.56% without it, 78.81% with
# it, on ViT-B/16 / ImageNet-1k at 50% of heads. Two epochs, not five.
pipe.recover(train_dataset, epochs=2, domain="same as eval")

report = pipe.validate()
print(report.table)
# ILLUSTRATIVE — one T4, one seed, no control, no recovery pass
# (scripts/modal_v050_test.py, GPT-2 Medium / SST-2). An API demonstration.
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
seeds — one model, one binary task (GPT-2 Medium / SST-2, 355M), five seeds,
**no recovery pass**. See [ROADMAP.md](ROADMAP.md) for the full trail.

**Replicated on ImageNet-1k, with a recovery pass, at three seeds.** The grid
above is one seed of GPT-2 Medium with no recovery. exp40b removed heads by
magnitude and at random on ViT-B/16, recovered each for two epochs, and scored
against the matched 2-epoch control (81.554%), 3 seeds per cell:

| heads removed | random | magnitude | random − magnitude | separated on 3/3? |
|---|---|---|---|---|
| 25% | −1.20pp | −1.30pp | +0.10pp | no — **a tie** |
| 33% | −1.70pp | −1.93pp | +0.23pp | no — **a tie** |
| **50%** | **−2.74pp** | **−3.91pp** | **+1.17pp** | **yes** (+1.60pp at INT4) |

**Random selection is the recommendation, and it is `compress()`'s default
already.** At 50% the separation is clean — random's worst seed (78.67%) beats
magnitude's best (77.81%) — and the comparison is *unmatched in random's
disfavour*, since random's seed also draws the head set while magnitude's
choice is deterministic. At 25% and 33% the ranges overlap and the honest word
is "tie", not "random wins".

Worth keeping in view: one-shot, the seed-to-seed spreads here are 14.7–17.9pp
and after two epochs of recovery they are 0.25–0.55pp. **The recovery pass
dominates the selection criterion by more than an order of magnitude.** Choose
random because it costs nothing to compute, not because it rescues a bad
pruning decision.
Full trail in `results/lumi_v060/EXP40B_RANDOM.md`.

#### The recipe those vision figures were measured under

`recover()`'s defaults are this recipe, with two elements that are yours
rather than its: the batch size, which is whatever your data yields when you
hand it batches, and the augmentation. Every element is stated because a
recovery at a different recipe is not comparable with the numbers above.

| | |
|---|---|
| optimizer | AdamW, weight decay 0.05 |
| learning rate | 5e-5 on the backbone, **20x that on the classifier** (`head_lr_multiplier`) |
| schedule | linear warmup over 5% of the total steps, then cosine to zero |
| gradient clipping | global norm 1.0 |
| batch size | 64 (the caller's; `batch_size=` applies only when `recover()` does the batching) |
| epochs | 2 |
| precision | **bfloat16** autocast while training, **float16** autocast while scoring |
| data order | reshuffled every epoch from a seeded generator |
| **augmentation** | **random horizontal flip, p=0.5 per image, applied to every training batch** |

**The augmentation is yours to supply, and it is not optional for
reproduction.** `recover()` does not augment: it takes whatever batches the
dataset yields, because the same stage has to recover a decoder on token
windows where flipping is meaningless. The measured figures above were
produced with that flip in the data pipeline, and leaving it out costs, on
ViT-B/16 at 50% of heads, **0.15 to 0.26pp** of final accuracy on 3 of 3
seeds, with training loss 0.049 lower in the first epoch and 0.039 in the
second, measured on six runs with a spread of 0.0006 across seeds. If you
compare your own recovery against these numbers, flip your training batches.

The classifier's 20x is the same story read from the other side: it shipped as
10x for one cycle, on the strength of a write-up sentence rather than the
recipe, and the release gate is what caught it.

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
| **> 30pp** — fragile | `sal_aligned` | **Magnitude + recovery if you can fine-tune after pruning; DropHead if you cannot.** With recovery: **77.57%** at 50% of heads after two epochs (exp29; exp32 replicates at 77.64%, 3 seeds), above both head methods, and SAL's or DropHead's weights add **+0.05pp / +0.07pp** to it, i.e. nothing. Without recovery, DropHead: **+54.95pp** over a standard fine-tune on random sets, **+63.47pp** on a fixed magnitude set, **+54.70pp** on SAL's own set (ViT-B/16, ImageNet-1k, 3 seeds). SAL only where the deployed set is fixed *before* training — +1.48pp over DropHead on that one set at 85.9M, and **−0.67pp** at 631M. |
| **10–30pp** — intermediate | `drophead` | **Magnitude + recovery if you can fine-tune after pruning; DropHead if you cannot.** Removing 33% of heads leaves DropHead +0.14pp above its own dense row at 85.9M and −0.04pp at 631M — against a matched dense control it is −2.75pp, since the training itself costs 2.90pp. |
| **< 10pp** — robust | `skip_heads` | **Skip head training — but this row is the one the evidence contradicts.** It was set from BERT-base's 5.15pp probe drop on ImageNet-100/SST-2 and phi-2 at one seed. Measured directly on BERT-base across four GLUE tasks (3 seeds), one-shot drops at 50% of heads are **−4.17 / −12.42 / −21.68 / −17.56pp** — three of four land in the 10–30pp band or above it, not in the skip band. And in that band magnitude + 2 epochs of recovery beat DropHead on **8 of 8 cells**. Treat `skip_heads` as unvalidated. |

The `route` strings are the values `probe()` returns and are kept for
compatibility; `sal_aligned` names the **regime** — fragile enough that head
damage is expensive — not the method. The route does not choose the method:
if you can fine-tune after pruning, magnitude + recovery is the strongest
option measured in every band; if you cannot, DropHead is what you train with.

Then, in order — **probe → head training *or* magnitude selection → FFN
neurons → recover → `torch.compile`**:

1. **Probe.** It measures whether head damage is expensive on *your* model,
   which is the quantity that orders the outcomes — not the architecture, not
   the parameter count, not the fragility index. It does **not** pick the
   method; step 2 does.
2. **Decide whether you can fine-tune after pruning. That decision dominates
   the training method.** If you can, **magnitude selection + recovery** is the
   strongest single-architecture option measured — **77.57%** at 50% of heads
   after **two** epochs of recovery (exp29; exp32 replicates it at **77.64%**
   on 3 seeds), against SAL's 76.51% and DropHead's 72.54% — and it needs no
   special training method at all. At a fixed head set neither SAL's nor
   DropHead's weights are worth more than **+0.05pp / +0.07pp** as a starting
   point for that fine-tune. On BERT-base across four GLUE tasks it beat
   DropHead on 8/8 cells. **If you cannot fine-tune after pruning**, head
   training is what you have: DropHead is general over head sets and stable
   across seeds (sd 0.02–0.92pp), where SAL is specific to one set and spans
   0.50–17.55pp. The "SAL when the deployed head set is fixed at training time"
   carve-out is **withdrawn** — exp28b measured exactly that cell on a 631M
   encoder and DropHead matched or beat SAL on 3/3 seeds. SAL remains
   supported; see [When to use SAL](#when-to-use-sal). **Neither head method is
   callable from this package yet** — see the note at the top.
3. **Prune FFN neurons.** This is where the bulk of the compression is, and it is
   roughly **five times cheaper per unit of size** than head pruning — see
   [MLP neurons vs. attention heads](#mlp-neurons-vs-attention-heads) below.
   The tooling for it (`MLPMasker`, `slice_mlp_neurons`) ships in
   [sal-torch-pro](#sal-torch-pro), not in this package.
4. **`torch.compile`.** Measured **on GPU**: 1.12x on a dense ViT-B/16 at batch
   32, 1.50x on a pruned one, and 2.2x at batch 1 — where pruning alone buys
   almost nothing, also on GPU.

**Report latency per batch size, and state the device and the backend.** At
batch 32 the measured pruning speedups run 1.15x (25% of FFN neurons) to
**1.74x on GPU** and **2.04x on CPU** (50% of heads + 50% of neurons, torch /
oneMKL — 1.95x through ONNX Runtime on the same CPU). **At batch 1 on a GPU
every pruning configuration is 1.00x** — a ViT-B/16 there is kernel-launch
bound, and the only speedup available is the compiler's. That is a GPU
statement and earlier revisions of this page did not say so: on CPU, where the
same shape is arithmetic-bound rather than launch-bound, batch 1 does benefit
from slicing (1.944x for the joint arm). The full three-pass grids, and the
reason a faster backend *lowers* the ratio, are in
[the speedups](#the-speedups-with-their-backend).

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

**That table is ImageNet-100 (and SST-2 for the two text models). ViT-B/16 is the
one row that has since been re-measured on ImageNet-1k**, three seeds, the full
50 000-image validation split — and it is the anchor the thresholds should be
read against:

| ViT-B/16, ImageNet-1k, at 50% of heads | vs the matched baseline |
|---|---|
| probe drop (standard fine-tune, random sets) | 80.44% → 20.38% = **−60.06pp** → route `sal_aligned` |
| **DropHead**, random sets | 75.33% = **+54.95pp** (+57.23pp at INT4) |
| **DropHead**, one fixed magnitude set | 72.54% = **+63.47pp** |
| **SAL**, its own trained-for set | 76.51% = **+56.18pp** |
| SAL, sets it did not train for | 34.52% = **+14.14pp** |
| SAL − DropHead, on SAL's own set | **+1.48pp** (+0.91pp at INT4) |

The probe called this model fragile and the route was right: both methods are
worth large amounts here. What the 1k replication adds is the **shape** of the
two gains — DropHead's +55 to +63pp is the same size wherever it is scored, and
SAL's +56pp exists on exactly one set and falls to +14pp off it. The
ImageNet-100 reading of `SAL − DropHead = +2.02pp` reproduces at **+1.48pp** on
SAL's own set, and is **−42.66pp** at the cell where the head set is not
SAL's to choose.

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

Head pruning is the *small* half of the compression — and on ImageNet-1k, with a
recovery pass and a matched control, the gap is wider than the ImageNet-100
reading suggested. Measured against **the same checkpoint that saw the same five
epochs with nothing pruned** (80.44% / 79.01% INT4), three seeds, full
validation split:

**At two epochs**, which is the peak of this recovery recipe (exp29), against
the matched 2-epoch control (**81.554%**, 3 seeds, nothing pruned):

| ViT-B/16, ImageNet-1k, + 2 epochs recovery | params | accuracy | **cost** | INT4 cost |
|---|---|---|---|---|
| **25% of FFN neurons** (magnitude) | **−16.4%** | **80.73%** | **−0.83pp** | −0.99pp |
| 25% of FFN neurons (random) | −16.4% | 80.30% | −1.25pp | −1.50pp |
| 50% of FFN neurons (random) | −32.7% | 78.28% | −3.28pp | −3.68pp |
| 50% of FFN neurons (magnitude) | −32.7% | 77.97% | −3.58pp | −3.97pp |
| 50% of heads (magnitude) | −16.5% | 77.64% | **−3.91pp** | −4.85pp |
| 50% heads + 50% FFN (magnitude) | −49.1% | 71.83% | **−9.73pp** | −11.27pp |

3 seeds per cell, full 50 000-image validation split
(`EXP35_FFN_AT_2EP.md`, 22.9 GPU-hours; in the public data archive, DOI [10.5281/zenodo.24391200](https://doi.org/10.5281/zenodo.24391200), as `data/pro/results/EXP35_FFN_AT_2EP.md`).

**For FFN neurons, no selection criterion dominates.** Magnitude is better at
25% of neurons (+0.43pp, separated on 3/3 seeds by 0.32pp) and random is
better at 50% (+0.30pp, separated on 3/3 but by only **0.07pp** — one more
seed could close it). That is the opposite shape from the head case, where
random wins at 50% by 1.17pp with no overlap. **A single "prune by X"
recommendation does not survive both structures**; prune heads at random and
choose the FFN criterion by the ratio you are using.

The spreads are not comparable to each other either: magnitude's seed only
moves the data order, while random's also draws the neuron set, so random's
spread (0.17–0.37pp) is 3–9x magnitude's (0.02–0.11pp) and the 50% comparison
rests on random's noisier side.

The earlier **five-epoch** table read −0.63pp for 25% of FFN neurons and
−3.17pp for 50% of heads against the 5-epoch control. Both readings are true
and must be stated together: two epochs gives a *better model* and a *larger
pruning cost*, because the control is itself better at two epochs.

**The FFN tooling is in `sal-torch-pro`, not in this package.** `MLPMasker` and
`slice_mlp_neurons` are part of the commercial extension
([sal-torch-pro](#sal-torch-pro)); every FFN number on this page was measured
with them (exp26, 22 jobs, 97.2 GPU-hours). Head masking, `slice_heads()`, the
probe and the Fragility Index are all in this package.

**At comparable size removed, FFN pruning costs 4.7x less than head pruning** —
0.83pp at −16.4% against 3.91pp at −16.5%, measured at two epochs against the
matched control. The joint arm is worse than the sum of its parts: −9.73pp
against −3.58 for the FFN half and −3.91 for the head half, so head and FFN
pruning **interact unfavourably**, by about 2.2pp.

### The speedups, with their backend

**2.04x is the largest speedup measured.** It is `50% of heads + 50% of FFN
neurons` at **batch 32 on CPU**, and every term in that sentence is
load-bearing:

| | value |
|---|---|
| speedup | **2.044x** (median of three passes: 1307.11 / 1307.02 / 1306.53 ms) |
| accuracy cost | **−9.725pp**, −11.270pp at INT4 |
| CPU | **AMD EPYC 7A53 64-Core**, 7 threads (`OMP_NUM_THREADS=MKL_NUM_THREADS=7`) |
| backend | torch 2.5.1, **oneMKL** (`BLAS_INFO=mkl`, `USE_MKLDNN=ON`), eager, AVX2 |
| batch | **32**. At batch 1 the same arm reads 1.944x, and on GPU 1.74x |

**A second backend does not reach 2x.** Through ONNX Runtime 1.31.1 on the same
CPU at the same thread count, the same arm reads **1.952x** — while being
*faster in absolute terms* than torch on every arm at every batch (1118 ms
against 1317 ms for this cell). The reason is the denominator: ORT speeds the
dense model up by 1.23x and the pruned one by 1.18x, so **a better backend
shrinks the measured benefit of pruning**. The 2x is a property of
torch/oneMKL on this CPU, not of the model.

So quote it as: *2.04x on torch/oneMKL, 1.95x on ONNX Runtime, both at batch 32
on an EPYC 7A53 at 7 threads, at a cost of −9.73pp.* Two backends, one honest
range, and no configuration measured clears 2x on both. Trail:
`EXP41_WIDTH.md` (three-pass CPU and GPU grids) and
`EXP43_ORT_CPU.md` (ORT), both in the public data archive, DOI [10.5281/zenodo.24391200](https://doi.org/10.5281/zenodo.24391200), under `data/pro/results/`.

**On GPU, every arm is 1.00x at batch 1** (8.09–8.37 ms against 8.16 ms dense),
because a ViT-B/16 there is kernel-launch bound. **Any latency claim must state
its batch size, its device and its backend**, or it is not a claim.

The reference matters here and is easy to get wrong: against the *pretrained*
checkpoint's 78.44%, 25% of FFN neurons reads **+1.37pp** — but that compares a
model with five epochs of ImageNet-1k against one with none. The matched control
is the one that answers "does pruning cost accuracy", and the answer is −0.63pp.

The earlier ImageNet-100 reading, without a recovery pass, at an identical 71.7M
parameters:

| ViT-B/16 at 71.7M parameters (−16.5%), ImageNet-100, no recovery | retention vs. the dense model |
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
on public PyPI). At equal parameter count (−16.5%) the ImageNet-100 reading
*without a recovery pass* is −3pp for MLP against −10pp for heads; the
ImageNet-1k table above, with recovery and a matched control, is the figure to
quote: **−0.63pp against −3.17pp**.

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

Short answer: **a pruning win before it is a quantization win, and superseded
where it overlaps ImageNet-1k.** Five seeds of GPT-2 Medium show a SAL-trained
model keeping 2.45pp more accuracy under 33% head pruning on 5 of 5 seeds at no
cost to clean accuracy; INT4 alone is a coin flip, and under LoRA the whole
thing loses.

> **That GPT-2 study is no longer the project's primary evidence.** It is one
> model, one binary task, 355M parameters and no recovery pass. The ImageNet-1k
> campaign — three seeds, the full 50 000-image validation split, a dense
> control trained for the same number of epochs — supersedes it wherever the
> cells overlap, and it reaches a different conclusion: against a matched
> control, with a recovery pass available, the training-time method is close to
> irrelevant. Read the GPT-2 table below as the historical record that motivated
> the campaign, not as the recommendation.

The numbers, including the rows SAL did not win, are in
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
pruning**, +2.45pp at 33% on every seed and +2.06pp at 50% — GPT-2 Medium /
SST-2, one binary task, five seeds, **no recovery pass**. Both combined
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

#### ViT-B/16 on ImageNet-100 — fifteen seeds, and what ImageNet-1k later took back

> **Superseded for every cell it shares with the ImageNet-1k series
> (exp23 to exp28b).** This study is ImageNet-100, 100-way, and the three-seed
> full-split ImageNet-1k campaign overrides it wherever they overlap. It is
> kept because its *method* lessons stand and because the retractions should
> be legible next to what they retract, not filed elsewhere. **Do not quote
> the figures in this subsection as current results.**

The most finely measured study of the ImageNet-100 era: ViT-B/16, 5 epochs,
**15 seeds**, every compressed cell averaged over **20 random head
selections** rather than the 3 used above
(`scripts/run_sal_qat_prototype.py`, full trail in
`results/lumi_v060/PHASE2{,A,B}.md`). Head pruning at 50% plus INT4 weights:

| arm | P50+I4 top-1 | vs baseline |
|---|---|---|
| standard fine-tune | 31.0% ± 1.4 | — |
| SAL | 43.2% ± 3.8 | +12.1pp, ImageNet-100 |
| SAL + QAT | 45.6% ± 4.3 | +14.5pp ± 4.3, 95% CI [+12.4, +16.7], 15/15 seeds, ImageNet-100 |

That +14.5pp **replicates** on ImageNet-1k at three seeds, where the same cell
reads +14.57pp — and on that same cell DropHead reads **+57.23pp**, which is
what makes the figure a measurement of SAL rather than a recommendation of it.
See [the ImageNet-1k results](#v06x--the-imagenet-1k-replication).

Three things came out of it that the thinner studies could not show. The first
is the one ImageNet-1k contradicted.

- **The dose-response is monotone on ImageNet-100, and the cost claim that
  travelled with it is withdrawn.** Sweeping `prune_fraction` over
  {0.1, 0.2, 0.3, 0.4, 0.5}, the gain at half-pruned + INT4 rises steadily:
  +6.2 → +9.0 → +12.7 → +16.6 → +20.8pp, with the best cells the ones where
  the *training* ratio matches the *deployment* ratio. **The accompanying
  "0.3 to 1.5pp of clean accuracy" is retracted outright**: measured on
  ImageNet-1k across three seeds, SAL's clean accuracy reads
  71.53 / 67.57 / 61.61%, a spread of **9.92pp**, against DropHead's
  77.60 / 77.32 / 77.71%. A method whose accuracy moves ten points with the
  training seed cannot carry a sub-2pp cost claim.
  **And "train at the ratio you intend to deploy at" is no longer the
  recommended path**, because training-time head masking is itself no longer
  the recommended path: it is the branch taken only when no post-pruning
  fine-tune is possible. Where you can recover, prune by magnitude or at
  random and recover for two epochs. The v0.6.0 default of
  `prune_fraction=0.50` still stands for callers who are on that branch, since
  at 0.33 most of the effect goes unused against a 50% target.
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

#### v0.7.0 — what the shipped pipeline reproduces

Every figure in this README until v0.7.0 was measured by the research scripts.
The release gate asked the only question that makes them product claims: does
`CompressionPipeline`, as installed from PyPI, land on the same numbers. Six
bands were written down before the runs — three cells, each scored twice, once
against the archived recipe (**paper mode**, two-sided, because a reproduction
landing outside the cell has not reproduced it) and once against the shipped
defaults (**default mode**, one-sided). The bands were never widened, and the
scorer refuses a band whose code is not an ancestor of what was published.

| band | cell | pre-registered | measured | |
|---|---|---|---|---|
| G1-paper | ViT-B/16, 50% heads, random, 2 ep | mean [78.668, 78.942], each seed [78.498, 79.112], cost [-2.908, -2.568] | 78.814%, seeds 78.718 / 78.672 / 79.052, cost -2.740 | **PASS** |
| G1-default | the same runs, shipped defaults | at or above 78.668 | 78.814% | **PASS** |
| G2-paper | ViT-B/16, 50% heads, magnitude, 2 ep | mean [77.44, 77.81], each seed [77.27, 77.98], cost [-4.08, -3.74] | 77.611%, seeds 77.510 / 77.616 / 77.708, cost -3.943 | **PASS** |
| G2-default | the same runs, shipped defaults | at or above 77.44 | 77.611% | **PASS** |
| G3-paper | OPT-1.3B, 50% heads, 32M C4 tokens, archived recipe | mean [21.783, 22.374], each seed [21.683, 22.474] | mean 21.996, seeds 21.498 / 21.896 / 22.596 | **FAIL** |
| G3-default | the same cell, shipped defaults | at or below 22.374 | mean 21.867 | **PASS** |

Vision accuracies are top-1 on the full 50 000-image validation split, against
the archived 2-epoch dense control (81.554%, 3 seeds) trained for the same
number of epochs; the decoder figures are WikiText-2 perplexity, against the
archived dense control after the same 32M C4 tokens (15.264). Three seeds
each, 42 / 123 / 456. Cost 8.83 billed GPU-h on one AMD MI250X GCD per job.

**G3-paper failed, and the reason is worth more than the verdict.** The mean
and the matched pruning cost both reproduce; what misses is two of three
per-seed windows, one low and one high. Those windows had been widened by a
tolerance of 0.10 perplexity that was **declared rather than measured**,
because no decoder cell in the campaign had ever been re-run at a fixed seed.
So the original script was re-run, unchanged, at the failing seed. Its
one-shot perplexity reproduced the archive's to every stored digit, which
proves it removed the same 384 heads, and its recovered perplexity came out at
21.574 against the archive's 21.883: a same-seed floor of **0.3093**, three
times what had been declared. The pipeline's own value for that seed, 21.498,
sits 0.076 from the original script's and 0.385 from the archive's. Since
0.385 exceeds 0.3093, the pre-registered rule records the band as **FAILED**
rather than explaining it away, and v0.7.0 was tagged on an explicit owner
decision that names the failure.

**What to take from it.** The decoder perplexities on this page are
reproducible to roughly 0.3, not 0.1. The vision accuracies are reproducible
to 0.040-0.170pp, which is measured over four same-seed control pairs. Any
difference smaller than that is noise, including differences this README
quotes. A band built on an unmeasured tolerance is now forbidden outright: a
reproduction band for a decoder cell needs a measured same-seed floor from at
least two pairs.

Two defects in the shipped code were found by the gate itself and fixed before
the tag: `recover()` defaulted the classifier learning rate to 10x the
backbone where every published vision figure was measured at **20x**
(`run_sal_qat_prototype.make_optimizer` runs 1e-3 against 5e-5), and
`recover()` shuffled only once rather than every epoch. Both would have been
invisible without a reproduction run, and both had been documented as
correct.

#### v0.6.x — the ImageNet-1k replication

**This is the most rigorous evidence in the project, and where the cells overlap
it supersedes every ImageNet-100 figure below.** ViT-B/16 on **ImageNet-1k**
(`evanarlian/imagenet_1k_resized_256`), 1 281 167 train / 50 000 val, 1000
classes, **three seeds (42, 123, 456), the full validation split**, 5 epochs,
`mask_ratio 0.5 --uniform-per-layer`. Full trail in
`results/lumi_v060/PHASE3_IMAGENET1K.md`, `EXP24_SAL_RECOVERY.md`,
`EXP25_ALIGNED.md`, `EXP27_DROPHEAD.md`.

The control for everything here is **the matched baseline**: the same images,
optimiser, schedule and epoch count with nothing masked — **80.44% / 79.01%
INT4**. Not the pretrained checkpoint, which has seen no fine-tuning.

**DropHead is general.** Scored against that baseline on the **identical** head
set, 3/3 seeds, every cell:

| head set removed | DropHead | baseline | **difference** |
|---|---|---|---|
| 33% of heads, one fixed magnitude set | **77.69%** | 51.24% | **+26.45pp** |
| 50% of heads, one fixed magnitude set | **72.54%** | 9.06% | **+63.47pp** |
| 50% of heads, **SAL's own** set | **75.03%** | 20.33% | **+54.70pp** |
| 50% of heads, 20 random sets | **75.33%** | 20.38% | **+54.95pp** |
| 50% + INT4, 20 random sets | **73.29%** | 16.06% | **+57.23pp** |

**Removing 33% of DropHead's heads leaves it +0.14pp above its own dense row** (77.69% vs 77.54%, and 76.12 vs 76.09 at INT4) — but that row is not the control. Against the dense checkpoint trained for the same five epochs with nothing pruned (80.44%) the same cell reads **−2.75pp**, because DropHead's training costs **2.90pp** of dense accuracy on its own. The earlier "free" wording compared an arm to itself and is withdrawn. At 50% it costs 5.01pp against its own dense row.

**DropHead is also stable, which SAL is not.** Seed-to-seed standard deviation
over six measured rows: **DropHead 0.02–0.92pp, SAL 0.50–17.55pp.** SAL's clean
accuracy across the three seeds is 71.53 / 67.57 / 61.61%; DropHead's is
77.60 / 77.32 / 77.71%. One SAL row reads 60.15 / 51.83 / 26.46%.

**SAL is a specialist, and that is a real effect with a hard boundary.** On the
one set it trained without, against the baseline on that identical set:

| at 50% of heads | SAL | baseline | **difference** |
|---|---|---|---|
| SAL's own trained-for set | **76.51%** (sd 0.50) | 20.33% | **+56.18pp** |
| 20 sets it never saw | 34.52% | 20.38% | +14.14pp |
| *the set, scored on the baseline vs a random set* | — | **−0.05pp** | — |

The last row is the control that settles it: **SAL's head set is worth 0.05pp to
a model that did not train for it.** The set is not a clever selection — it was
drawn at random. What SAL learned is not "operate with fewer heads" but "operate
without *these* heads", worth 42pp on that set and nothing off it.

**SAL's clean-accuracy cost is −3.93pp, not −13.54pp.** Its dense score (66.90%)
is not what it produces: remove the heads it trained without and it reads 76.51%,
**9.61pp better than its own dense score** at a 16.5% smaller model. Carrying
those 72 heads is actively harmful. Quote 80.44 → 76.51; any table using SAL's
dense number as its clean accuracy overstates the cost by 9.6pp.

**Where that leaves the four options**, at 50% of heads + INT4:

| method | accuracy | scored on | needs a post-prune fine-tune |
|---|---|---|---|
| SAL, random sets | 30.63% | 20 unseen sets | no |
| **DropHead**, random sets | **73.29%** ± 0.71 | 20 unseen sets | no |
| **SAL, its own set** | **73.95%** | 1 trained-for set | no |
| **magnitude + recovery** | **75.06%** | 1 fine-tuned-for set | **yes, 5 epochs** |
| SAL → aligned + recovery | 75.66% | 1 fine-tuned-for set | yes, 5 epochs |

**The two recovery rows are 5-epoch figures.** At the measured default of **two**
epochs the same magnitude arm reads **75.43% INT4 / 77.64% FP** (exp32, 3 seeds;
exp29 replicates at 75.55 / 77.57). Quote the epoch count with the number: the
costs these accuracies imply are only comparable against a dense control trained
for the *same* number of epochs, and the 2-epoch and 5-epoch controls differ by
1.11pp.

SAL on its own set is **+0.91pp over DropHead on that same set** and −1.11pp
under magnitude + recovery, which it beats on convenience — no fine-tuning pass.
Two conditions come attached and neither is small: **the head set must be fixed
at training time**, and **so must the ratio** — at 33%, a subset of its own set
and therefore an architecture it never trained for, SAL falls to 72.76%, below
magnitude + recovery's 78.85%.

**SAL is not a pre-conditioner.** At a *fixed* head set, neither SAL's nor
DropHead's weights beat the pretrained ones after recovery: **+0.05pp** and
**+0.07pp** at 50% + INT4. What survives is a head-*selection* signal worth
+0.60pp at 50%, and nothing at 33%, where the sign flips. If you can run a
recovery pass, no training-time masking method is buying you much.

**Two retractions.** The ImageNet-100 campaign's **+22pp** headline came from the
three most favourable seeds of five and is withdrawn (`PHASE2A.md`); the
replicated figure is +14.57pp at 50% + INT4, on the comparison SAL has always
made — against its *own* baseline, on sets it did not train for. And the
**"costs 0.3–1.5pp of clean accuracy"** claim does not survive three seeds: see
the 0.50–17.55pp spread above. A method whose accuracy moves thirty points with
the training seed cannot carry a sub-2pp cost claim.

#### v0.6.0 — two methods, two deployment regimes (ImageNet-100)

The v0.6.0 campaign on LUMI (AMD MI250X, ~85 GPU-hours) added the control arms
that had never been run, and they changed the picture. Everything below is
ViT-B/16 / ImageNet-100, 5 epochs, **10 seeds**, **20 shared head selections per
cell**, full trail in `results/lumi_v060/PHASE2{C,D,E,F}.md`. **Where it overlaps
the ImageNet-1k tables above, those supersede it** — in particular the
`SAL − DropHead = +2.02pp` reading, which holds only on SAL's own head set
(+1.48pp on 1k) and is −42.66pp where the set is not SAL's to choose.

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

**Both rows below assume you cannot fine-tune after pruning.** If you can,
neither is the recommendation — magnitude + 2 epochs of recovery beats both, and
SAL's or DropHead's weights add +0.05pp / +0.07pp to it. This table is the
ImageNet-100 reading of the head-training choice, kept for the trail.

| your deployment, with no recovery pass available | use | measured, ImageNet-100 |
|---|---|---|
| You choose which heads to remove, and can keep a list | **SAL** with `uniform_per_layer=True` (the default), saving `pruned_heads()` with the checkpoint and slicing exactly that set | 16.5% smaller, **+1.4pp** over the best alternative at the same size, 8–10/10 seeds |
| The pruning is out of your hands, or varies per deployment | **DropHead** (`mask_mode="drophead"` in the prototype script) | **+49.8pp** over a standard fine-tune at 50% + INT4, 10/10 seeds |

SAL costs ~1pp of clean accuracy (91.32% vs DropHead's 92.36%), so its net
advantage at deployment is ~1.4pp, not ~2pp. And SAL scored on a set it did not
train for is **27.6pp behind** DropHead. That gap is the price of specialisation
and nothing in v0.6.0 reduced it.

> **The ImageNet-1k replication keeps this shape and changes both magnitudes.**
> DropHead's advantage over a standard fine-tune is **+57.2pp** at 50% + INT4
> (3 seeds, full validation split), SAL's edge on its own set is **+0.91pp** at
> INT4 (+1.48pp in FP), and SAL on a set it did not train for is **42.66pp**
> behind DropHead, not 27.6pp. Use the 1k figures when quoting this table.

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
  not. On an arbitrary head set DropHead is ahead by **27.6pp** on ImageNet-100
  (82.16% vs 54.56% at mask_ratio 0.5, 10 seeds) and by **42.66pp** on
  ImageNet-1k at 50% + INT4 (73.29% vs 30.63%, 3 seeds). SAL's advantage exists
  only on the specific set it trained without, which means you must record that
  set and prune exactly it. If you cannot, use DropHead — and if you can
  fine-tune after pruning, neither: magnitude + recovery beats both.
- **That SAL's edge survives INT4.** Sliced and in FP16 it is +1.36pp, CI
  [+0.53, +2.18], 8/10 seeds on ImageNet-100. With INT4 weights stacked on top
  the same comparison is +0.49pp with the interval spanning zero, 6/10. On
  ImageNet-1k, SAL against DropHead **on SAL's own set** is +1.48pp in FP and
  **+0.91pp at INT4** over 3 seeds, with no interval published. The deployment
  claim holds for slicing; for slicing *plus* 4-bit weights it is a sub-point
  effect at every measurement and should not be sold as one.
- **That SAL's clean accuracy is stable enough to carry a cost claim.** It is
  not. Across three ImageNet-1k seeds SAL's dense accuracy is
  71.53 / 67.57 / 61.61%, and its seed-to-seed sd spans **0.50–17.55pp** over six
  measured rows against DropHead's 0.02–0.92pp. One row reads
  60.15 / 51.83 / 26.46%. The withdrawn "0.3–1.5pp" figure came from a setting
  with fewer seeds; the stable statement is the aligned-set one, **−3.93pp**.
- **That SAL's weights pre-condition a recovery fine-tune.** They do not. At a
  fixed head set, SAL pre-training is worth **+0.05pp** after five epochs of
  recovery, and DropHead's **+0.07pp** — i.e. nothing, against starting from the
  pretrained checkpoint. If a post-prune fine-tune is available to you, the
  training-time method is close to irrelevant.
- **That slicing helps at batch size 1.** It does not — it is marginally
  *slower* (0.96x), because that shape is bound by kernel launch overhead rather
  than arithmetic. The speedup at batch 32 is measured and real (1.20x eager,
  1.37x compiled); see [Does the smaller model actually run
  faster?](#does-the-smaller-model-actually-run-faster). Where the two regimes
  cross over is somewhere between 1 and 32 and has not been located, and the
  whole sweep is one GPU (MI250X), one architecture and one image size.
- **That 2x is reachable by pruning this model at all.** **The withdrawal was
  a GPU statement and it stands as one; on CPU 2x is reached.** On GPU, joint
  head+FFN pruning measures **1.74x at batch 32** with half the parameters
  removed and **1.00x at batch 1** in every configuration, so the GPU target
  stays withdrawn. On CPU the same arm reads **2.044x at batch 32** under
  torch/oneMKL and **1.952x** under ONNX Runtime — so it clears 2x on one
  backend and not the other, at a cost of −9.73pp against the matched
  2-epoch control. 1.37x is what 50% of heads buys with compilation on top.
  Attention is roughly a third of a transformer block, so the headroom is in
  the FFN — which is in [sal-torch-pro](#sal-torch-pro), not this package.
- **That the aligned/sliced result travels.** It has been measured on **one
  architecture** — ViT-B/16, now on both ImageNet-100 and ImageNet-1k, where the
  aligned-set effect replicated (+56.18pp over the matched baseline on the
  identical set, 3/3 seeds). That is a harder dataset, not a second
  architecture. The cross-architecture table covers DropHead on DINOv2 and
  GPT-2, not the aligned-pruning path.
- **Anything at all above ~631M.** 631M itself is now measured: the first
  attempt (exp28) collapsed every arm including the unmasked control and
  produced no comparison, but the corrected recipe (exp28b, frozen target
  encoder) holds the control on 3/3 seeds and the comparison goes through —
  DropHead ahead of SAL on every cell. **The earlier instruction to "treat every
  number in this README as applying to supervised fine-tuning at ≤350M" is
  withdrawn**; it predated exp28b and contradicted the 631M row in
  [When to use SAL](#when-to-use-sal). What remains unmeasured is above 631M,
  and self-supervised training at scale outside the one corrected recipe.
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

**Run `probe()` first, and answer one question before it.** The question is
whether you can run a fine-tuning pass after pruning: if you can, that dominates
every choice below and the training-time method is close to irrelevant
(+0.05pp / +0.07pp at a fixed head set). The probe then tells you how much is at
stake rather than which method to use. The old advice here was "use DropHead by
default"; six architectures and one matched-control campaign later, the default
is to measure your own model instead of inheriting someone else's regime.

| your setup | recommendation |
|---|---|
| **Any model, before anything else** | **`pipe.probe()`.** Two eval sweeps, no training. It measures how much head damage costs your model; it does **not** choose the method. **The method is chosen by whether you can fine-tune after pruning** — if you can, magnitude + recovery, in every band; if you cannot, DropHead, and SAL only where the deployed set is fixed before training (and even there DropHead matched or beat it at 631M). The probe tells you how much is at stake. See [Start with the probe](#start-with-the-probe-not-with-a-guess). |
| **You can run a fine-tuning pass after pruning** | **Magnitude + recovery, and skip the special training method.** On ImageNet-1k it is the strongest single-architecture option at 50% of heads: **77.57% / 75.55% INT4 after two epochs** of recovery (exp29; exp32 replicates at 77.64% / 75.43% on 3 seeds), against SAL's 76.51% / 73.95% on its own set. **Two epochs produces a better model than five and a larger pruning cost**, and both are true: 77.64% against 77.27%, but **−3.91pp** against the matched 2-epoch control where five epochs cost **−3.17pp** against its own — because the control is itself better at two epochs (81.55% vs 80.44%). Ten epochs is worse than one, **on a single seed** — that cell was not replicated, where the 1/2/3/5-epoch points are 3 seeds each. And SAL's weights are **not** a useful starting point — at a fixed head set SAL pre-training is worth **+0.05pp**. |
| **The probe says > 30pp, and you know which heads you will remove** | **If you can fine-tune after pruning, magnitude + recovery; otherwise DropHead first, and SAL only if you measure it ahead on your model.** This is SAL's best cell and the only one where it has ever led: +1.48pp over DropHead at INT4 on ViT-B/16 (below), and **−0.67pp** on the same cell at 631M (exp28b), so the lead does not survive a change of model. If you do use SAL, keep `uniform_per_layer=True` (the default). Save `pruned_heads()` with the checkpoint and slice that set. 16.5% fewer parameters, **+56.18pp** over a standard fine-tune on that identical set and **+0.91pp** over DropHead at INT4 (ImageNet-1k, 3 seeds). Without the saved set the advantage is unrecoverable: off its own set SAL reads 34.52% against DropHead's 75.33%. The **ratio** must match too — at 33% of heads, a subset of its own set, SAL falls to 72.76%. |
| **The probe says > 10pp and the pruning is out of your hands** | **DropHead, not SAL.** Per-step resampled head dropout: **+57.2pp** over a standard fine-tune at 50% pruning + INT4 on ViT-B/16 / ImageNet-1k, 3 seeds, and **+63.5pp** on a fixed magnitude set. 33% of heads leaves it **+0.14pp above its own dense row** and **−2.75pp against a matched dense control** — its training costs 2.90pp on its own. SAL on an arbitrary set is **42.7pp behind** it. DropHead is also the stable one: seed-to-seed sd 0.02–0.92pp against SAL's 0.50–17.55pp. |
| **The probe says < 10pp** | **Verify before trusting this row.** On BERT-base and phi-2 both head methods landed within 0.1pp of a standard fine-tune, and on I-JEPA DropHead cost 1.1pp of clean accuracy for no gain — but BERT-base, probed at −5.15pp, loses **−4.17 to −21.68pp** one-shot at 50% of heads across four GLUE tasks, mostly outside this band, and magnitude + 2 epochs of recovery beats DropHead there on 8/8 cells. Compress, then recover, and measure on your own task. |
| **Where your compression actually comes from** | **FFN neurons, not heads** — measured on ImageNet-1k with a matched control, **roughly five times cheaper per unit of size**: 25% of neurons costs 0.63pp at −16.4%, where 50% of heads costs 3.17pp at −16.5%. Not in this package; see [MLP neurons vs. attention heads](#mlp-neurons-vs-attention-heads). |
| **You need 2x** | **On CPU, yes, on one backend, at batch 32, for −9.73pp. On GPU, no.** CPU: 50% of heads + 50% of FFN neurons reads **2.044x at batch 32** under torch 2.5.1 / oneMKL on an AMD EPYC 7A53 at 7 threads (three passes within 0.6 ms), and **1.952x** through ONNX Runtime 1.31.1 on the same CPU — which is *faster in absolute terms* and yet a lower ratio, because a better backend speeds the dense denominator up more than the pruned arm. At batch 1 on CPU the same arm is 1.944x. It costs **−9.725pp** against the matched 2-epoch control, −11.270pp at INT4. GPU: the highest measured speedup is **1.74x** at batch 32 and every configuration is 1.00x at batch 1, so the GPU 2x target stays withdrawn. **No configuration measured clears 2x on both backends.** |
| **Full fine-tuning, and you prune heads** | **Measure, and prefer a recovery pass.** The GPT-2 Medium result (+2.45pp at 33% pruning on 5/5 seeds, no cost to clean accuracy) stands as measured, but it is one model on one binary task with no recovery pass, and ImageNet-1k supersedes it where the cells overlap: at a fixed head set SAL's weights are worth **+0.05pp** after recovery. If you can fine-tune after pruning, do that; the training method is close to irrelevant. |
| **Vision transformers, pruning at ≥50%** | **Yes, and read the ImageNet-1k numbers.** SAL over its own baseline at 50% + INT4 is **+14.57pp** on 3/3 seeds of ImageNet-1k, replicating the +14.5 ± 4.3 from 15 seeds of ImageNet-100 — but on the same cell DropHead reads +57.2pp. SAL's clean-accuracy cost is **−3.93pp** (80.44 → 76.51 on its own set, at a 16.5% smaller model); the earlier "0.3–1.5pp" claim is withdrawn, and so is the **+22pp** headline published from three seeds. |
| **Vision transformers, pruning at 33%** | **Measure first.** On ViT-B/16/ImageNet-100 the benefit is weak and inconsistent (one recurring seed below baseline), while on ViT-base/CIFAR-10 it was +14 to +20pp on 3/3 and on GPT-2 Medium +2.45pp on 5/5. Same ratio, three different answers — this is where your model and task decide. |
| **Full fine-tuning, quantization only** | **Measure first.** INT8 gave +0.73pp on GPT-2 (4/5 seeds) and nothing on ViT; INT4 alone was a coin flip (3/5, +0.21pp). Use `RobustnessTest` on your own model before committing. |
| **Self-supervised vision encoders** | **Yes, at 33–50% head pruning, with modest expectations.** I-JEPA ViT-H/14: +0.4 to +1.9pp, 35/36 wins over 3 seeds. No effect at 70%. DINOv2 is single-seed. |
| **LoRA / QLoRA adapters** | **Not recommended.** Measured worse than not using SAL at all, and it costs clean accuracy. The adapters are too small to redistribute what the masking removes. |
| **Models above ~1B** | **Measured at 631M, and DropHead wins there too.** The first attempt (exp28) collapsed *every* arm including the unmasked control — three-seed linear probe 46.07% pretrained → 15.32% control, 3.20% DropHead, 2.69% SAL — because the objective computed its target from the **same live weights it was training**, which has no fixed point. With a **frozen target encoder**, LR 5e-6, 10% warmup and cosine decay (exp28b), the control survives on 3/3 seeds and the comparison runs: at 50% head pruning DropHead is **+29.02pp** over the control against SAL's +7.89pp, at 33% **+11.88pp** against SAL's −0.22pp, and **+0.67pp on SAL's own trained-for set**. SAL also pays **12.35pp** of dense accuracy. 420.2 GPU-hours across both campaigns. See `results/lumi_v060/EXP28B_JEPA_CORRECTED.md`. |

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

> **This does not scale, yet.** The attempt to reproduce it at full size —
> I-JEPA ViT-H/14, two epochs of continued pretraining on ImageNet-1k — collapsed
> **every** arm including the unmasked control: linear probe 48.6% → 16.6% for a
> standard run, 3.0% DropHead, 2.3% SAL, with the training loss falling the whole
> way. The probe was verified independently (±0.58pp) and `slice_heads()` agrees
> with masking to 1e-06, so the fault is the recipe — learning rate, schedule and
> target-encoder EMA — not the methods or the package. The results above come
> from short runs where that recipe holds. ~181 GPU-hours, no comparison
> obtained; see `results/lumi_v060/EXP28_JEPA_1B.md`.

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
self-supervised encoders. v0.6.x added DropHead, `probe()` as the router, and
the ImageNet-1k replication — and withdrew the 2× speedup target, which is not
reachable by magnitude pruning on a ViT-B/16 (1.74× measured, at batch 32).
v0.7.0 is the pipeline the measurements imply, diagnose → route → compress
→ recover, and the release that put the shipped pipeline through a
pre-registered reproduction of the published cells: five of six bands
reproduced, one did not, and the one that did not is
[named with its numbers](#v070--what-the-shipped-pipeline-reproduces).

## sal-torch-pro

A separate commercial extension, licensed and not on public PyPI. It adds, on top
of everything above:

- **MLP neuron pruning** — `MLPMasker`, `slice_mlp_neurons`. **Validated on
  ImageNet-1k**, three seeds, full validation split, against a matched control:
  roughly **five times cheaper per unit of size** than head pruning — 25% of
  neurons costs **−0.63pp** at −16.4% of parameters (1.15× at batch 32) where
  50% of heads costs −3.17pp at −16.5% (1.10×). It composes with `HeadMasker`
  for joint head+MLP compression, whose ceiling is **1.74× at −6.82pp**; note
  that at a matched −32.7%, all-MLP beats head+MLP by 1.55pp, so the joint arm's
  extra throughput is not free.
- **`torch.compile` integration and latency benchmarking** — `benchmark_compile`
  across modes and batch sizes, with **ONNX export validated** against the eager
  model. Report per batch size: the compiled dense ViT-B/16 is 1.12× at batch 32
  and 2.2× at batch 1, where pruning alone buys nothing.
- **Quantization-aware training** — `SALQATTrainer` and NF4 QAT, which added
  +2.4pp ± 0.9 over post-hoc INT4 on ViT-B/16 (15/15 seeds).
- **An extended `CompressionPipeline`** that wires those stages in, and
  **compliance reports** and **fleet monitoring** for deployed models.

Everything documented elsewhere in this README — SAL head masking, aligned
pruning with `uniform_per_layer` and `slice_heads()`, `FIScanner`,
`PlasticityScanner`, `RobustnessTest`, `StructuralGuard`, `probe()` and
`CompressionPipeline` — is in this package and needs nothing else.

## Citation

The measurements in this README are from the preprint, and every figure in it is
generated from the result files in the data archive beside it.

- **Paper**: *The Price of Structure: Matched-Control Measurements of
  Transformer Compression on ViT, BERT and OPT*,
  [10.5281/zenodo.24390835](https://doi.org/10.5281/zenodo.24390835)
- **Data archive** (per-cell result files, the canonical tables, the generators
  and the verifiers): [10.5281/zenodo.24391200](https://doi.org/10.5281/zenodo.24391200)

```bibtex
@misc{martinventi2026price,
  title        = {The Price of Structure: Matched-Control Measurements of
                  Transformer Compression on ViT, BERT and OPT},
  author       = {Martin Venti, David},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24390835},
  url          = {https://doi.org/10.5281/zenodo.24390835},
  note         = {Preprint. Data archive: 10.5281/zenodo.24391200}
}

@misc{martinventi2026pricedata,
  title        = {Data archive for "The Price of Structure"},
  author       = {Martin Venti, David},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24391200},
  url          = {https://doi.org/10.5281/zenodo.24391200}
}
```

To cite the software itself rather than the measurements, see
[CITATION.cff](CITATION.cff).

Contact: david.martin@cognitive-engineering.dev

## License

BSL 1.1 — free for research and evaluation. Commercial production requires a license.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
