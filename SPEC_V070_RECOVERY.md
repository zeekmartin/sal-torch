# v0.7.0 item 1 — a recovery stage for `CompressionPipeline`

**Status: specification. Nothing here is implemented.** Raised by the pre-report
audit (pass 2) and approved as v0.7.0 item 1.

## The gap

`CompressionPipeline` has no recovery stage. In `sal/pipeline.py` the stages are
`probe → scan → sal_train → select_heads → compress → validate → report →
export`; in the pro package they are `train → compress → validate → export`.
In both, `compress()` slices heads and quantizes and the next thing that happens
is scoring. Nothing repairs the damage.

Every accuracy figure this project publishes for post-hoc pruning includes a
repair pass that the product does not perform:

| arm | one-shot | after recovery | what the product delivers |
|---|---|---|---|
| magnitude heads, P50 | **9.56%** | 77.642% (2 ep) | the 9.56% |
| SparseGPT 2:4, attention | 75.223% | 81.097% (2 ep) | the 75.223% |
| Wanda 2:4, attention | 72.871% | 80.999% (2 ep) | the 72.871% |
| unstructured, attention 50% | 77.210% | 81.397% (2 ep) | the 77.210% |
| MLP 25% | — | 79.81% (5 ep) | no recovery |

Sources: `results/lumi_v060/EXP32_RATIO_AT_2EP.md`,
`EXP33_SPARSE_VS_STRUCTURED.md`, `sal-torch-pro/results/EXP26_MLP.md`. The
one-shot magnitude figure is `exp32_ratios_s{42,123,456}/exp29_recovery.json`,
`ratios/0.50/accuracy_before_recovery`.

At P50 that is a **68pp** difference between what the documentation measures and
what the shipped pipeline produces. `sal_train` does not close it: those epochs
are masked *training*, paid before pruning, and exp24 established that they buy
nothing as a pre-conditioner (+0.05pp / +0.07pp at P50+INT4).

Two consequences beyond the headline number:

- **`accuracy_floor` fires on the wrong model.** `validate()` raises
  `PipelineError` below the floor and suggests "try a smaller `prune_ratio`".
  A user at P50 sees 9.56%, concludes the method does not work, and reduces the
  ratio — when two epochs would have put them at 77.6%. The error message
  actively misdirects.
- **Unstructured sparsity cannot be shipped at all.** It needs *mask-preserving*
  recovery: an ordinary fine-tune fills the zeros back in and the model silently
  stops being sparse while still carrying the label. `exp33` enforces this with
  a gradient hook plus a final re-apply and records `sparsity_drift` /
  `mask_preserved`. The product has neither, so it has no correct path to the
  cheapest compression the project has measured.

## What to build

### 1. `recover()` as a pipeline stage

```python
def recover(self, train_dataset, epochs: int = 2, *,
            lr: float = None, batch_size: int = None,
            preserve_sparsity: bool = True,
            control: "RecoveryControl | None" = None) -> Stage:
```

Placed between `compress()` and `validate()`. Raises `PipelineError` if called
before `compress()`, matching the existing stage discipline.

- **`epochs=2` is the default** and the number is measured, not chosen.
  `EXP29_RECOVERY_EPOCHS.md` found the curve peaks at 2 and declines, with 5
  below 2 on 3/3 seeds in FP and INT4, and 10 below 1. exp32 confirmed it at
  three ratios on three seeds. Two is both better and 39% of the cost of five.
- **The docstring must carry exp29's reframing.** The peak at 2 epochs is a
  property of *this fine-tuning recipe on this dataset*, not of recovery: the
  dense control with nothing pruned also peaks at 2, and by a larger margin.
  A user on another dataset must not assume the knee transfers, so the parameter
  is documented as "measured default, re-measure on your data", not "optimal".
- **Reuse the training recipe**, do not write a second one. `experiments/
  exp23_baselines.py:recover()` is the reference: AdamW, 10x LR on the
  classifier, linear warmup then cosine decay over `len(loader) * epochs`,
  bfloat16 autocast, horizontal flip, grad clip 1.0. The control and every arm
  in the ImageNet-1k series share it, which is why their comparison is sound;
  a different recipe in the product would make the published numbers
  unreproducible through the product.

### 2. `preserve_sparsity`, and why it defaults to true

For an unstructured or 2:4 compressed model, recovery must be masked:

- capture `weight.data != 0` per pruned layer at compress time;
- `weight.register_hook(lambda g: g * mask)` for the duration;
- re-apply the mask once at the end (weight decay and momentum are already
  neutral at zero, but the re-apply is cheap and makes the invariant
  unconditional);
- re-measure the zero fraction and record `sparsity_drift` and
  `mask_preserved`.

**`mask_preserved=False` must raise, not warn.** A model that reports itself
sparse and is not is the same class of defect as writing a diverged run's
majority-class accuracy to JSON. Measured on 24/24 recovery runs in exp33 the
drift was exactly `0.0`, so the invariant is achievable and a violation means
something is wrong.

For a structurally sliced model the heads are gone, so there is no mask and the
flag is a no-op. The parameter is still true by default because the wrong value
is silent and the right value costs nothing.

### 3. Matched-control reporting

This is the part that must not be optional, because the whole audit exists
because of it. `CompressionReport` gains two separately named quantities:

```python
@dataclass
class RecoveryControl:
    """The dense model trained for the SAME number of epochs, nothing pruned."""
    accuracy: float
    epochs: int
    seeds: int = 1
```

and the report gains:

- `fine_tuning_gain_pp` = control − pretrained. What the epochs bought with
  nothing pruned.
- `pruning_cost_pp` = compressed+recovered − control. What the compression cost.
- `epochs_matched: bool` — true only when `control.epochs == self.epochs`.

**`pruning_cost_pp` is emitted only when `epochs_matched` is true.** Otherwise
the field is absent and the report says which control is needed. This is the
`exp31_dense_control.py` guard promoted into the product: a 5-epoch arm against
a 2-epoch control was wrong by 0.97–1.12pp on three headline cells, and the
mistake survived in committed data (see `results/lumi_v060/CORRECTIONS.json`).

When no control is supplied, the report prints the pretrained delta **labelled
as the sum of two effects** and says a control is required for a pruning cost.
It must never print a bare "+1.37pp" that a reader will take for a pruning
result — that is how a model pruning damaged gets reported as improved, which
`CONTROL_CONVENTION.md` exists to prevent.

### 4. `validate()` and the floor

- `validate()` scores the recovered model when `recover()` has run, and says in
  the stage name which it scored.
- If `compress()` is followed directly by `validate()`, the floor error gains a
  second suggestion: "or call `recover()` — two epochs is worth up to 68pp at
  50% head pruning". The current message offers only a smaller `prune_ratio`.
- A `strict_recovery=True` pipeline option makes `validate()` refuse to score an
  unrecovered compressed model at all. Not the default: a user who genuinely
  wants the one-shot number should be able to ask for it.

## Tests

CPU-only, tiny fixture, no network — cardinal rule 4.

1. `recover()` before `compress()` raises `PipelineError`.
2. `epochs` defaults to 2 and the value lands in the stage record.
3. Loss decreases over two epochs on the tiny fixture.
4. **Sparsity survives**: build a model with a known unstructured mask, recover,
   assert the zero set is unchanged and `sparsity_drift == 0.0`.
5. **Sparsity loss raises**: with the hook disabled, assert the run raises
   rather than reporting a dense model as sparse.
6. `pruning_cost_pp` is present with a matched control and **absent** with a
   mismatched one, and `epochs_matched` reflects it.
7. `fine_tuning_gain_pp` equals control − pretrained exactly.
8. With no control, the report contains no field a reader could mistake for a
   pruning cost, and does contain the "sum of two effects" wording.
9. The floor error mentions recovery when `recover()` was not called.
10. A sliced (structural) model recovers with `preserve_sparsity=True` and the
    flag is a no-op rather than an error.

## Open questions for review

1. **Does `recover()` belong in the open package or in pro?** The recovery
   *recipe* is not a pro feature and the gap is in both pipelines, so the
   argument is for `sal`. But it makes `sal` depend on a training loop over a
   user dataset, which only `sal_train` currently does. Recommendation: `sal`,
   reusing `sal_train`'s loop machinery.
2. **Should a control be mandatory?** Making `RecoveryControl` required would
   force every user into the correct comparison and would also force them to
   pay for a dense training run they may not want. Recommendation: optional,
   with the no-control report wording above, and `strict_recovery=True` for
   users who want it enforced.
3. **MLP recovery.** `sal_pro.mlp` slices neurons; the same stage covers it with
   no change, but exp26's published MLP figures are at **5** epochs against a
   5-epoch control. The product defaulting to 2 means the product's numbers will
   not match `EXP26_MLP.md`. Either re-measure MLP at 2 epochs (exp35 does
   exactly this) or document the difference. Recommendation: wait for exp35.
4. **Token-budget recovery for language models.** For exp36's OPT-1.3B arms,
   "epochs" is the wrong unit; a fixed token budget is. Whether the stage grows
   a `tokens=` alternative or stays epoch-only is a decision for after exp36.
