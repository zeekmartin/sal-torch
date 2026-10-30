# v0.6.0 Phase 1 — ROCm / MI250X validation on LUMI

Goal: confirm every sal-torch code path that the v0.6.0 experiments depend on
produces the same numbers on AMD Instinct MI250X (ROCm 6.2) that it produced on
NVIDIA, **before** spending GPU-hours on new experiments.

Verdict: **all four gates pass.** Total cost 0.25 GPU-hours.

Machine: LUMI-G, `small-g`, 1 GCD per job (MI250X, 68.7 GB HBM2e visible),
PyTorch 2.5.1+rocm6.2, sal-torch 0.5.1.

| # | Gate | Job | Elapsed | Result |
|---|------|-----|---------|--------|
| exp01 | FI + plasticity scan, I-JEPA ViT-H/14 | 23315777 | 3:34 | **MATCH** vs published v0.5.1 |
| exp01 | FI + plasticity scan, DINOv2-base | 23315633 | 3:14 | new ROCm baseline |
| exp02 | SAL training + control + slicing, DINOv2-base | 23315870 | 3:15 | path OK (smoke) |
| exp03 | 4-arm SAL/QAT training, ViT-B/16 | 23315871 | 2:52 | QAT backend `torch.ao` (native) |
| exp03b | INT4 + P50+I4 compression path | 23315977 | 4:19 | all 6 configs compute |

## exp01 — the reproduction that matters

Probe held identical to the run behind the published figures
(`scripts/jepa_scan_results.json`): 256 CIFAR-10 test images, batch 8, encoder
in eager attention. FI is a function of activation correlation structure, so a
random-noise probe would give a different and unusable number.

| Quantity | v0.5.1 (NVIDIA) | LUMI MI250X | Δ |
|---|---|---|---|
| FI | 0.0040 | 0.003975 | 0.000025 |
| layer split | 1 immune / 4 buffer / 27 critical | 1 / 4 / 27 | — |
| plasticity | 8 elastic / 23 saturated / 1 hub | 8 / 23 / 1 | — |
| mean routing | 0.8656 | 0.866 | ~0 |
| mean MI | 0.0609 | 0.061 | ~0 |
| params | 630.8M | 630.8M | — |

ROCm reproduces the structural diagnostics to within float noise.

DINOv2-base (new baseline, no published figure): FI = 0.0650, 1 immune /
3 buffer / 8 critical, 0 elastic / 6 saturated / 6 hub, mean routing 0.665.

Peak GPU memory: **7.0 GB** for ViT-H/14 at batch 8 — a 64 GB GCD has very
large headroom for Phase 2.

## exp02 / exp03 / exp03b — path validation

Smoke-scale accuracies are meaningless by construction (100–200 images,
1 epoch) and are not quoted. What these gates establish:

- `SALTrainer`, `HeadMasker`, `slice_heads()`, `linear_probe`, `knn_accuracy`,
  `cka_similarity`, `measure_latency` and `FIScanner` all run on ROCm.
- Random vs magnitude head selection at 33% and 50% both execute.
- **QAT uses the native `torch.ao` backend on ROCm** (`qat_backend: "torch.ao"`),
  not the straight-through fallback. This was the main technical risk for
  Phase 2 and it is cleared.
- INT4 and the key **P50+I4** cell compute via the simulated per-channel
  round-trip. bitsandbytes is not installed, so there is no NF4 cross-check on
  LUMI; `int4_nf4` is absent from these JSONs.
- `--smoke` sets `skip_int4 = True`, so exp03b (a small *non-smoke* run) was
  needed to exercise INT4 at all.

## Environment repairs required

The environment was not in fact ready; three things had to be fixed.

1. **`datasets` was unusable.** pyarrow 26.0.0 requires numpy >= 2.0 but numpy
   was 1.24.4, so `import datasets` raised. scipy 1.10.1 and pandas 1.5.3 are
   numpy-1.x builds, so upgrading numpy was not an option. Pinned
   `pyarrow==16.1.0` + `datasets==3.1.0` instead.
2. **I-JEPA cannot load on transformers 4.46.0** — `IJepaModel` landed in 4.47.
   Rather than disturb the tested environment, a second env
   `.venv-tf47` carries transformers 4.47.1 and reuses the existing
   ROCm torch via a `.pth` pointing at `.venv`'s and Cray's site-packages.
   Select it with `SAL_VENV=.venv-tf47`. Only exp01's I-JEPA run used it.
3. **matplotlib was missing**, so report figures were skipped. Installed
   `matplotlib==3.7.5`, which moved numpy 1.24.4 → 1.26.4 (still numpy 1.x;
   scipy and pyarrow re-verified after).

Both environments were verified to import numpy/scipy/pandas/datasets/pyarrow/
torch/transformers/sal/timm/torchvision at the intended versions.

The `sal` installed on LUMI is byte-identical to this repo's `sal/` once CRLF
is normalised — the initial md5 mismatch was purely Windows line endings.

## Operational notes for LUMI

- Compute nodes have **no internet**. Everything is pre-fetched from a login
  node into `HF_HOME=/scratch/project_465003656/sal-torch/hf_cache` (19 GB:
  I-JEPA ViT-H/14, DINOv2-base, ViT-B/16, CIFAR-10, ImageNet-100 at
  126,689 train / 5,000 val). Jobs run with `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`.
- `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are pinned to `SLURM_CPUS_PER_TASK` (7).
  Unpinned, torch sees all 128 host cores and oversubscribes.
- The **first** training step costs ~60 s of one-time MIOpen kernel
  compilation; it is not per-step cost and it caches across jobs.

## Budget correction

The plan estimated exp04 at ~50 GPU-hours for 3 seeds. The v0.5.0 full-scale
run (ViT-B/16, 20k train / 5k val, 5 epochs, 4 arms, 3 prune seeds, all six
compression configs) took **723 s = 0.20 GPU-hours per seed** on NVIDIA.
Phase 2 is therefore roughly two orders of magnitude cheaper than planned, so
exp04 runs **five** seeds (42/123/456 reproduce the published runs; 137/2024
extend them) for an expected ~2–3 GPU-hours rather than a reduced matrix.
