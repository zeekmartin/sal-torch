#!/usr/bin/env bash
# Setup for the SAL+QAT prototype on RunPod, Lambda, vast.ai, or any Linux + GPU box.
#
#   bash scripts/setup_qat_prototype.sh
#   python scripts/run_sal_qat_prototype.py --smoke
#
# Stops at the first failure rather than letting you discover it mid-run.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/zeekmartin/sal-torch.git}"

echo "=== Setting up SAL+QAT prototype ==="

# --- repo -------------------------------------------------------------------
if [ -f "pyproject.toml" ] && [ -d "sal" ]; then
    echo "--- already inside a sal-torch checkout: $(pwd)"
elif [ -d "sal-torch" ]; then
    echo "--- reusing existing clone"
    cd sal-torch
else
    echo "--- cloning $REPO_URL"
    git clone "$REPO_URL"
    cd sal-torch
fi

# --- install ----------------------------------------------------------------
# torch is NOT installed here: the pod image ships a build matched to its CUDA
# driver, and `pip install torch` can replace it with one that does not work.
# timm is not needed — the script uses transformers' ViT, which sal-torch supports.
echo "--- installing sal-torch and prototype dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[dev,reports]"
python -m pip install --quiet transformers datasets torchvision safetensors accelerate
# Optional: real NF4 cross-check of the INT4 cells. Failure is not fatal.
python -m pip install --quiet "bitsandbytes>=0.43" \
    || echo "    bitsandbytes unavailable — INT4 cells will be simulated only"

# --- verify -----------------------------------------------------------------
echo "--- verifying GPU"
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("NO CUDA DEVICE. Only --smoke is practical on CPU.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, VRAM: {p.total_memory / 1e9:.1f}GB, "
      f"torch {torch.__version__}, CUDA {torch.version.cuda}")
PY

echo "--- verifying sal-torch"
python -c "import sal; from sal import SALTrainer, FIScanner, slice_heads; print(f'sal-torch {sal.__version__} OK')"

echo "--- verifying QAT"
python - <<'PY'
import torch.ao.quantization as tq
from torch.ao.nn.qat import Linear  # noqa: F401
print("torch.ao QAT OK")
PY

echo "--- verifying ViT-B/16 loads and is introspectable"
python - <<'PY'
from transformers import ViTForImageClassification
from sal import arch_support
m = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=100, ignore_mismatched_sizes=True)
projs = arch_support.get_output_projections(m)
print(f"ViT-B/16: {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M params, "
      f"{len(projs)} SAL hook points")
assert len(projs) == m.config.num_hidden_layers
PY

echo "--- verifying ImageNet-100 is reachable"
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("clane9/imagenet-100", streaming=True)
row = next(iter(ds["train"]))
print(f"dataset: splits={list(ds)}, columns={list(row)}")
PY

echo
echo "=== Ready. Run: python scripts/run_sal_qat_prototype.py --smoke ==="
echo "    Then:      python scripts/run_sal_qat_prototype.py --epochs 5 --output results/"
echo "    Containers: export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 first (CPU latency)"
echo "    Disk:       export HF_HOME=/workspace/hf and use --output /workspace/results"
