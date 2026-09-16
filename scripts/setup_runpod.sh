#!/usr/bin/env bash
# Setup for RunPod, Lambda, vast.ai, or any fresh Linux + GPU machine.
#
#   bash scripts/setup_runpod.sh
#   python scripts/run_jepa_sal.py --smoke
#
# Every step is checked, and the script stops at the first failure rather than
# leaving you to discover it four hours into a paid run.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/zeekmartin/sal-torch.git}"

echo "=== Setting up SAL-JEPA benchmark ==="

# --- repo -------------------------------------------------------------------
# Works both from inside an existing clone and on a bare machine.
if [ -f "pyproject.toml" ] && [ -d "sal" ]; then
    echo "--- already inside a sal-torch checkout: $(pwd)"
elif [ -d "sal-torch" ]; then
    echo "--- reusing existing clone"
    cd sal-torch
else
    echo "--- cloning $REPO_URL"
    # Private repo? Use a token URL or scp the directory across instead:
    #   REPO_URL=https://<token>@github.com/zeekmartin/sal-torch.git bash scripts/setup_runpod.sh
    git clone "$REPO_URL"
    cd sal-torch
fi

# --- install ----------------------------------------------------------------
# torch is deliberately NOT installed here: a RunPod/Lambda image ships a build
# matched to its CUDA driver, and `pip install torch` can replace it with one
# that does not work on the machine.
echo "--- installing sal-torch and benchmark dependencies"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e ".[dev,reports]"
python -m pip install --quiet transformers datasets torchvision safetensors accelerate

# --- verify -----------------------------------------------------------------
echo "--- verifying GPU"
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("NO CUDA DEVICE. The benchmark needs a GPU; run_jepa_sal.py will fall")
    print("back to CPU and take days on ViT-H/14.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
vram = p.total_memory / 1e9          # note: total_memory, not total_mem
print(f"GPU: {p.name}, VRAM: {vram:.1f}GB, torch {torch.__version__}, CUDA {torch.version.cuda}")
if vram < 20:
    print(f"WARNING: {vram:.1f}GB is tight for ViT-H/14. Expect to need")
    print("         --batch-size 1, and possibly a bigger card.")
PY

echo "--- verifying sal-torch"
python -c "
import sal
from sal import SALTrainer, FIScanner, PlasticityScanner, linear_probe, cka_similarity
print(f'sal-torch {sal.__version__} OK')
"

echo "--- verifying I-JEPA downloads and is introspectable"
# Config first: a few KB, and it fails fast on a network or auth problem before
# committing to the 2.5GB checkpoint.
python - <<'PY'
from transformers import AutoConfig
from sal import arch_support
cfg = AutoConfig.from_pretrained("facebook/ijepa_vith14_1k")
print(f"config: {cfg.model_type}, {cfg.num_hidden_layers}x{cfg.num_attention_heads} heads, "
      f"patch {cfg.patch_size}, image {cfg.image_size}")
assert cfg.model_type in arch_support.supported_architectures(), \
    f"{cfg.model_type} not in SAL's registry"
print("architecture: registered with sal-torch")
PY

python - <<'PY'
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/ijepa_vith14_1k", attn_implementation="eager")
n = sum(p.numel() for p in m.parameters())
print(f"I-JEPA: {n/1e6:.0f}M params")
from sal import arch_support
projs = arch_support.get_output_projections(m)
print(f"SAL hook points: {len(projs)} attention output projections")
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
echo "=== Ready. Run: python scripts/run_jepa_sal.py --smoke ==="
echo "    Then:      python scripts/run_jepa_sal.py --epochs 5 --output results/"
echo "    Retrieve:  scp -r <host>:$(pwd)/results ."
