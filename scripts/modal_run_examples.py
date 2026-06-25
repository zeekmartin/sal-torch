"""Run the examples/ scripts on a Modal T4 GPU to verify they work end-to-end.

    modal run scripts/modal_run_examples.py

(On Windows, prefix with PYTHONUTF8=1 so the Modal CLI can print its glyphs.)
"""
import sys
import modal

app = modal.App("sal-torch-examples")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "scipy", "numpy", "accelerate>=1.1.0")
    .add_local_dir("sal", "/root/sal-torch/sal", copy=True)
    .add_local_dir("examples", "/root/sal-torch/examples", copy=True)
    .add_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/sal-torch/README.md", copy=True)
    .run_commands("cd /root/sal-torch && pip install -e .")
)

EXAMPLES = ["quickstart", "standalone_fi", "full_control", "compare_with_without_sal"]


@app.function(image=image, gpu="T4", timeout=1200)
def run_examples():
    import importlib
    import traceback

    sys.path.insert(0, "/root/sal-torch/examples")
    import torch
    print(f"=== running examples on {'cuda' if torch.cuda.is_available() else 'cpu'} ===", flush=True)

    results = {}
    for name in EXAMPLES:
        print(f"\n========== {name} ==========", flush=True)
        try:
            importlib.import_module(name).main()
            results[name] = "PASS"
        except Exception as e:  # noqa: BLE001 — report, keep going
            traceback.print_exc()
            results[name] = f"FAIL: {type(e).__name__}: {e}"
        print(f"---------- {name}: {results[name]} ----------", flush=True)

    print("\n========== SUMMARY ==========", flush=True)
    for name, status in results.items():
        print(f"{name:<28} {status}", flush=True)
    assert all(v == "PASS" for v in results.values()), "some examples failed"
    return results


@app.local_entrypoint()
def main():
    print(run_examples.remote())
