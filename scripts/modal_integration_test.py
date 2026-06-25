"""Run sal-torch integration tests on a remote GPU via Modal.

These call the exact same check logic as the local pytest integration tests
(tests/test_arch_integration.py and tests/test_training_integration.py); the
Modal functions just execute them remotely on a T4 and print results to stdout.

Usage:
    modal run scripts/modal_integration_test.py                 # run both
    modal run scripts/modal_integration_test.py::test_architectures
    modal run scripts/modal_integration_test.py::test_sal_training

T4 is intentional — these are plumbing checks, not benchmarks.
"""
import sys
import modal

app = modal.App("sal-torch-integration")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "datasets", "scipy", "numpy")
    # copy_local_* runs at build time so `pip install -e .` below can see the package
    .copy_local_dir("sal", "/root/sal-torch/sal")
    .copy_local_dir("tests", "/root/sal-torch/tests")
    .copy_local_file("pyproject.toml", "/root/sal-torch/pyproject.toml")
    .copy_local_file("README.md", "/root/sal-torch/README.md")
    .run_commands("cd /root/sal-torch && pip install -e .")
)


def _import_tests():
    # tests/ is not an installed package; add it to the path and import directly.
    sys.path.insert(0, "/root/sal-torch/tests")
    import test_arch_integration as arch
    import test_training_integration as training
    return arch, training


@app.function(image=image, gpu="T4", timeout=600)
def test_architectures():
    """SALConfig.auto + HeadMasker on real models (DistilBERT, GPT-2, ViT, BERT)."""
    import torch
    arch, _ = _import_tests()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== architecture integration on {device} ===", flush=True)
    results = arch.run_all(verbose=True, device=device)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n=== architectures: {passed}/{len(results)} passed ===", flush=True)
    for r in results:
        print(r, flush=True)
    assert passed == len(results), "some architecture checks failed"
    return results


@app.function(image=image, gpu="T4", timeout=900)
def test_sal_training():
    """End-to-end SAL training pipeline on DistilBERT + SST-2 (plumbing only)."""
    import torch
    _, training = _import_tests()
    print(f"=== SAL training integration (cuda={torch.cuda.is_available()}) ===", flush=True)
    result = training.run(verbose=True)
    print(f"\n=== training result ===\n{result}", flush=True)
    assert result.get("passed"), f"training pipeline failed: {result.get('error')}"
    return result


@app.local_entrypoint()
def main():
    arch_results = test_architectures.remote()
    training_result = test_sal_training.remote()
    print("\n========== SUMMARY ==========")
    for r in arch_results:
        print("arch    ", r)
    print("training", training_result)
