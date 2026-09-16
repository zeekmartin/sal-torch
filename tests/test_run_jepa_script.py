"""End-to-end test of scripts/run_jepa_sal.py (v0.5.1).

The real script costs GPU-hours and downloads 2.5GB before it reaches its first
bug. This runs `main()` in full on CPU against a 2-layer random-weight I-JEPA
and synthetic images, with only two things faked: the model loader and the data
builder. Everything else — the scans, SALTrainer with the custom step,
per-epoch checkpointing, resume, pruning, scoring, figures, save_pretrained, the
JSON and the summary table — is the code that will run on the GPU.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_jepa_sal.py"

IMG, PATCH, NL, NH, HS = 32, 16, 2, 4, 64


@pytest.fixture(scope="module")
def mod():
    """Import the script as a module (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("run_jepa_sal", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["run_jepa_sal"] = m
    spec.loader.exec_module(m)
    return m


def _tiny_model():
    from transformers import IJepaConfig, IJepaModel
    torch.manual_seed(0)
    return IJepaModel(IJepaConfig(hidden_size=HS, num_hidden_layers=NL,
                                  num_attention_heads=NH, intermediate_size=HS * 2,
                                  image_size=IMG, patch_size=PATCH))


def _fake_data(n_classes=3):
    """Labelled synthetic images with real class structure, so probes can score."""
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(0)
    proto = torch.randn(n_classes, 3, IMG, IMG, generator=g)

    def split(n, seed, bs=4):
        gg = torch.Generator().manual_seed(seed)
        y = torch.arange(n) % n_classes
        x = proto[y] + 0.4 * torch.randn(n, 3, IMG, IMG, generator=gg)
        return DataLoader(TensorDataset(x, y), batch_size=bs)

    val_x = torch.cat([b[0] for b in split(12, 2)])
    loaders = {
        "train": split(12, 1),
        "probe_train": split(12, 3),
        "probe_val": split(12, 4),
        "cka": DataLoader(TensorDataset(val_x[:8]), batch_size=4),
    }
    fi_batches = [{"pixel_values": val_x[i:i + 4]} for i in range(0, 8, 4)]
    return loaders, fi_batches, n_classes


@pytest.fixture
def patched(mod, monkeypatch):
    monkeypatch.setattr(mod, "load_model", lambda device, grad_ckpt: _tiny_model())
    monkeypatch.setattr(mod, "build_data",
                        lambda sizes, batch_size, seed: _fake_data())
    monkeypatch.setattr(mod, "LATENCY_RUNS", 2)
    return mod


# ------------------------------------------------------------------ helpers
def test_mask_patches_zeroes_whole_blocks(mod):
    torch.manual_seed(0)
    x = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
    out = mod.mask_patches(x, 2, 0.5)
    for i in (0, 2):
        for j in (0, 2):
            block = out[0, 0, i:i + 2, j:j + 2]
            assert bool((block == 0).all()) or bool((block != 0).all()), \
                "a patch was partially masked"


def test_mask_patches_masks_each_image_independently(mod):
    torch.manual_seed(0)
    x = torch.ones(32, 1, 8, 8)
    out = mod.mask_patches(x, 2, 0.5)
    per_image = [(out[i] == 0).float().mean().item() for i in range(32)]
    assert len(set(per_image)) > 1, "every image got the same mask pattern"


def test_mask_patches_ratio_is_approximately_right(mod):
    torch.manual_seed(0)
    out = mod.mask_patches(torch.ones(64, 1, 8, 8), 2, 0.4)
    assert 0.3 < (out == 0).float().mean().item() < 0.5


@pytest.mark.parametrize("vram,expected", [(80.0, 32), (40.0, 16), (24.0, 8), (8.0, 4)])
def test_batch_size_ladder(mod, vram, expected):
    assert next(b for floor, b in mod._VRAM_BATCH if vram >= floor) == expected


def test_remap_window_shifts_into_remaining_run(mod):
    from sal import SALConfig
    cfg = SALConfig(num_layers=2, num_heads_per_layer=4,
                    prune_start_ratio=0.10, prune_end_ratio=0.80)
    mod.remap_window(cfg, epochs_done=2, total_epochs=5)
    # 40% done: the window's remaining span maps into the last 60%.
    assert cfg.prune_start_ratio == pytest.approx(0.0)
    assert cfg.prune_end_ratio == pytest.approx((0.80 - 0.4) / 0.6)
    assert cfg.prune_start_ratio < cfg.prune_end_ratio


def test_remap_window_stays_valid_at_the_last_epoch(mod):
    """SALConfig rejects start >= end, so the remap must never produce one."""
    from sal import SALConfig
    cfg = SALConfig(num_layers=2, num_heads_per_layer=4,
                    prune_start_ratio=0.10, prune_end_ratio=0.80)
    mod.remap_window(cfg, epochs_done=4, total_epochs=5)
    assert 0.0 <= cfg.prune_start_ratio < cfg.prune_end_ratio <= 1.0


def test_summary_table_renders_and_handles_missing(mod):
    results = {
        "original": {"linear_probe": 0.762, "knn_accuracy": 0.731,
                     "cka_similarity": 1.0, "latency_gpu_ms": 42.0,
                     "latency_cpu_ms": 310.0},
        "sal+random-33%": {"linear_probe": 0.748, "knn_accuracy": 0.715,
                           "cka_similarity": 0.967, "latency_gpu_ms": None,
                           "latency_cpu_ms": 240.0},
    }
    out = mod.summary_table(results, ["original", "sal+random-33%", "absent"])
    assert "Variant" in out and "76.2%" in out and "0.967" in out
    assert "n/a" in out                      # the missing GPU latency
    assert "absent" not in out
    # top, header, mid, 2 rows, bottom = 6 lines = 5 newlines
    assert out.count("\n") == 5
    assert len({len(line) for line in out.splitlines()}) == 1, "ragged borders"


# ------------------------------------------------------------- the whole run
@pytest.mark.parametrize("control", [False, True], ids=["no-control", "control"])
def test_smoke_run_end_to_end(patched, tmp_path, control):
    mod = patched
    out = tmp_path / "results"
    argv = ["--smoke", "--output", str(out), "--seed", "1"]
    if control:
        argv.append("--control")

    payload = mod.main(argv)

    # every artefact the brief promises
    assert (out / "jepa_sal_benchmark.json").is_file()
    assert (out / "pre_sal_scan.json").is_file()
    assert (out / "post_sal_scan.json").is_file()
    assert (out / "checkpoint.pt").is_file()
    assert (out / "model" / "config.json").is_file()
    assert (out / "model" / "model.safetensors").is_file()

    saved = json.loads((out / "jepa_sal_benchmark.json").read_text())
    assert saved["smoke"] is True
    assert saved["results"].keys() == payload["results"].keys()

    expected = {"original", "sal-trained",
                "sal+random-33%", "sal+magnitude-33%",
                "sal+random-50%", "sal+magnitude-50%"}
    if control:
        expected |= {"control-trained", "ctrl+random-33%", "ctrl+magnitude-33%",
                     "ctrl+random-50%", "ctrl+magnitude-50%"}
    assert set(payload["results"]) == expected

    for name, row in payload["results"].items():
        assert 0.0 <= row["linear_probe"] <= 1.0, name
        assert 0.0 <= row["knn_accuracy"] <= 1.0, name
        assert 0.0 <= row["cka_similarity"] <= 1.0, name
        assert row["params"] > 0
        assert row["latency_cpu_ms"] > 0

    assert payload["results"]["original"]["cka_similarity"] == 1.0
    assert payload["comparison"].startswith(
        "SAL-training" if control else "head SELECTION")


def test_smoke_run_prunes_but_does_not_finish_the_ramp(patched, tmp_path):
    """A 3-step smoke run cannot reach the full prune fraction, by design.

    The ramp hits its target only once progress passes prune_end_ratio (0.80).
    With three optimizer steps the last one sits at progress 0.67, so the smoke
    run stops short. That is the schedule working, not a bug — the real run has
    thousands of steps and gets there. Pinned so nobody reads a short run's
    masker stats as a failure.
    """
    total = NL * NH
    stats = patched.main(["--smoke", "--output", str(tmp_path / "r"),
                          "--mask-ratio", "0.5", "--seed", "1"])["masker_stats"]
    assert stats["total_heads"] == total
    assert 0 < stats["pruned_heads"] <= int(total * 0.5)


def test_longer_run_reaches_the_full_prune_fraction(patched, tmp_path):
    """Given enough steps, the ramp does arrive at the configured target."""
    total = NL * NH
    stats = patched.main(["--smoke", "--epochs", "12", "--output", str(tmp_path / "r2"),
                          "--mask-ratio", "0.5", "--seed", "1"])["masker_stats"]
    assert stats["pruned_heads"] == int(total * 0.5)


def test_figures_are_written(patched, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "results"
    payload = patched.main(["--smoke", "--output", str(out), "--seed", "1"])
    figs = out / "figures"
    for name in ("feature_maps_comparison.png", "compression_table.png",
                 "fi_before_after.png", "latency_comparison.png"):
        assert (figs / name).is_file(), f"{name} missing"
        assert (figs / name).stat().st_size > 0
        assert name in payload["figures"]


def test_checkpoint_roundtrip_and_resume(patched, tmp_path):
    """A resumed run continues rather than restarting, and keeps its pruned set."""
    mod = patched
    out = tmp_path / "results"
    mod.main(["--smoke", "--output", str(out), "--seed", "1"])

    ckpt = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert ckpt["model_id"] == mod.MODEL_ID
    assert len(ckpt["masks"]) == NL
    pruned_before = sum(int((m == 0).sum()) for m in ckpt["masks"].values())
    assert pruned_before > 0

    # Resume for a longer run: epoch 1 is done, so 2 more should execute.
    out2 = tmp_path / "resumed"
    payload = mod.main(["--smoke", "--epochs", "3", "--output", str(out2),
                        "--resume", str(out / "checkpoint.pt"), "--seed", "1"])
    assert len(payload["sal_losses"]) == 3          # 1 restored + 2 run
    ck2 = torch.load(out2 / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck2["epoch"] == 3
    after = sum(int((m == 0).sum()) for m in ck2["masks"].values())
    assert after >= pruned_before, "resume lost the accumulated pruned set"


def test_resume_of_a_finished_run_skips_training(patched, tmp_path, capsys):
    mod = patched
    out = tmp_path / "results"
    mod.main(["--smoke", "--output", str(out), "--seed", "1"])
    capsys.readouterr()

    mod.main(["--smoke", "--epochs", "1", "--output", str(tmp_path / "again"),
              "--resume", str(out / "checkpoint.pt"), "--seed", "1"])
    assert "already complete" in capsys.readouterr().out


def test_summary_table_falls_back_to_ascii(mod, monkeypatch):
    """A non-UTF-8 console must not lose the summary after hours of compute."""
    class Cp1252Stdout:
        encoding = "cp1252"

    monkeypatch.setattr(mod.sys, "stdout", Cp1252Stdout())
    results = {"original": {"linear_probe": 0.5, "knn_accuracy": 0.5,
                            "cka_similarity": 1.0, "latency_gpu_ms": 1.0,
                            "latency_cpu_ms": 2.0}}
    out = mod.summary_table(results, ["original"])
    out.encode("cp1252")                       # the whole point: this must not raise
    assert "+" in out and "|" in out
    assert "┌" not in out
