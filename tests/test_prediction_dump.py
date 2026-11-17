# -*- coding: utf-8 -*-
"""Every evaluation can dump its per-image predictions, from one env var.

The collector inside ``accuracy()`` existed for a year and exactly one
gate ever set it, so the example-level question -- do two arms that end
at the same accuracy get the same images right? -- stayed unanswerable
while the data was one assignment away. ``SAL_DUMP_PREDICTIONS`` turns
it on for every call in a process, so an experiment does not have to opt
in and a later wave cannot forget to.

These tests pin the three things a join depends on: that a dump happens
at all, that each evaluation gets its own file rather than overwriting
the last, and that the dumped vectors reproduce the returned accuracy.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class TinyClassifier(nn.Module):
    """Something with ``.logits`` and a deterministic answer."""

    def __init__(self, n_classes=4, pixels=3 * 4 * 4):
        super().__init__()
        self.fc = nn.Linear(pixels, n_classes)
        self.n_classes = n_classes

    def forward(self, pixel_values=None, **_kw):
        x = pixel_values.flatten(1)
        out = self.fc(x)

        class Out:
            pass
        o = Out()
        o.logits = out
        return o


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    """The module, with the env var set before it is imported."""
    monkeypatch.setenv("SAL_DUMP_PREDICTIONS", str(tmp_path / "dumps"))
    import run_sal_qat_prototype as P
    P = importlib.reload(P)
    # normalize() and batches() are the real ones; the split is tiny.
    monkeypatch.setattr(P, "normalize", lambda x: x)
    monkeypatch.setattr(P, "LOADER_WORKERS", 0)
    return P


def _split(n=16, classes=4):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 4, 4)
    y = torch.randint(0, classes, (n,))
    return (x, y)


def test_a_dump_appears_without_the_experiment_opting_in(harness, tmp_path):
    P = harness
    model = TinyClassifier()
    acc = P.accuracy(model, _split(), torch.device("cpu"), False,
                     batch_size=8, label="arm-a")
    files = sorted((tmp_path / "dumps").glob("*.npz"))
    assert len(files) == 1, [f.name for f in files]
    assert "arm-a" in files[0].name
    blob = np.load(files[0])
    assert float((blob["pred"] == blob["label"]).mean()) == pytest.approx(
        acc, abs=1e-12)


def test_each_evaluation_gets_its_own_file(harness, tmp_path):
    """A run scores many models; one filename would keep only the last."""
    P = harness
    split = _split()
    for name in ("one_shot", "recovered", "control"):
        P.accuracy(TinyClassifier(), split, torch.device("cpu"), False,
                   batch_size=8, label=name)
    files = sorted((tmp_path / "dumps").glob("*.npz"))
    assert len(files) == 3, [f.name for f in files]
    assert [f.name[:3] for f in files] == ["001", "002", "003"]
    assert {n for n in ("one_shot", "recovered", "control")
            if any(n in f.name for f in files)} == {
        "one_shot", "recovered", "control"}


def test_the_labels_are_the_evaluation_order(harness, tmp_path):
    """The join pairs images by position, so the order must be stable."""
    P = harness
    split = _split()
    P.accuracy(TinyClassifier(), split, torch.device("cpu"), False,
               batch_size=8, label="a")
    P.accuracy(TinyClassifier(), split, torch.device("cpu"), False,
               batch_size=4, label="b")
    a, b = sorted((tmp_path / "dumps").glob("*.npz"))
    la, lb = np.load(a)["label"], np.load(b)["label"]
    assert np.array_equal(la, lb), (
        "two evaluations of the same split in a different batch size "
        "produced different label orders; a positional join would be "
        "comparing different images")
    assert np.array_equal(la, split[1].numpy().astype(la.dtype))


def test_an_explicit_path_still_wins(harness, tmp_path):
    """A driver that wants one named file is not overridden by the var."""
    P = harness
    target = tmp_path / "named" / "predictions.npz"
    P.COLLECT_PREDICTIONS = str(target)
    try:
        P.accuracy(TinyClassifier(), _split(), torch.device("cpu"), False,
                   batch_size=8, label="ignored")
    finally:
        P.COLLECT_PREDICTIONS = None
    assert target.is_file()
    assert not list((tmp_path / "dumps").glob("*.npz"))


def test_nothing_is_written_when_the_var_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("SAL_DUMP_PREDICTIONS", raising=False)
    import run_sal_qat_prototype as P
    P = importlib.reload(P)
    monkeypatch.setattr(P, "normalize", lambda x: x)
    monkeypatch.setattr(P, "LOADER_WORKERS", 0)
    assert P.DUMP_PREDICTIONS_DIR is None
    P.accuracy(TinyClassifier(), _split(), torch.device("cpu"), False,
               batch_size=8, label="x")
    assert not list(tmp_path.rglob("*.npz"))
