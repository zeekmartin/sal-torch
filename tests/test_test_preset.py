# -*- coding: utf-8 -*-
"""A test flag must never silently override an argument given explicitly.

This has cost the project twice. ``--smoke`` once also disabled INT4, so a
smoke run never exercised the cell the published table reads and the first
real batch failed there. ``--gate`` in exp34 reset ``--n-train``/
``--n-val`` to 512/256 *after* the caller asked for 64, so a micro-test ran
twelve times its intended size, overran its budget and was killed before
reaching the cell it existed to exercise.

Both times the operator believed their argument had taken effect. The rule
is therefore not "prefer the explicit value" -- that is just as silent --
but **refuse to start**, naming the flag and the two values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_sal_qat_prototype as P  # noqa: E402


class _Args:
    """A stand-in for an argparse namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_preset_applies_what_the_caller_did_not_give():
    a = _Args(n_train=1000, n_val=500)
    applied = P.apply_test_preset(a, {"n_train": 512, "n_val": 256},
                                  "--gate", argv=["--gate"])
    assert a.n_train == 512 and a.n_val == 256
    assert applied == {"n_train": 512, "n_val": 256}


def test_preset_refuses_when_the_caller_gave_the_same_argument():
    a = _Args(n_train=64, n_val=500)
    with pytest.raises(SystemExit) as e:
        P.apply_test_preset(a, {"n_train": 512, "n_val": 256}, "--gate",
                            argv=["--gate", "--n-train", "64"])
    msg = str(e.value)
    assert "--n-train" in msg, msg
    assert "512" in msg and "64" in msg, msg
    # And it must not have applied anything: a partial application is the
    # same silent redefinition in a smaller package.
    assert a.n_train == 64 and a.n_val == 500


def test_preset_refuses_the_equals_form_too():
    a = _Args(n_train=64)
    with pytest.raises(SystemExit):
        P.apply_test_preset(a, {"n_train": 512}, "--gate",
                            argv=["--gate", "--n-train=64"])


def test_preset_refuses_an_underscored_destination():
    """``--skip-int4`` maps to ``skip_int4``; the option spelling is what
    appears on the command line."""
    a = _Args(skip_int4=False)
    with pytest.raises(SystemExit):
        P.apply_test_preset(a, {"skip_int4": True}, "--smoke",
                            argv=["--smoke", "--skip-int4"])


def test_preset_names_every_conflicting_argument():
    a = _Args(n_train=64, n_val=32, epochs=1)
    with pytest.raises(SystemExit) as e:
        P.apply_test_preset(a, {"n_train": 512, "n_val": 256, "epochs": 3},
                            "--gate",
                            argv=["--gate", "--n-train", "64",
                                  "--n-val", "32"])
    msg = str(e.value)
    assert "--n-train" in msg and "--n-val" in msg, msg
    # epochs was NOT given, so it must not be reported as a conflict.
    assert "--epochs" not in msg, msg


def test_preset_is_quiet_about_arguments_it_does_not_set():
    a = _Args(n_train=1000, seed=7)
    P.apply_test_preset(a, {"n_train": 512}, "--gate",
                        argv=["--gate", "--seed", "7"])
    assert a.n_train == 512
    assert a.seed == 7


def test_the_exp34_case_that_motivated_this():
    """The exact invocation that silently ran twelve times its size."""
    a = _Args(n_train=64, n_val=64, epochs=1, recover_epochs=1)
    with pytest.raises(SystemExit) as e:
        P.apply_test_preset(
            a, {"n_train": 512, "n_val": 256, "epochs": 1,
                "recover_epochs": 1}, "--gate",
            argv=["--gate", "--recover-drophead", "--n-train", "64",
                  "--n-val", "64", "--epochs", "1", "--recover-epochs", "1"])
    assert "--n-train" in str(e.value)
