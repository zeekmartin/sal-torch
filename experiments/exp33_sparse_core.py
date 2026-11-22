#!/usr/bin/env python3
"""SparseGPT and Wanda pruning cores. **Now a re-export of `sal.sparsity`.**

The implementation moved into the shipped package in v0.7.1, and this module
re-exports it so that exactly ONE implementation exists. That matters more
than it sounds: `exp33` produced 48 unstructured runs and 12 at 2:4 over
three seeds, 60.05 GPU-hours, and `exp36` produced four decoder cells, all
against the code that used to live here. If promoting it into `sal/` had left
a copy behind, the figures in `EXP33_SPARSE_VS_STRUCTURED.md` would describe
a module that is no longer the one the package runs, and the layer-level
equivalence test in `tests/test_exp33_sparse_equivalence.py` would certify
the wrong one.

So: one implementation, in `sal/sparsity.py`, reviewable where it ships, and
every existing importer -- `exp33_sparse_vs_structured.py`,
`exp37_fi_compressed.py`, the two test modules here, and three modules in the
research repository -- keeps working unchanged.

The attribution, the licences and the reference-implementation correspondence
are in `sal/sparsity.py`'s own docstring and in `THIRD_PARTY_LICENSES`.
"""
from __future__ import annotations

from sal.sparsity import (SCOPES, SparseGPTPruner, SparsityError,  # noqa: F401
                          WandaPruner, attention_linear_layers,
                          linear_layers, prune_2of4, prune_model,
                          prune_unstructured, sparsity_report)

#: Re-exported too, because `exp36_opt_sparse` reads them to record what it
#: ran with rather than hardcoding them.
from sal.sparsity import BLOCKSIZE, PERCDAMP                       # noqa: F401

__all__ = ["SparseGPTPruner", "WandaPruner", "prune_model",
           "linear_layers", "attention_linear_layers", "sparsity_report",
           "prune_unstructured", "prune_2of4", "SparsityError", "SCOPES",
           "BLOCKSIZE", "PERCDAMP"]
