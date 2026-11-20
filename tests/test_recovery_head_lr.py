"""The classifier's learning rate must be the archived recipe's, not a story.

Every published vision recovery figure was measured by
`run_sal_qat_prototype.make_optimizer`, which runs `LR_HEAD` 1e-3 against
`LR_BACKBONE` 5e-5: a factor of **20**. `recover()` shipped a default of 10
for one cycle, on the strength of a write-up sentence that said "10x on the
classifier". The release gate caught it, as a 0.15-0.26pp deficit against
exp40 on 3/3 seeds alongside a missing augmentation.

So the ratio is not asserted against a literal here. It is RECOMPUTED from the
harness constants the archive actually used, which is the only form of this
test that fails when either side drifts.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "scripts"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from sal.recovery import (ARCHIVED_HEAD_LR_MULTIPLIER,  # noqa: E402
                          _param_groups, recover_model)


def _archived_ratio() -> float:
    import run_sal_qat_prototype as P
    return P.LR_HEAD / P.LR_BACKBONE


class _Tiny(nn.Module):
    """A body and a `classifier`, which is what HEAD_PREFIXES matches."""

    def __init__(self, dim=8, classes=3):
        super().__init__()
        self.body = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, pixel_values=None, labels=None, **_):
        logits = self.classifier(self.body(pixel_values))
        loss = (nn.functional.cross_entropy(logits, labels)
                if labels is not None else None)
        return type("Out", (), {"logits": logits, "loss": loss})()


def test_the_default_is_the_archived_ratio_recomputed_from_the_harness():
    assert ARCHIVED_HEAD_LR_MULTIPLIER == _archived_ratio(), (
        "the shipped default and the recipe every figure was measured under "
        "have drifted apart")


def test_the_signature_default_is_that_constant_and_not_a_copy_of_it():
    import inspect
    sig = inspect.signature(recover_model)
    assert (sig.parameters["head_lr_multiplier"].default
            is ARCHIVED_HEAD_LR_MULTIPLIER)


def test_the_classifier_group_really_gets_that_rate():
    groups = _param_groups(_Tiny(), 5e-5, ARCHIVED_HEAD_LR_MULTIPLIER)
    assert len(groups) == 2
    body, head = groups
    assert body["lr"] == 5e-5
    assert head["lr"] == 5e-5 * _archived_ratio()
    # And the split is the one the archive made: `classifier` apart, rest body.
    assert len(head["params"]) == 2 and len(body["params"]) == 2


def test_a_recovery_records_the_multiplier_it_ran():
    model = _Tiny()
    data = [{"pixel_values": torch.randn(4, 8),
             "labels": torch.randint(0, 3, (4,))} for _ in range(3)]
    rec = recover_model(model, data, domain="unit test", epochs=1,
                        batch_size=4, preserve_sparsity=False)
    assert rec["head_lr_multiplier"] == ARCHIVED_HEAD_LR_MULTIPLIER
    assert rec["steps"] == 3


def test_a_model_with_no_head_gets_one_group():
    """Tied embeddings: OPT's lm_head is its embedding, so there is no head."""
    body = nn.Sequential(nn.Linear(4, 4))
    groups = _param_groups(body, 1e-5, ARCHIVED_HEAD_LR_MULTIPLIER)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-5
