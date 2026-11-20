"""What the router tells a user, and what it is no longer allowed to tell them.

`ProbeSummary.recommendation` is a published claim: a user reads it and ships.
So each of the three routes is generated here and read, and the figures in it
are required to have a source. Two statements this file forbids were in the
text until v0.7.0 and were false, not merely stale:

* that SAL leads DropHead on the head set it trained for. exp28b measured that
  cell on a 631M encoder and DropHead leads it by +0.67pp on 3/3 seeds.
* that DropHead is not callable from this package. It is, as
  `mask_mode="drophead"`.
"""
import re

import pytest
import torch

from sal.pipeline import (ARCHIVE_DOI, NOT_MEASUREMENTS, SOURCES,
                          CompressionPipeline, ProbeSummary)

ROUTES = ("sal_aligned", "drophead", "skip_heads")


def _summary(route, drop_pp=50.0):
    """A ProbeSummary on each route, built the way probe() builds one."""
    clean = 0.80
    return ProbeSummary(clean=clean, pruned=clean - drop_pp / 100,
                        fraction=0.5, heads_removed=72, route=route,
                        metric="accuracy")


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_names_recovery_before_head_training(route):
    """Recovery is worth up to 68pp and head training +0.05pp, so it goes first."""
    text = _summary(route).recommendation
    first_recover = text.find("recover")
    assert first_recover >= 0, "every route must mention recovery"
    for later in ("drophead", "mask_mode", "head masking"):
        at = text.find(later)
        if at >= 0:
            assert first_recover < at, (
                f"{route}: '{later}' is named before recovery")


@pytest.mark.parametrize("route", ROUTES)
def test_no_withdrawn_claim_survives(route):
    text = _summary(route).recommendation.lower()
    forbidden = [
        "not callable",                      # it is callable now
        "v0.7.0 item 2",                     # shipped
        "no recover() stage",                # shipped
        "sal led by",                        # exp28b reversed that cell
        "+1.48pp over drophead",             # inverted wording, withdrawn
        "+2.6pp on the identical set",       # ImageNet-100, superseded
        "train at the ratio you intend to deploy at",
    ]
    for phrase in forbidden:
        assert phrase not in text, f"{route} still says {phrase!r}"


def test_the_fixed_set_carve_out_is_stated_as_withdrawn():
    text = _summary("sal_aligned").recommendation
    assert "carve-out is withdrawn" in text
    assert "+0.67pp above SAL" in text


def test_drophead_is_offered_as_a_callable_mode():
    text = _summary("sal_aligned").recommendation
    assert "mask_mode='drophead'" in text
    # And it really is callable, by the name the text gives.
    from sal.config import SALConfig
    assert SALConfig(num_layers=4, num_heads_per_layer=8,
                     mask_mode="drophead").mask_mode == "drophead"


def test_the_ffn_default_is_magnitude_and_says_why():
    text = _summary("skip_heads").recommendation
    assert "by MAGNITUDE" in text
    assert "1.91pp BETTER one-shot" in text, (
        "the inversion is the reason a user must not benchmark selections "
        "before recovery")


def test_the_head_default_is_random_and_the_overlap_is_stated():
    doc = CompressionPipeline.select_heads.__doc__
    assert "default because it is" in doc
    assert "ranges DO overlap" in doc, (
        "random only wins at 50%; at 25% and 33% the ranges overlap and the "
        "docstring must not imply otherwise")


def test_select_heads_default_strategy_is_random(tiny_model):
    """Read off a call, not off the signature."""
    data = [{"input_ids": torch.randint(0, 100, (4, 16)),
             "labels": torch.randint(0, 100, (4, 16))} for _ in range(2)]
    pipe = CompressionPipeline(tiny_model, data, metric="loss", batch_size=4,
                               seed=0)
    default = pipe.select_heads(0.25)
    seeded = pipe.select_heads(0.25, strategy="random")
    magnitude = pipe.select_heads(0.25, strategy="magnitude")
    assert default == seeded
    assert default != magnitude, (
        "if these agree the test proves nothing; pick a fixture where the "
        "two strategies differ")


# ------------------------------------------------------------ the citations
# A word boundary after "%" never matches, because "%" is not a word
# character. The first version of this pattern ended in \b and so skipped
# every percentage in the text, which is most of the figures in it.
FIGURE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:pp\b|x\b|%)")

#: What the caller's own probe prints back to it, which is not a claim about
#: anything measured elsewhere.
OWN_PROBE = {"50.00pp", "50%", "0.80", "0.30"}


def test_every_source_entry_points_into_the_archive():
    assert ARCHIVE_DOI == "10.5281/zenodo.24391200"
    assert SOURCES, "the mapping must not be empty"
    for figure, path in SOURCES.items():
        assert path.startswith(("data/oss/", "data/pro/", "tables/",
                                "campaign/")), (figure, path)
        assert not path.startswith("/") and ":" not in path, (
            f"{figure}: a source is an archive-relative path, never a local one")
        assert FIGURE.fullmatch(figure), (
            f"{figure!r}: keys are figures as the text prints them, unsigned")


def test_the_routing_text_quotes_no_figure_without_a_source():
    """A contract, not a substitute for having checked the numbers.

    It cannot tell a right number from a wrong one. What it stops is a figure
    arriving in user-facing text with nothing behind it, which is how the two
    false claims this release removed got in.
    """
    missing = {}
    for route in ROUTES:
        for figure in FIGURE.findall(_summary(route).recommendation):
            token = figure.lstrip("+-")
            if (token in SOURCES or token in NOT_MEASUREMENTS
                    or token in OWN_PROBE):
                continue
            missing[token] = route
    assert not missing, f"figures with no entry in SOURCES: {sorted(missing)}"


def test_the_contract_can_fail():
    """The check above is only worth having if an uncited figure trips it."""
    token = "99.99pp"
    assert FIGURE.fullmatch(token)
    assert token not in SOURCES
    assert token not in NOT_MEASUREMENTS and token not in OWN_PROBE
    # So the branch that records a miss is reachable: the loop body above
    # would put this token in `missing`.
    missing = {}
    for figure in FIGURE.findall(f"a new {token} with nothing behind it"):
        tok = figure.lstrip("+-")
        if tok in SOURCES or tok in NOT_MEASUREMENTS or tok in OWN_PROBE:
            continue
        missing[tok] = "synthetic"
    assert missing == {token: "synthetic"}
