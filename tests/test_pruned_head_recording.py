"""HeadMasker records which heads it killed, and SALTrainer reports them.

The set matters because SAL's adaptation is specific to it: a model trained
without heads A is not thereby prepared to lose heads B. Without this
bookkeeping the aligned-pruning comparison cannot be made at all.
"""
import pytest
import torch

from sal import HeadMasker, SALConfig, SALTrainer


@pytest.fixture
def labelled_data():
    """Causal-LM batches for the tiny fixture, which takes input_ids + labels."""
    torch.manual_seed(7)
    return [{"input_ids": (x := torch.randint(0, 100, (4, 16))), "labels": x}
            for _ in range(6)]


def test_pruned_heads_empty_before_pruning(tiny_model, tiny_config):
    masker = HeadMasker(tiny_model, tiny_config, seed=0)
    masker.install()
    assert masker.pruned_heads() == []
    assert all(v == [] for v in masker.pruned_heads_by_layer().values())
    masker.remove()


def test_pruned_heads_matches_the_masks(tiny_model, tiny_config):
    masker = HeadMasker(tiny_model, tiny_config, seed=1)
    masker.install()
    masker.activate()

    pruned = masker.pruned_heads()
    assert len(pruned) == tiny_config.num_heads_to_prune
    assert len(set(pruned)) == len(pruned), "a head was reported twice"

    # Every reported head is zero in the masks, and nothing else is.
    by_layer = masker.pruned_heads_by_layer()
    for layer, mask in masker._masks.items():
        zeros = (mask == 0.0).nonzero(as_tuple=True)[0].tolist()
        assert by_layer[layer] == zeros
        for head in zeros:
            assert (layer, head) in pruned
    assert sum(len(v) for v in by_layer.values()) == len(pruned)
    masker.remove()


def test_pruned_heads_is_sorted(tiny_model, tiny_config):
    masker = HeadMasker(tiny_model, tiny_config, seed=2)
    masker.install()
    masker.activate()
    pruned = masker.pruned_heads()
    assert pruned == sorted(pruned)
    masker.remove()


def test_pruned_heads_reproducible_from_seed(tiny_model, tiny_config):
    """The set is a deterministic function of the seed, not of training."""
    def run(seed):
        model = type(tiny_model)()
        m = HeadMasker(model, tiny_config, seed=seed)
        m.install()
        m.activate()
        out = m.pruned_heads()
        m.remove()
        return out

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_accumulates_monotonically(tiny_model, tiny_config):
    masker = HeadMasker(tiny_model, tiny_config, seed=3)
    masker.install()
    seen = []
    for step in range(60):
        masker.step(step, 60)
        seen.append(set(masker.pruned_heads()))
    # Once a head is pruned it stays pruned: every set contains the previous one.
    for earlier, later in zip(seen, seen[1:]):
        assert earlier <= later, "a pruned head came back"
    masker.remove()


def test_trainer_reports_pruned_heads(tiny_model, tiny_config, labelled_data):
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
    trainer = SALTrainer(tiny_model, tiny_config, opt, labelled_data, seed=5)
    history = trainer.train(num_epochs=2)

    assert "pruned_heads" in history
    pruned = history["pruned_heads"]
    assert pruned, "nothing was recorded as pruned"
    assert len(pruned) == history["masker_stats"]["pruned_heads"]
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pruned)
    layers = {layer for layer, _ in pruned}
    assert layers <= set(range(tiny_config.num_layers))


def test_trainer_reports_pruned_heads_in_custom_mode(tiny_model, tiny_config, labelled_data):
    def custom_step(model, batch, optimizer, mask_module):
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return out.loss

    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
    trainer = SALTrainer(tiny_model, tiny_config, opt, labelled_data, seed=6,
                         train_step=custom_step)
    history = trainer.train(num_epochs=2)
    assert len(history["pruned_heads"]) == history["masker_stats"]["pruned_heads"]


def test_masking_the_recorded_set_reproduces_training_time_damage(
        tiny_model, tiny_config, probe_data):
    """Re-masking the recorded set must give the same outputs it gave in training."""
    masker = HeadMasker(tiny_model, tiny_config, seed=9)
    masker.install()
    masker.activate()
    recorded = masker.pruned_heads()
    with torch.no_grad():
        during = tiny_model(**probe_data[0]).logits.clone()
    masker.remove()

    # Fresh masker, no schedule: just switch off exactly the recorded heads.
    replay = HeadMasker(tiny_model, tiny_config, seed=999)
    replay.install()
    for layer, head in recorded:
        replay._masks[layer][head] = 0.0
    replay.apply_mask()
    with torch.no_grad():
        after = tiny_model(**probe_data[0]).logits
    replay.remove()

    assert torch.allclose(during, after, atol=1e-6)
