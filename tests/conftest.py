"""Test fixtures — tiny transformer for CPU tests."""
import pytest
import torch
import torch.nn as nn


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run integration tests (require network + model downloads, slow)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires network + model downloads (and ideally GPU); "
        "skipped unless --run-integration is passed")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration (network + model downloads)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

class TinyConfig:
    model_type = "gpt2"
    n_layer = 4; n_head = 8; n_embd = 64
    num_hidden_layers = 4; num_attention_heads = 8; hidden_size = 64

class TinyAttn(nn.Module):
    def __init__(self, hs=64, nh=8):
        super().__init__()
        self.nh = nh; self.hd = hs // nh
        self.q_proj = nn.Linear(hs, hs, bias=False)
        self.k_proj = nn.Linear(hs, hs, bias=False)
        self.v_proj = nn.Linear(hs, hs, bias=False)
        self.out_proj = nn.Linear(hs, hs, bias=False)

    def forward(self, x):
        bs, sl, _ = x.shape
        q = self.q_proj(x).view(bs, sl, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(bs, sl, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(bs, sl, self.nh, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-2, -1) / self.hd**0.5, dim=-1)
        self._last_attn = a.detach()  # [bs, nh, sl, sl] — read when output_attentions=True
        o = (a @ v).transpose(1, 2).contiguous().view(bs, sl, -1)
        return self.out_proj(o)

class TinyBlock(nn.Module):
    def __init__(self, hs=64, nh=8):
        super().__init__()
        self.attn = TinyAttn(hs, nh)
        self.ln = nn.LayerNorm(hs)
        self.mlp = nn.Sequential(nn.Linear(hs, hs*4), nn.GELU(), nn.Linear(hs*4, hs))
    def forward(self, x):
        x = x + self.attn(self.ln(x))
        return x + self.mlp(self.ln(x))

class TinyTransformer(nn.Module):
    def __init__(self, nl=4, hs=64, nh=8, vs=100, msl=32):
        super().__init__()
        self.config = TinyConfig()
        self.config.n_layer = nl; self.config.n_head = nh; self.config.n_embd = hs
        self.config.num_hidden_layers = nl; self.config.num_attention_heads = nh; self.config.hidden_size = hs
        self.embed = nn.Embedding(vs, hs)
        self.pos_embed = nn.Embedding(msl, hs)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([TinyBlock(hs, nh) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(hs)
        self.head = nn.Linear(hs, vs, bias=False)

    def forward(self, input_ids=None, labels=None, output_attentions=False,
                output_hidden_states=False, **kw):
        bs, sl = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed(torch.arange(sl, device=input_ids.device).unsqueeze(0))
        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for block in self.transformer.h:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(block.attn._last_attn)
        logits = self.head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[...,:-1,:].contiguous().view(-1, logits.size(-1)),
                                               labels[...,1:].contiguous().view(-1))
        class Out: pass
        o = Out(); o.logits = logits; o.loss = loss
        o.attentions = tuple(attentions) if output_attentions else None
        o.hidden_states = tuple(hidden_states) if output_hidden_states else None
        return o

@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return TinyTransformer()

@pytest.fixture
def tiny_config():
    from sal.config import SALConfig
    return SALConfig(num_layers=4, num_heads_per_layer=8, attention_pattern="transformer.h.{}.attn", prune_fraction=0.33)

@pytest.fixture
def probe_data():
    torch.manual_seed(42)
    return [{"input_ids": torch.randint(0, 100, (4, 16))} for _ in range(10)]
