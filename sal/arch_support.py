"""Architecture auto-detection and attention-module introspection for SAL.

Module finding is centralized here so that the masker (which perturbs heads) and
the FI graph builder (which measures them) always operate on the *same* output
projections.

Attention patterns are written **relative to the base model** (the HF
``model.base_model``), so the same pattern works whether the model is bare
(``BertModel``) or wrapped by a task head (``BertForSequenceClassification``).
"""
from __future__ import annotations
from dataclasses import dataclass


class SALArchitectureError(Exception):
    pass


@dataclass
class ArchInfo:
    model_type: str
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    attention_pattern: str

    @property
    def total_heads(self) -> int:
        return self.num_layers * self.num_heads


# Attention-container path relative to model.base_model. The masker/FI then pull
# the output projection out of each container via get_output_projection().
_REGISTRY = {
    "llama": "layers.{}.self_attn",
    "mistral": "layers.{}.self_attn",
    "gpt2": "h.{}.attn",
    "bert": "encoder.layer.{}.attention",
    "roberta": "encoder.layer.{}.attention",
    "distilbert": "transformer.layer.{}.attention",
    "vit": "layers.{}.attention",
    "phi": "layers.{}.self_attn",
    "phi3": "layers.{}.self_attn",
    "gemma": "layers.{}.self_attn",
    "gemma2": "layers.{}.self_attn",
    "qwen2": "layers.{}.self_attn",
}

# Fallback attention-container patterns, tried in order when no pattern is known.
_FALLBACK_PATTERNS = [
    "layers.{}.self_attn",              # llama / mistral / phi / gemma / qwen
    "h.{}.attn",                        # gpt2 (relative to base model)
    "transformer.h.{}.attn",            # gpt2-like from root (e.g. tiny test model)
    "encoder.layer.{}.attention",       # bert / roberta
    "transformer.layer.{}.attention",   # distilbert
    "layers.{}.attention",              # vit (transformers >= 5)
    "encoder.layer.{}.attention.self",  # legacy bert layout
]

# Output-projection attribute names, in priority order. ``out_lin`` is DistilBERT;
# ``output`` is the BERT/ViT-style wrapper whose ``.dense`` is the projection.
_OUTPUT_PROJ_NAMES = ["o_proj", "out_proj", "c_proj", "out_lin", "dense", "output"]


def _standard_arch(model_type, config, pattern):
    nl = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', getattr(config, 'n_layers', None)))
    nh = getattr(config, 'num_attention_heads', getattr(config, 'n_head', getattr(config, 'n_heads', None)))
    hs = getattr(config, 'hidden_size', getattr(config, 'n_embd', getattr(config, 'dim', None)))
    hd = getattr(config, 'head_dim', hs // nh if hs and nh else None)
    if nl is None or nh is None or hs is None:
        raise SALArchitectureError(f"Cannot extract arch params for '{model_type}'")
    return ArchInfo(model_type=model_type, num_layers=nl, num_heads=nh, head_dim=hd,
                    hidden_size=hs, attention_pattern=pattern)


def detect_architecture(model) -> ArchInfo:
    config = getattr(model, 'config', None)
    if config is None:
        raise SALArchitectureError("Model has no .config. Use SALConfig manually.")
    model_type = getattr(config, 'model_type', None)
    if model_type is None:
        raise SALArchitectureError("Model config has no model_type. Use SALConfig manually.")
    pattern = _REGISTRY.get(model_type)
    if pattern is None:
        raise SALArchitectureError(
            f"'{model_type}' not supported. Supported: {sorted(_REGISTRY.keys())}")
    return _standard_arch(model_type, config, pattern)


def supported_architectures() -> list[str]:
    return sorted(_REGISTRY.keys())


# ----------------------------------------------------- module/projection finding
def _resolve_root(model):
    """The HF base model (strips a task-head prefix) or the model itself."""
    base = getattr(model, "base_model", None)
    return base if base is not None else model


def _walk(root, path):
    mod = root
    for attr in path.split("."):
        mod = mod[int(attr)] if attr.isdigit() else getattr(mod, attr)
    return mod


def _find_by_pattern(model, pattern):
    roots = [_resolve_root(model)]
    if model is not roots[0]:
        roots.append(model)
    for root in roots:
        mods = []
        for i in range(2000):
            try:
                mods.append(_walk(root, pattern.format(i)))
            except (AttributeError, IndexError, TypeError, KeyError):
                break
        if mods:
            return mods
    return []


def get_attention_modules(model, attention_pattern: str | None = None) -> list:
    """Locate per-layer attention container modules."""
    if attention_pattern:
        mods = _find_by_pattern(model, attention_pattern)
        if mods:
            return mods
    for pat in _FALLBACK_PATTERNS:
        mods = _find_by_pattern(model, pat)
        if mods:
            return mods
    return []


def get_output_projection(attn_module):
    """Return the attention output projection module, or None."""
    for name in _OUTPUT_PROJ_NAMES:
        proj = getattr(attn_module, name, None)
        if proj is not None:
            # BERT/ViT wrappers expose the projection one level down as `.dense`.
            if hasattr(proj, "dense"):
                return proj.dense
            return proj
    return None


def get_output_projections(model, attention_pattern: str | None = None) -> list:
    """Per-layer attention output projections (the hook points for SAL and FI)."""
    projs = []
    for m in get_attention_modules(model, attention_pattern):
        p = get_output_projection(m)
        if p is not None:
            projs.append(p)
    return projs


# Q/K/V projection names. Separate triples are tried first (Linear modules);
# fused names hold Q, K, V concatenated in a single projection.
_QKV_TRIPLES = [
    ("q_proj", "k_proj", "v_proj"),   # llama / mistral / phi / gemma / qwen / tiny test
    ("query", "key", "value"),        # bert / roberta / vit (often under `.self`)
    ("q_lin", "k_lin", "v_lin"),      # distilbert
    ("q", "k", "v"),
]
_QKV_FUSED = ["c_attn", "qkv_proj", "in_proj", "Wqkv", "query_key_value"]


def get_qkv_projections(attn_module):
    """Locate the Q/K/V projection(s) for an attention container.

    Returns one of:
      * ``{"mode": "separate", "q": mod, "k": mod, "v": mod}`` — three modules, or
      * ``{"mode": "fused", "qkv": mod, "name": str}`` — one module holding Q|K|V, or
      * ``None`` if nothing recognizable is found.

    BERT/ViT keep Q/K/V one level down in ``.self``; both levels are searched.
    """
    candidates = [attn_module]
    sub = getattr(attn_module, "self", None)
    if sub is not None:
        candidates.append(sub)

    for c in candidates:
        for qn, kn, vn in _QKV_TRIPLES:
            q, k, v = getattr(c, qn, None), getattr(c, kn, None), getattr(c, vn, None)
            if q is not None and k is not None and v is not None:
                return {"mode": "separate", "q": q, "k": k, "v": v}
    for c in candidates:
        for name in _QKV_FUSED:
            m = getattr(c, name, None)
            if m is not None:
                return {"mode": "fused", "qkv": m, "name": name}
    return None
