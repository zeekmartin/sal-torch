"""sal-torch — Structurally Adaptive Learning for PyTorch.

(c) 2026 Cognitive Engineering — cognitive-engineering.dev
Licensed under BSL 1.1.
"""
__version__ = "0.2.0"

from sal.config import SALConfig
from sal.masker import HeadMasker
from sal.callback import SALCallback
from sal.fi import compute_fi, extract_activation_graph, classify_layers, LayerClass
from sal.scanner import FIScanner, FIMonitor
from sal.plasticity import PlasticityScanner, PlasticityMap, Recommendation
from sal.compare import compare

import sal.license as _lic

_LICENSE_INFO = None

def set_license(path: str):
    global _LICENSE_INFO
    _LICENSE_INFO = _lic.verify_license(path)

def license_info() -> dict:
    if _LICENSE_INFO:
        return _LICENSE_INFO.to_dict()
    return {"tier": "community", "organization": None, "expires": None,
            "features": ["sal", "fi", "scanner", "plasticity", "report"],
            "note": "Community — full features, non-commercial use only."}

def _check_env():
    import os
    p = os.environ.get("SAL_LICENSE_FILE")
    if p:
        try: set_license(p)
        except Exception: pass

_check_env()

__all__ = ["SALConfig", "SALCallback", "HeadMasker", "FIScanner", "FIMonitor",
           "compute_fi", "extract_activation_graph", "classify_layers", "LayerClass",
           "PlasticityScanner", "PlasticityMap", "Recommendation", "compare",
           "set_license", "license_info"]
