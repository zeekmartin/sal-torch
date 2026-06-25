"""Report generation stub."""
from __future__ import annotations

def generate_fi_report(scan_result, output_path):
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("pip install sal-torch[reports]")
    # Stub — will be implemented in Phase 5
    raise NotImplementedError("PDF report generation coming in v0.1.0")
