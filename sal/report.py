"""PDF report generation — thin wrapper over sal.visualize."""
from __future__ import annotations


def generate_fi_report(scan_result, output_path):
    """Render an FI scan to a one-page PDF. Requires sal-torch[reports]."""
    from sal.visualize import render_fi_pdf
    render_fi_pdf(scan_result, str(output_path))
