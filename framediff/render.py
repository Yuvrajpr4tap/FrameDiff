"""
Rendering utilities for DiffReport (terminal, Jupyter, HTML).

Most rendering is handled by DiffReport.__repr__() and DiffReport._repr_html__(),
but this module provides utilities for custom rendering and formatting.
"""
from typing import Optional
import pandas as pd

from .report import DiffReport


def to_terminal_string(report: DiffReport, use_rich: bool = True) -> str:
    """
    Render DiffReport as terminal string.

    Args:
        report: DiffReport to render
        use_rich: If True, use rich tables if available

    Returns:
        Formatted string for terminal output
    """
    if use_rich:
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            return repr(report)
        except ImportError:
            pass

    # Fallback to plain text
    return repr(report)


def to_html_string(report: DiffReport, include_samples: bool = True) -> str:
    """
    Render DiffReport as HTML string.

    Args:
        report: DiffReport to render
        include_samples: Whether to include sample data tables

    Returns:
        HTML string
    """
    html = report._repr_html_()

    if include_samples:
        html += _render_samples_html(report)

    return html


def _render_samples_html(report: DiffReport) -> str:
    """Render sample data tables as HTML."""
    html = """
    <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
        <h4>Sample Data</h4>
    """

    if not report.rows.sample_added.empty:
        html += "<h5>Added Rows (Sample)</h5>"
        html += report.rows.sample_added.to_html()

    if not report.rows.sample_removed.empty:
        html += "<h5>Removed Rows (Sample)</h5>"
        html += report.rows.sample_removed.to_html()

    if not report.rows.sample_modified.empty:
        html += "<h5>Modified Rows (Sample)</h5>"
        html += report.rows.sample_modified.to_html()

    html += "</div>"
    return html


def table_from_dict(data: dict, title: str = "Table") -> str:
    """
    Convert dict to formatted table string.

    Args:
        data: Dict mapping keys to values
        title: Table title

    Returns:
        Formatted table string
    """
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title=title)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        console = Console()
        return str(console.render_str(table.__str__()))
    except ImportError:
        # Fallback to plain text
        lines = [f"=== {title} ==="]
        for key, value in data.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
