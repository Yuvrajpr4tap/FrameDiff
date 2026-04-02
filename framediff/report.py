"""
DiffReport — the main object returned by compare().
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import json
import hashlib

from .schema import SchemaDiff, DiffIssue
from .stats import StatDiff
from .rows import RowDiff
from .exceptions import DiffThresholdError


@dataclass
class DiffReport:
    """
    Complete diff report combining schema, stats, and row-level changes.
    """

    schema: SchemaDiff
    stats: Dict[str, StatDiff]
    rows: RowDiff
    issues: List[DiffIssue] = field(default_factory=list)

    def __post_init__(self):
        """Aggregate all issues and compute severity."""
        # Collect issues from all sources
        all_issues = []
        all_issues.extend(self.schema.issues)
        for stat_diff in self.stats.values():
            if stat_diff.severity != "info":
                all_issues.append(
                    DiffIssue(
                        severity=stat_diff.severity,
                        category="stats",
                        message=f"Column '{stat_diff.column}': {stat_diff.distribution_label}",
                    )
                )
        self.issues = all_issues

    @property
    def severity(self) -> str:
        """
        Aggregate severity: highest level found across all issues.
        Returns one of: "info", "warning", "critical"
        """
        if not self.issues:
            return "info"

        severity_order = {"critical": 2, "warning": 1, "info": 0}
        max_severity = max(severity_order.get(issue.severity, 0) for issue in self.issues)

        for sev, order in severity_order.items():
            if order == max_severity:
                return sev
        return "info"

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        added = self.rows.added_count
        removed = self.rows.removed_count
        modified = self.rows.modified_count
        schema_changes = len(
            self.schema.added_columns
            + self.schema.removed_columns
            + list(self.schema.type_changes.keys())
        )

        parts = []
        if schema_changes > 0:
            parts.append(f"{schema_changes} schema change(s)")
        if added > 0:
            parts.append(f"{added} rows added")
        if removed > 0:
            parts.append(f"{removed} rows removed")
        if modified > 0:
            parts.append(f"{modified} rows modified")

        if not parts:
            return "No changes detected"

        return " | ".join(parts)

    def to_dict(self) -> dict:
        """
        Convert to fully JSON-serializable dict.
        """
        return {
            "schema": self.schema.to_dict(),
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "rows": self.rows.to_dict(),
            "severity": self.severity,
            "summary": self.summary,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
        }

    def to_json(self) -> str:
        """Return JSON string representation."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def fingerprint(self) -> str:
        """
        Deterministic SHA256 hash of the diff content.
        """
        dict_repr = self.to_dict()
        json_str = json.dumps(dict_repr, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def __repr__(self) -> str:
        """
        Terminal-friendly representation using rich if available.
        """
        try:
            import io
            from rich.table import Table
            from rich.console import Console

            # Create a string buffer and console that writes to it
            buffer = io.StringIO()
            console = Console(file=buffer, highlight=False)
            
            table = Table(title=f"DiffReport [{self.severity.upper()}]")

            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Summary", self.summary)
            table.add_row("Severity", self.severity)
            table.add_row("Rows Added", str(self.rows.added_count))
            table.add_row("Rows Removed", str(self.rows.removed_count))
            table.add_row("Rows Modified", str(self.rows.modified_count))
            table.add_row(
                "Schema Changes",
                str(
                    len(self.schema.added_columns)
                    + len(self.schema.removed_columns)
                    + len(self.schema.type_changes)
                ),
            )
            table.add_row("Fingerprint", self.fingerprint[:16] + "...")

            # Render table to buffer
            console.print(table)
            return buffer.getvalue()

        except ImportError:
            # Fallback to plain text
            return (
                f"DiffReport[{self.severity.upper()}]\n"
                f"  Summary: {self.summary}\n"
                f"  Rows: +{self.rows.added_count}, -{self.rows.removed_count}, ~{self.rows.modified_count}\n"
                f"  Schema: {len(self.schema.added_columns) + len(self.schema.removed_columns) + len(self.schema.type_changes)} changes\n"
                f"  Fingerprint: {self.fingerprint[:16]}..."
            )

    def _repr_html_(self) -> str:
        """Jupyter HTML representation with histograms."""
        severity_colors = {"info": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"}
        severity_color = severity_colors.get(self.severity, "#95a5a6")

        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <h3>
                <span style="display: inline-block; padding: 5px 10px; border-radius: 3px; background-color: {severity_color}; color: white;">
                    {self.severity.upper()}
                </span>
                DiffReport
            </h3>
            <p><strong>{self.summary}</strong></p>
            <hr>
            
            <h4>Row Changes</h4>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f8f9fa;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Count</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">%</th>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Added</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{self.rows.added_count}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{self.rows.added_pct:.2f}%</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="border: 1px solid #ddd; padding: 8px;">Removed</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{self.rows.removed_count}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{self.rows.removed_pct:.2f}%</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Modified</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{self.rows.modified_count}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{0 if self.rows.total_before == 0 else 100.0 * self.rows.modified_count / self.rows.total_before:.2f}%</td>
                </tr>
            </table>

            <h4>Schema Changes</h4>
            <ul>
        """

        if self.schema.removed_columns:
            html += f"<li><strong style='color: #e74c3c;'>Removed:</strong> {', '.join(self.schema.removed_columns)}</li>"
        if self.schema.added_columns:
            html += f"<li><strong style='color: #2ecc71;'>Added:</strong> {', '.join(self.schema.added_columns)}</li>"
        if self.schema.type_changes:
            type_strs = [
                f"{col}: {before} → {after}"
                for col, (before, after) in self.schema.type_changes.items()
            ]
            html += f"<li><strong style='color: #f39c12;'>Type Changes:</strong> {', '.join(type_strs)}</li>"

        html += """
            </ul>

            <h4>Top Issues</h4>
            <ul>
        """

        for issue in sorted(self.issues, key=lambda x: {"critical": 0, "warning": 1, "info": 2}.get(x.severity, 3))[:5]:
            color = severity_colors.get(issue.severity, "#95a5a6")
            html += f"<li><span style='color: {color};'><strong>[{issue.severity.upper()}]</strong></span> {issue.message}</li>"

        html += """
            </ul>
            <hr>
            <p style="font-size: 12px; color: #7f8c8d;">
                Fingerprint: <code style="background-color: #ecf0f1; padding: 2px 5px; border-radius: 3px;">{}</code>
            </p>
        </div>
        """.format(self.fingerprint[:20] + "...")

        return html

    def assert_within(
        self,
        max_rows_removed_pct: Optional[float] = None,
        max_rows_added_pct: Optional[float] = None,
        max_null_rate_increase: Optional[float] = None,
        max_psi: Optional[float] = None,
        no_type_changes: bool = False,
        no_removed_columns: bool = False,
        no_critical: bool = False,
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Raise DiffThresholdError if report violates any specified thresholds.

        Args:
            max_rows_removed_pct: Max % of rows that can be removed
            max_rows_added_pct: Max % of rows that can be added
            max_null_rate_increase: Max increase in null rate per column
            max_psi: Max PSI score per column
            no_type_changes: If True, fail on any type change
            no_removed_columns: If True, fail if any columns removed
            no_critical: If True, fail on any critical severity
            columns: Restrict checks to specific columns

        Raises:
            DiffThresholdError: If any assertion fails
        """
        violations = []

        # Rows removed check
        if max_rows_removed_pct is not None:
            if self.rows.removed_pct > max_rows_removed_pct:
                violations.append(
                    f"Rows removed {self.rows.removed_pct:.2f}% exceeds max {max_rows_removed_pct:.2f}%"
                )

        # Rows added check
        if max_rows_added_pct is not None:
            if self.rows.added_pct > max_rows_added_pct:
                violations.append(
                    f"Rows added {self.rows.added_pct:.2f}% exceeds max {max_rows_added_pct:.2f}%"
                )

        # Null rate increase check
        if max_null_rate_increase is not None:
            for stat_diff in self.stats.values():
                if columns and stat_diff.column not in columns:
                    continue
                if stat_diff.null_rate_delta > max_null_rate_increase:
                    violations.append(
                        f"Column '{stat_diff.column}' null rate increased by {stat_diff.null_rate_delta:.3f} (max {max_null_rate_increase:.3f})"
                    )

        # PSI check
        if max_psi is not None:
            for stat_diff in self.stats.values():
                if columns and stat_diff.column not in columns:
                    continue
                if (
                    stat_diff.distribution_method == "psi"
                    and stat_diff.distribution_score > max_psi
                ):
                    violations.append(
                        f"Column '{stat_diff.column}' PSI {stat_diff.distribution_score:.4f} exceeds max {max_psi:.4f}"
                    )

        # Type changes check
        if no_type_changes and self.schema.type_changes:
            for col, (before, after) in self.schema.type_changes.items():
                if not columns or col in columns:
                    violations.append(f"Column '{col}' dtype changed: {before} → {after}")

        # Removed columns check
        if no_removed_columns and self.schema.removed_columns:
            violations.append(
                f"Removed columns not allowed: {', '.join(self.schema.removed_columns)}"
            )

        # Critical severity check
        if no_critical and self.severity == "critical":
            violations.append(
                f"Critical severity issues found: {len([i for i in self.issues if i.severity == 'critical'])} issue(s)"
            )

        if violations:
            raise DiffThresholdError(
                "Diff assertions failed:\n  " + "\n  ".join(violations),
                violations=violations,
            )
