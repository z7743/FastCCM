#!/usr/bin/env python3
"""Helpers for publishing benchmark script output into scripts/README.md."""

from __future__ import annotations

from pathlib import Path


REPORT_PATH = Path(__file__).with_name("README.md")


def format_setting_value(value: object) -> str:
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value!r}`"


def render_settings(settings: dict[str, object]) -> list[str]:
    lines = ["Settings:"]
    for key, value in settings.items():
        lines.append(f"- `{key}`: {format_setting_value(value)}")
    return lines


def render_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return [header, divider, *body]


def update_report_section(
    *,
    section_id: str,
    title: str,
    script_name: str,
    settings: dict[str, object],
    columns: list[str],
    rows: list[list[str]],
    report_path: Path = REPORT_PATH,
) -> None:
    start_marker = f"<!-- benchmark-report:{section_id} start -->"
    end_marker = f"<!-- benchmark-report:{section_id} end -->"

    content = report_path.read_text(encoding="utf-8")
    replacement_lines = [
        start_marker,
        f"## {title}",
        "",
        f"Source: `{script_name}`",
        "",
        *render_settings(settings),
        "",
        "Results:",
        "",
        *render_table(columns, rows),
        end_marker,
    ]
    replacement = "\n".join(replacement_lines)

    start = content.find(start_marker)
    end = content.find(end_marker)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Could not find report markers for section {section_id!r}.")
    end += len(end_marker)
    updated = content[:start] + replacement + content[end:]
    report_path.write_text(updated + ("\n" if not updated.endswith("\n") else ""), encoding="utf-8")
