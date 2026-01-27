"""
Utilities for loading MCP (Model Context Protocol) server configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MCPConfigError(ValueError):
    """Raised when an MCP config file is missing or invalid."""


def load_mcp_config(path: Path) -> dict[str, Any]:
    """
    Load a MCP config JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Parsed config dictionary.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MCPConfigError(f"MCP config not found: {path}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"Invalid JSON in MCP config: {path}") from exc

    if "mcpServers" not in data or not isinstance(data["mcpServers"], dict):
        raise MCPConfigError("MCP config must contain a 'mcpServers' object.")

    return data


def summarize_mcp_servers(data: dict[str, Any]) -> list[str]:
    """
    Summarize MCP servers from a loaded config.

    Args:
        data: Parsed MCP config dictionary.

    Returns:
        List of human-readable server summaries.
    """
    servers = data.get("mcpServers", {})
    summaries = []

    for name, config in servers.items():
        command = str(config.get("command", "")).strip()
        args = config.get("args", [])
        if isinstance(args, list):
            args_str = " ".join(str(arg) for arg in args)
        else:
            args_str = str(args)
        line = f"{name}: {command} {args_str}".strip()
        summaries.append(line)

    return summaries
