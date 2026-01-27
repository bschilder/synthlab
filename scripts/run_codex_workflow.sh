#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=. python scripts/run_codex_workflow.py --mcp-config configs/mcp.servers.json
