# SKILLS.md

Minimal skills for Codex agents in SynthLab.

## Workflow runner
- Run: `scripts/run_codex_workflow.sh`
- Purpose: validates the offline test plan, MCP config, and UKB synthetic dataset listing.

## MCP config check
- Run: `synthlab-agent mcp --config configs/mcp.servers.json`
- Purpose: verify MCP server entries parse cleanly.

## UK Biobank synthetic (offline)
- Run: `synthlab-agent ukbiobank --summary`
- Purpose: list file categories and counts without downloading data.
