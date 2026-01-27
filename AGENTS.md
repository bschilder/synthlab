# AGENTS.md

Guidance for GPT-5.2-Codex agents working in this repository.

## Core rules
- Do not download real datasets into this repo. Only prepare scripts/configs; no .xlsx, .csv, .h5, .npz data files.
- Every actionable step must be captured in a script under `scripts/`.
- After creating or updating a script, run it to verify the workflow is reproducible.
- Prefer the CLI entry point (`synthlab-agent`) and `scripts/run_codex_workflow.sh` for repeatable runs.
- Keep changes minimal, explicit, and offline-safe by default.

## MCP integration
- MCP server definitions live in `configs/mcp.servers.json`.
- Use `synthlab-agent mcp --config configs/mcp.servers.json` to validate formatting.

## Dataset pipeline expectations
- Use the UK Biobank synthetic dataset pipeline in dry-run mode only.
- Do not fetch data; only list available files and verify local logic.

## References
- `CLAUDE.md` provides architecture and module conventions.
