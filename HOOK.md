# HOOK.md

Agent execution hook for reproducible work.

- All actionable steps must be written into scripts under `scripts/`.
- After script changes, run the script to validate the workflow.
- Keep runs offline-safe; no dataset downloads in this repo.
- Prefer `synthlab-agent` for CLI-driven workflows.
