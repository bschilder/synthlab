# MCP Integration

SynthLab uses MCP (Model Context Protocol) servers for optional enrichment and ontology lookup. This repo only ships configuration and scripts; no MCP servers are started automatically.

## Config
The default config lives at `configs/mcp.servers.json`.

Example entries:
- `ols-mcp-server`: https://github.com/seandavi/ols-mcp-server

## Validate
Run the offline-safe check:

```bash
synthlab-agent mcp --config configs/mcp.servers.json
```

## Notes
- MCP servers may require separate installation steps (do not install or download data in this repo).
- Keep config changes reproducible and script-driven.
