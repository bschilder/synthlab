"""Reproducible, offline-safe workflow runner for Codex agents."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_plan_modules():
    coherent = _load_module("synthlab_coherent", "synthlab/coherent.py")
    imaging = _load_module("synthlab_imaging", "synthlab/imaging.py")
    snomed = _load_module("synthlab_snomed", "synthlab/snomed.py")
    ukb = _load_module("synthlab_ukb", "synthlab/download_ukbiobank_synthetic.py")
    mcp = _load_module("synthlab_mcp", "synthlab/mcp.py")
    return coherent, imaging, snomed, ukb, mcp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SynthLab Codex workflow (offline-safe)"
    )
    parser.add_argument(
        "--mcp-config",
        default="configs/mcp.servers.json",
        help="Path to MCP config JSON",
    )
    args = parser.parse_args()

    coherent, imaging, snomed, ukb, mcp = _load_plan_modules()

    print("[1/3] Plan checks")
    version = "dev"
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            tomllib = None
        if tomllib is not None:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            version = data.get("project", {}).get("version", version)
    print(f"Version: {version}")
    components = coherent.list_coherent_components()
    print(f"Coherent components: {', '.join(components)}")
    datasets = imaging.list_imaging_datasets()
    print(f"Imaging datasets: {len(datasets)}")
    concepts = snomed.get_sample_snomed_concepts()
    print(f"Sample SNOMED concepts: {len(concepts)}")
    ukb_files = ukb.list_available_files()
    summary = ", ".join(
        f"{category}:{len(files)}" for category, files in ukb_files.items()
    )
    print(f"UK Biobank synthetic files: {summary}")

    print("[2/3] MCP config")
    config_path = Path(args.mcp_config)
    mcp_data = mcp.load_mcp_config(config_path)
    summaries = mcp.summarize_mcp_servers(mcp_data)
    print(f"MCP config: {config_path}")
    for line in summaries:
        print(f"  - {line}")

    print("[3/3] UK Biobank synthetic summary")
    for category, names in ukb_files.items():
        print(f"{category}: {len(names)} files")

    print("Workflow done. No data downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
