"""SynthLab CLI utilities for agent-friendly workflows."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

from synthlab.coherent import list_coherent_components
from synthlab.download_ukbiobank_synthetic import list_available_files
from synthlab.imaging import list_imaging_datasets
from synthlab.mcp import load_mcp_config, summarize_mcp_servers
from synthlab.snomed import get_sample_snomed_concepts

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _get_version() -> str:
    try:
        return importlib.metadata.version("synthlab")
    except importlib.metadata.PackageNotFoundError:
        pass

    pyproject = Path("pyproject.toml")
    if pyproject.exists() and tomllib is not None:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return data.get("project", {}).get("version", "dev")

    return "dev"


def run_plan(_: argparse.Namespace) -> int:
    print("SynthLab Codex Plan (offline-safe)")
    print("=" * 40)
    print(f"Version: {_get_version()}")

    # Core catalogs
    components = list_coherent_components()
    print(f"Coherent components: {', '.join(components)}")

    imaging = list_imaging_datasets()
    print(f"Imaging datasets: {len(imaging)}")

    concepts = get_sample_snomed_concepts()
    print(f"Sample SNOMED concepts: {len(concepts)}")

    ukb_files = list_available_files()
    ukb_summary = ", ".join(
        f"{category}:{len(files)}" for category, files in ukb_files.items()
    )
    print(f"UK Biobank synthetic files: {ukb_summary}")

    # Optional dependency presence (no imports)
    print("Optional dependencies:")
    optional = {
        "biomcp": "biomcp",
        "bioinfomcp": "bioinfomcp",
        "faiss": "faiss",
        "pydicom": "pydicom",
        "boto3": "boto3",
    }
    for label, module in optional.items():
        status = "yes" if _module_available(module) else "no"
        print(f"  - {label}: {status}")

    print("Plan complete. No data downloaded.")
    return 0


def run_mcp(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    data = load_mcp_config(config_path)
    summaries = summarize_mcp_servers(data)

    print(f"MCP config: {config_path}")
    for line in summaries:
        print(f"  - {line}")
    return 0


def run_ukbiobank(args: argparse.Namespace) -> int:
    files = sl.list_available_files()
    if args.summary:
        for category, names in files.items():
            print(f"{category}: {len(names)} files")
        return 0

    for category, names in files.items():
        print(f"{category}:")
        for name in names:
            print(f"  - {name}")
    return 0


def run_workflow(args: argparse.Namespace) -> int:
    print("Running Codex workflow...")
    run_plan(args)
    run_mcp(argparse.Namespace(config=args.mcp_config))
    run_ukbiobank(argparse.Namespace(summary=True))
    print("Workflow complete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SynthLab CLI for agent-friendly workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Run offline-safe checks")
    plan_parser.set_defaults(func=run_plan)

    mcp_parser = subparsers.add_parser("mcp", help="Inspect MCP config")
    mcp_parser.add_argument(
        "--config",
        default="configs/mcp.servers.json",
        help="Path to MCP config JSON",
    )
    mcp_parser.set_defaults(func=run_mcp)

    ukb_parser = subparsers.add_parser("ukbiobank", help="List UKB synthetic files")
    ukb_parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print category counts",
    )
    ukb_parser.set_defaults(func=run_ukbiobank)

    workflow_parser = subparsers.add_parser(
        "workflow", help="Run the default Codex workflow"
    )
    workflow_parser.add_argument(
        "--mcp-config",
        default="configs/mcp.servers.json",
        help="Path to MCP config JSON",
    )
    workflow_parser.set_defaults(func=run_workflow)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
