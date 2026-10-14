"""Run the optional numerical producer without loading synthlab's EHR stack.

Use the hash-locked simulation environment described in docs/genealogy_simulation.md.
This standalone checkout/source-archive entry point runs outside CuEvo.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType


def load_recipe_writer() -> ModuleType:
    """Import only the optional numerical modules in this source checkout."""
    namespace = "_synthlab_simulation_cli"
    package = ModuleType(namespace)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "synthlab")]
    sys.modules[namespace] = package
    return importlib.import_module(f"{namespace}.association_recipes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-output-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.recipe.stat().st_size > 16 * 1024**2:
        raise ValueError("recipe byte bound exceeded")
    producer = load_recipe_writer()
    recipe = producer.load_recipe_json(args.recipe.read_bytes())
    receipt = producer.write_association_recipe(
        args.destination,
        recipe,
        source_revision=args.source_revision,
        max_output_bytes=args.max_output_bytes,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
