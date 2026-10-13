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

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-output-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.recipe.stat().st_size > 16 * 1024**2:
        raise ValueError("recipe byte bound exceeded")
    recipe = json.loads(args.recipe.read_text())
    if not isinstance(recipe, dict) or set(recipe) != {
        "genealogy",
        "reference",
        "contig",
        "trait_units",
        "phenotypes",
    }:
        raise ValueError("recipe requires genealogy, reference, contig, trait_units and phenotypes")
    namespace = "_synthlab_simulation_cli"
    package = ModuleType(namespace)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "synthlab")]
    sys.modules[namespace] = package
    genealogy = importlib.import_module(f"{namespace}.genealogy")
    quantitative = importlib.import_module(f"{namespace}.genotype_phenotypes")
    bundle = importlib.import_module(f"{namespace}.association_bundle")
    truth = genealogy.simulate_genealogy(genealogy.GenealogySimConfig(**recipe["genealogy"]))
    variants = tuple(f"site-{int(position)}" for position in truth.site_positions)
    options = dict(recipe["phenotypes"])
    n, p = len(truth.sample_ids), len(options["trait_ids"])
    k, b, q = (
        len(options["covariate_ids"]),
        len(options.get("context_ids", ())),
        len(options.get("residual_factor_ids", ())),
    )
    shapes = {
        "covariates": (n, k),
        "covariate_effects": (k, p),
        "residual_covariance": (p, p),
        "contexts": (n, b),
        "residual_row_factors": (n, q),
        "environment_means": (n, p),
        "observed_probabilities": (n, p),
    }
    for name, shape in shapes.items():
        if options.get(name) is not None:
            values = np.asarray(options[name], dtype=np.float64)
            options[name] = values.reshape(shape) if values.size == 0 and 0 in shape else values
    for name, cls in (
        ("effects", quantitative.AlleleEffect),
        ("dominance_effects", quantitative.DominanceEffect),
        ("interaction_effects", quantitative.InteractionEffect),
        ("context_effects", quantitative.ContextEffect),
        ("compound_effects", quantitative.CompoundHetEffect),
    ):
        options[name] = tuple(cls(**value) for value in options.get(name, ()))
    receipt = bundle.write_association_bundle(
        args.destination,
        truth,
        variant_ids=variants,
        reference=recipe["reference"],
        contig=recipe["contig"],
        trait_units=recipe["trait_units"],
        source_revision=args.source_revision,
        phenotype_options=options,
        max_output_bytes=args.max_output_bytes,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(receipt.manifest_path.resolve()),
                "manifest_sha256": receipt.manifest_sha256,
                "total_file_bytes": receipt.total_file_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
