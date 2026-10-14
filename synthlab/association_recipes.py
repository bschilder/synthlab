"""Independent, JSON-defined genealogy/quantitative simulation recipes.

Demes defines demographic histories; explicit descendant factors define shared
environmental noise. No CuEvo code or fitted association model is imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import genotype_phenotypes as quantitative
from .association_bundle import write_association_bundle
from .genealogical_covariance import genealogical_environment_factors
from .genealogy import (
    GenealogySimConfig,
    _finite_real,
    _identifiers,
    _positive_integer,
    _require_msprime,
    simulate_genealogy,
)


def _closed(value: Any, required: set[str], optional: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= set(value) <= required | optional:
        raise ValueError(f"{name} contains missing or unsupported fields")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON recipe key")
        result[key] = value
    return result


def _no_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON recipe constant {value}")


def load_recipe_json(data: str | bytes) -> Any:
    """Reject overwritten keys and nonfinite JSON constants before generation."""
    return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_no_constant)


def write_association_recipe(
    destination: str | Path,
    recipe: dict[str, Any],
    *,
    source_revision: str,
    max_output_bytes: int = 256 * 1024**2,
) -> dict[str, Any]:
    """Execute one fixed recipe and publish a fresh verified version0 bundle.

    Required fields match the original standalone single-deme recipe. Optional
    demography={graph: <Demes mapping>, sample_demes: <person-aligned names>}
    defines splits, admixture, migration and size histories through Demes.
    Deme labels describe generation and are not prescribed analysis classes.

    genealogical_environment is an ordered list with unique id and nonnegative
    variance per component; optional lower_time, upper_time (null=infinity),
    center and normalization follow the independent branch oracle. Scaled factors
    are appended to any explicitly supplied household/environment factors,
    preserving all IDs. Their covariance contributions add, with no cross terms
    between independent factor columns. A zero coefficient remains recorded.

    The returned receipt pins canonical JSON recipe bytes (sorted keys, compact
    separators, finite numbers), the transport manifest and each unscaled
    environmental oracle's provenance/signature. Preserve this receipt with the
    original recipe: these additional mechanisms do not change the closed
    version0 bundle schema. The bundle stores actual scaled row factors.

    Optional effect_rules plant signed additive coefficients using the first
    coordinate-ordered biallelic locus meeting a declared minor-copy threshold.
    Selection reads complete genotype truth only, before any phenotype draw.
    A missing eligible locus fails the registered recipe without resimulation.

    Component factor outputs are bounded by max_output_bytes before aggregation.
    The existing individual oracle and bundle bounds still apply; the output
    limit is not a process-RSS, simulator-event or wall-clock guarantee.
    """
    _positive_integer(max_output_bytes, "max_output_bytes")
    value = _closed(
        recipe,
        {"genealogy", "reference", "contig", "trait_units", "phenotypes"},
        {"demography", "genealogical_environment", "effect_rules"},
        "recipe",
    )
    canonical = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    config = GenealogySimConfig(**value["genealogy"])
    rules = value.get("effect_rules", [])
    if not isinstance(rules, list):
        raise TypeError("effect_rules must be an ordered list")
    for rule in rules:
        _closed(rule, {"rule", "minimum_copies", "effects"}, set(), "effect rule")
        if rule["rule"] != "first_biallelic_minor_copies":
            raise ValueError("unknown or outcome-dependent effect rule")
        _positive_integer(rule["minimum_copies"], "minimum_copies")
    environments = value.get("genealogical_environment", [])
    if not isinstance(environments, list):
        raise TypeError("genealogical_environment must be an ordered list")
    validated = []
    for item in environments:
        item = _closed(
            item,
            {"id", "variance"},
            {"lower_time", "upper_time", "center", "normalization"},
            "environmental component",
        )
        identifier = _identifiers((item["id"],), "environment component ID")[0]
        variance = _finite_real(item["variance"], "environment variance", allow_zero=True)
        options = {
            key: item[key]
            for key in ("lower_time", "upper_time", "center", "normalization")
            if key in item
        }
        if options.get("upper_time", math.inf) is None:
            options["upper_time"] = math.inf
        validated.append((identifier, variance, options))
    if validated:
        _identifiers(tuple(item[0] for item in validated), "environment component IDs")
    model, labels = None, None
    if "demography" in value:
        specification = _closed(value["demography"], {"graph", "sample_demes"}, set(), "demography")
        if not isinstance(specification["graph"], dict) or not isinstance(
            specification["sample_demes"], list
        ):
            raise ValueError("Demes graph mapping and person-aligned deme list required")
        import demes

        graph = demes.Graph.fromdict(copy.deepcopy(specification["graph"]))
        model = _require_msprime().Demography.from_demes(graph)
        labels = specification["sample_demes"]
    truth = simulate_genealogy(config, demography=model, sample_demes=labels)
    variants = tuple(f"site-{int(position)}" for position in truth.site_positions)
    options = dict(value["phenotypes"])
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
            array = np.asarray(options[name], dtype=np.float64)
            options[name] = array.reshape(shape) if array.size == 0 and 0 in shape else array
    for name, cls in (
        ("effects", quantitative.AlleleEffect),
        ("dominance_effects", quantitative.DominanceEffect),
        ("interaction_effects", quantitative.InteractionEffect),
        ("context_effects", quantitative.ContextEffect),
        ("compound_effects", quantitative.CompoundHetEffect),
    ):
        options[name] = tuple(cls(**item) for item in options.get(name, ()))
    resolved = []
    for rule in rules:
        for index, alleles in enumerate(truth.alleles):
            if len(alleles) != 2:
                continue
            copies = int((truth.genotypes[:, index, :] == 1).sum())
            if min(copies, 2 * n - copies) >= rule["minimum_copies"]:
                options["effects"] += (
                    quantitative.AlleleEffect(variants[index], 1, tuple(rule["effects"])),
                )
                resolved.append({**rule, "variant_id": variants[index], "allele_index": 1})
                break
        else:
            raise ValueError("no eligible effect-rule locus; no resimulation or seed replacement")
    factors: list[np.ndarray] = []
    factor_ids = list(options.get("residual_factor_ids", ()))
    existing = options.get("residual_row_factors")
    if existing is not None:
        if existing.shape != (n, q) or not np.all(np.isfinite(existing)):
            raise ValueError("aligned finite explicit residual factors required")
        factors.append(existing)
    elif q:
        raise ValueError("explicit residual factor IDs require factors")
    records = []
    for identifier, variance, settings in validated:
        if 8 * n * (len(factor_ids) + n) > max_output_bytes:
            raise ValueError("combined environmental factor byte bound exceeded")
        environment = genealogical_environment_factors(
            truth.tree_sequence,
            sample_ids=truth.sample_ids,
            sample_node_pairs=truth.sample_node_pairs,
            factor_prefix=identifier,
            **settings,
        )
        scaled = environment.factors * math.sqrt(variance)
        if not np.all(np.isfinite(scaled)):
            raise ValueError("unrepresentable environmental scale")
        factors.append(scaled)
        factor_ids.extend(environment.factor_ids)
        records.append(
            {
                "variance": variance,
                "scientific_signature": environment.scientific_signature,
                "provenance": json.loads(environment.provenance),
            }
        )
    if validated:
        options["residual_factor_ids"] = (
            _identifiers(tuple(factor_ids), "combined residual factor IDs") if factor_ids else ()
        )
        options["residual_row_factors"] = np.column_stack(factors) if factors else np.empty((n, 0))
    receipt = write_association_bundle(
        destination,
        truth,
        variant_ids=variants,
        reference=value["reference"],
        contig=value["contig"],
        trait_units=value["trait_units"],
        source_revision=source_revision,
        phenotype_options=options,
        max_output_bytes=max_output_bytes,
    )
    return {
        "manifest_path": str(receipt.manifest_path.resolve()),
        "manifest_sha256": receipt.manifest_sha256,
        "total_file_bytes": receipt.total_file_bytes,
        "recipe_sha256": hashlib.sha256(canonical).hexdigest(),
        "genealogical_environment": records,
        "resolved_effect_rules": resolved,
    }
