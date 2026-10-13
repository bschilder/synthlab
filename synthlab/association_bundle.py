"""Verified oracle inputs for external association benchmarks.

The writer generates phenotypes directly from the verified genealogy's complete
diploid truth. It publishes a manifest last, with transport checksums separate
from logical simulation signatures. It neither imports nor fits an association
library. This module belongs in the external simulation environment.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .genealogy import (
    GenealogySimulation,
    _diploid_sample_pairs,
    _identifiers,
    _positive_integer,
    _scientific_signature,
)
from .genotype_phenotypes import simulate_quantitative_phenotypes


@dataclass(frozen=True)
class AssociationBundleReceipt:
    """Transport pins to preserve separately from the emitted bundle."""

    manifest_path: Path
    manifest_sha256: str
    total_file_bytes: int


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _verify_truth(truth: GenealogySimulation, max_genotype_checks: int) -> None:
    if not isinstance(truth, GenealogySimulation):
        raise ValueError("genealogy must be a GenealogySimulation")  # noqa: TRY004
    ts = truth.tree_sequence
    n, v = len(truth.sample_ids), len(truth.alleles)
    _identifiers(truth.sample_ids, "sample_ids")
    specification = json.loads(truth.provenance)
    if specification["config"]["sample_ids"] != list(truth.sample_ids):
        raise ValueError("genealogy person IDs disagree with original simulation recipe")
    if (
        ts.time_units != "generations"
        or ts.num_individuals != n
        or ts.num_sites != v
        or ts.sequence_length != specification["config"]["sequence_length"]
        or truth.genotypes.shape != (n, v, 2)
        or truth.genotypes.dtype.kind not in "iu"
        or truth.sample_node_pairs.shape != (n, 2)
        or truth.sample_node_pairs.dtype.kind not in "iu"
        or truth.site_positions.shape != (v,)
        or truth.site_positions.dtype.kind not in "iu"
    ):
        raise ValueError("genealogy truth dimensions, coordinates or units disagree")
    if n * v > max_genotype_checks:
        raise ValueError("genealogy genotype verification work bound exceeded")
    if not np.array_equal(truth.sample_node_pairs, _diploid_sample_pairs(ts, range(n))):
        raise ValueError("genealogy person-to-node join differs from original individual order")
    if not np.array_equal(truth.site_positions, ts.sites_position):
        raise ValueError("genealogy site positions disagree with tree sequence")
    if _scientific_signature(ts, truth.provenance) != truth.scientific_signature:
        raise ValueError("genealogy logical signature disagrees with tree sequence/recipe")
    for column, variant in enumerate(ts.variants(samples=truth.sample_node_pairs.ravel().tolist())):
        if tuple(variant.alleles) != truth.alleles[column] or not np.array_equal(
            variant.genotypes.reshape(n, 2), truth.genotypes[:, column, :]
        ):
            raise ValueError("genealogy literal alleles/genotypes disagree with tree sequence")


def write_association_bundle(
    destination: str | Path,
    genealogy: GenealogySimulation,
    *,
    variant_ids: Sequence[str],
    reference: str,
    contig: str,
    trait_units: Sequence[str],
    source_revision: str,
    phenotype_options: Mapping[str, Any],
    max_output_bytes: int = 256 * 1024**2,
    max_manifest_bytes: int = 16 * 1024**2,
    max_genotype_checks: int = 1_000_000,
) -> AssociationBundleReceipt:
    """Generate and emit joined truth into a new, exclusively created directory.

    ``phenotype_options`` supplies the quantitative generator's keyword options;
    the four person/variant/allele/genotype join arguments cannot be overridden.
    Exact variant IDs follow the genealogy's site order. Trait units are explicit
    and follow trait_ids. Source revision is the external generator's Git SHA,
    recorded as provenance rather than an authenticity certificate.

    All arrays are uncompressed .npy with pickle disabled; the original .trees
    preserves tables/provenance. A JSON manifest pins shape, dtype, byte count
    and SHA256 for every file, plus ordered IDs, conventions and simulation
    recipes. Publish the returned manifest digest separately as the reader's
    trust pin. Checksums detect corruption, not an untrusted producer's lies.

    The manifest is published last. Failure removes only the fresh directory
    created by this call; existing paths are never overwritten. Output byte and
    verification-work limits are separate from phenotype memory/work preflights.
    """
    for value, name in (
        (max_output_bytes, "max_output_bytes"),
        (max_manifest_bytes, "max_manifest_bytes"),
        (max_genotype_checks, "max_genotype_checks"),
    ):
        _positive_integer(value, name)
    _identifiers((reference,), "reference")
    _identifiers((contig,), "contig")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise ValueError("source_revision must be an exact lowercase 40-character Git SHA")
    if isinstance(variant_ids, (str, bytes, set, frozenset, dict)):
        raise ValueError("variant_ids must be ordered")  # noqa: TRY004
    variants = tuple(variant_ids)
    if variants:
        _identifiers(variants, "variant_ids")
    if len(variants) != len(genealogy.alleles):
        raise ValueError("variant IDs must match genealogy sites exactly")
    if not isinstance(phenotype_options, Mapping):
        raise ValueError("phenotype_options must be a keyword mapping")  # noqa: TRY004
    options = dict(phenotype_options)
    if {"sample_ids", "variant_ids", "alleles", "genotypes"} & options.keys():
        raise ValueError("phenotype options cannot override the genotype truth join")
    if isinstance(trait_units, (str, bytes, set, frozenset, dict)):
        raise ValueError("trait_units must be ordered")  # noqa: TRY004
    units = tuple(trait_units)
    for unit in units:
        _identifiers((unit,), "trait unit")
    if genealogy.tree_sequence.nbytes > max_output_bytes:
        raise ValueError("association bundle byte bound exceeded")
    _verify_truth(genealogy, max_genotype_checks)
    phenotypes = simulate_quantitative_phenotypes(
        genealogy.sample_ids,
        variants,
        genealogy.alleles,
        genealogy.genotypes,
        **options,
    )
    if len(units) != len(phenotypes.trait_ids):
        raise ValueError("trait units must match ordered traits")
    n = len(genealogy.sample_ids)
    arrays = {
        "sample_node_pairs": genealogy.sample_node_pairs,
        "site_positions": genealogy.site_positions,
        "genotypes": genealogy.genotypes,
        "covariates": np.asarray(options["covariates"], dtype=np.float64),
        "covariate_effects": np.asarray(options["covariate_effects"], dtype=np.float64),
        "contexts": np.asarray(options["contexts"], dtype=np.float64)
        if options.get("contexts") is not None
        else np.empty((n, 0)),
        "residual_row_factors": np.asarray(options["residual_row_factors"], dtype=np.float64)
        if options.get("residual_row_factors") is not None
        else np.empty((n, 0)),
    }
    arrays.update(
        {
            name: getattr(phenotypes, name)
            for name in phenotypes.__dataclass_fields__
            if isinstance(getattr(phenotypes, name), np.ndarray)
        }
    )
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    try:
        tree_path = destination / "truth.trees"
        genealogy.tree_sequence.dump(tree_path)
        files = {"tree_sequence": _file_record(tree_path)}
        size = files["tree_sequence"]["bytes"]
        for name, values in arrays.items():
            if size + values.nbytes + 4096 > max_output_bytes:
                raise ValueError("association bundle byte bound exceeded")
            path = destination / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            files[name] = {
                **_file_record(path),
                "dtype": values.dtype.str,
                "shape": list(values.shape),
            }
            size += files[name]["bytes"]
        manifest = {
            "schema_version": 0,
            "format": "synthlab.association-bundle",
            "source_revision": source_revision,
            "reference": reference,
            "contig": contig,
            "coordinate_offset": 0,
            "sequence_length": genealogy.tree_sequence.sequence_length,
            "time_units": "generations",
            "phase": "complete true simulated homologues",
            "genotype_axes": ["person", "variant", "homologue"],
            "sample_ids": list(genealogy.sample_ids),
            "variant_ids": list(variants),
            "alleles": [list(value) for value in genealogy.alleles],
            "trait_ids": list(phenotypes.trait_ids),
            "trait_units": list(units),
            "covariate_ids": list(phenotypes.covariate_ids),
            "context_ids": list(phenotypes.context_ids),
            "residual_factor_ids": list(phenotypes.residual_factor_ids),
            "genealogy": {
                "scientific_signature": genealogy.scientific_signature,
                "provenance": json.loads(genealogy.provenance),
            },
            "phenotypes": {
                "scientific_signature": phenotypes.scientific_signature,
                "provenance": json.loads(phenotypes.provenance),
            },
            "files": files,
        }
        raw = (json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
        if len(raw) > max_manifest_bytes or size + len(raw) > max_output_bytes:
            raise ValueError("association bundle manifest/output byte bound exceeded")
        partial = destination / "manifest.json.partial"
        partial.write_bytes(raw)
        path = destination / "manifest.json"
        partial.rename(path)
        return AssociationBundleReceipt(path, hashlib.sha256(raw).hexdigest(), size + len(raw))
    except BaseException:
        shutil.rmtree(destination)
        raise
