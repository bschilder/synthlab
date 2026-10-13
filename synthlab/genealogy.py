"""Independent, optional genealogy and phased genotype truth for benchmarks.

This is an msprime adapter, not an ARG inference method. It keeps true marginal
genealogies and complete literal-allele genotypes together. Deme labels describe
the simulator and are never prescribed as ancestry classes for an analysis.
The full recombination-event history is not requested; these are true marginal
trees, sufficient for the association operator's oracle comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


def _identifiers(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, set, frozenset, dict)):
        raise ValueError(f"{name} must be an explicit ordered sequence")  # noqa: TRY004 - one input-error contract
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an explicit ordered sequence") from error
    if (
        not result
        or any(
            not isinstance(value, str) or not value or any(ord(c) < 32 for c in value)
            for value in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain nonempty unique IDs without control characters")
    return result


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_real(value: float, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")  # noqa: TRY004 - one input-error contract
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{name} is outside its finite nonnegative/positive domain")
    return result


@dataclass(frozen=True)
class GenealogySimConfig:
    """A discrete genome and explicit diploid people, in zero-based coordinates.

    Rates are per base per generation. Recent DTWF generations precede the
    standard coalescent; zero selects coalescent only. ``population_size`` is
    used only in the single-deme case, not with a supplied demographic model.
    """

    sample_ids: tuple[str, ...]
    sequence_length: int
    seed: int
    population_size: float = 10000
    recombination_rate: float = 1e-8
    mutation_rate: float = 1.2e-8
    dtwf_generations: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", _identifiers(self.sample_ids, "sample_ids"))
        _positive_integer(self.sequence_length, "sequence_length")
        if self.sequence_length > 2**53:
            raise ValueError("sequence_length exceeds exact float64 integer coordinates")
        _positive_integer(self.seed, "seed")
        if self.seed >= 2**63:
            raise ValueError("seed must fit a positive signed 64-bit integer")
        if type(self.dtwf_generations) is not int or self.dtwf_generations < 0:
            raise ValueError("dtwf_generations must be a nonnegative integer")
        for name in ("population_size", "recombination_rate", "mutation_rate"):
            value = _finite_real(getattr(self, name), name, allow_zero=name != "population_size")
            object.__setattr__(self, name, value)


def simulation_stream_seeds(seed: int) -> Mapping[str, int]:
    """Stable named positive uint32 streams, independent of rates and masks.

    SHA256 separates channels; deterministic collision resolution also excludes
    zero, which msprime does not accept. Python's process-randomized hash is not
    used. Returned values, rather than a statistical independence claim about
    arbitrary PRNGs, are recorded in every result.
    """
    _positive_integer(seed, "seed")
    if seed >= 2**63:
        raise ValueError("seed must fit a positive signed 64-bit integer")
    result = {}
    for channel in (
        "ancestry",
        "mutation",
        "phenotype",
        "missingness",
        "genotype_error",
        "phase_error",
    ):
        counter = 0
        while True:
            payload = json.dumps(["synthlab-simulation-seed-v0", seed, channel, counter]).encode()
            value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
            if value and value not in result.values():
                result[channel] = value
                break
            counter += 1
    return MappingProxyType(result)


def _require_msprime() -> Any:
    try:
        import msprime
    except ImportError as error:
        raise ImportError(
            "Genealogy simulation requires the optional synthlab[simulation] extra"
        ) from error
    return msprime


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"simulation provenance contains unsupported {type(value).__name__}")


def _scientific_signature(tree_sequence: Any, specification: str) -> str:
    """Hash canonical table content plus recipe; exclude runtime provenance rows.

    File UUIDs and msprime's provenance timestamps vary between identical runs.
    This logical content hash does not replace a transport file SHA256. Software
    versions are part of the recipe; cross-version bit reproducibility is not
    asserted. Arrays use little-endian bytes with explicit dtype and shape.
    """
    digest = hashlib.sha256(specification.encode())

    def add(value: Any) -> None:
        if isinstance(value, dict):
            digest.update(b"dict" + len(value).to_bytes(8, "little"))
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, np.ndarray):
            canonical = np.ascontiguousarray(
                value.astype(value.dtype.newbyteorder("<"), copy=False)
            )
            digest.update(json.dumps(["array", canonical.dtype.str, canonical.shape]).encode())
            digest.update(canonical.tobytes())
        elif isinstance(value, bytes):
            digest.update(b"bytes" + len(value).to_bytes(8, "little") + value)
        else:
            encoded = json.dumps(
                value, default=_json_default, sort_keys=True, allow_nan=False
            ).encode()
            digest.update(b"json" + len(encoded).to_bytes(8, "little") + encoded)

    tables = tree_sequence.dump_tables().asdict()
    tables.pop("provenances", None)
    add(tables)
    return digest.hexdigest()


def _diploid_sample_pairs(tree_sequence: Any, individual_ids: Sequence[int]) -> np.ndarray:
    """Map two distinct sample nodes through each declared individual ID."""
    expected = tuple(individual_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("individual IDs must be explicit, nonempty and unique")
    by_individual = {individual: [] for individual in expected}
    for node_id in tree_sequence.samples():
        node = tree_sequence.node(int(node_id))
        if node.individual not in by_individual:
            raise ValueError("sample node lacks a declared individual mapping")
        by_individual[node.individual].append(int(node_id))
    if any(len(nodes) != 2 for nodes in by_individual.values()):
        raise ValueError("each declared diploid person must map to exactly two sample nodes")
    return np.asarray([by_individual[individual] for individual in expected], dtype=np.int64)


@dataclass(frozen=True)
class GenealogySimulation:
    """Complete phased truth; no genotype calling errors or missingness applied."""

    tree_sequence: Any
    sample_ids: tuple[str, ...]
    sample_node_pairs: np.ndarray
    site_positions: np.ndarray
    alleles: tuple[tuple[str, ...], ...]
    genotypes: np.ndarray
    stream_seeds: Mapping[str, int]
    scientific_signature: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("sample_node_pairs", "site_positions", "genotypes"):
            array = getattr(self, name).copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "stream_seeds", MappingProxyType(dict(self.stream_seeds)))


def simulate_genealogy(
    config: GenealogySimConfig,
    *,
    demography: Any | None = None,
    sample_demes: Sequence[str] | None = None,
    max_individuals: int = 2048,
    max_sites: int = 100000,
    max_output_bytes: int = 256 * 1024**2,
) -> GenealogySimulation:
    """Generate true marginal trees, then independent finite-allele mutations.

    With a custom msprime Demography, supply one exact population name per
    sample ID. The caller model is copied. One explicit diploid SampleSet per
    person preserves input person order, with each pair extracted through its
    generated individual ID rather than adjacent-node arithmetic.

    Request bounds and a conservative principal output-array check are enforced;
    they cannot bound msprime's internal event tables/allocations before it
    returns. Large simulations require a remote subprocess with separate time
    and process-memory controls. No sample/site downsampling happens here.
    """
    if not isinstance(config, GenealogySimConfig):
        raise ValueError("config must be GenealogySimConfig")  # noqa: TRY004 - one input-error contract
    for value, name in (
        (max_individuals, "max_individuals"),
        (max_sites, "max_sites"),
        (max_output_bytes, "max_output_bytes"),
    ):
        _positive_integer(value, name)
    n = len(config.sample_ids)
    if n > max_individuals or 64 * n > max_output_bytes:
        raise ValueError("simulation individual/output bound exceeded")
    if (demography is None) != (sample_demes is None):
        raise ValueError("demography and exact sample_demes must be supplied together")
    msprime = _require_msprime()
    if demography is None:
        model = msprime.Demography.isolated_model([config.population_size])
        labels = tuple(model.populations[0].name for _ in config.sample_ids)
    else:
        if not isinstance(demography, msprime.Demography) or isinstance(
            sample_demes, (str, bytes, set, frozenset, dict)
        ):
            raise ValueError("custom demography and explicit population-name sequence required")
        labels = tuple(sample_demes)
        model = demography.copy()
        if len(labels) != n or any(
            not isinstance(label, str) or label not in {p.name for p in model.populations}
            for label in labels
        ):
            raise ValueError("sample_demes must match every person with a known population name")
    model.validate()
    seeds = simulation_stream_seeds(config.seed)
    models = [msprime.StandardCoalescent()]
    if config.dtwf_generations:
        models.insert(0, msprime.DiscreteTimeWrightFisher(duration=config.dtwf_generations))
    provenance = json.dumps(
        {
            "schema_version": 0,
            "generator": "msprime oracle marginal genealogies and phased genotypes",
            "config": asdict(config),
            "single_deme_population_size_used": demography is None,
            "demography": model.asdict(),
            "sample_demes": labels,
            "models": [type(value).__name__ for value in models],
            "mutation_model": "JC69",
            "discrete_genome": True,
            "record_full_arg": False,
            "time_units": "generations",
            "reference_convention": "simulated ancestral allele is REF; not a human reference",
            "stream_seeds": dict(seeds),
            "versions": {name: version(name) for name in ("msprime", "tskit", "numpy")},
            "genotypes": "complete true haplotype allele indices; no observation contamination",
        },
        default=_json_default,
        sort_keys=True,
        allow_nan=False,
    )
    ancestry = msprime.sim_ancestry(
        samples=[msprime.SampleSet(1, population=label, ploidy=2) for label in labels],
        demography=model,
        sequence_length=config.sequence_length,
        discrete_genome=True,
        recombination_rate=config.recombination_rate,
        model=models,
        ploidy=2,
        random_seed=seeds["ancestry"],
        record_full_arg=False,
    )
    ts = msprime.sim_mutations(
        ancestry,
        rate=config.mutation_rate,
        random_seed=seeds["mutation"],
        model=msprime.JC69(),
        discrete_genome=True,
    )
    if ts.time_units != "generations" or ts.num_individuals != n:
        raise ValueError("simulator did not preserve declared time units/diploid individuals")
    count = ts.num_sites
    required = 4 * ts.nbytes + 8 * n * count + 64 * (count + n)
    if count > max_sites or required > max_output_bytes:
        raise ValueError("simulation site/output bound exceeded; no silent downsampling")
    pairs = _diploid_sample_pairs(ts, range(n))
    population_ids = {population.name: index for index, population in enumerate(model.populations)}
    for label, pair in zip(labels, pairs):
        if any(ts.node(int(node)).population != population_ids[label] for node in pair):
            raise ValueError("simulator person order differs from the declared sample-deme join")
    positions = np.asarray(ts.sites_position, dtype=np.uint64)
    if np.any(ts.sites_position != positions) or np.any(positions >= config.sequence_length):
        raise ValueError("oracle sites must be exact discrete zero-based coordinates")
    genotypes = np.empty((n, count, 2), dtype=np.int8)
    alleles = []
    for col, variant in enumerate(ts.variants(samples=pairs.reshape(-1).tolist())):
        vocabulary = tuple(variant.alleles)
        if any(allele not in ("A", "C", "G", "T") for allele in vocabulary):
            raise ValueError("oracle JC69 allele vocabulary contains unknown/nonliteral states")
        if np.any(variant.genotypes < 0) or np.any(variant.genotypes >= len(vocabulary)):
            raise ValueError(
                "oracle haplotypes unexpectedly have missing or invalid allele indices"
            )
        alleles.append(vocabulary)
        genotypes[:, col, :] = variant.genotypes.reshape(n, 2)
    return GenealogySimulation(
        ts,
        config.sample_ids,
        pairs,
        positions,
        tuple(alleles),
        genotypes,
        seeds,
        _scientific_signature(ts, provenance),
        provenance,
    )
