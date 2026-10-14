"""Known quantitative genotype/phenotype mechanisms with separate observation masks.

Inputs are complete diploid genotype truth, including exact multiallelic allele
indices. Outputs keep genetic mechanisms, measured/unmeasured nuisance means,
independent/shared environmental residuals, complete outcomes and recorded cells
separate. No association method, ancestry classifier or imputer is fitted here.
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

from .genealogy import _identifiers, _positive_integer


def _ordered(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, set, frozenset, dict)):
        raise ValueError(f"{name} must be an explicit ordered sequence")  # noqa: TRY004
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an explicit ordered sequence") from error


def _ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = _ordered(values, name)
    return () if allow_empty and not result else _identifiers(result, name)


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite in float64")
    return result


def _allele_index(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("allele_index must be a nonnegative integer")


def _effect_vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(_finite_number(value, "effect") for value in _ordered(values, "effects"))
    if not result:
        raise ValueError("effect vector must have at least one trait")
    return result


@dataclass(frozen=True)
class AlleleEffect:
    """Signed effect per exact allele copy, one coefficient per declared trait."""

    variant_id: str
    allele_index: int
    effects: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifiers((self.variant_id,), "variant_id")
        _allele_index(self.allele_index)
        object.__setattr__(self, "effects", _effect_vector(self.effects))


@dataclass(frozen=True)
class DominanceEffect(AlleleEffect):
    """Effect when exactly one homologue carries the named allele.

    This is an explicit one-allele-versus-all-other-alleles heterozygote coding,
    including at multiallelic sites, rather than an unspecified dominance model.
    """


@dataclass(frozen=True)
class InteractionEffect:
    """Effect times (dosage1-center1)*(dosage2-center2), with caller-fixed centers."""

    first_variant_id: str
    first_allele_index: int
    second_variant_id: str
    second_allele_index: int
    effects: tuple[float, ...]
    first_center: float = 0.0
    second_center: float = 0.0

    def __post_init__(self) -> None:
        for name in ("first_variant_id", "second_variant_id"):
            _identifiers((getattr(self, name),), name)
        _allele_index(self.first_allele_index)
        _allele_index(self.second_allele_index)
        object.__setattr__(self, "effects", _effect_vector(self.effects))
        for name in ("first_center", "second_center"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))


@dataclass(frozen=True)
class ContextEffect:
    """Allele dosage times a declared continuous context deviation."""

    variant_id: str
    allele_index: int
    context_id: str
    effects: tuple[float, ...]
    context_center: float = 0.0

    def __post_init__(self) -> None:
        _identifiers((self.variant_id,), "variant_id")
        _allele_index(self.allele_index)
        _identifiers((self.context_id,), "context_id")
        object.__setattr__(self, "effects", _effect_vector(self.effects))
        object.__setattr__(
            self, "context_center", _finite_number(self.context_center, "context_center")
        )


@dataclass(frozen=True)
class CompoundHetEffect:
    """Two distinct nonreference alleles, each carried once, on opposite homologues.

    The two alleles may be at different sites or at the same multiallelic site.
    This contrast excludes homozygous and cis two-allele configurations; broader
    recessive mechanisms need their own explicitly declared effect coding.
    """

    first_variant_id: str
    first_allele_index: int
    second_variant_id: str
    second_allele_index: int
    effects: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("first_variant_id", "second_variant_id"):
            _identifiers((getattr(self, name),), name)
        for allele in (self.first_allele_index, self.second_allele_index):
            _allele_index(allele)
            if allele == 0:
                raise ValueError("compound heterozygote alleles must be nonreference")
        if (self.first_variant_id, self.first_allele_index) == (
            self.second_variant_id,
            self.second_allele_index,
        ):
            raise ValueError("compound heterozygote alleles must be distinct")
        object.__setattr__(self, "effects", _effect_vector(self.effects))


def _array(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or values.shape != shape
        or values.dtype.kind not in "fiu"
    ):
        raise ValueError(f"{name} must be a real array with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and representable in float64")
    return result


def _covariance_factor(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Factor a covariance, declaring any numerical-rank projection explicitly."""
    p = len(values)
    tolerance = 64 * np.finfo(np.float64).eps * p
    variances = np.diag(values)
    if np.any(variances < 0):
        raise ValueError("residual_covariance must have nonnegative variances")
    positive = variances > 0
    if np.any(values[~positive, :] != 0) or np.any(values[:, ~positive] != 0):
        raise ValueError("residual_covariance zero-variance rows/columns must be zero")
    scale = np.sqrt(variances[positive])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correlation = values[np.ix_(positive, positive)] / scale[:, None] / scale[None, :]
    if not np.all(np.isfinite(correlation)) or np.any(np.abs(correlation) > 1 + tolerance):
        raise ValueError("residual_covariance implies an invalid correlation")
    if correlation.size and np.max(np.abs(correlation - correlation.T)) > tolerance:
        raise ValueError("residual_covariance must be symmetric")
    symmetric = correlation * 0.5 + correlation.T * 0.5
    try:
        normalized_factor = np.linalg.cholesky(symmetric)
        method, rank = "Cholesky on positive-variance correlation matrix", len(scale)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        if eigenvalues[0] < -tolerance:
            raise ValueError("residual_covariance must be positive semidefinite") from None
        keep = eigenvalues > 0
        normalized_factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        method, rank = (
            "eigenfactor; negative roundoff omitted; every positive eigenvalue retained",
            int(keep.sum()),
        )
    factor = np.zeros((p, rank))
    factor[positive, :] = normalized_factor * scale[:, None]
    sampling = factor @ factor.T
    return (
        factor,
        sampling,
        {
            "method": method,
            "symmetrization": "(correlation+correlation.T)/2 before restoring caller units",
            "roundoff_tolerance": tolerance,
            "tolerance_units": "dimensionless correlation scale",
            "zero_variance_policy": "all covariances with a zero-variance trait must be exactly zero",
            "sampling_rank": rank,
            "maximum_requested_sampling_difference": float(np.max(np.abs(values - sampling))),
            "jitter_added": False,
        },
    )


def quantitative_stream_seeds(seed: int) -> Mapping[str, int]:
    """Named positive uint128 PCG64 seeds from the original signed64 master.

    Avoid the uint32 intermediary required by msprime's separate tree streams.
    Expanding a repeated uint32 parent cannot restore independent state keys.
    Derivation depends on master, versioned namespace and role only; permutation
    and observation-probability changes do not change these keys. Recorded keys
    support reproducible pseudorandom scheduling, not an independence theorem.
    """
    _positive_integer(seed, "seed")
    if seed >= 2**63:
        raise ValueError("seed must fit a positive signed 64-bit integer")
    result = {}
    for channel in ("residual_iid", "residual_shared", "missingness"):
        counter = 0
        while True:
            payload = json.dumps(
                ["quantitative-pcg64-128-v1", seed, channel, counter], separators=(",", ":")
            ).encode()
            value = int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")
            if value and value not in result.values():
                result[channel] = value
                break
            counter += 1
    return MappingProxyType(result)


def _hash_array(digest: Any, name: str, values: np.ndarray, *, row_block: int) -> None:
    dtype = "<i8" if values.dtype.kind in "iu" else ("|b1" if values.dtype.kind == "b" else "<f8")
    header = json.dumps([name, dtype, values.shape]).encode()
    digest.update(len(header).to_bytes(8, "little") + header)
    digest.update((values.size * np.dtype(dtype).itemsize).to_bytes(8, "little"))
    for start in range(0, len(values), row_block):
        block = np.ascontiguousarray(values[start : start + row_block], dtype=dtype)
        digest.update(block.tobytes())


@dataclass(frozen=True)
class QuantitativePhenotypeSimulation:
    """Complete truth and recorded cells, in exact caller person/trait order."""

    sample_ids: tuple[str, ...]
    trait_ids: tuple[str, ...]
    variant_ids: tuple[str, ...]
    covariate_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    residual_factor_ids: tuple[str, ...]
    additive_means: np.ndarray
    dominance_means: np.ndarray
    interaction_means: np.ndarray
    context_means: np.ndarray
    compound_means: np.ndarray
    genetic_means: np.ndarray
    fixed_means: np.ndarray
    environment_means: np.ndarray
    nuisance_means: np.ndarray
    independent_residuals: np.ndarray
    structured_residuals: np.ndarray
    residuals: np.ndarray
    shared_factor_values: np.ndarray
    requested_residual_covariance: np.ndarray
    sampling_covariance: np.ndarray
    complete_values: np.ndarray
    observation_probabilities: np.ndarray
    observed: np.ndarray
    observed_values: np.ndarray
    stream_seeds: Mapping[str, int]
    scientific_signature: str
    provenance: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                owned = value.copy()
                owned.flags.writeable = False
                object.__setattr__(self, name, owned)
        object.__setattr__(self, "stream_seeds", MappingProxyType(dict(self.stream_seeds)))


def simulate_quantitative_phenotypes(
    sample_ids: Sequence[str],
    variant_ids: Sequence[str],
    alleles: Sequence[Sequence[str]],
    genotypes: np.ndarray,
    *,
    trait_ids: Sequence[str],
    effects: Sequence[AlleleEffect],
    covariate_ids: Sequence[str],
    covariates: np.ndarray,
    covariate_effects: np.ndarray,
    residual_covariance: np.ndarray,
    seed: int,
    dominance_effects: Sequence[DominanceEffect] = (),
    interaction_effects: Sequence[InteractionEffect] = (),
    context_ids: Sequence[str] = (),
    contexts: np.ndarray | None = None,
    context_effects: Sequence[ContextEffect] = (),
    compound_effects: Sequence[CompoundHetEffect] = (),
    environment_means: np.ndarray | None = None,
    residual_factor_ids: Sequence[str] = (),
    residual_row_factors: np.ndarray | None = None,
    observed_probabilities: np.ndarray | None = None,
    missingness_description: str | None = None,
    max_traits: int = 256,
    max_work: int = 1_000_000_000,
    max_memory_bytes: int = 256 * 1024**2,
) -> QuantitativePhenotypeSimulation:
    """Generate signed mechanisms plus Gaussian outcomes without deleting rows.

    Genotypes are [person, variant, homologue] complete integer allele indices.
    Caller order is preserved; effect vectors follow trait_ids. No intercept,
    scale transform, allele frequency or effect center is inferred. Columns of
    covariates, contexts and residual factors have exact explicit IDs.

    Residuals have covariance sampling_covariance tensor (I + A A.T), where A
    is the supplied residual_row_factors. This explicit low-rank environmental
    structure is independent of genetic effects. The actual numerical sampling
    covariance, including any declared roundoff projection of a PSD boundary,
    is returned alongside the requested covariance; no jitter is added.

    IID, shared-factor and observation-mask RNGs are separate. Canonically sorted
    person/trait/factor IDs make row/trait/factor permutations preserve individual
    realizations. Changes to the cohort or trait set need not preserve draws.
    Observed probabilities require a description; MAR/MNAR can be supplied using
    two deterministic passes with fixed seed. Masks are generation mechanisms,
    not corrections for informative missingness.

    Principal array preflight excludes resident caller arrays and metadata/BLAS
    allocator overhead. Genotype validation and hashing are row-blocked. These
    bounds and this reference generator do not establish biobank throughput.
    """
    people = _ids(sample_ids, "sample_ids")
    variants = _ids(variant_ids, "variant_ids", allow_empty=True)
    traits = _ids(trait_ids, "trait_ids")
    covariate_names = _ids(covariate_ids, "covariate_ids", allow_empty=True)
    context_names = _ids(context_ids, "context_ids", allow_empty=True)
    factor_names = _ids(residual_factor_ids, "residual_factor_ids", allow_empty=True)
    vocabularies = tuple(_ids(value, "alleles") for value in _ordered(alleles, "alleles"))
    n, v, p = len(people), len(variants), len(traits)
    if len(vocabularies) != v:
        raise ValueError("allele vocabulary must match every ordered variant")
    if (
        not isinstance(genotypes, np.ndarray)
        or genotypes.shape != (n, v, 2)
        or genotypes.dtype.kind not in "iu"
    ):
        raise ValueError("genotypes must be complete integer [person,variant,homologue] truth")
    for value, name in (
        (max_traits, "max_traits"),
        (max_work, "max_work"),
        (max_memory_bytes, "max_memory_bytes"),
    ):
        _positive_integer(value, name)
    if p > max_traits:
        raise ValueError("trait bound exceeded")
    families = (
        ("additive", _ordered(effects, "effects"), AlleleEffect),
        ("dominance", _ordered(dominance_effects, "dominance_effects"), DominanceEffect),
        ("interaction", _ordered(interaction_effects, "interaction_effects"), InteractionEffect),
        ("context", _ordered(context_effects, "context_effects"), ContextEffect),
        ("compound", _ordered(compound_effects, "compound_effects"), CompoundHetEffect),
    )
    total_effects = sum(len(family) for _, family, _ in families)
    k, b, q = len(covariate_names), len(context_names), len(factor_names)
    work = n * (2 * v + 8 * p * total_effects + p * (k + b + q + p)) + q * p * p
    if work > max_work:
        raise ValueError("simulation work bound exceeded")
    required = 512 * n * p + 32 * n * (k + b + q) + 32 * p * (k + q) + 128 * p * p + 64 * max(v, 1)
    if required > max_memory_bytes:
        raise ValueError("simulation principal array memory bound exceeded")
    row_block = max(1, min(n, (max_memory_bytes - required) // (64 * max(v, 1)) + 1))
    limits = np.asarray([len(value) for value in vocabularies], dtype=np.int64)
    for start in range(0, n, row_block):
        block = genotypes[start : start + row_block]
        if np.any(block < 0) or np.any(block >= limits[None, :, None]):
            raise ValueError("genotype truth contains missing or out-of-vocabulary allele indices")
    x = _array(covariates, (n, k), "covariates")
    coefficients = _array(covariate_effects, (k, p), "covariate_effects")
    z = _array(np.empty((n, 0)) if contexts is None else contexts, (n, b), "contexts")
    a = _array(
        np.empty((n, 0)) if residual_row_factors is None else residual_row_factors,
        (n, q),
        "residual_row_factors",
    )
    environment = _array(
        np.zeros((n, p)) if environment_means is None else environment_means,
        (n, p),
        "environment_means",
    )
    requested_covariance = _array(residual_covariance, (p, p), "residual_covariance")
    if observed_probabilities is None:
        probabilities = np.ones((n, p))
        description = "all observed" if missingness_description is None else missingness_description
    else:
        probabilities = _array(observed_probabilities, (n, p), "observed_probabilities")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("observed_probabilities must lie between zero and one")
        if missingness_description is None:
            raise ValueError("supplied observation probabilities require a missingness_description")
        description = missingness_description
    _identifiers((description,), "missingness_description")
    lookup = {value: index for index, value in enumerate(variants)}
    context_lookup = {value: index for index, value in enumerate(context_names)}

    def dosage(variant_id: str, allele: int) -> np.ndarray:
        if variant_id not in lookup or allele >= len(vocabularies[lookup[variant_id]]):
            raise ValueError("effect refers to an unknown variant or allele")
        return (genotypes[:, lookup[variant_id], :] == allele).sum(axis=1).astype(float)

    means: dict[str, np.ndarray] = {}
    records = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name, family, expected_type in families:
            values = np.zeros((n, p))
            seen = set()
            for effect in family:
                if type(effect) is not expected_type or len(effect.effects) != p:
                    raise ValueError(
                        "each effect must match its declared mechanism and trait dimension"
                    )
                record = asdict(effect)
                key = json.dumps(
                    {key: value for key, value in record.items() if key != "effects"},
                    sort_keys=True,
                )
                if key in seen:
                    raise ValueError("duplicate effect feature within a mechanism family")
                seen.add(key)
                if name in ("additive", "dominance", "context"):
                    feature = dosage(effect.variant_id, effect.allele_index)
                    if name == "dominance":
                        feature = (feature == 1).astype(float)
                    elif name == "context":
                        if effect.context_id not in context_lookup:
                            raise ValueError("context effect refers to an unknown context ID")
                        feature *= z[:, context_lookup[effect.context_id]] - effect.context_center
                else:
                    first = dosage(effect.first_variant_id, effect.first_allele_index)
                    second = dosage(effect.second_variant_id, effect.second_allele_index)
                    if name == "interaction":
                        feature = (first - effect.first_center) * (second - effect.second_center)
                    else:
                        left = genotypes[:, lookup[effect.first_variant_id], :]
                        right = genotypes[:, lookup[effect.second_variant_id], :]
                        trans = (
                            (left[:, 0] == effect.first_allele_index)
                            & (right[:, 1] == effect.second_allele_index)
                        ) | (
                            (left[:, 1] == effect.first_allele_index)
                            & (right[:, 0] == effect.second_allele_index)
                        )
                        feature = ((first == 1) & (second == 1) & trans).astype(float)
                values += feature[:, None] * np.asarray(effect.effects)[None, :]
            means[name] = values
            records[name] = [asdict(effect) for effect in family]
        genetic = sum(means.values(), start=np.zeros((n, p)))
        fixed = x @ coefficients
        nuisance = fixed + environment
    seeds = quantitative_stream_seeds(seed)
    row_order = np.asarray(sorted(range(n), key=people.__getitem__), dtype=np.int64)
    trait_order = np.asarray(sorted(range(p), key=traits.__getitem__), dtype=np.int64)
    factor_order = np.asarray(sorted(range(q), key=factor_names.__getitem__), dtype=np.int64)
    factor, canonical_covariance, covariance_checks = _covariance_factor(
        requested_covariance[np.ix_(trait_order, trait_order)]
    )
    independent = np.empty((n, p))
    iid_draws = np.random.Generator(np.random.PCG64(seeds["residual_iid"])).standard_normal(
        (n, factor.shape[1])
    )
    independent[np.ix_(row_order, trait_order)] = iid_draws @ factor.T
    shared_values = np.empty((q, p))
    shared_draws = np.random.Generator(np.random.PCG64(seeds["residual_shared"])).standard_normal(
        (q, factor.shape[1])
    )
    shared_values[np.ix_(factor_order, trait_order)] = shared_draws @ factor.T
    sampling = np.empty((p, p))
    sampling[np.ix_(trait_order, trait_order)] = canonical_covariance
    with np.errstate(over="ignore", invalid="ignore"):
        structured = a @ shared_values
        residuals = independent + structured
        complete = genetic + nuisance + residuals
    for value in (
        *means.values(),
        genetic,
        fixed,
        nuisance,
        independent,
        shared_values,
        structured,
        residuals,
        complete,
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError("simulated means/outcomes must remain finite in float64")
    uniforms = np.random.Generator(np.random.PCG64(seeds["missingness"])).random((n, p))
    observed = np.empty((n, p), dtype=bool)
    observed[np.ix_(row_order, trait_order)] = (
        uniforms < probabilities[np.ix_(row_order, trait_order)]
    )
    recorded = np.where(observed, complete, np.nan)
    provenance = json.dumps(
        {
            "schema_version": 0,
            "generator": "known quantitative genotype mechanisms and Gaussian residuals",
            "sample_ids": people,
            "trait_ids": traits,
            "variant_ids": variants,
            "alleles": vocabularies,
            "covariate_ids": covariate_names,
            "context_ids": context_names,
            "residual_factor_ids": factor_names,
            "mechanisms": records,
            "genotype_axes": ["person", "variant", "homologue"],
            "effects": "signed caller units; no automatic scaling, intercept or frequency estimation",
            "residual_model": "sampling_covariance tensor (I + A A.T); environmental, independent of genetic mean",
            "covariance_factorization": covariance_checks,
            "seed": seed,
            "rng_scheme": "quantitative-pcg64-128-v1",
            "stream_seeds": {name: f"0x{value:032x}" for name, value in seeds.items()},
            "stream_seed_encoding": "lossless hexadecimal uint128; Python result retains integer seed values",
            "rng": "NumPy PCG64; canonical person,trait,factor order; separate iid/shared/mask streams",
            "missingness_description": description,
            "numpy_version": version("numpy"),
        },
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(provenance.encode())
    arrays = {
        "genotypes": genotypes,
        "covariates": x,
        "covariate_effects": coefficients,
        "contexts": z,
        "environment_means": environment,
        "residual_row_factors": a,
        "requested_covariance": requested_covariance,
        "sampling_covariance": sampling,
        "probabilities": probabilities,
        "complete_values": complete,
        "observed": observed,
        "observed_values": recorded,
    }
    for name, value in arrays.items():
        _hash_array(digest, name, value, row_block=row_block)
    return QuantitativePhenotypeSimulation(
        people,
        traits,
        variants,
        covariate_names,
        context_names,
        factor_names,
        means["additive"],
        means["dominance"],
        means["interaction"],
        means["context"],
        means["compound"],
        genetic,
        fixed,
        environment,
        nuisance,
        independent,
        structured,
        residuals,
        shared_values,
        requested_covariance,
        sampling,
        complete,
        probabilities,
        observed,
        recorded,
        seeds,
        digest.hexdigest(),
        provenance,
    )
