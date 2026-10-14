"""Independent small-panel genealogy-correlated Gaussian environment oracle.

This explicit descendant-column implementation belongs to the external
simulation environment. It imports no CuEvo code. It is a bounded dense truth
reference, not a biobank constructor or an estimate of causal heritability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from typing import Any, Literal

import numpy as np

from .genealogy import _finite_real, _identifiers, _positive_integer


@dataclass(frozen=True)
class GenealogicalEnvironmentTruth:
    """Owned factor B and its actual numerical covariance B B.T in person order."""

    sample_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    factors: np.ndarray
    covariance: np.ndarray
    raw_mean_diagonal: float
    scientific_signature: str
    provenance: str

    def __post_init__(self) -> None:
        for name in ("factors", "covariance"):
            array = getattr(self, name).copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)


def genealogical_environment_factors(
    tree_sequence: Any,
    *,
    sample_ids: Sequence[str],
    sample_node_pairs: np.ndarray,
    lower_time: float = 0,
    upper_time: float = math.inf,
    center: bool = True,
    normalization: Literal["none", "mean_trace"] = "mean_trace",
    factor_prefix: str = "genealogy",
    max_samples: int = 2048,
    max_branch_features: int = 1_000_000,
    max_work: int = 1_000_000_000,
    max_memory_bytes: int = 256 * 1024**2,
) -> GenealogicalEnvironmentTruth:
    """Compress explicit span/time-weighted branch carriers without eigen repair.

    A branch column is sqrt(span/L * clipped_branch_time) times the fraction
    (0, 1/2 or 1) of each person's declared haplotypes below that branch.
    Optional centering uses all declared people, never phenotype availability.
    QR of the transposed feature matrix returns B=R.T, preserving Phi Phi.T
    with at most n factor columns even when the centered covariance is singular.
    Positive mean-trace normalization divides B by its pre-normalization RMS.

    Input row order is preserved. QR uses canonical person-ID order internally
    so row permutations preserve factor realizations attached to explicit IDs.
    Factors are compressed basis coordinates, not interpreted ancestral nodes.
    Reference-only tree samples remain in the graph without becoming trait rows.

    Principal-array bounds include descendant columns, QR/factor/covariance
    arrays and tree workspaces, excluding resident tree tables and library
    allocator overhead. Work uses node/sample visits plus 2*n*n*branches+2*n^3
    as a conservative scalar-operation proxy, not a wall-clock guarantee.
    Empty bands return zero factors only with explicit normalization='none'.
    """
    try:
        import tskit
    except ImportError as error:
        raise ImportError("Genealogical covariance requires synthlab[simulation]") from error
    if not isinstance(tree_sequence, tskit.TreeSequence):
        raise TypeError("tree_sequence must be an actual tskit TreeSequence")
    ts = tree_sequence
    ids = _identifiers(sample_ids, "sample_ids")
    prefix = _identifiers((factor_prefix,), "factor_prefix")[0]
    for value, name in (
        (max_samples, "max_samples"),
        (max_branch_features, "max_branch_features"),
        (max_work, "max_work"),
        (max_memory_bytes, "max_memory_bytes"),
    ):
        _positive_integer(value, name)
    n = len(ids)
    if n > max_samples:
        raise ValueError("dense truth sample bound exceeded")
    if type(center) is not bool or normalization not in ("none", "mean_trace"):
        raise ValueError("explicit boolean centering and known normalization required")
    lower = _finite_real(lower_time, "lower_time", allow_zero=True)
    if isinstance(upper_time, bool) or not isinstance(upper_time, Real):
        raise TypeError("upper_time must be finite or positive infinity")
    upper = float(upper_time)
    if math.isnan(upper) or upper <= lower:
        raise ValueError("time band must be strictly ordered")
    if ts.time_units in ("", "unknown"):
        raise ValueError("known explicit tree time units required")
    span = float(ts.sequence_length)
    if not math.isfinite(span) or span <= 0:
        raise ValueError("positive finite genomic span required")
    if (
        not isinstance(sample_node_pairs, np.ndarray)
        or sample_node_pairs.shape != (n, 2)
        or sample_node_pairs.dtype.kind not in "iu"
        or np.any(sample_node_pairs < 0)
        or np.any(sample_node_pairs >= ts.num_nodes)
        or len(np.unique(sample_node_pairs)) != 2 * n
    ):
        raise ValueError("unique aligned diploid sample-node pairs required")
    if any(not ts.node(int(node)).is_sample() for node in sample_node_pairs.reshape(-1)):
        raise ValueError("every mapped node must be a tree sample")
    order = np.array(sorted(range(n), key=ids.__getitem__), dtype=np.int64)
    inverse_order = np.argsort(order)
    pairs = sample_node_pairs.astype(np.int64, copy=True)
    person_for_node = {int(node): rank for rank, pair in enumerate(pairs[order]) for node in pair}
    columns: list[np.ndarray] = []
    visits = 0

    def preflight(features: int) -> None:
        required = 96 * ts.num_nodes + 8 * (4 * n * features + 12 * n * n)
        if required > max_memory_bytes:
            raise ValueError("dense truth memory bound exceeded")
        if features > max_branch_features:
            raise ValueError("branch feature bound exceeded")
        if visits + 2 * n * n * features + 2 * n**3 > max_work:
            raise ValueError("dense truth traversal/compression work bound exceeded")

    preflight(0)
    for tree in ts.trees(sample_lists=True):
        for node in sorted(tree.nodes()):
            visits += 1
            preflight(len(columns))
            parent = tree.parent(node)
            if parent == tskit.NULL:
                continue
            child_time, parent_time = tree.time(node), tree.time(parent)
            if not math.isfinite(child_time) or not math.isfinite(parent_time):
                raise ValueError("finite branch times required")
            length = max(0.0, min(parent_time, upper) - max(child_time, lower))
            if not length:
                continue
            carrier = np.zeros(n)
            for tip in tree.samples(node):
                visits += 1
                preflight(len(columns))
                person = person_for_node.get(int(tip))
                if person is not None:
                    carrier[person] += 0.5
            if not np.any(carrier):
                continue
            preflight(len(columns) + 1)
            columns.append(carrier * math.sqrt(tree.span / span * length))
    if columns:
        features = np.column_stack(columns)
        if center:
            features -= features.mean(axis=0)[None, :]
        with np.errstate(over="ignore", invalid="ignore"):
            raw_mean = float(np.sum(features * features) / n)
        if not math.isfinite(raw_mean):
            raise ValueError("unrepresentable covariance scale")
        if normalization == "mean_trace" and raw_mean <= 0:
            raise ValueError("time band has no sample-varying covariance to normalize")
        r = np.linalg.qr(features.T, mode="r")
        # Fix QR basis signs using each row's largest-magnitude coordinate.
        for row in r:
            if row[int(np.argmax(np.abs(row)))] < 0:
                row *= -1
        factors = np.asarray(r.T[inverse_order], dtype=np.float64)
        if normalization == "mean_trace":
            factors /= math.sqrt(raw_mean)
    else:
        raw_mean = 0.0
        if normalization == "mean_trace":
            raise ValueError("time band has no sample-varying covariance to normalize")
        factors = np.empty((n, 0))
    covariance = factors @ factors.T
    if not np.all(np.isfinite(factors)) or not np.all(np.isfinite(covariance)):
        raise ValueError("nonfinite covariance truth")
    factor_ids = tuple(f"{prefix}:{index:06d}" for index in range(factors.shape[1]))
    provenance = json.dumps(
        {
            "schema_version": 0,
            "method": "independent explicit branch-carrier columns; reduced QR compression",
            "sample_ids": ids,
            "sample_node_pairs": pairs.tolist(),
            "factor_ids": factor_ids,
            "factor_basis": "canonical person-ID order; QR row-sign canonicalization; not ancestral-node labels",
            "time_band": [lower, upper if math.isfinite(upper) else None],
            "time_units": ts.time_units,
            "sequence_length": span,
            "tree_count": ts.num_trees,
            "contributing_branch_columns": len(columns),
            "centering_ids": ids if center else [],
            "normalization": normalization,
            "raw_mean_diagonal": raw_mean,
            "covariance": "actual returned factor B @ B.T; no jitter or eigenvalue clipping",
            "interpretation": "genealogy-correlated environmental nuisance, not causal heritability",
            "versions": {name: version(name) for name in ("numpy", "tskit")},
        },
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(provenance.encode())
    digest.update(factors.astype("<f8", copy=False).tobytes())
    digest.update(covariance.astype("<f8", copy=False).tobytes())
    return GenealogicalEnvironmentTruth(
        ids, factor_ids, factors, covariance, raw_mean, digest.hexdigest(), provenance
    )
