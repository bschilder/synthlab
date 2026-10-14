"""Independent hand/tree-coalescence checks of the external covariance oracle."""

from __future__ import annotations

import math

import pytest

from tests.numerical_modules import numerical_module

np = pytest.importorskip("numpy")
tskit = pytest.importorskip("tskit")


def tree_sequence(two_trees=False):
    tables = tskit.TableCollection(10)
    tables.time_units = "generations"
    for _ in range(8):
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
    for time in (1, 2, 3, 4, 5, 6, 10):
        tables.nodes.add_row(time=time)
    topology = (
        (8, 0),
        (8, 1),
        (9, 2),
        (9, 3),
        (10, 4),
        (10, 5),
        (11, 6),
        (11, 7),
        (12, 8),
        (12, 9),
        (13, 10),
        (13, 11),
        (14, 12),
        (14, 13),
    )
    intervals = ((0, 3, False), (3, 10, True)) if two_trees else ((0, 10, False),)
    permutation = {2: 4, 3: 5, 4: 2, 5: 3}
    for left, right, swap in intervals:
        for parent, child in topology:
            tables.edges.add_row(
                left, right, parent, permutation.get(child, child) if swap else child
            )
    tables.sort()
    return tables.tree_sequence()


def truth(ts=None, **options):
    module = numerical_module("genealogical_covariance")
    kwargs = {
        "sample_ids": ("a", "b", "c", "d"),
        "sample_node_pairs": np.arange(8, dtype=np.int32).reshape(4, 2),
    }
    kwargs.update(options)
    return module.genealogical_environment_factors(tree_sequence() if ts is None else ts, **kwargs)


def expected_centered():
    # Raw diploid covariance diagonal is (9.5,9,8.5,8), AB=5, CD=4.
    # Subtract row/column means; mean centered diagonal =87/16.
    return (
        np.array(
            [[89, 19, -55, -53], [19, 85, -53, -51], [-55, -53, 89, 19], [-53, -51, 19, 85]],
            dtype=float,
        )
        / 87
    )


def test_hand_calculated_diploid_covariance_and_compressed_factors():
    result = truth()
    np.testing.assert_allclose(result.covariance, expected_centered(), rtol=2e-13, atol=3e-14)
    np.testing.assert_allclose(
        result.factors @ result.factors.T, result.covariance, rtol=2e-14, atol=1e-15
    )
    assert result.raw_mean_diagonal == pytest.approx(87 / 16)
    assert result.factors.shape[0] == 4 and result.factors.shape[1] <= 4
    assert len(result.factor_ids) == result.factors.shape[1]
    assert result.sample_ids == ("a", "b", "c", "d")
    assert not result.covariance.flags.writeable
    assert not result.factors.flags.writeable


def test_two_local_trees_are_weighted_by_physical_span():
    result = truth(tree_sequence(True))
    first = expected_centered()
    order = [0, 2, 1, 3]
    np.testing.assert_allclose(
        result.covariance, 0.3 * first + 0.7 * first[np.ix_(order, order)], rtol=2e-13, atol=3e-14
    )


def test_rank_deficient_time_band_needs_no_jitter_or_eigenvalue_repair():
    result = truth(lower_time=6, upper_time=10)
    contrast = np.array([1, 1, -1, -1])
    np.testing.assert_allclose(
        result.covariance, np.outer(contrast, contrast), rtol=2e-14, atol=2e-14
    )
    assert result.raw_mean_diagonal == pytest.approx(2)
    assert result.factors.shape[1] <= 4


def test_reference_only_tips_do_not_become_trait_rows_or_implicit_joins():
    pairs = np.array([[6, 7], [0, 1], [2, 3]], dtype=np.int32)
    result = truth(
        sample_ids=("d", "a", "b"), sample_node_pairs=pairs, center=False, normalization="none"
    )
    np.testing.assert_allclose(result.covariance, [[8, 0, 0], [0, 9.5, 5], [0, 5, 9]], atol=2e-14)
    assert result.sample_ids == ("d", "a", "b")


def test_homologue_swapping_preserves_diploid_covariance():
    first = truth()
    second = truth(sample_node_pairs=np.arange(8, dtype=np.int32).reshape(4, 2)[:, ::-1])
    np.testing.assert_allclose(second.covariance, first.covariance, atol=2e-14)


def test_empty_time_band_is_explicit_and_never_regularized():
    empty = truth(lower_time=20, normalization="none")
    assert empty.factors.shape == (4, 0)
    np.testing.assert_array_equal(empty.covariance, np.zeros((4, 4)))
    assert empty.raw_mean_diagonal == 0
    with pytest.raises(ValueError, match="sample-varying"):
        truth(lower_time=20)


@pytest.mark.parametrize(
    "pairs",
    [
        np.array([[0, 1], [0, 3], [4, 5], [6, 7]]),
        np.array([[0, 1], [2, 3], [4, 5], [6, 14]]),
        np.array([[-1, 1], [2, 3], [4, 5], [6, 7]]),
        np.arange(8, dtype=float).reshape(4, 2),
        np.arange(4).reshape(4, 1),
    ],
)
def test_invalid_duplicate_or_non_sample_node_mapping_is_rejected(pairs):
    with pytest.raises(ValueError):
        truth(sample_node_pairs=pairs)


@pytest.mark.parametrize(
    "options",
    [
        {"max_memory_bytes": 1},
        {"max_branch_features": 1},
        {"max_work": 1},
        {"max_samples": 3},
        {"sample_ids": ("a", "a", "c", "d")},
        {"lower_time": math.nan},
        {"upper_time": math.nan},
        {"lower_time": -1},
        {"lower_time": 5, "upper_time": 5},
        {"center": 1},
        {"normalization": "automatic"},
    ],
)
def test_explicit_input_and_resource_guards(options):
    with pytest.raises(ValueError):
        truth(**options)


def test_unknown_time_units_are_rejected():
    tables = tree_sequence().dump_tables()
    tables.time_units = "unknown"
    with pytest.raises(ValueError, match="time units"):
        truth(tables.tree_sequence())


def test_simulated_covariance_matches_independent_pairwise_coalescence_oracle():
    pytest.importorskip("msprime")
    engine = numerical_module("genealogy")
    panel = engine.simulate_genealogy(
        engine.GenealogySimConfig(
            sample_ids=("d", "b", "a", "c"),
            sequence_length=100,
            seed=260014,
            population_size=20,
            recombination_rate=1e-3,
            mutation_rate=0,
            dtwf_generations=3,
        )
    )
    result = truth(
        panel.tree_sequence, sample_ids=panel.sample_ids, sample_node_pairs=panel.sample_node_pairs
    )
    raw = np.zeros((4, 4))
    for tree in panel.tree_sequence.trees():
        root_time = tree.time(tree.root)
        weight = tree.span / panel.tree_sequence.sequence_length
        for i in range(4):
            for j in range(4):
                raw[i, j] += (
                    weight
                    * sum(
                        root_time - tree.tmrca(int(a), int(b))
                        for a in panel.sample_node_pairs[i]
                        for b in panel.sample_node_pairs[j]
                    )
                    / 4
                )
    centered = raw - raw.mean(axis=0)[None, :] - raw.mean(axis=1)[:, None] + raw.mean()
    np.testing.assert_allclose(
        result.covariance, centered / (np.trace(centered) / 4), rtol=3e-13, atol=3e-14
    )


def test_person_reordering_preserves_compressed_factor_realizations_by_id():
    first = truth()
    order = [3, 1, 0, 2]
    pairs = np.arange(8, dtype=np.int32).reshape(4, 2)[order]
    second = truth(sample_ids=tuple(first.sample_ids[i] for i in order), sample_node_pairs=pairs)
    np.testing.assert_array_equal(second.factors, first.factors[order])
    assert second.factor_ids == first.factor_ids
