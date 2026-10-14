"""Exact planted mechanisms, known residual draws and independent observation masks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.numerical_modules import numerical_module

np = pytest.importorskip("numpy")


def engine():
    return numerical_module("genotype_phenotypes")


def fixture():
    return {
        "sample_ids": ("c", "a", "b"),
        "variant_ids": ("one", "two"),
        "alleles": (("A", "C"), ("A", "G", "T")),
        "genotypes": np.array(
            [[[0, 0], [0, 1]], [[0, 1], [1, 2]], [[1, 1], [2, 2]]], dtype=np.int8
        ),
        "trait_ids": ("height", "expression"),
        "effects": (),
        "covariate_ids": ("intercept",),
        "covariates": np.ones((3, 1)),
        "covariate_effects": np.array([[5.0, -2.0]]),
        "residual_covariance": np.zeros((2, 2)),
        "seed": 91,
    }


def simulate(**options):
    inputs = fixture()
    inputs.update(options)
    return engine().simulate_quantitative_phenotypes(**inputs)


def test_exact_signed_allele_effects_and_covariates_have_separate_truth():
    result = simulate(effects=(engine().AlleleEffect("one", 1, (2.0, -1.0)),))
    np.testing.assert_array_equal(result.genetic_means, [[0.0, 0.0], [2.0, -1.0], [4.0, -2.0]])
    np.testing.assert_array_equal(result.nuisance_means, [[5.0, -2.0]] * 3)
    np.testing.assert_array_equal(result.complete_values, [[5.0, -2.0], [7.0, -3.0], [9.0, -4.0]])
    np.testing.assert_array_equal(result.observed_values, result.complete_values)
    assert result.observed.all()
    assert result.sample_ids == ("c", "a", "b")
    assert result.trait_ids == ("height", "expression")
    assert not result.complete_values.flags.writeable


def test_multiallelic_and_reference_allele_effects_use_exact_copy_counts():
    result = simulate(
        effects=(
            engine().AlleleEffect("two", 2, (3.0, 1.0)),
            engine().AlleleEffect("one", 0, (-1.0, 2.0)),
        )
    )
    np.testing.assert_array_equal(result.genetic_means, [[-2.0, 4.0], [2.0, 3.0], [6.0, 2.0]])


def test_dominance_and_continuous_context_are_explicit_mechanisms():
    result = simulate(
        dominance_effects=(engine().DominanceEffect("one", 1, (4.0, -2.0)),),
        context_ids=("background",),
        contexts=np.array([[-1.0], [0.0], [2.0]]),
        context_effects=(
            engine().ContextEffect("one", 1, "background", (3.0, 1.0), context_center=1.0),
        ),
    )
    np.testing.assert_array_equal(result.dominance_means, [[0.0, 0.0], [4.0, -2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(result.context_means, [[0.0, 0.0], [-3.0, -1.0], [6.0, 2.0]])
    np.testing.assert_array_equal(
        result.genetic_means, result.dominance_means + result.context_means
    )


def test_balanced_pure_epistasis_has_no_marginal_linear_signal():
    gt = np.array(
        [[[0, 0], [0, 0]], [[0, 0], [1, 1]], [[1, 1], [0, 0]], [[1, 1], [1, 1]]], dtype=np.int8
    )
    result = simulate(
        sample_ids=("a", "b", "c", "d"),
        genotypes=gt,
        alleles=(("A", "C"), ("A", "G")),
        covariates=np.ones((4, 1)),
        interaction_effects=(
            engine().InteractionEffect(
                "one", 1, "two", 1, (2.0, -1.0), first_center=1.0, second_center=1.0
            ),
        ),
    )
    np.testing.assert_array_equal(
        result.interaction_means, [[2.0, -1.0], [-2.0, 1.0], [-2.0, 1.0], [2.0, -1.0]]
    )
    np.testing.assert_array_equal((gt.sum(axis=2) - 1).T @ result.genetic_means, np.zeros((2, 2)))


def test_compound_heterozygosity_distinguishes_cis_and_trans_and_swapped_homologues():
    gt = np.array([[[0, 1], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 1]]], dtype=np.int8)
    options = {
        "genotypes": gt,
        "alleles": (("A", "C"), ("A", "G")),
        "compound_effects": (engine().CompoundHetEffect("one", 1, "two", 1, (2.0, -1.0)),),
    }
    result = simulate(**options)
    np.testing.assert_array_equal(result.compound_means, [[0.0, 0.0], [2.0, -1.0], [0.0, 0.0]])
    swapped = simulate(**{**options, "genotypes": gt[:, :, ::-1]})
    np.testing.assert_array_equal(result.genetic_means, swapped.genetic_means)


def test_residual_draws_match_independent_canonical_order_cholesky_formula():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = simulate(residual_covariance=covariance)
    normal = np.random.Generator(
        np.random.PCG64(result.stream_seeds["residual_iid"])
    ).standard_normal((3, 2))
    # Canonical rows a,b,c and traits expression,height, then restore caller order.
    expected = normal @ np.linalg.cholesky(covariance[::-1, ::-1]).T
    expected = expected[[2, 0, 1]][:, ::-1]
    np.testing.assert_allclose(result.independent_residuals, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.sampling_covariance, covariance, atol=1e-14)
    np.testing.assert_allclose(result.residuals, result.independent_residuals)


def test_distinct_master_seeds_do_not_reuse_known_colliding_gaussian_streams():
    # Metadata audit of the frozen draft found the same uint32 intermediate.
    first = simulate(seed=7494693928700945653, residual_covariance=np.eye(2))
    second = simulate(seed=7062127190484666175, residual_covariance=np.eye(2))
    assert first.stream_seeds["residual_iid"] != second.stream_seeds["residual_iid"]
    assert first.stream_seeds["residual_shared"] != second.stream_seeds["residual_shared"]
    assert not np.array_equal(first.independent_residuals, second.independent_residuals)


def test_distinct_master_seeds_do_not_reuse_known_colliding_mask_state():
    first = simulate(seed=2881303302047735690)
    second = simulate(seed=6228104137962913151)
    assert first.stream_seeds["missingness"] != second.stream_seeds["missingness"]


def test_numpy_quantitative_streams_use_full_width_immutable_named_seeds():
    seeds = engine().quantitative_stream_seeds(91)
    assert set(seeds) == {"residual_iid", "residual_shared", "missingness"}
    assert len(set(seeds.values())) == 3
    assert all(type(value) is int and 2**64 < value < 2**128 for value in seeds.values())
    # Fixed compact-JSON/SHA256/little-endian v1 protocol, independently calculated.
    assert {name: f"0x{value:032x}" for name, value in seeds.items()} == {
        "residual_iid": "0xc0b83cc3d9b730b4bde9d993eb8639e2",
        "residual_shared": "0xf6c6925d1d1f40bbf9f4e8ed13c9370b",
        "missingness": "0x4b4f7cbd3d2e79baa4374d0d99a55364",
    }
    assert seeds == engine().quantitative_stream_seeds(91)
    with pytest.raises(TypeError):
        seeds["missingness"] = 1


def test_full_width_seed_provenance_is_lossless_hex_not_json_float_precision():
    result = simulate(residual_covariance=np.eye(2))
    provenance = json.loads(result.provenance)
    assert provenance["rng_scheme"] == "quantitative-pcg64-128-v1"
    recorded = provenance["stream_seeds"]
    assert all(
        isinstance(value, str) and len(value) == 34 and value.startswith("0x")
        for value in recorded.values()
    )
    assert {name: int(value, 16) for name, value in recorded.items()} == result.stream_seeds


@pytest.mark.parametrize("seed", [True, -1, 0, 2**63])
def test_full_width_stream_helper_preserves_master_seed_input_contract(seed):
    with pytest.raises(ValueError):
        engine().quantitative_stream_seeds(seed)


def test_row_and_trait_permutations_preserve_person_specific_realizations():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    probabilities = np.array([[0.2, 0.8], [1.0, 0.0], [0.5, 0.5]])
    original = simulate(
        residual_covariance=covariance,
        observed_probabilities=probabilities,
        missingness_description="supplied heterogeneous observation probabilities",
    )
    rows = np.array([1, 2, 0])
    inputs = fixture()
    permuted = simulate(
        sample_ids=tuple(inputs["sample_ids"][i] for i in rows),
        genotypes=inputs["genotypes"][rows],
        covariates=inputs["covariates"][rows],
        trait_ids=inputs["trait_ids"][::-1],
        covariate_effects=inputs["covariate_effects"][:, ::-1],
        residual_covariance=covariance[::-1, ::-1],
        observed_probabilities=probabilities[rows][:, ::-1],
        missingness_description="supplied heterogeneous observation probabilities",
    )
    np.testing.assert_allclose(
        permuted.complete_values, original.complete_values[rows][:, ::-1], atol=1e-14
    )
    np.testing.assert_array_equal(permuted.observed, original.observed[rows][:, ::-1])


def test_observation_probabilities_change_masks_without_changing_complete_truth():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    complete = simulate(residual_covariance=covariance)
    missing = simulate(
        residual_covariance=covariance,
        observed_probabilities=np.zeros((3, 2)),
        missingness_description="MCAR: no recorded measurements",
    )
    np.testing.assert_array_equal(missing.complete_values, complete.complete_values)
    assert not missing.observed.any()
    assert np.isnan(missing.observed_values).all()
    assert missing.sample_ids == complete.sample_ids
    mnar_probabilities = 1 / (1 + np.exp(-complete.complete_values))
    mnar = simulate(
        residual_covariance=covariance,
        observed_probabilities=mnar_probabilities,
        missingness_description="MNAR: depends on complete simulated outcome",
    )
    np.testing.assert_array_equal(mnar.complete_values, complete.complete_values)
    np.testing.assert_array_equal(
        mnar.observed_values[mnar.observed], complete.complete_values[mnar.observed]
    )


def test_shared_environment_factors_are_separate_from_genetic_means():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    factors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    result = simulate(
        residual_covariance=covariance,
        residual_factor_ids=("household-a", "household-b"),
        residual_row_factors=factors,
        environment_means=np.array([[3.0, 0.0], [3.0, 0.0], [-3.0, 0.0]]),
    )
    np.testing.assert_array_equal(result.genetic_means, np.zeros((3, 2)))
    np.testing.assert_array_equal(result.structured_residuals[0], result.structured_residuals[1])
    np.testing.assert_allclose(
        result.structured_residuals, factors @ result.shared_factor_values, atol=1e-14
    )
    np.testing.assert_allclose(
        result.nuisance_means, np.array([[8.0, -2.0], [8.0, -2.0], [2.0, -2.0]])
    )
    np.testing.assert_allclose(
        result.residuals, result.independent_residuals + result.structured_residuals
    )


def test_zero_variant_panel_can_generate_environment_only_null_outcomes():
    result = simulate(
        variant_ids=(),
        alleles=(),
        genotypes=np.empty((3, 0, 2), dtype=np.int8),
        environment_means=np.array([[1.0, 2.0], [0.0, 0.0], [-1.0, -2.0]]),
    )
    np.testing.assert_array_equal(result.genetic_means, np.zeros((3, 2)))
    np.testing.assert_array_equal(result.complete_values, [[6.0, 0.0], [5.0, -2.0], [4.0, -4.0]])


def test_generation_is_deterministic_and_does_not_use_global_numpy_rng():
    np.random.seed(712)
    before = np.random.get_state()
    first = simulate(residual_covariance=np.eye(2))
    second = simulate(residual_covariance=np.eye(2))
    after = np.random.get_state()
    np.testing.assert_array_equal(first.complete_values, second.complete_values)
    assert first.scientific_signature == second.scientific_signature
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    "options",
    [
        {"genotypes": np.full((3, 2, 2), -1, dtype=np.int8)},
        {"genotypes": np.full((3, 2, 2), 9, dtype=np.int8)},
        {"genotypes": np.zeros((3, 2, 2), dtype=float)},
        {"genotypes": np.zeros((3, 2, 1), dtype=np.int8)},
        {"sample_ids": ("a", "a", "b")},
        {"sample_ids": "abc"},
        {"variant_ids": ("one", "one")},
        {"alleles": (("A", "A"), ("A", "G", "T"))},
        {"covariates": np.full((3, 1), np.nan)},
        {"covariate_effects": np.ones((2, 2))},
        {"residual_covariance": np.array([[1.0, 2.0], [2.0, 1.0]])},
        {"residual_covariance": np.array([[1.0, 0.2], [0.3, 1.0]])},
        {"observed_probabilities": np.full((3, 2), 1.1), "missingness_description": "invalid"},
        {"observed_probabilities": np.ones((3, 2))},
        {"seed": True},
        {"max_memory_bytes": 1},
        {"max_work": 1},
    ],
)
def test_invalid_truth_or_misaligned_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        simulate(**options)


def test_unknown_effects_and_trait_dimension_errors_are_rejected():
    for effect in (
        engine().AlleleEffect("unknown", 1, (1.0, 1.0)),
        engine().AlleleEffect("one", 2, (1.0, 1.0)),
        engine().AlleleEffect("one", 1, (1.0,)),
    ):
        with pytest.raises(ValueError):
            simulate(effects=(effect,))
    with pytest.raises(ValueError):
        engine().AlleleEffect("one", -1, (1.0, 1.0))
    with pytest.raises(ValueError):
        engine().AlleleEffect("one", 1, (float("inf"), 1.0))


def test_singular_covariance_preserves_every_positive_variance_in_caller_units():
    covariance = np.diag([1.0, 1e-20, 0.0])
    result = simulate(
        trait_ids=("a", "b", "c"),
        covariate_effects=np.zeros((1, 3)),
        residual_covariance=covariance,
    )
    assert result.sampling_covariance[1, 1] == pytest.approx(1e-20, rel=1e-10, abs=0)
    assert result.sampling_covariance[2, 2] == 0


def test_exactly_shared_trait_noise_and_declared_boundary_factorization():
    result = simulate(residual_covariance=np.ones((2, 2)))
    np.testing.assert_allclose(result.residuals[:, 0], result.residuals[:, 1], atol=1e-14)
    record = json.loads(result.provenance)["covariance_factorization"]
    assert record["sampling_rank"] == 1
    assert record["jitter_added"] is False


def test_factor_column_permutation_preserves_shared_environment_draws():
    options = {
        "residual_covariance": np.eye(2),
        "residual_factor_ids": ("z", "a"),
        "residual_row_factors": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
    }
    original = simulate(**options)
    changed = simulate(
        **{
            **options,
            "residual_factor_ids": ("a", "z"),
            "residual_row_factors": options["residual_row_factors"][:, ::-1],
        }
    )
    np.testing.assert_allclose(changed.complete_values, original.complete_values, atol=1e-14)


def test_hashing_block_size_and_homologue_exchange_do_not_change_realizations():
    original = simulate(residual_covariance=np.eye(2))
    small_blocks = simulate(residual_covariance=np.eye(2), max_memory_bytes=3874)
    assert original.scientific_signature == small_blocks.scientific_signature
    gt = fixture()["genotypes"]
    before = gt.copy()
    changed = simulate(genotypes=gt[:, :, ::-1], residual_covariance=np.eye(2))
    np.testing.assert_array_equal(changed.complete_values, original.complete_values)
    np.testing.assert_array_equal(gt, before)


def test_perfectly_linked_variants_retain_distinct_planted_coefficients():
    gt = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 1], [1, 1]]], dtype=np.int8)
    result = simulate(
        genotypes=gt,
        alleles=(("A", "C"), ("A", "G")),
        effects=(
            engine().AlleleEffect("one", 1, (2.0, -1.0)),
            engine().AlleleEffect("two", 1, (3.0, 1.0)),
        ),
    )
    np.testing.assert_array_equal(result.genetic_means, [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    assert len(json.loads(result.provenance)["mechanisms"]["additive"]) == 2


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1.0, -1e-20]),
        np.array([[1.0, 1e-10], [1e-10, 1e-30]]),
        np.array([[1.0, 1e-20], [1e-20, 0.0]]),
    ],
)
def test_small_scale_invalid_covariance_is_not_hidden_by_large_trait_variance(covariance):
    with pytest.raises(ValueError, match="residual_covariance"):
        simulate(residual_covariance=covariance)


def test_memory_preflight_counts_wide_covariate_coefficient_matrix():
    with pytest.raises(ValueError, match="principal array memory"):
        simulate(
            sample_ids=("a",),
            genotypes=fixture()["genotypes"][:1],
            trait_ids=tuple(f"trait-{i}" for i in range(32)),
            covariate_ids=tuple(f"covariate-{i}" for i in range(10000)),
            covariates=np.ones((1, 10000)),
            covariate_effects=np.zeros((10000, 32)),
            residual_covariance=np.eye(32),
            max_memory_bytes=1024**2,
        )


def test_phenotype_ci_gate_executes_this_module_and_tracks_helper_changes():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/simulation.yml"
    ).read_text()
    assert "pytest tests/test_genealogy.py tests/test_genotype_phenotypes.py" in workflow
    assert "'tests/numerical_modules.py'" in workflow
    assert "'synthlab/genotype_phenotypes.py'" in workflow


def test_simulation_source_manifest_includes_numerical_test_support():
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text()
    assert "tests/numerical_modules.py" in manifest
    assert "tests/__init__.py" in manifest
