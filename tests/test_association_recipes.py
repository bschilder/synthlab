"""External recipe wiring, actual covariance and retained streamed failures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.numerical_modules import numerical_module

np = pytest.importorskip("numpy")
tskit = pytest.importorskip("tskit")
pytest.importorskip("msprime")


def recipe():
    return {
        "genealogy": {
            "sample_ids": ["d", "a", "c", "b"],
            "sequence_length": 100,
            "seed": 91,
            "population_size": 100,
            "mutation_rate": 0,
            "dtwf_generations": 3,
        },
        "reference": "synthetic ancestral-reference coordinates",
        "contig": "simulation-1",
        "trait_units": ["expression", "height"],
        "phenotypes": {
            "trait_ids": ["expression", "height"],
            "effects": [],
            "covariate_ids": ["intercept"],
            "covariates": [[1], [1], [1], [1]],
            "covariate_effects": [[5, -2]],
            "residual_covariance": [[1, 0.2], [0.2, 2]],
            "seed": 113,
            "observed_probabilities": [[1, 0], [0, 1], [0, 0], [1, 1]],
            "missingness_description": "fixed exogenous missingness, including an unobserved person",
        },
    }


def environment_recipe():
    result = recipe()
    result["genealogical_environment"] = [
        {"id": "whole", "variance": 0.5},
        {"id": "recent", "variance": 2, "lower_time": 0, "upper_time": 20, "normalization": "none"},
    ]
    result["phenotypes"].update(
        residual_factor_ids=["household:0"], residual_row_factors=[[1], [1], [0], [0]]
    )
    return result


def demes_graph():
    return {
        "time_units": "generations",
        "demes": [
            {"name": "root", "epochs": [{"start_size": 200}]},
            {"name": "a", "ancestors": ["root"], "start_time": 40, "epochs": [{"start_size": 100}]},
            {"name": "b", "ancestors": ["root"], "start_time": 40, "epochs": [{"start_size": 100}]},
            {
                "name": "mix",
                "ancestors": ["a", "b"],
                "proportions": [0.25, 0.75],
                "start_time": 5,
                "epochs": [{"start_size": 100}],
            },
        ],
        "migrations": [{"demes": ["a", "b"], "rate": 0.01}],
    }


def write(tmp_path, value):
    return numerical_module("association_recipes").write_association_recipe(
        tmp_path / "bundle", value, source_revision="a" * 40
    )


def load_arrays(receipt):
    path = Path(receipt["manifest_path"])
    manifest = json.loads(path.read_text())
    arrays = {
        role: np.load(path.parent / record["path"], allow_pickle=False)
        for role, record in manifest["files"].items()
        if role != "tree_sequence"
    }
    return manifest, arrays


def test_combined_independent_time_bands_and_household_covariance_are_exact(tmp_path):
    value = environment_recipe()
    original = json.dumps(value, sort_keys=True)
    receipt = write(tmp_path, value)
    manifest, arrays = load_arrays(receipt)
    assert json.dumps(value, sort_keys=True) == original
    ts = tskit.load(Path(receipt["manifest_path"]).parent / "truth.trees")
    oracle = numerical_module("genealogical_covariance").genealogical_environment_factors
    common = {
        "sample_ids": manifest["sample_ids"],
        "sample_node_pairs": arrays["sample_node_pairs"],
    }
    whole = oracle(ts, factor_prefix="whole", **common)
    recent = oracle(ts, upper_time=20, normalization="none", **common)
    household = np.array([[1], [1], [0], [0]])
    expected = 0.5 * whole.covariance + 2 * recent.covariance + household @ household.T
    a = arrays["residual_row_factors"]
    np.testing.assert_allclose(a @ a.T, expected, rtol=3e-13, atol=3e-14)
    assert manifest["residual_factor_ids"][0] == "household:0"
    assert receipt["genealogical_environment"][0]["variance"] == 0.5
    assert (
        receipt["genealogical_environment"][0]["scientific_signature"] == whole.scientific_signature
    )
    assert receipt["genealogical_environment"][0]["provenance"]["interpretation"].startswith(
        "genealogy-correlated environmental"
    )
    assert not arrays["observed"][2].any()
    assert np.isnan(arrays["observed_values"][2]).all()
    np.testing.assert_allclose(
        arrays["residuals"], arrays["independent_residuals"] + arrays["structured_residuals"]
    )


def test_demes_admixture_and_migration_keep_explicit_person_deme_join(tmp_path):
    value = recipe()
    labels = ["mix", "a", "b", "mix"]
    value["demography"] = {"graph": demes_graph(), "sample_demes": labels}
    receipt = write(tmp_path, value)
    manifest, arrays = load_arrays(receipt)
    provenance = manifest["genealogy"]["provenance"]
    assert provenance["sample_demes"] == labels
    assert not provenance["single_deme_population_size_used"]
    assert manifest["sample_ids"] == value["genealogy"]["sample_ids"]
    assert arrays["sample_node_pairs"].shape == (4, 2)
    assert manifest["time_units"] == "generations"
    canonical = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    assert receipt["recipe_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_outcome_blind_effect_rule_resolves_exact_first_eligible_allele(tmp_path):
    value = recipe()
    value["genealogy"]["mutation_rate"] = 1e-4
    value["effect_rules"] = [
        {"rule": "first_biallelic_minor_copies", "minimum_copies": 1, "effects": [0.8, -0.4]}
    ]
    receipt = write(tmp_path, value)
    manifest, arrays = load_arrays(receipt)
    candidates = []
    for i, alleles in enumerate(manifest["alleles"]):
        dosage = (arrays["genotypes"][:, i] == 1).sum(axis=1)
        if len(alleles) == 2 and min(int(dosage.sum()), 8 - int(dosage.sum())) >= 1:
            candidates.append((i, dosage))
    assert candidates
    index, dosage = candidates[0]
    resolved = receipt["resolved_effect_rules"][0]
    assert resolved["variant_id"] == manifest["variant_ids"][index]
    assert resolved["allele_index"] == 1
    np.testing.assert_allclose(arrays["additive_means"], dosage[:, None] * np.array([[0.8, -0.4]]))
    assert np.count_nonzero(arrays["interaction_means"]) == 0
    nested = tmp_path / "different-mask"
    nested.mkdir()
    value["phenotypes"]["observed_probabilities"] = [[1, 1]] * 4
    other = write(nested, value)
    _, observed = load_arrays(other)
    assert other["resolved_effect_rules"] == receipt["resolved_effect_rules"]
    np.testing.assert_array_equal(arrays["complete_values"], observed["complete_values"])


def test_unavailable_registered_effect_rule_fails_without_replacing_seed(tmp_path):
    value = recipe()
    value["effect_rules"] = [
        {"rule": "first_biallelic_minor_copies", "minimum_copies": 1, "effects": [0.8, -0.4]}
    ]
    with pytest.raises(ValueError, match="eligible.*no resimulation"):
        write(tmp_path, value)
    assert value["genealogy"]["seed"] == 91
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("rule", ["association_pvalue", "best_trait_correlation"])
def test_outcome_dependent_or_unknown_effect_selection_is_rejected(tmp_path, rule):
    value = recipe()
    value["effect_rules"] = [{"rule": rule, "minimum_copies": 1, "effects": [0.8, -0.4]}]
    with pytest.raises(ValueError, match="effect rule"):
        write(tmp_path, value)


@pytest.mark.parametrize(
    "change",
    [
        "negative_variance",
        "boolean_variance",
        "duplicate_id",
        "unknown_environment_field",
        "unknown_recipe_field",
        "bad_deme_join",
    ],
)
def test_invalid_generating_recipe_is_rejected_without_publishing_a_bundle(tmp_path, change):
    value = environment_recipe()
    if change == "negative_variance":
        value["genealogical_environment"][0]["variance"] = -1
    elif change == "boolean_variance":
        value["genealogical_environment"][0]["variance"] = True
    elif change == "duplicate_id":
        value["genealogical_environment"][1]["id"] = "whole"
    elif change == "unknown_environment_field":
        value["genealogical_environment"][0]["ancestry_class"] = "invented"
    elif change == "unknown_recipe_field":
        value["silently_ignored"] = 1
    else:
        value["demography"] = {"graph": demes_graph(), "sample_demes": ["mix"]}
    with pytest.raises(ValueError):
        write(tmp_path, value)
    assert not (tmp_path / "bundle").exists()


def test_named_zero_weight_component_does_not_change_complete_outcomes(tmp_path):
    first = write(tmp_path, recipe())
    nested = tmp_path / "second"
    nested.mkdir()
    value = recipe()
    value["genealogical_environment"] = [{"id": "zero", "variance": 0}]
    second = write(nested, value)
    _, original = load_arrays(first)
    _, other = load_arrays(second)
    np.testing.assert_array_equal(original["complete_values"], other["complete_values"])
    np.testing.assert_array_equal(original["observed"], other["observed"])
    assert np.count_nonzero(other["residual_row_factors"]) == 0


def test_single_cli_accepts_demes_and_independent_environment_recipe(tmp_path):
    value = environment_recipe()
    value["demography"] = {"graph": demes_graph(), "sample_demes": ["mix", "a", "b", "mix"]}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    script = Path(__file__).resolve().parents[1] / "scripts/generate_association_bundle.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recipe",
            str(path),
            "--destination",
            str(tmp_path / "bundle"),
            "--source-revision",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    receipt = json.loads(process.stdout)
    assert len(receipt["genealogical_environment"]) == 2
    assert (
        receipt["manifest_sha256"]
        == hashlib.sha256(Path(receipt["manifest_path"]).read_bytes()).hexdigest()
    )


def test_explicit_empty_environment_band_preserves_zero_factor_axes(tmp_path):
    value = recipe()
    value["genealogical_environment"] = [
        {"id": "empty", "variance": 1, "lower_time": 1e9, "normalization": "none"}
    ]
    receipt = write(tmp_path, value)
    manifest, arrays = load_arrays(receipt)
    assert manifest["residual_factor_ids"] == []
    assert arrays["residual_row_factors"].shape == (4, 0)
    assert np.count_nonzero(arrays["structured_residuals"]) == 0


def test_demes_years_are_converted_to_generation_times(tmp_path):
    value = recipe()
    graph = demes_graph()
    graph.update(time_units="years", generation_time=25)
    for deme in graph["demes"]:
        if "start_time" in deme:
            deme["start_time"] *= 25
    graph["migrations"][0]["rate"] /= 25
    value["demography"] = {"graph": graph, "sample_demes": ["mix", "a", "b", "mix"]}
    manifest, _ = load_arrays(write(tmp_path, value))
    events = manifest["genealogy"]["provenance"]["demography"]["events"]
    assert {event["time"] for event in events} == {5, 40}
    assert manifest["time_units"] == "generations"


def test_streaming_overlong_line_reports_prefix_and_unattempted_remainder(tmp_path):
    source = tmp_path / "recipes.jsonl"
    source.write_text(json.dumps({"case_id": "large", "recipe": recipe()}) + "\n")
    destination = tmp_path / "batch"
    script = Path(__file__).resolve().parents[1] / "scripts/generate_association_batch.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recipes",
            str(source),
            "--destination",
            str(destination),
            "--source-revision",
            "a" * 40,
            "--max-recipe-bytes",
            "64",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 1, process.stderr
    completion = json.loads((destination / "completed.json").read_text())
    assert completion["attempted"] == 1 and not completion["complete_input"]
    assert (destination / "recipes.jsonl").read_bytes() == source.read_bytes()[:65]
    row = json.loads((destination / "records.jsonl").read_text())
    assert row["input_line_complete"] is False
    assert "input_line_sha256" not in row
    assert row["input_prefix_sha256"] == hashlib.sha256(source.read_bytes()[:65]).hexdigest()


def test_streaming_cli_retains_failed_recipe_and_runs_later_registered_cases(tmp_path):
    first = recipe()
    bad = environment_recipe()
    bad["genealogical_environment"][0]["variance"] = -1
    third = recipe()
    third["genealogy"]["seed"] = 92
    entries = [
        {"case_id": "first", "recipe": first},
        {"case_id": "failed", "recipe": bad},
        {"case_id": "third", "recipe": third},
    ]
    source = tmp_path / "recipes.jsonl"
    source.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    script = Path(__file__).resolve().parents[1] / "scripts/generate_association_batch.py"
    destination = tmp_path / "batch"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recipes",
            str(source),
            "--destination",
            str(destination),
            "--source-revision",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 1, process.stderr
    assert (destination / "recipes.jsonl").read_bytes() == source.read_bytes()
    journal = [
        json.loads(line) for line in (destination / "records.jsonl").read_text().splitlines()
    ]
    assert [row["case_id"] for row in journal] == ["first", "failed", "third"]
    assert [row["status"] for row in journal] == ["success", "failed", "success"]
    assert journal[1]["error_type"] == "ValueError"
    assert not (destination / "failed").exists()
    assert json.loads((destination / "completed.json").read_text())["failed"] == 1
    for row, expected in ((journal[0], 91), (journal[2], 92)):
        manifest = json.loads(Path(row["receipt"]["manifest_path"]).read_text())
        assert manifest["genealogy"]["provenance"]["config"]["seed"] == expected


def test_streaming_cli_preserves_existing_destination(tmp_path):
    source = tmp_path / "recipes.jsonl"
    source.write_text(json.dumps({"case_id": "first", "recipe": recipe()}) + "\n")
    destination = tmp_path / "batch"
    destination.mkdir()
    sentinel = destination / "user.txt"
    sentinel.write_text("preserve")
    script = Path(__file__).resolve().parents[1] / "scripts/generate_association_batch.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recipes",
            str(source),
            "--destination",
            str(destination),
            "--source-revision",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode != 0
    assert "FileExistsError" in process.stderr
    assert sentinel.read_text() == "preserve"
