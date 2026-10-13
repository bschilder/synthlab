"""Real oracle-to-artifact joins, separate truth arrays and closed publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.numerical_modules import numerical_module

np = pytest.importorskip("numpy")
pytest.importorskip("msprime")


def inputs(*, mutation_rate=1e-4):
    genealogy = numerical_module("genealogy")
    truth = genealogy.simulate_genealogy(
        genealogy.GenealogySimConfig(
            ("c", "a", "b"),
            100,
            91,
            population_size=100,
            mutation_rate=mutation_rate,
            dtwf_generations=5,
        )
    )
    variants = tuple(f"site-{int(position)}" for position in truth.site_positions)
    return truth, {
        "variant_ids": variants,
        "reference": "synthetic ancestral-reference coordinates",
        "contig": "simulation-1",
        "trait_units": ("expression units", "height units"),
        "source_revision": "a" * 40,
        "phenotype_options": {
            "trait_ids": ("expression", "height"),
            "effects": (),
            "covariate_ids": ("intercept",),
            "covariates": np.ones((3, 1)),
            "covariate_effects": np.array([[5.0, -2.0]]),
            "residual_covariance": np.array([[1.0, 0.2], [0.2, 2.0]]),
            "seed": 113,
            "observed_probabilities": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            "missingness_description": "fixed observation pattern; retain wholly unobserved donor",
        },
    }


def test_bundle_files_pin_exact_truth_and_keep_observed_cells_separate(tmp_path):
    truth, options = inputs()
    receipt = numerical_module("association_bundle").write_association_bundle(
        tmp_path / "bundle", truth, **options
    )
    raw = receipt.manifest_path.read_bytes()
    manifest = json.loads(raw)
    assert receipt.manifest_sha256 == hashlib.sha256(raw).hexdigest()
    assert manifest["sample_ids"] == ["c", "a", "b"]
    assert manifest["variant_ids"] == list(options["variant_ids"])
    assert manifest["alleles"] == [list(alleles) for alleles in truth.alleles]
    assert manifest["coordinate_offset"] == 0
    assert manifest["time_units"] == "generations"
    assert manifest["phase"] == "complete true simulated homologues"
    assert manifest["genealogy"]["scientific_signature"] == truth.scientific_signature
    arrays = {}
    for role, record in manifest["files"].items():
        path = receipt.manifest_path.parent / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if role != "tree_sequence":
            arrays[role] = np.load(path, allow_pickle=False)
            assert arrays[role].dtype.str == record["dtype"]
            assert list(arrays[role].shape) == record["shape"]
    np.testing.assert_array_equal(arrays["genotypes"], truth.genotypes)
    np.testing.assert_array_equal(arrays["sample_node_pairs"], truth.sample_node_pairs)
    assert not arrays["observed"][2].any()
    assert np.isnan(arrays["observed_values"][~arrays["observed"]]).all()
    np.testing.assert_array_equal(
        arrays["observed_values"][arrays["observed"]],
        arrays["complete_values"][arrays["observed"]],
    )
    np.testing.assert_array_equal(arrays["covariates"], options["phenotype_options"]["covariates"])


def test_bundle_zero_mutation_panel_retains_empty_genotype_axes(tmp_path):
    truth, options = inputs(mutation_rate=0)
    receipt = numerical_module("association_bundle").write_association_bundle(
        tmp_path / "bundle", truth, **options
    )
    manifest = json.loads(receipt.manifest_path.read_text())
    assert manifest["variant_ids"] == []
    assert manifest["files"]["genotypes"]["shape"] == [3, 0, 2]


@pytest.mark.parametrize("change", ["genotype", "pairs", "sample_ids"])
def test_bundle_rejects_genealogy_truth_misalignment_before_publication(tmp_path, change):
    truth, options = inputs()
    if change == "genotype":
        calls = truth.genotypes.copy()
        assert calls.shape[1] > 0
        calls[0, 0, 0] = (calls[0, 0, 0] + 1) % len(truth.alleles[0])
        truth = replace(truth, genotypes=calls)
    elif change == "pairs":
        truth = replace(truth, sample_node_pairs=truth.sample_node_pairs[::-1])
    else:
        truth = replace(truth, sample_ids=truth.sample_ids[::-1])
    with pytest.raises(ValueError):
        numerical_module("association_bundle").write_association_bundle(
            tmp_path / "bundle", truth, **options
        )
    assert not (tmp_path / "bundle").exists()


def test_bundle_forbids_overridden_genotype_join_and_preserves_existing_destination(tmp_path):
    truth, options = inputs()
    options["phenotype_options"]["sample_ids"] = ("a", "b", "c")
    with pytest.raises(ValueError, match="join"):
        numerical_module("association_bundle").write_association_bundle(
            tmp_path / "new", truth, **options
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "user.txt"
    sentinel.write_text("preserve")
    options["phenotype_options"].pop("sample_ids")
    with pytest.raises(FileExistsError):
        numerical_module("association_bundle").write_association_bundle(existing, truth, **options)
    assert sentinel.read_text() == "preserve"


def test_bundle_resource_rejection_cleans_only_its_new_destination(tmp_path):
    truth, options = inputs()
    with pytest.raises(ValueError, match="byte bound"):
        numerical_module("association_bundle").write_association_bundle(
            tmp_path / "bundle", truth, max_output_bytes=1, **options
        )
    assert not (tmp_path / "bundle").exists()


def test_isolated_producer_cli_uses_recipe_without_importing_the_ehr_stack(tmp_path):
    truth, options = inputs(mutation_rate=0)
    recipe = {
        "genealogy": json.loads(truth.provenance)["config"],
        "reference": options["reference"],
        "contig": options["contig"],
        "trait_units": options["trait_units"],
        "phenotypes": {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in options["phenotype_options"].items()
        },
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    script = Path(__file__).resolve().parents[1] / "scripts/generate_association_bundle.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--recipe",
            str(path),
            "--destination",
            str(tmp_path / "cli-bundle"),
            "--source-revision",
            "a" * 40,
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    manifest = Path(receipt["manifest_path"])
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert json.loads(manifest.read_text())["files"]["genotypes"]["shape"] == [3, 0, 2]


def test_bundle_ci_gate_tracks_writer_cli_and_runs_artifact_tests():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/simulation.yml").read_text()
    assert "tests/test_association_bundle.py -q" in workflow
    assert "'synthlab/association_bundle.py'" in workflow
    assert "'scripts/generate_association_bundle.py'" in workflow
    assert "scripts/generate_association_bundle.py" in (root / "MANIFEST.in").read_text()
