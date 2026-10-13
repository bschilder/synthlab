"""Tiny tests of the optional engine, independent of synthlab's EHR/LLM imports.

Loading this leaf module directly lets the focused simulation gate install only
its declared numerical dependencies. The existing full-package gate is retained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("msprime")


def engine():
    name = "_synthlab_genealogy_test"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "synthlab/genealogy.py"
        if not path.is_file():
            raise ModuleNotFoundError("synthlab.genealogy is not implemented")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def config(**options):
    values = {
        "sample_ids": ("query", "reference", "relative"),
        "sequence_length": 1000,
        "seed": 91,
        "population_size": 100,
        "mutation_rate": 1e-4,
        "dtwf_generations": 5,
    }
    values.update(options)
    return engine().GenealogySimConfig(**values)


def test_deterministic_diploid_truth_retains_genealogy_and_alleles():
    first = engine().simulate_genealogy(config())
    second = engine().simulate_genealogy(config())
    assert first.scientific_signature == second.scientific_signature
    assert first.sample_ids == ("query", "reference", "relative")
    assert first.genotypes.shape == (3, len(first.alleles), 2)
    assert first.tree_sequence.time_units == "generations"
    assert len(first.alleles) > 0
    assert not first.genotypes.flags.writeable
    assert not first.site_positions.flags.writeable
    assert not first.sample_node_pairs.flags.writeable
    flat_nodes = first.sample_node_pairs.reshape(-1).tolist()
    for col, variant in enumerate(first.tree_sequence.variants(samples=flat_nodes)):
        assert first.site_positions[col] == variant.site.position
        assert first.alleles[col] == variant.alleles
        np.testing.assert_array_equal(first.genotypes[:, col, :].reshape(-1), variant.genotypes)
    for pair in first.sample_node_pairs:
        individuals = {first.tree_sequence.node(int(node)).individual for node in pair}
        assert len(individuals) == 1
    assert len(set(first.stream_seeds.values())) == len(first.stream_seeds)
    provenance = json.loads(first.provenance)
    assert provenance["models"] == ["DiscreteTimeWrightFisher", "StandardCoalescent"]
    assert (
        provenance["reference_convention"]
        == "simulated ancestral allele is REF; not a human reference"
    )


def test_zero_mutations_are_a_valid_empty_phased_panel():
    result = engine().simulate_genealogy(config(mutation_rate=0))
    assert result.genotypes.shape == (3, 0, 2)
    assert result.alleles == ()
    assert result.site_positions.size == 0
    assert result.tree_sequence.num_samples == 6


def test_mutation_stream_does_not_change_ancestry():
    first = engine().simulate_genealogy(config(mutation_rate=0))
    second = engine().simulate_genealogy(config())
    for name in ("left", "right", "parent", "child"):
        np.testing.assert_array_equal(
            getattr(first.tree_sequence.tables.edges, name),
            getattr(second.tree_sequence.tables.edges, name),
        )
    np.testing.assert_array_equal(
        first.tree_sequence.tables.nodes.time, second.tree_sequence.tables.nodes.time
    )
    assert first.scientific_signature != second.scientific_signature
    assert first.stream_seeds == second.stream_seeds


def test_multiallelic_recurrent_mutations_remain_literal_categories():
    result = engine().simulate_genealogy(
        config(sequence_length=10, population_size=20, mutation_rate=0.1)
    )
    assert any(len(alleles) > 2 for alleles in result.alleles)
    assert np.all(result.genotypes >= 0)
    for col, alleles in enumerate(result.alleles):
        assert 1 <= len(alleles) <= 4
        assert all(len(allele) == 1 and allele in "ACGT" for allele in alleles)
        assert np.max(result.genotypes[:, col]) < len(alleles)


def test_custom_demography_preserves_explicit_interleaved_person_order():
    msprime = pytest.importorskip("msprime")
    demography = msprime.Demography()
    for name in ("left", "right", "old"):
        demography.add_population(name=name, initial_size=100)
    demography.add_mass_migration(time=3, source="left", dest="right", proportion=0.25)
    demography.add_population_split(time=20, derived=["left", "right"], ancestral="old")
    original = demography.asdict()
    cfg = config(sample_ids=("r1", "l1", "r2", "l2"))
    labels = ("right", "left", "right", "left")
    result = engine().simulate_genealogy(cfg, demography=demography, sample_demes=labels)
    assert result.sample_ids == cfg.sample_ids
    assert demography.asdict() == original
    lookup = {population.name: index for index, population in enumerate(demography.populations)}
    for label, pair in zip(labels, result.sample_node_pairs):
        assert {result.tree_sequence.node(int(node)).population for node in pair} == {lookup[label]}
    assert json.loads(result.provenance)["sample_demes"] == list(labels)


def test_coalescent_only_is_explicit_and_different_seed_changes_truth():
    first = engine().simulate_genealogy(config(dtwf_generations=0))
    second = engine().simulate_genealogy(config(dtwf_generations=0, seed=92))
    assert first.scientific_signature != second.scientific_signature
    assert json.loads(first.provenance)["models"] == ["StandardCoalescent"]


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_ids": "abc"},
        {"sample_ids": ()},
        {"sample_ids": ("a", "a")},
        {"sample_ids": ("a\nb",)},
        {"sequence_length": 0},
        {"sequence_length": True},
        {"sequence_length": 1.5},
        {"seed": 0},
        {"seed": True},
        {"seed": -1},
        {"dtwf_generations": -1},
        {"dtwf_generations": 1.5},
        {"population_size": 0},
        {"population_size": float("inf")},
        {"recombination_rate": -1},
        {"mutation_rate": float("nan")},
        {"mutation_rate": True},
    ],
)
def test_configuration_rejects_ambiguous_or_invalid_values(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_explicit_resource_limits_fail_without_silent_downsampling():
    for options in ({"max_individuals": 2}, {"max_output_bytes": 1}, {"max_sites": 1}):
        with pytest.raises(ValueError, match="bound|limit"):
            engine().simulate_genealogy(config(), **options)


def test_demography_labels_require_an_exact_declared_join():
    msprime = pytest.importorskip("msprime")
    demography = msprime.Demography.isolated_model([100])
    for options in (
        {"sample_demes": ("pop_0",) * 3},
        {"demography": demography},
        {"demography": demography, "sample_demes": ("wrong",) * 3},
        {"demography": demography, "sample_demes": ("pop_0",)},
        {"demography": demography, "sample_demes": "pop_0"},
    ):
        with pytest.raises(ValueError):
            engine().simulate_genealogy(config(), **options)


def test_optional_absence_has_actionable_installation_message(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_msprime(name, *args, **kwargs):
        if name == "msprime":
            raise ImportError("simulated missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_msprime)
    with pytest.raises(ImportError, match=r"synthlab\[simulation\]"):
        engine().simulate_genealogy(config())


def test_pairs_use_individuals_when_homologue_nodes_are_nonadjacent():
    tskit = pytest.importorskip("tskit")
    tables = tskit.TableCollection(10)
    tables.individuals.add_row()
    tables.individuals.add_row()
    for individual in (0, 1, 0, 1):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=individual)
    root = tables.nodes.add_row(time=10)
    for node in range(4):
        tables.edges.add_row(0, 10, root, node)
    tables.sort()
    tree = tables.tree_sequence()
    np.testing.assert_array_equal(engine()._diploid_sample_pairs(tree, (1, 0)), [[1, 3], [0, 2]])
    with pytest.raises(ValueError, match="declared"):
        engine()._diploid_sample_pairs(tree, (0,))


def test_scientific_hash_preserves_nested_dictionary_boundaries():
    from types import SimpleNamespace

    # Two different nested records must not have the same byte serialization.
    first = {"a": {"b": 1}, "c": 2}
    second = {"a": {"b": 1, "c": 2}}
    tree = lambda payload: SimpleNamespace(
        dump_tables=lambda: SimpleNamespace(asdict=lambda: payload)
    )
    assert engine()._scientific_signature(tree(first), "recipe") != engine()._scientific_signature(
        tree(second), "recipe"
    )


@pytest.mark.parametrize("ids", [(["a"],), {"a", "b"}])
def test_sample_ids_reject_unhashable_and_unordered_sources(ids):
    with pytest.raises(ValueError):
        config(sample_ids=ids)


def test_deme_labels_reject_nonstring_values_as_input_errors():
    msprime = pytest.importorskip("msprime")
    demography = msprime.Demography.isolated_model([100])
    with pytest.raises(ValueError):
        engine().simulate_genealogy(config(), demography=demography, sample_demes=(["pop_0"],) * 3)


def test_simulation_dependency_is_an_explicit_optional_extra():
    import tomllib

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())[
        "project"
    ]
    assert project["optional-dependencies"]["simulation"] == ["msprime>=1.4.4,<2", "numpy>=2,<3"]
    assert not any(value.startswith("msprime") for value in project["dependencies"])


def test_focused_ci_uses_a_hash_checked_lock_and_retains_the_existing_suite():
    root = Path(__file__).resolve().parents[1]
    workflow = root / ".github/workflows/simulation.yml"
    assert workflow.is_file()
    text = workflow.read_text()
    assert "--require-hashes requirements/simulation-test.lock" in text
    assert "tests/test_genealogy.py" in text
    assert "pytest tests/ -v" in (root / ".github/workflows/test.yml").read_text()
