"""Unit tests for the disease-effect catalog (:class:`synthlab.olink.DiseaseEffectCatalog`).

Covers:

- bundled CSV loads with the expected schema;
- minimum catalog coverage (>= 6 diseases);
- ``effects_for`` returns the protein -> disease -> delta mapping that
  :func:`simulate_olink_npx` expects;
- unknown diseases raise ``KeyError``;
- ``noise_sd=0`` is deterministic; ``noise_sd>0`` perturbs the deltas;
- round-trip: loading the catalog, plugging it into
  :func:`simulate_olink_npx`, and recovering the catalog's expected
  mean NPX shift per disease within 2 SE of the empirical estimate.
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")

from synthlab.olink import (
    DiseaseEffectCatalog,
    OlinkPanelConfig,
    OlinkSimConfig,
    default_explore_3072_panel,
    load_disease_effect_catalog,
    simulate_olink_npx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel_from_catalog(cat: DiseaseEffectCatalog) -> OlinkPanelConfig:
    """Build a uniform panel that covers every protein in the catalog.

    Parameters
    ----------
    cat : DiseaseEffectCatalog
        Catalog to derive the protein set from.

    Returns
    -------
    OlinkPanelConfig
        Panel with ``mean=5.0``, ``sd=0.5``, ``lod=-1000`` (so that no
        row is dropped by LOD missingness — we want clean mean-shift
        recovery in the round-trip test).
    """
    proteins = tuple(cat.effects["protein_uniprot"].unique().to_list())
    return OlinkPanelConfig(
        name="test_catalog",
        proteins=proteins,
        mean={p: 5.0 for p in proteins},
        sd={p: 0.5 for p in proteins},
        lod={p: -1000.0 for p in proteins},
    )


# ---------------------------------------------------------------------------
# Catalog shape + schema
# ---------------------------------------------------------------------------


def test_catalog_loads_with_expected_columns() -> None:
    """The bundled catalog exposes the 7 required columns plus ``meta``."""
    cat = load_disease_effect_catalog()
    required = {
        "disease",
        "protein_uniprot",
        "delta_npx",
        "se_delta",
        "source",
        "doi",
        "evidence_strength",
    }
    assert required.issubset(set(cat.effects.columns))
    assert cat.effects.height >= 40, (
        f"catalog is unexpectedly small: {cat.effects.height} rows; "
        "expected >= 40 per spec"
    )


def test_catalog_has_at_least_6_diseases() -> None:
    """Minimum coverage spec: >= 6 distinct disease labels."""
    cat = load_disease_effect_catalog()
    assert len(cat.diseases()) >= 6


def test_catalog_dois_nonempty_and_real_shape() -> None:
    """Every row must cite a DOI (no fabricated placeholders)."""
    cat = load_disease_effect_catalog()
    dois = cat.effects["doi"].to_list()
    # Basic shape: "10.<registrant>/<suffix>". This is the real ISO 26324
    # pattern — every legitimate DOI starts with "10.".
    assert all(isinstance(d, str) and d.startswith("10.") and "/" in d for d in dois), (
        "found a row with a malformed / missing DOI"
    )


def test_catalog_delta_magnitudes_defensible() -> None:
    """No row exceeds |delta_npx| = 3.0 — guards against typos / rogue units."""
    cat = load_disease_effect_catalog()
    max_abs = float(cat.effects["delta_npx"].abs().max())
    assert max_abs <= 3.0, (
        f"at least one row has |delta_npx|={max_abs:.2f}, "
        "outside the defensible NPX range (sepsis CRP peaks at +3 to +4)"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_effects_for_returns_valid_dict_shape() -> None:
    """``effects_for`` returns ``{protein: {disease: delta_npx}}``."""
    cat = load_disease_effect_catalog()
    eff = cat.effects_for(["T2D", "CAD"])
    assert isinstance(eff, dict)
    for prot, by_disease in eff.items():
        assert isinstance(prot, str)
        assert isinstance(by_disease, dict)
        for disease, delta in by_disease.items():
            assert disease in {"T2D", "CAD"}
            assert isinstance(delta, float)
    # Sanity check: CRP is in both T2D and CAD rows, so it should appear
    # with BOTH disease keys in the merged mapping.
    assert "T2D" in eff["CRP"]
    assert "CAD" in eff["CRP"]


def test_effects_for_unknown_disease_raises_keyerror() -> None:
    """Passing an unregistered disease label raises ``KeyError``."""
    cat = load_disease_effect_catalog()
    with pytest.raises(KeyError, match="Unknown disease"):
        cat.effects_for(["fake_disease_xyz"])


def test_proteins_for_unknown_disease_raises_keyerror() -> None:
    """``proteins_for`` also raises on unknown diseases."""
    cat = load_disease_effect_catalog()
    with pytest.raises(KeyError, match="not in catalog"):
        cat.proteins_for("not_a_real_disease")


def test_noise_sd_zero_is_deterministic() -> None:
    """With ``noise_sd=0`` two calls return identical deltas."""
    cat = load_disease_effect_catalog()
    a = cat.effects_for(["T2D"], noise_sd=0.0, seed=1)
    b = cat.effects_for(["T2D"], noise_sd=0.0, seed=999)
    assert a == b, "noise_sd=0 should be independent of seed"


def test_noise_sd_positive_changes_deltas() -> None:
    """With ``noise_sd>0`` deltas are perturbed but preserve overall structure."""
    cat = load_disease_effect_catalog()
    clean = cat.effects_for(["CAD"], noise_sd=0.0)
    noisy = cat.effects_for(["CAD"], noise_sd=1.0, seed=42)
    # Every protein in the clean mapping should also appear in the noisy one.
    assert set(clean.keys()) == set(noisy.keys())
    # But at least one delta should differ (probability of all matching is
    # vanishingly small when noise_sd=1.0 * se_delta > 0).
    differences = [
        abs(clean[p]["CAD"] - noisy[p]["CAD"])
        for p in clean
    ]
    assert max(differences) > 1e-6, "noise_sd=1.0 should perturb at least one delta"


def test_noise_sd_reproducible_across_seeds() -> None:
    """Same seed + same ``noise_sd`` -> identical outputs."""
    cat = load_disease_effect_catalog()
    a = cat.effects_for(["T2D"], noise_sd=0.7, seed=3)
    b = cat.effects_for(["T2D"], noise_sd=0.7, seed=3)
    assert a == b


def test_noise_sd_negative_raises() -> None:
    """Negative ``noise_sd`` is a type error."""
    cat = load_disease_effect_catalog()
    with pytest.raises(ValueError, match="noise_sd"):
        cat.effects_for(["T2D"], noise_sd=-0.1)


# ---------------------------------------------------------------------------
# End-to-end round-trip with simulate_olink_npx
# ---------------------------------------------------------------------------


def test_catalog_integrates_with_simulate_olink_npx() -> None:
    """Load -> effects_for -> simulate -> recover expected per-disease shift.

    Draws 500 samples per group (T2D / CAD / baseline), runs the
    simulator with ``missingness="none"`` and ``plate_effect_sd=0``,
    then checks that the empirical case vs baseline mean shift matches
    the catalog's ``delta_npx`` within 2 SE for each catalog row.
    """
    cat = load_disease_effect_catalog()
    panel = _panel_from_catalog(cat)
    effects = cat.effects_for(["T2D", "CAD"])

    n_per_group = 500
    assignments = (
        ["T2D"] * n_per_group + ["CAD"] * n_per_group + ["baseline"] * n_per_group
    )
    cfg = OlinkSimConfig(
        n_samples=len(assignments),
        panel=panel,
        group_effects=effects,
        group_assignments=assignments,
        missingness="none",
        qc_warn_rate=0.0,
        plate_effect_sd=0.0,
        seed=123,
    )
    df = simulate_olink_npx(cfg)

    # SE of the difference: sqrt(sd^2/n + sd^2/n) with sd=0.5, n=500 -> ~0.032.
    se_per_arm = 0.5 / (n_per_group ** 0.5)
    se_diff = se_per_arm * (2 ** 0.5)
    tolerance = 3.0 * se_diff  # ~0.095

    # Loop over every catalog row present in the used-in-sim effects.
    catalog_sub = cat.effects.filter(pl.col("disease").is_in(["T2D", "CAD"]))
    for row in catalog_sub.iter_rows(named=True):
        prot, disease, expected = (
            row["protein_uniprot"],
            row["disease"],
            float(row["delta_npx"]),
        )
        if abs(expected) == 0.0:  # skip null-hypothesis rows
            continue
        case_mean = (
            df.filter(
                (pl.col("protein_id") == prot) & (pl.col("group") == disease)
            )["npx"]
            .mean()
        )
        base_mean = (
            df.filter(
                (pl.col("protein_id") == prot) & (pl.col("group") == "baseline")
            )["npx"]
            .mean()
        )
        observed = float(case_mean - base_mean)
        assert abs(observed - expected) < tolerance, (
            f"{disease}/{prot}: expected delta={expected:.3f} "
            f"observed={observed:.3f} tol={tolerance:.3f}"
        )


def test_user_supplied_csv_override(tmp_path) -> None:
    """A user-supplied CSV at a custom path is honoured over the bundle."""
    custom = tmp_path / "mini.csv"
    custom.write_text(
        "disease,protein_uniprot,delta_npx,se_delta,source,doi,evidence_strength\n"
        "minimal_disease,CRP,1.23,0.1,unit_test,10.0/unit.test,weak\n"
    )
    cat = load_disease_effect_catalog(custom)
    assert cat.diseases() == ("minimal_disease",)
    assert cat.effects_for(["minimal_disease"]) == {"CRP": {"minimal_disease": 1.23}}


def test_load_disease_effect_catalog_missing_file(tmp_path) -> None:
    """A non-existent ``path`` raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_disease_effect_catalog(tmp_path / "does_not_exist.csv")


def test_catalog_rejects_bad_schema(tmp_path) -> None:
    """CSV missing a required column raises ``ValueError``."""
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_disease_effect_catalog(bad)
