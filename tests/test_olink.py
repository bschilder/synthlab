"""Unit tests for :mod:`synthlab.olink`.

Covers determinism, the group-effects model, LOD-driven missingness,
MCAR overlay, plate intercepts, QC warning rate, parquet round-trip,
and empty-frame edge cases. No network / filesystem heavy dependencies
— everything runs on in-memory polars DataFrames.
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")

from synthlab.olink import (
    OlinkPanelConfig,
    OlinkSimConfig,
    default_explore_3072_panel,
    load_olink_parquet,
    simulate_olink_npx,
    write_olink_parquet,
)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _uniform_panel(
    proteins: tuple[str, ...],
    mean: float = 5.0,
    sd: float = 0.6,
    lod: float = 3.0,
) -> OlinkPanelConfig:
    """Build a small OlinkPanelConfig with uniform priors across proteins."""
    return OlinkPanelConfig(
        name="test",
        proteins=proteins,
        lod={p: lod for p in proteins},
        mean={p: mean for p in proteins},
        sd={p: sd for p in proteins},
    )


# ---------------------------------------------------------------------------
# Core simulator tests
# ---------------------------------------------------------------------------


def test_simulate_is_deterministic_under_seed() -> None:
    """Two calls with the same seed return bit-identical DataFrames."""
    panel = default_explore_3072_panel()
    cfg = OlinkSimConfig(n_samples=20, panel=panel, seed=42)
    df1 = simulate_olink_npx(cfg)
    df2 = simulate_olink_npx(cfg)
    assert df1.equals(df2)


def test_group_assignments_mean_shift() -> None:
    """A +2.0 delta on CRP for "case" group moves the case NPX mean by ~2.0."""
    # Zero plate SD and no LOD loss so the mean is clean to compare.
    panel = _uniform_panel(("CRP", "IL6"), mean=5.0, sd=0.5, lod=-1000.0)
    assignments = ["case"] * 500 + ["control"] * 500
    cfg = OlinkSimConfig(
        n_samples=1000,
        panel=panel,
        group_effects={"CRP": {"case": 2.0}},
        group_assignments=assignments,
        missingness="none",
        qc_warn_rate=0.0,
        plate_effect_sd=0.0,
        seed=1,
    )
    df = simulate_olink_npx(cfg)
    crp = df.filter(pl.col("protein_id") == "CRP")
    case_mean = crp.filter(pl.col("group") == "case")["npx"].mean()
    ctrl_mean = crp.filter(pl.col("group") == "control")["npx"].mean()
    delta = case_mean - ctrl_mean
    # With sd=0.5, n=500 per arm, SE of the difference ~ 0.5 * sqrt(2/500) ~ 0.032.
    # Shift should be ~ 2.0 within 2 standard errors.
    assert abs(delta - 2.0) < 0.1, f"expected ~2.0 shift; got {delta:.3f}"


def test_lod_missingness_kicks_in() -> None:
    """With mean << lod, >60% of rows are dropped under mnar_lod."""
    # mean=1.0, lod=2.0, sd=0.5 → most values fall below LOD.
    panel = _uniform_panel(("CRP", "IL6"), mean=1.0, sd=0.5, lod=2.0)
    cfg = OlinkSimConfig(
        n_samples=200,
        panel=panel,
        missingness="mnar_lod",
        missing_rate=0.0,  # isolate LOD effect
        qc_warn_rate=0.0,
        plate_effect_sd=0.0,
        seed=7,
    )
    df = simulate_olink_npx(cfg)
    kept = df.height
    total = 200 * 2
    drop_rate = 1 - kept / total
    assert drop_rate > 0.6, (
        f"expected >60% LOD drop with mean << lod; got {drop_rate:.2f}"
    )


def test_mcar_missingness_adds_rate() -> None:
    """With missing_rate=0.3 on top of LOD, observed drop exceeds 0.3."""
    panel = _uniform_panel(("CRP", "IL6"), mean=5.0, sd=0.3, lod=3.0)
    cfg = OlinkSimConfig(
        n_samples=300,
        panel=panel,
        missingness="mnar_lod",
        missing_rate=0.3,
        qc_warn_rate=0.0,
        plate_effect_sd=0.0,
        seed=11,
    )
    df = simulate_olink_npx(cfg)
    drop_rate = 1 - df.height / (300 * 2)
    # LOD contributes a tiny bit since mean=5, lod=3 → ~0 sub-LOD; MCAR
    # contributes ~0.3. Combined rate should exceed 0.3.
    assert drop_rate > 0.3, f"expected drop > 0.3; got {drop_rate:.3f}"


def test_plate_effects_vary_by_plate() -> None:
    """2 plates (192 samples) get different means at non-zero plate_effect_sd."""
    panel = _uniform_panel(("CRP",), mean=5.0, sd=0.2, lod=-1000.0)
    cfg_off = OlinkSimConfig(
        n_samples=192,
        panel=panel,
        missingness="none",
        qc_warn_rate=0.0,
        plate_effect_sd=0.0,
        seed=3,
    )
    df_off = simulate_olink_npx(cfg_off)
    p0_off = df_off.filter(pl.col("plate_id") == "plate_0")["npx"].mean()
    p1_off = df_off.filter(pl.col("plate_id") == "plate_1")["npx"].mean()
    # With plate_effect_sd=0 the two plate means are independent samples
    # from the same protein noise — their difference is small but may be
    # nonzero from the Gaussian noise. We check that plates EXIST.
    assert p0_off is not None and p1_off is not None

    # With plate_effect_sd=1.0 the plates typically differ by > 0.2 (most
    # draws from |N(0,1) - N(0,1)| exceed 0.2 by a wide margin).
    cfg_on = OlinkSimConfig(
        n_samples=192,
        panel=panel,
        missingness="none",
        qc_warn_rate=0.0,
        plate_effect_sd=1.0,
        seed=3,
    )
    df_on = simulate_olink_npx(cfg_on)
    p0_on = df_on.filter(pl.col("plate_id") == "plate_0")["npx"].mean()
    p1_on = df_on.filter(pl.col("plate_id") == "plate_1")["npx"].mean()
    assert abs(p0_on - p1_on) > abs(p0_off - p1_off), (
        "plate_effect_sd=1.0 should produce larger inter-plate spread "
        f"than 0.0; got |{p0_on - p1_on}| vs |{p0_off - p1_off}|"
    )


def test_qc_warning_bernoulli() -> None:
    """qc_warn_rate=0.1 yields ~10% qc_warning=True rows (within tolerance)."""
    panel = _uniform_panel(("CRP", "IL6", "TNF"), mean=5.0, sd=0.2, lod=-1000.0)
    cfg = OlinkSimConfig(
        n_samples=500,
        panel=panel,
        missingness="none",
        qc_warn_rate=0.1,
        plate_effect_sd=0.0,
        seed=13,
    )
    df = simulate_olink_npx(cfg)
    observed = df["qc_warning"].sum() / df.height
    # With 1500 rows, expected SE ~ sqrt(0.1 * 0.9 / 1500) ~ 0.008. Allow 3 SE.
    assert abs(observed - 0.1) < 0.03, (
        f"qc_warning rate out of tolerance: {observed:.4f}"
    )


def test_parquet_roundtrip(tmp_path) -> None:
    """write then load returns a frame bit-equivalent to the source."""
    panel = default_explore_3072_panel()
    cfg = OlinkSimConfig(n_samples=12, panel=panel, seed=5)
    df = simulate_olink_npx(cfg)
    out = write_olink_parquet(df, tmp_path / "npx.parquet")
    assert out.is_file()
    loaded = load_olink_parquet(out)
    assert loaded.equals(df)


def test_schema_types() -> None:
    """Column dtypes are (str, str, f64, bool, str, str) in that order."""
    panel = default_explore_3072_panel()
    cfg = OlinkSimConfig(n_samples=3, panel=panel, seed=1)
    df = simulate_olink_npx(cfg)
    assert df.columns == [
        "sample_id",
        "protein_id",
        "npx",
        "qc_warning",
        "group",
        "plate_id",
    ]
    assert df.dtypes == [pl.Utf8, pl.Utf8, pl.Float64, pl.Boolean, pl.Utf8, pl.Utf8]


def test_default_panel_sizes() -> None:
    """Default panel has exactly 50 proteins and complete lod/mean/sd maps."""
    panel = default_explore_3072_panel()
    assert panel.name == "explore_3072"
    assert len(panel.proteins) == 50
    assert len(set(panel.proteins)) == 50  # unique
    for p in panel.proteins:
        assert p in panel.lod
        assert p in panel.mean
        assert p in panel.sd


def test_empty_samples() -> None:
    """n_samples=0 returns an empty frame with the canonical schema."""
    panel = default_explore_3072_panel()
    cfg = OlinkSimConfig(n_samples=0, panel=panel, seed=1)
    df = simulate_olink_npx(cfg)
    assert df.shape == (0, 6)
    assert df.columns == [
        "sample_id",
        "protein_id",
        "npx",
        "qc_warning",
        "group",
        "plate_id",
    ]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_panel_rejects_duplicates() -> None:
    """Duplicate protein IDs are rejected in __post_init__."""
    with pytest.raises(ValueError, match="unique"):
        OlinkPanelConfig(
            name="bad",
            proteins=("CRP", "CRP"),
            lod={"CRP": 3.0},
            mean={"CRP": 5.0},
            sd={"CRP": 0.6},
        )


def test_panel_rejects_nonpositive_sd() -> None:
    """sd <= 0 raises."""
    with pytest.raises(ValueError, match="sd"):
        OlinkPanelConfig(
            name="bad",
            proteins=("CRP",),
            lod={"CRP": 3.0},
            mean={"CRP": 5.0},
            sd={"CRP": 0.0},
        )


def test_sim_config_rejects_bad_missingness() -> None:
    """Unknown missingness mode is rejected."""
    panel = default_explore_3072_panel()
    with pytest.raises(ValueError, match="missingness"):
        OlinkSimConfig(
            n_samples=10,
            panel=panel,
            missingness="bogus",  # type: ignore[arg-type]
        )


def test_sim_config_rejects_mismatched_assignments() -> None:
    """group_assignments length must match n_samples."""
    panel = default_explore_3072_panel()
    with pytest.raises(ValueError, match="group_assignments"):
        OlinkSimConfig(
            n_samples=5,
            panel=panel,
            group_assignments=["case", "control"],
        )


def test_load_parquet_missing_file(tmp_path) -> None:
    """load_olink_parquet raises FileNotFoundError on a missing path."""
    with pytest.raises(FileNotFoundError):
        load_olink_parquet(tmp_path / "nope.parquet")


def test_load_parquet_bad_schema(tmp_path) -> None:
    """A parquet without the expected columns triggers ValueError."""
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"foo": [1, 2, 3]}).write_parquet(bad)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_olink_parquet(bad)
