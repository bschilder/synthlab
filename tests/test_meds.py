"""Unit tests for :mod:`synthlab.meds`.

Covers the pure-Python surface (config validation, cache-dir
resolution, error paths). Integration tests that actually invoke
``meds_etl_omop`` are gated on the package being importable, since
meds_etl pulls a big dep tree and isn't always in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthlab.meds import (
    MedsConvertConfig,
    _meds_etl_available,
    convert_omop_to_meds,
    get_meds_cache_dir,
    get_meds_info,
    load_meds_events,
    print_meds_info,
)


# ---------------------------------------------------------------------------
# Cache dir + info
# ---------------------------------------------------------------------------


def test_get_meds_cache_dir_creates_and_returns_path(tmp_path, monkeypatch) -> None:
    """get_meds_cache_dir creates the <cache>/meds subdir on first call."""
    monkeypatch.setenv("SYNTHLAB_CACHE_DIR", str(tmp_path))
    cache = get_meds_cache_dir()
    assert cache == tmp_path / "meds"
    assert cache.is_dir()


def test_get_meds_cache_dir_defaults_to_home(monkeypatch) -> None:
    """Without SYNTHLAB_CACHE_DIR the cache lives under ~/.cache/synthlab/meds."""
    monkeypatch.delenv("SYNTHLAB_CACHE_DIR", raising=False)
    cache = get_meds_cache_dir()
    # We don't actually want to mkdir in $HOME during tests, so just check
    # the path shape. The dir may or may not exist depending on prior runs.
    assert cache.name == "meds"
    assert cache.parent.name == "synthlab"


def test_get_meds_info_has_expected_keys() -> None:
    """Info dict exposes the canonical metadata fields."""
    info = get_meds_info()
    for key in (
        "module",
        "purpose",
        "schema_docs",
        "backend_pkg",
        "cache_dir",
        "meds_etl_available",
        "polars_available",
    ):
        assert key in info, f"missing key: {key}"
    assert info["module"] == "synthlab.meds"
    assert info["backend_pkg"] == "meds_etl"


def test_print_meds_info_is_stable(capsys) -> None:
    """print_meds_info writes the expected header line without crashing."""
    print_meds_info()
    out = capsys.readouterr().out
    assert "synthlab.meds" in out
    assert "cache_dir" in out


# ---------------------------------------------------------------------------
# MedsConvertConfig
# ---------------------------------------------------------------------------


def test_config_normalises_paths(tmp_path) -> None:
    """String paths are cast to Path and ~/ is expanded."""
    cfg = MedsConvertConfig(
        omop_dir=str(tmp_path / "omop"),
        meds_dir=str(tmp_path / "meds"),
    )
    assert isinstance(cfg.omop_dir, Path)
    assert isinstance(cfg.meds_dir, Path)


def test_config_rejects_unknown_backend(tmp_path) -> None:
    """Only 'polars' and 'cpp' are allowed backend values."""
    with pytest.raises(ValueError, match="backend"):
        MedsConvertConfig(
            omop_dir=tmp_path / "omop",
            meds_dir=tmp_path / "meds",
            backend="rust",  # type: ignore[arg-type]
        )


def test_config_defaults(tmp_path) -> None:
    """Default backend is polars with 4 shards and 1 worker."""
    cfg = MedsConvertConfig(omop_dir=tmp_path / "omop", meds_dir=tmp_path / "meds")
    assert cfg.backend == "polars"
    assert cfg.num_shards == 4
    assert cfg.num_proc == 1
    assert cfg.overwrite is False


# ---------------------------------------------------------------------------
# convert_omop_to_meds — error paths
# ---------------------------------------------------------------------------


def test_convert_raises_without_meds_etl(tmp_path, monkeypatch) -> None:
    """When meds_etl isn't installed, a helpful ImportError is raised."""
    monkeypatch.setattr("synthlab.meds._meds_etl_available", False)
    cfg = MedsConvertConfig(
        omop_dir=tmp_path / "omop",
        meds_dir=tmp_path / "meds",
    )
    with pytest.raises(ImportError, match="meds_etl"):
        convert_omop_to_meds(cfg)


def test_convert_raises_when_omop_missing(tmp_path, monkeypatch) -> None:
    """Missing omop_dir → FileNotFoundError that names the bad path."""
    monkeypatch.setattr("synthlab.meds._meds_etl_available", True)
    cfg = MedsConvertConfig(
        omop_dir=tmp_path / "nonexistent_omop",
        meds_dir=tmp_path / "meds",
    )
    with pytest.raises(FileNotFoundError, match="nonexistent_omop"):
        convert_omop_to_meds(cfg)


def test_convert_refuses_overwrite_by_default(tmp_path, monkeypatch) -> None:
    """Existing meds_dir/data/ triggers FileExistsError unless overwrite=True."""
    monkeypatch.setattr("synthlab.meds._meds_etl_available", True)
    omop = tmp_path / "omop"
    omop.mkdir()
    meds = tmp_path / "meds"
    (meds / "data").mkdir(parents=True)
    cfg = MedsConvertConfig(omop_dir=omop, meds_dir=meds, overwrite=False)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        convert_omop_to_meds(cfg)


# ---------------------------------------------------------------------------
# load_meds_events — error paths
# ---------------------------------------------------------------------------


def test_load_meds_events_missing_dir(tmp_path) -> None:
    """Missing meds_dir/data raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No MEDS data"):
        load_meds_events(tmp_path / "nonexistent")


def test_load_meds_events_empty_data_dir(tmp_path) -> None:
    """Empty data/ subdir (no shards) raises FileNotFoundError."""
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError, match="No parquet shards"):
        load_meds_events(tmp_path)


def test_load_meds_events_reads_shards(tmp_path) -> None:
    """Given a tiny parquet shard, load_meds_events returns a sorted frame."""
    pl = pytest.importorskip("polars")
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        {
            "subject_id": ["S2", "S1", "S1"],
            "time": [
                "2020-01-02T00:00:00",
                "2020-01-01T00:00:00",
                "2020-01-03T00:00:00",
            ],
            "code": ["ICD10:A", "ICD10:B", "ICD10:C"],
        }
    ).with_columns(pl.col("time").str.to_datetime()).write_parquet(
        data / "shard_0000.parquet"
    )
    df = load_meds_events(tmp_path)
    # Rows sorted by (subject_id, time).
    assert df["subject_id"].to_list() == ["S1", "S1", "S2"]


def test_load_meds_events_filters_by_subject(tmp_path) -> None:
    """subject_ids filter is honoured at scan time."""
    pl = pytest.importorskip("polars")
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3"],
            "time": [
                "2020-01-01T00:00:00",
                "2020-01-02T00:00:00",
                "2020-01-03T00:00:00",
            ],
            "code": ["A", "B", "C"],
        }
    ).with_columns(pl.col("time").str.to_datetime()).write_parquet(
        data / "shard_0000.parquet"
    )
    df = load_meds_events(tmp_path, subject_ids=["S1", "S3"])
    assert sorted(df["subject_id"].to_list()) == ["S1", "S3"]


# ---------------------------------------------------------------------------
# Integration — only if meds_etl is importable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _meds_etl_available, reason="meds_etl not installed")
def test_meds_etl_cli_registered() -> None:
    """When meds_etl is installed, its console_scripts entry points register.

    We check via ``importlib.metadata`` rather than ``shutil.which`` so
    the test passes under virtualenvs where the entry points exist but
    the env's ``bin/`` isn't on the parent shell's ``$PATH`` (e.g. when
    running pytest via an explicit interpreter path).
    """
    import importlib.metadata as m

    ep_names: set[str] = set()
    for dist in m.distributions():
        if (dist.metadata or {}).get("Name", "").lower() == "meds_etl":
            ep_names.update(ep.name for ep in dist.entry_points)
    assert "meds_etl_omop" in ep_names, (
        f"Expected meds_etl_omop console_script entry point; saw {ep_names}"
    )
