"""Medical Event Data Standard (MEDS) conversion utilities.

MEDS is a machine-learning-native schema for clinical event data:
https://medical-event-data-standard.github.io. Models like
[SMB-v1](https://huggingface.co/standardmodelbio) and
[MOTOR](https://github.com/som-shahlab/motor) consume MEDS parquet
as their canonical patient-history input, so being able to convert
synthetic EHR (Synthea, UKB Synthetic) into MEDS opens the door to
plugging SynthLab's output directly into any MEDS-compatible model.

This module is a thin wrapper around
[meds_etl](https://pypi.org/project/meds-etl/) — the official
open-source ETL library maintained by the MEDS community — with
SynthLab-specific convenience (cache-dir discovery, config
dataclass, loading helpers that return polars DataFrames ready
for downstream tokenizers).

Pipeline
--------

```
SyntheaRunner.run()                          # synthlab.synthea
    → Synthea CSV output
convert_synthea_to_omop(synthea_dir, ...)    # synthlab.synthea
    → OMOP v5.4 CDM CSVs
convert_omop_to_meds(omop_dir, meds_dir)     # synthlab.meds (this module)
    → MEDS parquet shards
load_meds_events(meds_dir)                   # synthlab.meds
    → polars DataFrame ready for SMB-v1 / MOTOR / etc.
```

Example
-------

```python
from synthlab.meds import convert_omop_to_meds, load_meds_events

convert_omop_to_meds(
    omop_dir="~/.cache/synthlab/synthea/omop_10",
    meds_dir="~/.cache/synthlab/meds/synthea_10",
)
df = load_meds_events("~/.cache/synthlab/meds/synthea_10")
print(df.head())
```
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy deps
_polars_available = False
_meds_etl_available = False

try:
    import polars as pl  # noqa: F401
    _polars_available = True
except ImportError:
    pass

try:
    import meds_etl  # noqa: F401 -- import-time availability check
    _meds_etl_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Cache dir + info helpers (mirrors the pattern of other synthlab modules)
# ---------------------------------------------------------------------------


def get_meds_cache_dir() -> Path:
    """Return the cache directory for MEDS-format datasets.

    Mirrors ``synthlab.synthea.get_synthea_cache_dir`` — resolves to
    ``$SYNTHLAB_CACHE_DIR/meds`` when the env-var is set, otherwise
    ``~/.cache/synthlab/meds``. Creates the directory if absent.

    Returns
    -------
    pathlib.Path
        Absolute path to the cache directory.

    Examples
    --------
    >>> p = get_meds_cache_dir()
    >>> p.is_dir()
    True
    >>> p.name
    'meds'
    """
    base = os.environ.get("SYNTHLAB_CACHE_DIR", str(Path.home() / ".cache" / "synthlab"))
    cache_dir = Path(base).expanduser() / "meds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_meds_info() -> dict:
    """Return a metadata dict describing the MEDS converter module.

    Returns
    -------
    dict
        Keys: ``module``, ``purpose``, ``schema_docs``, ``backend_pkg``,
        ``cache_dir``, ``meds_etl_available``, ``polars_available``.

    Examples
    --------
    >>> info = get_meds_info()
    >>> info["module"]
    'synthlab.meds'
    """
    return {
        "module": "synthlab.meds",
        "purpose": "Convert OMOP v5.4 CSVs to MEDS parquet for ML-ready EHR encoding.",
        "schema_docs": "https://medical-event-data-standard.github.io",
        "backend_pkg": "meds_etl",
        "cache_dir": str(get_meds_cache_dir()),
        "meds_etl_available": _meds_etl_available,
        "polars_available": _polars_available,
    }


def print_meds_info() -> None:
    """Pretty-print :func:`get_meds_info` to stdout.

    Examples
    --------
    >>> print_meds_info()  # doctest: +SKIP
    synthlab.meds
    - purpose: ...
    ...
    """
    info = get_meds_info()
    print(f"{info['module']}")
    print(f"- purpose: {info['purpose']}")
    print(f"- schema_docs: {info['schema_docs']}")
    print(f"- backend_pkg: {info['backend_pkg']}")
    print(f"- cache_dir: {info['cache_dir']}")
    print(f"- meds_etl installed: {info['meds_etl_available']}")
    print(f"- polars installed: {info['polars_available']}")


# ---------------------------------------------------------------------------
# Config + conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MedsConvertConfig:
    """Config for :func:`convert_omop_to_meds`.

    Attributes
    ----------
    omop_dir : pathlib.Path
        Path to a directory of OMOP v5.4 CSV files (one per OMOP table —
        ``person.csv``, ``condition_occurrence.csv``, etc.). Typically
        produced by :func:`synthlab.synthea.convert_synthea_to_omop`.
    meds_dir : pathlib.Path
        Destination directory for the MEDS parquet output. Will be
        created. If ``overwrite=False`` and the directory already
        contains ``data/``, conversion is skipped.
    backend : {"polars", "cpp"}, default ``"polars"``
        :mod:`meds_etl` has two backends. ``polars`` is Python-only
        and works anywhere; ``cpp`` is faster but requires the
        optional ``meds_etl[cpp]`` install.
    num_shards : int, default ``4``
        Number of shards the MEDS dataset is split into. Lower
        shard counts use more memory per worker; see the
        :mod:`meds_etl` README for tuning.
    num_proc : int, default ``1``
        Worker processes. Match ``num_shards`` for the cpp backend,
        keep low for polars to avoid OOM.
    overwrite : bool, default ``False``
        If ``True`` and ``meds_dir`` already exists, it is removed
        before conversion.

    Examples
    --------
    >>> cfg = MedsConvertConfig(omop_dir="/tmp/omop", meds_dir="/tmp/meds")
    >>> cfg.backend
    'polars'
    """

    omop_dir: Path
    meds_dir: Path
    backend: Literal["polars", "cpp"] = "polars"
    num_shards: int = 4
    num_proc: int = 1
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Normalise paths to :class:`Path` + validate backend choice."""
        # frozen dataclass: __setattr__ goes through object
        object.__setattr__(self, "omop_dir", Path(self.omop_dir).expanduser())
        object.__setattr__(self, "meds_dir", Path(self.meds_dir).expanduser())
        if self.backend not in {"polars", "cpp"}:
            raise ValueError(
                f"backend must be 'polars' or 'cpp'; got {self.backend!r}"
            )


def _require_meds_etl() -> None:
    """Raise :class:`ImportError` with a helpful message if missing.

    Raises
    ------
    ImportError
        When :mod:`meds_etl` cannot be imported. Message names the
        right ``pip install`` invocation.
    """
    if not _meds_etl_available:
        raise ImportError(
            "meds_etl is required for MEDS conversion. Install with "
            "`pip install meds_etl` (or `pip install 'meds_etl[cpp]'` "
            "for the faster C++ backend). See "
            "https://github.com/Medical-Event-Data-Standard/meds_etl."
        )


def convert_omop_to_meds(config: MedsConvertConfig) -> Path:
    """Convert an OMOP v5.4 CSV directory into a MEDS parquet dataset.

    Thin wrapper around the :mod:`meds_etl` ``meds_etl_omop`` CLI.
    The CLI is invoked via ``subprocess`` so a failure surfaces the
    full ``meds_etl`` traceback without this wrapper swallowing it.

    Parameters
    ----------
    config : MedsConvertConfig

    Returns
    -------
    pathlib.Path
        Path to ``config.meds_dir`` after successful conversion.

    Raises
    ------
    ImportError
        If :mod:`meds_etl` isn't installed.
    FileNotFoundError
        If ``config.omop_dir`` doesn't exist.
    RuntimeError
        If the ``meds_etl_omop`` CLI exits non-zero. The wrapped
        ``stderr`` is attached to the error message.
    FileExistsError
        If ``config.meds_dir`` already has a ``data/`` subdir and
        ``config.overwrite`` is ``False``.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> cfg = MedsConvertConfig(omop_dir="omop/", meds_dir="meds/")
    >>> convert_omop_to_meds(cfg)
    PosixPath('meds')
    """
    _require_meds_etl()
    if not config.omop_dir.is_dir():
        raise FileNotFoundError(
            f"omop_dir does not exist: {config.omop_dir}"
        )
    existing_data = config.meds_dir / "data"
    if existing_data.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{existing_data} already contains MEDS data; pass "
                "overwrite=True on MedsConvertConfig to replace it."
            )
        shutil.rmtree(config.meds_dir)
    config.meds_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "meds_etl_omop",
        str(config.omop_dir),
        str(config.meds_dir),
        "--backend",
        config.backend,
        "--num_shards",
        str(config.num_shards),
        "--num_proc",
        str(config.num_proc),
    ]
    logger.info("Running meds_etl_omop: %s", " ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"meds_etl_omop exited with {completed.returncode}. "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    logger.info("MEDS conversion complete: %s", config.meds_dir)
    return config.meds_dir


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_meds_events(
    meds_dir: Path | str,
    *,
    subject_ids: Optional[list[str]] = None,
):
    """Load MEDS event rows from a converted dataset into a polars DataFrame.

    Reads every parquet shard under ``<meds_dir>/data/`` and concatenates
    them. Filtering by ``subject_ids`` pushes the filter into the polars
    scan, so a cohort of, say, 500 patients out of 100k costs roughly
    ``O(patient_count)`` rather than ``O(dataset)``.

    Parameters
    ----------
    meds_dir : pathlib.Path or str
        Directory produced by :func:`convert_omop_to_meds` (must
        contain a ``data/`` subdirectory of parquet shards).
    subject_ids : list[str], optional
        If given, only rows with ``subject_id`` in this list are
        returned. Accepts int-valued subject IDs too — they're cast
        to string before comparison because parquet schema varies
        across ETL backends.

    Returns
    -------
    polars.DataFrame
        Columns: ``subject_id``, ``time``, ``code``, plus any
        auxiliary columns produced by the ETL (``value``, ``table``,
        ``numeric_value``, …). Sorted by ``(subject_id, time)``
        which is the order MEDS-native consumers expect.

    Raises
    ------
    ImportError
        If :mod:`polars` isn't installed.
    FileNotFoundError
        If ``meds_dir/data`` doesn't exist or has no parquet shards.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> df = load_meds_events("~/.cache/synthlab/meds/synthea_100")
    >>> df.columns[:3]
    ['subject_id', 'time', 'code']
    """
    if not _polars_available:
        raise ImportError(
            "polars is required for load_meds_events. "
            "`pip install polars` (already listed in synthlab's core deps)."
        )
    import polars as pl

    data_dir = Path(meds_dir).expanduser() / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"No MEDS data subdirectory at {data_dir}; "
            "did convert_omop_to_meds() run successfully?"
        )
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No parquet shards under {data_dir}."
        )
    lf = pl.scan_parquet([str(s) for s in shards])
    if subject_ids is not None:
        sid_strs = [str(s) for s in subject_ids]
        lf = lf.filter(pl.col("subject_id").cast(pl.Utf8).is_in(sid_strs))
    return lf.sort(["subject_id", "time"]).collect()
