"""Back-compat shim — the SNOMED downloader now lives in :mod:`biodb.snomed`.

The full implementation (with tqdm progress, the 3-strategy auth
flow, and the OHDSI ``CONCEPT.csv`` loader) was relocated to bioDB on
2026-05-18. The GitHub Release asset also moved — same bytes, same
SHA-256, new home at ``bschilder/bioDB`` (release ``vocab-v1``).

This module re-exports the public names so any existing
``from synthlab.download_snomed import ...`` keeps working. New code
should import directly from :mod:`biodb.snomed`.
"""

from __future__ import annotations

import warnings

from biodb.snomed import (
    CACHE_DIR as _BIODB_CACHE_DIR,
)
from biodb.snomed import (
    GITHUB_ASSET_NAME,
    GITHUB_RELEASE_TAG,
    GITHUB_REPO,
    SNOMED_RELEASE_URL,
    download_concept_csv as _download_concept_csv,
)
from biodb.snomed import (
    get_concept_csv_path as _get_concept_csv_path,
)
from biodb.snomed import (
    get_snomed_data_dir as _get_snomed_data_dir,
)
from biodb.snomed import (
    is_available as _is_available,
)

# Back-compat alias for the cache directory constant.
DEFAULT_SNOMED_DATA_DIR = _BIODB_CACHE_DIR

# Re-export under the original synthlab names. The original module had
# slightly different function signatures (``output_dir`` first, plus
# ``url=`` and ``verbose=`` kwargs) — wrap to preserve those callers.


def get_snomed_data_dir():  # type: ignore[no-redef]
    """Re-export of :func:`biodb.snomed.get_snomed_data_dir`."""
    return _get_snomed_data_dir()


def get_concept_csv_path():  # type: ignore[no-redef]
    """Re-export of :func:`biodb.snomed.get_concept_csv_path`."""
    return _get_concept_csv_path()


def is_snomed_available() -> bool:
    """Back-compat name for :func:`biodb.snomed.is_available`."""
    return _is_available()


def download_snomed_vocabulary(
    output_dir=None,
    url: str = SNOMED_RELEASE_URL,
    verbose: bool = True,
    force: bool = False,
):
    """Back-compat wrapper for :func:`biodb.snomed.download_concept_csv`.

    The ``url`` argument is accepted for ABI compatibility but is now
    ignored — :mod:`biodb.snomed` always uses the release URL on the
    bioDB repo. If you were overriding ``url`` to point at a private
    mirror, set ``GITHUB_TOKEN`` instead and bioDB will use the token
    auth flow against the same release tag.
    """
    if url != SNOMED_RELEASE_URL:
        warnings.warn(
            f"synthlab.download_snomed.download_snomed_vocabulary(url=...) is "
            f"ignored — biodb.snomed always uses {SNOMED_RELEASE_URL}. "
            f"Set GITHUB_TOKEN / GH_TOKEN for private-mirror access.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _download_concept_csv(output_dir=output_dir, force=force, progress=verbose)


__all__ = [
    "DEFAULT_SNOMED_DATA_DIR",
    "GITHUB_ASSET_NAME",
    "GITHUB_RELEASE_TAG",
    "GITHUB_REPO",
    "SNOMED_RELEASE_URL",
    "download_snomed_vocabulary",
    "get_concept_csv_path",
    "get_snomed_data_dir",
    "is_snomed_available",
]
