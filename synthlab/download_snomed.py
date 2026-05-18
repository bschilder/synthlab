"""Back-compat shim — SNOMED bulk downloader was retired in 2026-05-18.

The 175 MB OHDSI ``CONCEPT.csv`` used to be shipped as a GitHub release
asset. SNOMED CT's license (free in Member countries via UMLS / IHTSDO,
paid Affiliate license elsewhere) doesn't permit onward redistribution
from a public mirror, so the asset was deleted and the in-package
downloader removed.

What replaces it:

* **Per-concept lookups** — ``biodb.snomed.query_concept`` /
  ``search_concepts`` / ``get_descendants`` / ``get_ancestors`` / etc.
  route through EBI's OLS4. EBI handles SNOMED CT licensing
  server-side, so callers don't need their own UMLS/IHTSDO license.
* **Bulk data** — obtain a ``CONCEPT.csv`` from `OHDSI Athena
  <https://athena.ohdsi.org>`_ after accepting the SNOMED CT license,
  then load it with ``biodb.snomed.load_concept_csv`` (or
  ``load_concept_csv_from_zip`` for the raw Athena bundle).

This module re-exports the OLS-backed query helpers and the parsers
so any existing ``from synthlab.download_snomed import ...`` import
keeps working. The old ``download_snomed_vocabulary`` /
``is_snomed_available`` / ``get_concept_csv_path`` names are kept as
deprecated stubs that raise :class:`RuntimeError` with migration
guidance — they cannot be transparently emulated since the asset is
gone.
"""

from __future__ import annotations

from biodb.snomed import (
    ATHENA_DOWNLOAD_PAGE,
    CACHE_DIR as DEFAULT_SNOMED_DATA_DIR,
    get_ancestors,
    get_children,
    get_descendants,
    get_parents,
    get_snomed_data_dir,
    load_concept_csv,
    load_concept_csv_from_zip,
    query_concept,
    search_concepts,
)

_MIGRATION_NOTE = (
    "synthlab.download_snomed.{name} is retired (2026-05-18). The OHDSI "
    f"CONCEPT.csv is no longer redistributed from a public mirror — SNOMED "
    f"CT's license doesn't permit it. Get a vocabulary bundle from "
    f"{ATHENA_DOWNLOAD_PAGE} (accept the SNOMED CT license first), then "
    f"call ``biodb.snomed.load_concept_csv(path)`` or "
    f"``biodb.snomed.load_concept_csv_from_zip(zip_path)``."
)


def download_snomed_vocabulary(*args, **kwargs):
    """Retired — see module docstring + the error message for the migration path."""
    raise RuntimeError(_MIGRATION_NOTE.format(name="download_snomed_vocabulary"))


def is_snomed_available() -> bool:
    """Retired — see module docstring + the error message for the migration path."""
    raise RuntimeError(_MIGRATION_NOTE.format(name="is_snomed_available"))


def get_concept_csv_path():
    """Retired — see module docstring + the error message for the migration path."""
    raise RuntimeError(_MIGRATION_NOTE.format(name="get_concept_csv_path"))


__all__ = [
    "ATHENA_DOWNLOAD_PAGE",
    "DEFAULT_SNOMED_DATA_DIR",
    # OLS-backed query helpers (the live, working surface):
    "get_ancestors",
    "get_children",
    "get_descendants",
    "get_parents",
    "get_snomed_data_dir",
    "load_concept_csv",
    "load_concept_csv_from_zip",
    "query_concept",
    "search_concepts",
    # Deprecated stubs (raise RuntimeError on call):
    "download_snomed_vocabulary",
    "get_concept_csv_path",
    "is_snomed_available",
]
