"""Olink-NPX proteomics simulator.

Simulate case/control Olink proteomics data (subject x protein -> NPX)
with limit-of-detection (LOD) driven missingness, group effects, and
per-plate batch effects. NPX ("Normalized Protein eXpression") is
Olink's log2 relative quantification unit used across all
[Olink Explore](https://olink.com/products/olink-explore) panels.

As of 2026-04 there is no widely-used open-source Olink simulator;
closest analogues are
[MSstatsSampleSize](https://bioconductor.org/packages/MSstatsSampleSize/)
(LC-MS/MS, not NPX) and
[OlinkAnalyze](https://github.com/Olink-Proteomics/OlinkRPackage) (NPX
demo tables, no simulator). Priors are informed by the
[UKB-PPP paper](https://www.nature.com/articles/s41586-023-06592-6)
(Sun et al. 2023, 2,923 proteins x ~54k participants) and the
OlinkAnalyze ``npx_data1`` / ``npx_data2`` tables.

Model: ``NPX[i, j] = mean[j] + plate_eff[i] + group_shift[i, j] + eps``
with ``plate_eff ~ N(0, plate_effect_sd^2)`` (96 samples / plate) and
``eps ~ N(0, sd[j]^2)``. ``mnar_lod`` drops 80% of sub-LOD values
(soft LOD); ``mcar``/``mar`` add uniform ``missing_rate`` drop (MAR
currently aliases MCAR). Full MAR, multi-plate batch effects, and a
realistic PEA dilution noise model are deferred to a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import polars as pl

__all__ = [
    "OlinkPanelConfig",
    "OlinkSimConfig",
    "simulate_olink_npx",
    "default_explore_3072_panel",
    "write_olink_parquet",
    "load_olink_parquet",
    "DiseaseEffectCatalog",
    "load_disease_effect_catalog",
]

# Columns every disease-effects catalog CSV must provide. Any extra columns
# are preserved untouched (useful for meta-notes / study-design tags).
_CATALOG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "disease",
    "protein_uniprot",
    "delta_npx",
    "se_delta",
    "source",
    "doi",
    "evidence_strength",
)

# 50 protein symbols drawn from UKB-PPP Explore 3072 — illustrative mix of
# inflammation / cardiovascular / oncology markers. Downstream users should
# override with biologically-relevant subsets where needed.
_DEFAULT_PANEL_PROTEINS: tuple[str, ...] = (
    "CRP", "IL6", "TNF", "IL1B", "IL10", "IL8", "IFNG", "IL17A", "BNP",
    "NT-proBNP", "TROPT", "TROPI", "LEP", "ADIPOQ", "GDF15", "NEFL", "TAU",
    "AB42", "AB40", "NRGN", "PSA", "CA125", "CEA", "AFP", "CA199", "VEGFA",
    "PDGFA", "FGF2", "EGF", "HGF", "TGFB1", "BMP2", "IGF1", "IGFBP3", "INS",
    "GLP1", "GIP", "GHRL", "CORT", "FABP4", "APOA1", "APOB", "LDLR", "PCSK9",
    "MMP9", "MMP2", "TIMP1", "SERPINE1", "VWF", "FGB",
)


def _unique_preserve_order(items: Sequence[str]) -> tuple[str, ...]:
    """De-duplicate ``items`` while preserving first-seen order.

    Parameters
    ----------
    items : Sequence[str]
        Strings that may contain duplicates.

    Returns
    -------
    tuple[str, ...]
        First-occurrence order preserved.
    """
    seen: set[str] = set()
    return tuple(x for x in items if not (x in seen or seen.add(x)))


@dataclass(frozen=True)
class OlinkPanelConfig:
    """An Olink panel specification — proteins + per-protein priors.

    Built-in presets (:func:`default_explore_3072_panel`) hard-code
    priors from the UKB-PPP paper (Sun et al. 2023) and the OlinkAnalyze
    ``npx_data1`` / ``npx_data2`` demo tables: NPX ~ ``N(mean, sd^2)``
    with ``mean ~ 5`` log2-units, ``sd ~ 0.6``, ``lod ~ mean - 2*sd``.

    Attributes
    ----------
    name : str
        Panel label, e.g. ``"explore_3072"`` / ``"explore_ht"`` / custom.
    proteins : tuple[str, ...]
        Ordered tuple of protein IDs (UniProt IDs or gene symbols);
        duplicates rejected.
    lod : Mapping[str, float]
        Per-protein limit of detection on the NPX scale; values below
        LOD are candidates for ``mnar_lod`` missingness.
    mean : Mapping[str, float]
        Per-protein baseline NPX mean.
    sd : Mapping[str, float]
        Per-protein NPX stddev (must be ``> 0``).
    """

    name: str
    proteins: tuple[str, ...]
    lod: Mapping[str, float]
    mean: Mapping[str, float]
    sd: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate uniqueness, key coverage, and ``sd > 0``."""
        if len(set(self.proteins)) != len(self.proteins):
            raise ValueError("OlinkPanelConfig.proteins must be unique.")
        miss_l = [p for p in self.proteins if p not in self.lod]
        miss_m = [p for p in self.proteins if p not in self.mean]
        miss_s = [p for p in self.proteins if p not in self.sd]
        if miss_l or miss_m or miss_s:
            raise ValueError(
                f"OlinkPanelConfig missing lod={miss_l}, "
                f"mean={miss_m}, sd={miss_s}."
            )
        for p in self.proteins:
            if self.sd[p] <= 0:
                raise ValueError(
                    f"OlinkPanelConfig.sd[{p!r}] must be > 0; got {self.sd[p]}."
                )


@dataclass(frozen=True)
class OlinkSimConfig:
    """Configuration for :func:`simulate_olink_npx`.

    Attributes
    ----------
    n_samples : int
        Number of subjects to simulate (``>= 0``).
    panel : OlinkPanelConfig
        Panel specification.
    group_effects : Mapping[str, Mapping[str, float]], optional
        Per-protein NPX shift per group label, e.g.
        ``{"CRP": {"case": 1.2}}``. Proteins not listed get no shift.
    group_assignments : Sequence[str], optional
        Length-``n_samples`` sequence of group labels; ``None`` means
        every sample is labelled ``"baseline"``.
    missingness : {"mcar", "mar", "mnar_lod", "none"}, default "mnar_lod"
        Missingness model; ``"mar"`` currently aliases ``"mcar"`` (full
        MAR is deferred).
    missing_rate : float, default 0.05
        Extra MCAR rate on top of LOD-driven missingness; ``[0, 1]``.
    qc_warn_rate : float, default 0.01
        Per-row Bernoulli rate for ``qc_warning=True``.
    plate_effect_sd : float, default 0.15
        Stddev of the per-plate intercept (96 samples / plate).
    seed : int, default 0
        NumPy RNG seed; deterministic under fixed seed.
    """

    n_samples: int
    panel: OlinkPanelConfig
    group_effects: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    group_assignments: Sequence[str] | None = None
    missingness: Literal["mcar", "mar", "mnar_lod", "none"] = "mnar_lod"
    missing_rate: float = 0.05
    qc_warn_rate: float = 0.01
    plate_effect_sd: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate scalar bounds and ``group_assignments`` length."""
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0; got {self.n_samples}.")
        if not (0.0 <= self.missing_rate <= 1.0):
            raise ValueError(f"missing_rate must be in [0, 1]; got {self.missing_rate}.")
        if not (0.0 <= self.qc_warn_rate <= 1.0):
            raise ValueError(f"qc_warn_rate must be in [0, 1]; got {self.qc_warn_rate}.")
        if self.plate_effect_sd < 0:
            raise ValueError(f"plate_effect_sd must be >= 0; got {self.plate_effect_sd}.")
        if self.missingness not in {"mcar", "mar", "mnar_lod", "none"}:
            raise ValueError(
                f"missingness must be one of 'mcar'/'mar'/'mnar_lod'/'none'; "
                f"got {self.missingness!r}."
            )
        if (
            self.group_assignments is not None
            and len(self.group_assignments) != self.n_samples
        ):
            raise ValueError(
                f"group_assignments length ({len(self.group_assignments)}) "
                f"must equal n_samples ({self.n_samples})."
            )


_SCHEMA: dict[str, pl.DataType] = {
    "sample_id": pl.Utf8,
    "protein_id": pl.Utf8,
    "npx": pl.Float64,
    "qc_warning": pl.Boolean,
    "group": pl.Utf8,
    "plate_id": pl.Utf8,
}


def simulate_olink_npx(config: OlinkSimConfig) -> pl.DataFrame:
    """Simulate an Olink-style long-form NPX DataFrame.

    Pipeline: (1) assign each sample a plate (96 per plate, in order);
    (2) draw plate intercepts ``~ N(0, plate_effect_sd**2)``; (3) per
    cell, ``npx = mean[p] + plate_eff + group_effect[p][group] + eps``
    with ``eps ~ N(0, sd[p]**2)``; (4) apply missingness — ``mnar_lod``
    drops 80% of sub-LOD values (soft LOD), ``mcar`` adds uniform
    ``missing_rate`` drop, ``mar`` aliases ``mcar``, ``none`` skips;
    (5) flag Bernoulli ``qc_warn_rate`` QC warnings on surviving rows.
    Deterministic under fixed seed.

    Parameters
    ----------
    config : OlinkSimConfig
        Full simulation spec.

    Returns
    -------
    polars.DataFrame
        Long-form frame with columns ``sample_id`` (``str``),
        ``protein_id`` (``str``), ``npx`` (``f64``), ``qc_warning``
        (``bool``), ``group`` (``str``), ``plate_id`` (``str``).

    Examples
    --------
    >>> from synthlab.olink import default_explore_3072_panel
    >>> cfg = OlinkSimConfig(n_samples=3, panel=default_explore_3072_panel(), seed=1)
    >>> df = simulate_olink_npx(cfg)
    >>> sorted(df.columns)
    ['group', 'npx', 'plate_id', 'protein_id', 'qc_warning', 'sample_id']
    """
    panel = config.panel
    n_proteins = len(panel.proteins)
    n = config.n_samples
    if n == 0 or n_proteins == 0:
        return pl.DataFrame(schema=_SCHEMA)
    rng = np.random.default_rng(config.seed)
    sample_ids = np.array([f"S{i:04d}" for i in range(n)], dtype=object)
    plate_idx = np.arange(n) // 96
    plate_labels = np.array([f"plate_{p}" for p in plate_idx], dtype=object)
    groups = (
        np.full(n, "baseline", dtype=object)
        if config.group_assignments is None
        else np.array(list(config.group_assignments), dtype=object)
    )
    # Plate intercepts per sample.
    n_plates = int(plate_idx.max()) + 1
    plate_intercepts = (
        rng.normal(0.0, config.plate_effect_sd, size=n_plates)
        if config.plate_effect_sd > 0
        else np.zeros(n_plates, dtype=np.float64)
    )
    plate_eff = plate_intercepts[plate_idx]
    # Per-protein priors in panel.proteins order.
    mean_vec = np.array([panel.mean[p] for p in panel.proteins], dtype=np.float64)
    sd_vec = np.array([panel.sd[p] for p in panel.proteins], dtype=np.float64)
    lod_vec = np.array([panel.lod[p] for p in panel.proteins], dtype=np.float64)
    # Group shift matrix; nonzero only for (protein, group) pairs in config.
    group_shift = np.zeros((n, n_proteins), dtype=np.float64)
    for j, p in enumerate(panel.proteins):
        mapping = config.group_effects.get(p)
        if not mapping:
            continue
        for i, g in enumerate(groups):
            if g in mapping:
                group_shift[i, j] = mapping[g]
    # NPX = mean + plate + group + eps.
    eps = rng.standard_normal(size=(n, n_proteins)) * sd_vec
    npx = mean_vec[np.newaxis, :] + plate_eff[:, np.newaxis] + group_shift + eps
    # Missingness masks.
    keep = np.ones_like(npx, dtype=bool)
    if config.missingness == "mnar_lod":
        below_lod = npx < lod_vec[np.newaxis, :]
        keep &= ~(below_lod & (rng.random(size=npx.shape) < 0.8))
    if (
        config.missingness in {"mcar", "mar", "mnar_lod"}
        and config.missing_rate > 0
    ):
        keep &= ~(rng.random(size=npx.shape) < config.missing_rate)
    # Flatten (row-major), mask, build frame.
    flat_keep = keep.reshape(-1)
    flat_samples = np.repeat(sample_ids, n_proteins)
    flat_plates = np.repeat(plate_labels, n_proteins)
    flat_groups = np.repeat(groups, n_proteins)
    flat_proteins = np.tile(np.asarray(panel.proteins, dtype=object), n)
    flat_npx = npx.reshape(-1)
    n_kept = int(flat_keep.sum())
    qc_flags = (
        rng.random(size=n_kept) < config.qc_warn_rate
        if n_kept > 0 and config.qc_warn_rate > 0
        else np.zeros(n_kept, dtype=bool)
    )
    return pl.DataFrame(
        {
            "sample_id": pl.Series("sample_id", flat_samples[flat_keep], dtype=pl.Utf8),
            "protein_id": pl.Series("protein_id", flat_proteins[flat_keep], dtype=pl.Utf8),
            "npx": pl.Series("npx", flat_npx[flat_keep], dtype=pl.Float64),
            "qc_warning": pl.Series("qc_warning", qc_flags, dtype=pl.Boolean),
            "group": pl.Series("group", flat_groups[flat_keep], dtype=pl.Utf8),
            "plate_id": pl.Series("plate_id", flat_plates[flat_keep], dtype=pl.Utf8),
        }
    )


def default_explore_3072_panel(
    mean: float = 5.0,
    sd: float = 0.6,
    lod: float = 3.0,
) -> OlinkPanelConfig:
    """Return an :class:`OlinkPanelConfig` mirroring Olink Explore 3072.

    Uses 50 named proteins — a tiny subset of the real 3,072-plex panel
    chosen for test stability. Priors reflect aggregate UKB-PPP NPX
    statistics (Sun et al. 2023) and the OlinkAnalyze ``npx_data1`` /
    ``npx_data2`` demo tables: NPX ~ ``N(5, 0.6^2)``, LOD typically
    ``mean - 2*sd``.

    Parameters
    ----------
    mean : float, default 5.0
        Baseline NPX mean applied to every protein.
    sd : float, default 0.6
        NPX stddev applied to every protein (``> 0``).
    lod : float, default 3.0
        Limit of detection applied to every protein.

    Returns
    -------
    OlinkPanelConfig
        50-protein panel with name ``"explore_3072"``.
    """
    proteins = _unique_preserve_order(_DEFAULT_PANEL_PROTEINS)
    if len(proteins) < 50:
        proteins += tuple(f"PROT{i:04d}" for i in range(50 - len(proteins)))
    elif len(proteins) > 50:
        proteins = proteins[:50]
    return OlinkPanelConfig(
        name="explore_3072",
        proteins=proteins,
        lod={p: float(lod) for p in proteins},
        mean={p: float(mean) for p in proteins},
        sd={p: float(sd) for p in proteins},
    )


def write_olink_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Write an Olink NPX DataFrame to parquet (snappy compression).

    Parameters
    ----------
    df : polars.DataFrame
        Long-form NPX DataFrame — typically the output of
        :func:`simulate_olink_npx`.
    path : str or pathlib.Path
        Output path; parent directory is created if absent.

    Returns
    -------
    pathlib.Path
        The resolved output path.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="snappy")
    return out


def load_olink_parquet(path: str | Path) -> pl.DataFrame:
    """Load an Olink NPX parquet and validate its schema.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a parquet file written by :func:`write_olink_parquet`.

    Returns
    -------
    polars.DataFrame
        DataFrame with the canonical Olink NPX schema: ``sample_id``,
        ``protein_id``, ``npx``, ``qc_warning``, ``group``,
        ``plate_id``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the loaded frame is missing expected columns.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Olink parquet not found: {p}")
    df = pl.read_parquet(p)
    missing = [c for c in _SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(
            f"Olink parquet at {p} missing expected columns: {missing}. "
            f"Found: {df.columns}."
        )
    return df


# ---------------------------------------------------------------------------
# Disease-effect catalog — curated per-disease protein NPX shifts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiseaseEffectCatalog:
    """Curated per-disease protein effect sizes from published Olink / plasma
    proteomics literature.

    Loaded from the bundled
    [`synthlab/data/olink_disease_effects.csv`](../data/olink_disease_effects.csv)
    (or a user-supplied CSV with the same schema). Effects are **log2 NPX-unit
    mean shifts** of cases vs a demographically-matched baseline; ``se_delta``
    reflects between-study / between-cohort variability. Every row is
    annotated with a ``source`` label and a real DOI — see the CSV itself for
    citation-level sourcing of every estimate.

    The catalog plugs directly into :func:`simulate_olink_npx` via
    :meth:`effects_for` — see the Examples. Row schema is validated on
    construction (``disease``, ``protein_uniprot``, ``delta_npx``,
    ``se_delta``, ``source``, ``doi``, ``evidence_strength``).

    Parameters
    ----------
    effects : polars.DataFrame
        The loaded catalog as a polars frame. Must contain every column
        listed in ``_CATALOG_REQUIRED_COLUMNS``; extra columns (e.g. the
        bundled ``meta`` notes column) are preserved unmodified.

    Examples
    --------
    Load the bundled catalog and feed it into the simulator:

    >>> from synthlab import (
    ...     OlinkSimConfig,
    ...     default_explore_3072_panel,
    ...     load_disease_effect_catalog,
    ...     simulate_olink_npx,
    ... )
    >>> cat = load_disease_effect_catalog()
    >>> effects = cat.effects_for(["T2D", "CAD"])
    >>> cfg = OlinkSimConfig(
    ...     n_samples=10,
    ...     panel=default_explore_3072_panel(),
    ...     group_effects=effects,
    ...     group_assignments=["T2D"] * 5 + ["baseline"] * 5,
    ...     seed=42,
    ... )
    >>> df = simulate_olink_npx(cfg)
    >>> sorted(df.columns)
    ['group', 'npx', 'plate_id', 'protein_id', 'qc_warning', 'sample_id']
    """

    effects: pl.DataFrame

    def __post_init__(self) -> None:
        """Validate required columns + dtypes on construction.

        Raises
        ------
        ValueError
            If any required column is missing, or if ``delta_npx`` /
            ``se_delta`` are not castable to floats.
        """
        missing = [c for c in _CATALOG_REQUIRED_COLUMNS if c not in self.effects.columns]
        if missing:
            raise ValueError(
                f"DiseaseEffectCatalog is missing required columns: {missing}. "
                f"Found: {self.effects.columns}."
            )
        for col in ("delta_npx", "se_delta"):
            if not self.effects[col].dtype.is_numeric():
                raise ValueError(
                    f"DiseaseEffectCatalog column {col!r} must be numeric; "
                    f"got dtype={self.effects[col].dtype}."
                )

    def diseases(self) -> tuple[str, ...]:
        """Return the sorted list of registered disease labels.

        Returns
        -------
        tuple[str, ...]
            Unique disease labels from the ``disease`` column, sorted
            alphabetically. Useful for populating dropdowns / docs.

        Examples
        --------
        >>> from synthlab import load_disease_effect_catalog
        >>> cat = load_disease_effect_catalog()
        >>> "T2D" in cat.diseases()
        True
        """
        return tuple(sorted(self.effects["disease"].unique().to_list()))

    def proteins_for(self, disease: str) -> tuple[str, ...]:
        """Return the proteins with non-zero effects in ``disease``.

        Parameters
        ----------
        disease : str
            Disease label, e.g. ``"T2D"``. Must be a member of
            :meth:`diseases`; a ``KeyError`` is raised otherwise.

        Returns
        -------
        tuple[str, ...]
            Protein UniProt / gene-symbol IDs with ``|delta_npx| > 0`` for
            the requested disease, in the CSV's row order. Rows with
            exactly-zero ``delta_npx`` (e.g. null-hypothesis placeholder
            rows) are omitted.

        Raises
        ------
        KeyError
            If ``disease`` is not present in the catalog.

        Examples
        --------
        >>> from synthlab import load_disease_effect_catalog
        >>> cat = load_disease_effect_catalog()
        >>> "CRP" in cat.proteins_for("IBD")
        True
        """
        if disease not in self.effects["disease"].unique().to_list():
            raise KeyError(
                f"Disease {disease!r} not in catalog. Registered: {self.diseases()}."
            )
        sub = self.effects.filter(
            (pl.col("disease") == disease) & (pl.col("delta_npx").abs() > 0.0)
        )
        return tuple(sub["protein_uniprot"].to_list())

    def effects_for(
        self,
        diseases: Sequence[str],
        *,
        noise_sd: float = 0.0,
        seed: int = 0,
    ) -> Mapping[str, Mapping[str, float]]:
        """Return a ``{protein_uniprot: {disease: delta_npx}}`` mapping.

        The returned mapping plugs directly into
        :attr:`OlinkSimConfig.group_effects`, which expects the exact shape
        ``{protein: {group_label: delta_npx}}``. Each ``disease`` label in
        ``diseases`` is treated as a group label at the simulator layer.

        Parameters
        ----------
        diseases : Sequence[str]
            One or more disease labels; each must be present in
            :meth:`diseases`. A ``KeyError`` is raised for unknowns.
        noise_sd : float, default 0.0
            If ``> 0``, add Gaussian noise ``N(0, (noise_sd * se_delta)^2)``
            to every delta — useful for Monte-Carlo ablations over
            effect-size uncertainty. ``0.0`` returns the catalog's point
            estimates verbatim.
        seed : int, default 0
            NumPy RNG seed. Only consulted when ``noise_sd > 0``.

        Returns
        -------
        Mapping[str, Mapping[str, float]]
            Keys are protein IDs that have *any* non-zero effect in the
            requested diseases; values map each disease label to its
            (possibly noise-perturbed) delta-NPX. Proteins present in the
            CSV with ``delta_npx == 0`` are omitted from the output so
            that :func:`simulate_olink_npx` skips them in its inner loop.

        Raises
        ------
        KeyError
            If any element of ``diseases`` is not in the catalog.
        ValueError
            If ``noise_sd < 0``.

        Examples
        --------
        >>> from synthlab import load_disease_effect_catalog
        >>> cat = load_disease_effect_catalog()
        >>> eff = cat.effects_for(["T2D"])
        >>> "CRP" in eff and "T2D" in eff["CRP"]
        True
        """
        if noise_sd < 0:
            raise ValueError(f"noise_sd must be >= 0; got {noise_sd}.")
        registered = set(self.diseases())
        unknown = [d for d in diseases if d not in registered]
        if unknown:
            raise KeyError(
                f"Unknown disease labels: {unknown}. "
                f"Registered: {sorted(registered)}."
            )
        sub = self.effects.filter(
            pl.col("disease").is_in(list(diseases))
            & (pl.col("delta_npx").abs() > 0.0)
        )
        # Draw per-row noise (only used when noise_sd > 0). Draw in the
        # CSV's row order for reproducibility.
        if noise_sd > 0:
            rng = np.random.default_rng(seed)
            ses = sub["se_delta"].to_numpy()
            noise = rng.normal(0.0, noise_sd * ses)
        else:
            noise = np.zeros(sub.height, dtype=float)
        out: dict[str, dict[str, float]] = {}
        proteins = sub["protein_uniprot"].to_list()
        dlist = sub["disease"].to_list()
        deltas = sub["delta_npx"].to_numpy().astype(float)
        for prot, disease, delta, noise_val in zip(proteins, dlist, deltas, noise):
            out.setdefault(prot, {})[disease] = float(delta + noise_val)
        return out


def load_disease_effect_catalog(
    path: str | Path | None = None,
) -> DiseaseEffectCatalog:
    """Load the curated per-disease Olink effect-size catalog.

    Reads the bundled
    [`synthlab/data/olink_disease_effects.csv`](../data/olink_disease_effects.csv)
    (or a user-supplied CSV with the same schema) and wraps it in a
    :class:`DiseaseEffectCatalog`. Every bundled row cites a real DOI —
    see the CSV for sourcing.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Override CSV path. If ``None`` (the default), the bundled catalog
        is loaded via :mod:`importlib.resources` — this works inside
        installed wheels without requiring any download.

    Returns
    -------
    DiseaseEffectCatalog
        Frozen dataclass wrapping the loaded polars frame.

    Raises
    ------
    FileNotFoundError
        If ``path`` is supplied but does not exist.
    ValueError
        Propagated from :class:`DiseaseEffectCatalog` if the CSV schema is
        invalid.

    Examples
    --------
    Default load (bundled CSV):

    >>> from synthlab import load_disease_effect_catalog
    >>> cat = load_disease_effect_catalog()
    >>> len(cat.diseases()) >= 6
    True

    User-supplied override:

    >>> # cat = load_disease_effect_catalog("my_custom_catalog.csv")
    """
    if path is None:
        # Ship-in-wheel path via importlib.resources — works regardless of
        # install location (editable, wheel, zip-app).
        src = resources.files("synthlab.data").joinpath("olink_disease_effects.csv")
        with resources.as_file(src) as csv_path:
            df = pl.read_csv(csv_path)
    else:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Disease-effect catalog not found: {p}")
        df = pl.read_csv(p)
    return DiseaseEffectCatalog(effects=df)
