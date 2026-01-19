"""
SynthLab - Python tools for working with Synthea synthetic healthcare data.

SynthLab provides a Python wrapper for Synthea, a synthetic patient population
simulator that generates realistic synthetic patient records.
"""

from synthlab.synthea import (
    SyntheaRunner,
    SyntheaConfig,
    convert_synthea_to_omop,
    add_concept_names,
    download_synthea_jar,
    SYNTHEA_JAR_URL,
)

from synthlab.download_synthea_omop import (
    list_synthea_datasets,
    download_dataset,
    convert_csv_to_parquet,
)

__version__ = "0.1.0"

__all__ = [
    "SyntheaRunner",
    "SyntheaConfig",
    "convert_synthea_to_omop",
    "add_concept_names",
    "download_synthea_jar",
    "SYNTHEA_JAR_URL",
    "list_synthea_datasets",
    "download_dataset",
    "convert_csv_to_parquet",
]
