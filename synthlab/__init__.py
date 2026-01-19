"""
SynthLab - Python tools for working with synthetic healthcare datasets.

SynthLab provides Python interfaces for working with major synthetic healthcare datasets:
- Synthea: A synthetic patient population simulator
- UK Biobank Synthetic Dataset: Large-scale synthetic dataset for system testing
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

from synthlab.download_ukbiobank_synthetic import (
    list_available_files,
    download_file,
    download_category,
    load_tabular_data,
    load_medical_records,
    load_genetic_dictionary,
    get_cache_dir,
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
    "list_available_files",
    "download_file",
    "download_category",
    "load_tabular_data",
    "load_medical_records",
    "load_genetic_dictionary",
    "get_cache_dir",
]
