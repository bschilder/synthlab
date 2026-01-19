# SynthLab

Python tools for working with synthetic healthcare datasets.

SynthLab provides Python interfaces for working with major synthetic healthcare datasets, including:
- **[Synthea](https://github.com/synthetichealth/synthea)**: A synthetic patient population simulator that generates realistic synthetic patient records
- **[UK Biobank Synthetic Dataset](https://biobank.ndph.ox.ac.uk/synthetic_dataset/)**: A large-scale synthetic dataset designed for system testing with UK Biobank-compatible data

## Features

### Synthea Support
- **Synthea Runner**: Easy-to-use Python interface for running Synthea simulations
- **OMOP Conversion**: Convert Synthea CSV output to OMOP CDM format
- **AWS Dataset Download**: Download pre-generated Synthea OMOP datasets from AWS
- **Configuration Management**: Flexible configuration system with validation

### UK Biobank Synthetic Dataset Support
- **Dataset Download**: Download tabular, medical, genetic, and bulk data files
- **Automatic Caching**: Files are automatically cached in `~/.cache/synthlab/ukbiobank_synthetic/`
- **MD5 Verification**: Automatic checksum verification for downloaded files
- **Data Loading**: Load data into Polars DataFrames for efficient analysis
- **Category Management**: Organized downloads by category (tabular, medical, genetic, bulk)

## Installation

```bash
pip install synthlab
```

For AWS dataset download functionality:
```bash
pip install synthlab[aws]
```

## Requirements

- Python 3.8+
- Java 11 or newer (required by Synthea)
- Polars (for efficient data loading)
- Requests (for dataset downloads)

## Quick Start

### Synthea: Generate Synthetic Patient Data

```python
from synthlab import SyntheaRunner, SyntheaConfig

# Create a runner (downloads Synthea JAR automatically)
runner = SyntheaRunner()

# Configure a simulation
config = SyntheaConfig(
    population_size=100,
    state="Massachusetts",
    seed=12345,
    output_dir="output/synthea_data"
)

# Run the simulation
result = runner.run(config)

if result['returncode'] == 0:
    print(f"Generated data in: {result['output_dir']}")
```

### Synthea: Convert to OMOP CDM

```python
from synthlab import convert_synthea_to_omop

# Convert Synthea CSV to OMOP CDM format
output_files = convert_synthea_to_omop(
    synthea_csv_dir="output/synthea_data",
    output_dir="output/omop",
    cdm_version="5.4",
    output_format="parquet"
)

print(f"Generated {len(output_files)} OMOP tables")
```

### Synthea: Download Pre-generated Datasets

```python
from synthlab import list_synthea_datasets, download_dataset

# List available datasets
datasets = list_synthea_datasets()

# Download a dataset
download_dataset("synthea1k", output_dir="data/synthea1k")
```

### UK Biobank Synthetic Dataset: Download and Load Data

```python
from synthlab import download_category, load_tabular_data, get_cache_dir

# Download tabular data (saved to ~/.cache/synthlab/ukbiobank_synthetic/tabular/)
download_category("tabular", verify_md5=True)

# Load the data into Polars DataFrames
tabular_data = load_tabular_data(sample_rows=1000)  # Load first 1000 rows for demo

# Access individual files
death_data = tabular_data["dates_death"]
integer_data = tabular_data["integer_no_arrays"]

print(f"Loaded {len(tabular_data)} tabular files")
print(f"Cache directory: {get_cache_dir()}")
```

### UK Biobank Synthetic Dataset: Download Specific Files

```python
from synthlab import download_file

# Download a single file
download_file("dates_death.tsv", category="tabular")

# Files are automatically saved to ~/.cache/synthlab/ukbiobank_synthetic/tabular/
```

## Documentation

### SyntheaRunner

The main class for running Synthea simulations.

```python
runner = SyntheaRunner(
    jar_path=None,          # Path to existing JAR (auto-downloads if None)
    jar_url=SYNTHEA_JAR_URL,  # URL to download JAR from
    cache_dir=None,         # Cache directory (defaults to OS cache)
    java_executable="java"  # Java executable path
)
```

### SyntheaConfig

Configuration class for Synthea simulations.

```python
config = SyntheaConfig(
    population_size=100,     # Number of patients
    seed=12345,              # Random seed
    state="Massachusetts",   # US state
    city="Boston",           # Optional city
    min_age=0,              # Minimum age
    max_age=100,            # Maximum age
    gender="M",             # "M", "F", or None
    output_dir="output"     # Output directory
)
```

### UK Biobank Synthetic Dataset Functions

```python
from synthlab import (
    list_available_files,
    download_file,
    download_category,
    load_tabular_data,
    load_medical_records,
    load_genetic_dictionary,
    get_cache_dir,
)

# List available files
files = list_available_files(category="tabular")

# Download entire category
download_category("tabular")  # Downloads to ~/.cache/synthlab/ukbiobank_synthetic/tabular/

# Download single file
download_file("dates_death.tsv", category="tabular")

# Load data (uses cache directory by default)
data = load_tabular_data(sample_rows=1000)
medical = load_medical_records(sample_rows=10000)
genetic_dict = load_genetic_dictionary()

# Get cache directory
cache_dir = get_cache_dir()  # Returns ~/.cache/synthlab/ukbiobank_synthetic/
```

### Convenience Methods

```python
# Synthea quick test run
runner.run_quick(population_size=10, state="Massachusetts")

# Synthea custom location
runner.run_custom_location(state="California", city="San Francisco", population_size=100)

# Synthea age-specific population
runner.run_age_specific(min_age=25, max_age=65, population_size=100)
```

## Dataset Information

### Synthea

Synthea generates synthetic patient records with:
- Demographics
- Medical history
- Medications
- Lab results
- Procedures
- Encounters

**Reference**: [Synthea GitHub](https://github.com/synthetichealth/synthea)

### UK Biobank Synthetic Dataset

The UK Biobank Synthetic Dataset contains:

1. **Tabular Records** (23 TSV files): Main phenotype data (~600K participants × ~27K columns)
   - Survey responses, measurements, clinical data
   - Files: `dates_death.tsv`, `integer_no_arrays.tsv`, `real_fields1.tsv`, etc.

2. **Medical Records** (6 text files): GP clinical records (~400M rows)
   - Diagnosis codes (Read 2/3), visit data, clinical events

3. **Genetic Records**: SNP genotype data (~600K participants × 840K SNPs)
   - Dictionary file + 26 chromosome files (compressed)

4. **Bulk Files** (37 zip archives): ~6M files for system testing

**Important**: This is synthetic data and may not be internally consistent (e.g., events after death, prostate cancer in females).

**Reference**: [UK Biobank Synthetic Dataset](https://biobank.ndph.ox.ac.uk/synthetic_dataset/)

## Examples

See the `examples/` and `notebooks/` directories for detailed examples:
- `examples/basic_usage.py` - Basic Synthea usage
- `examples/ukbiobank_synthetic_example.py` - UK Biobank Synthetic Dataset examples
- `notebooks/Synthea.ipynb` - Comprehensive Synthea tutorial
- `notebooks/UKBiobank_Synthetic.ipynb` - UK Biobank Synthetic Dataset tutorial

## License

MIT License

## References

### Synthea
- [Synthea GitHub](https://github.com/synthetichealth/synthea)
- [OMOP CDM](https://ohdsi.github.io/CommonDataModel/)
- [OHDSI ETL-Synthea](https://github.com/OHDSI/ETL-Synthea)
- [AWS Synthea OMOP Dataset](https://registry.opendata.aws/synthea-omop/)

### UK Biobank
- [UK Biobank Synthetic Dataset](https://biobank.ndph.ox.ac.uk/synthetic_dataset/)
- [UK Biobank Showcase](http://biobank.ndph.ox.ac.uk/showcase/schema.cgi) (for field definitions)
- [Read Code Mappings](https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=592)
