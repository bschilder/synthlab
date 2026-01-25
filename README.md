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

### Synthea Coherent Data Set (Multimodal) - NEW!
- **EHR + Imaging + Genomics**: The only publicly available synthetic dataset combining all three modalities
- **FHIR Records**: Complete patient records with demographics, conditions, medications, encounters
- **MRI DICOM**: Synthetic brain imaging linked to patients
- **Familial Genomes**: VCF files with genetic variants for patients and family members
- **Clinical Notes**: SOAP-style clinical documentation
- **Physiological Data**: Time-series vital signs data

### UK Biobank Synthetic Dataset Support
- **Dataset Download**: Download tabular, medical, genetic, and bulk data files
- **Automatic Caching**: Files are automatically cached in `~/.cache/synthlab/ukbiobank_synthetic/`
- **MD5 Verification**: Automatic checksum verification for downloaded files
- **Data Loading**: Load data into Polars DataFrames for efficient analysis
- **Category Management**: Organized downloads by category (tabular, medical, genetic, bulk)

### Genomics Data Support (NEW)
- **HAPNEST Integration**: Download and load HAPNEST synthetic genomics data (1M+ individuals, 6.8M variants)
- **Synthetic Genotype Generation**: Generate simple synthetic genotypes for testing
- **PLINK Format Support**: Work with standard genomics file formats (.pgen, .pvar, .psam)

### Medical Imaging Catalog (NEW)
- **Dataset Discovery**: Catalog of 15+ publicly available medical imaging datasets
- **Multi-modal Coverage**: CT, MRI, X-ray, and histopathology datasets
- **Access Information**: Clear documentation of open vs. registration-required datasets
- **Download Utilities**: Helpers for downloading select open-access datasets

## Installation

```bash
pip install synthlab
```

For AWS dataset download functionality:
```bash
pip install synthlab[aws]
```

For imaging notebooks (DICOM):
```bash
pip install synthlab[imaging]
```

For genomics helpers:
```bash
pip install synthlab[genomics]
```

Install everything:
```bash
pip install synthlab[all]
```

Optional GPU performance (FlashAttention):
```bash
# Requires a supported NVIDIA GPU + CUDA toolchain
# See https://github.com/Dao-AILab/flash-attention for compatibility details
pip install flash-attn --no-build-isolation
```

Optional GPU performance (FlashAttention 2):
```bash
# FlashAttention 2 is provided by the same package; install as usual
# See https://github.com/Dao-AILab/flash-attention for compatibility details
pip install flash-attn --no-build-isolation
```

## Requirements

- Python 3.8+
- Java 11 or newer (required by Synthea)
- Polars (for efficient data loading)
- Requests (for dataset downloads)
- Matplotlib (plots)
- Optional (imaging notebooks): pydicom

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

### Genomics: Download HAPNEST Synthetic Data

```python
from synthlab import download_hapnest_small, load_hapnest_variants, load_hapnest_samples

# Download small HAPNEST test dataset (600 individuals)
data_dir = download_hapnest_small()

# Load variant and sample information
variants = load_hapnest_variants(data_dir)
samples = load_hapnest_samples(data_dir)

print(f"Variants: {len(variants)}")
print(f"Samples: {len(samples)}")
```

### Genomics: Generate Synthetic Genotypes

```python
from synthlab import generate_synthetic_genotypes

# Generate random synthetic genotype data for testing
data_dir = generate_synthetic_genotypes(
    n_samples=1000,
    n_variants=10000,
    seed=42
)
```

### Coherent Data Set: Multimodal EHR + Imaging + Genomics

```python
from synthlab import (
    download_coherent_dataset,
    load_fhir_patients,
    print_coherent_info,
)

# See what's available
print_coherent_info()

# Download specific components (FHIR records + genomics)
download_coherent_dataset(components=['fhir', 'genomics'])

# Or download everything (several GB)
# download_coherent_dataset()

# Load FHIR patient records
patients = load_fhir_patients(max_patients=10)
print(f"Loaded {len(patients)} patient bundles")
```

### Medical Imaging: Explore Available Datasets

```python
from synthlab import print_dataset_catalog, list_imaging_datasets, get_dataset_info

# Print catalog of all datasets
print_dataset_catalog()

# Filter by modality
histology = list_imaging_datasets(modality="Histopathology")

# Get info about specific dataset
mhist_info = get_dataset_info("mhist")
print(f"MHIST: {mhist_info.n_images} images, {mhist_info.size_gb} GB")
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
- `notebooks/Coherent_Dataset.ipynb` - Multimodal EHR + Imaging + Genomics tutorial

## License

MIT License

## References

### Synthea
- [Synthea GitHub](https://github.com/synthetichealth/synthea)
- [OMOP CDM](https://ohdsi.github.io/CommonDataModel/)
- [OHDSI ETL-Synthea](https://github.com/OHDSI/ETL-Synthea)
- [AWS Synthea OMOP Dataset](https://registry.opendata.aws/synthea-omop/)

### Synthea Coherent Data Set (Multimodal)
- [AWS Open Data Registry](https://registry.opendata.aws/synthea-coherent-data/)
- [Paper: "The Coherent Data Set" (MDPI Electronics, 2022)](https://www.mdpi.com/2079-9292/11/8/1199)
- S3 Bucket: `s3://synthea-open-data/coherent/`

### UK Biobank
- [UK Biobank Synthetic Dataset](https://biobank.ndph.ox.ac.uk/synthetic_dataset/)
- [UK Biobank Showcase](http://biobank.ndph.ox.ac.uk/showcase/schema.cgi) (for field definitions)
- [Read Code Mappings](https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=592)

### Genomics
- [HAPNEST Paper](https://academic.oup.com/bioinformatics/article/39/9/btad535/7255913) - Synthetic genotype/phenotype generation
- [HAPNEST BioStudies](https://www.ebi.ac.uk/biostudies/studies/S-BSST936) - Full dataset download
- [PLINK 2](https://www.cog-genomics.org/plink/2.0/) - Genomics analysis toolkit

### Medical Imaging
- [TCIA - The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)
- [Stanford AIMI Datasets](https://aimi.stanford.edu/shared-datasets)
- [MHIST Histopathology Dataset](https://bmirds.github.io/MHIST/)
- [Grand Challenge](https://grand-challenge.org/) - Medical imaging challenges and datasets
