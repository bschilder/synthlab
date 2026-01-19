# SynthLab

Python tools for working with Synthea synthetic healthcare data.

SynthLab provides a Python wrapper for [Synthea](https://github.com/synthetichealth/synthea), a synthetic patient population simulator that generates realistic synthetic patient records.

## Features

- **Synthea Runner**: Easy-to-use Python interface for running Synthea simulations
- **OMOP Conversion**: Convert Synthea CSV output to OMOP CDM format
- **AWS Dataset Download**: Download pre-generated Synthea OMOP datasets from AWS
- **Configuration Management**: Flexible configuration system with validation

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

## Quick Start

### Generate Synthetic Patient Data

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

### Convert to OMOP CDM

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

### Download Pre-generated Datasets

```python
from synthlab import list_synthea_datasets, download_dataset

# List available datasets
datasets = list_synthea_datasets()

# Download a dataset
download_dataset("synthea1k", output_dir="data/synthea1k")
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

### Convenience Methods

```python
# Quick test run
runner.run_quick(population_size=10, state="Massachusetts")

# Custom location
runner.run_custom_location(state="California", city="San Francisco", population_size=100)

# Age-specific population
runner.run_age_specific(min_age=25, max_age=65, population_size=100)
```

## Examples

See the `examples/` directory for more detailed examples.

## License

MIT License

## References

- [Synthea GitHub](https://github.com/synthetichealth/synthea)
- [OMOP CDM](https://ohdsi.github.io/CommonDataModel/)
- [OHDSI ETL-Synthea](https://github.com/OHDSI/ETL-Synthea)
- [AWS Synthea OMOP Dataset](https://registry.opendata.aws/synthea-omop/)
