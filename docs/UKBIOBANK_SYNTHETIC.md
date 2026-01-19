# UK Biobank Synthetic Dataset

## What is this?

The UK Biobank Synthetic Dataset is a **fake version** of the real UK Biobank dataset. It's designed for:
- Testing software systems that need to handle large healthcare datasets
- Practicing data processing pipelines without privacy concerns
- System performance testing

**Important**: The data is randomly generated and **not internally consistent**. For example, it might have:
- Prostate cancer in female participants
- Medical events after death
- Dates without corresponding diagnoses

## What files do you need?

The dataset is split into 4 categories. You probably don't need all of them:

### 1. **Tabular Records** (Most Important) ✅
**What it is**: The main phenotype data - all the survey responses, measurements, and clinical data for ~600,000 participants.

**Files**: 23 TSV files (tab-separated values)
- Each file contains a subset of fields (columns)
- All participants appear in every file
- Total: ~27,000 columns × 600,000 rows

**When to use**: If you want to analyze participant characteristics, health outcomes, or build predictive models.

**Example files**:
- `integer_no_arrays.tsv` - Integer survey responses
- `real_fields1.tsv` - Real-valued measurements (e.g., BMI, blood pressure)
- `string_fields1.tsv` - Text responses
- `dates_death.tsv` - Death dates
- `datetime_fields.tsv` - Various date/time fields

### 2. **Medical Records** (GP Clinical Data)
**What it is**: General practitioner (GP) clinical records - what doctors recorded during visits.

**Files**: 6 text files
- Total: ~400 million rows
- Columns: EID (participant ID), data provider, event date, Read codes (diagnosis codes), values

**When to use**: If you need detailed clinical history, diagnosis codes, or want to analyze medical events over time.

**Warning**: This is **huge** (~400M rows). Only download if you specifically need GP clinical data.

### 3. **Genetic Records** (SNP Data)
**What it is**: Genotype data - genetic variants (SNPs) for each participant.

**Files**: 
- `gene_dic.dat` - Dictionary mapping SNP IDs to variants
- 26 compressed chromosome files (`rand_chr*.dat.gz`)
- Total: ~600,000 participants × 840,000 SNPs

**When to use**: If you're doing genetic association studies, GWAS, or polygenic risk score analysis.

**Warning**: This is **extremely large**. Each chromosome file is several GB compressed. Only download if you need genetic data.

### 4. **Bulk Files** (System Testing)
**What it is**: A collection of ~6 million small files simulating the UK Biobank bulk file repository.

**Files**: 37 zip archives containing millions of files

**When to use**: **Only for system testing** - testing file handling, pseudonymization pipelines, etc. Not for actual data analysis.

**Warning**: This is for infrastructure testing, not analysis. Skip unless you're building/testing data processing systems.

## Quick Start

### Download Tabular Data (Recommended)

```python
from synthlab import download_category
from pathlib import Path

# Download all tabular files
output_dir = Path("data/ukbiobank_synthetic/tabular")
download_category("tabular", output_dir)
```

### Load Tabular Data

```python
from synthlab import load_tabular_data

# Load all tabular files
data = load_tabular_data("data/ukbiobank_synthetic/tabular")

# Access individual files
integer_data = data["integer_no_arrays"]
real_data = data["real_fields1"]
```

### Download Specific Files

```python
from synthlab import download_file

# Download a single file
download_file("dates_death.tsv", "data/ukbiobank_synthetic/tabular", category="tabular")
```

## File Format Details

### Tabular Files
- **Format**: TSV (tab-separated values)
- **First column**: EID (7-digit participant identifier)
- **Other columns**: FieldID.InstanceID.ArrayID (e.g., "31.0.0" = Field 31, Instance 0, Array 0)
- **Header row**: Contains "EID" followed by field names

### Medical Records
- **Format**: Tab-separated text
- **Columns**: 
  1. EID (participant ID)
  2. data_provider (integer)
  3. event_date (YYYYMMDD format)
  4. read2 (Read 2 diagnosis code)
  5. read3 (Read 3 diagnosis code)
  6. value1, value2, value3 (text values)

### Genetic Data
- **Dictionary file** (`gene_dic.dat`): Tab-separated with columns:
  - Affymetrix ID, Chromosome, Index, Variant 0, Variant 1, Variant 2, Variant 3
- **Data files** (`rand_chr*.dat.gz`): Compressed text files
  - Each line: `EID genotype_values`
  - Genotype values are indices (0, 1, 2, 3) corresponding to variants in dictionary

## MD5 Checksums

All files have MD5 checksums for verification. The download functions automatically verify checksums to ensure file integrity.

## More Information

- Official documentation: https://biobank.ndph.ox.ac.uk/synthetic_dataset/
- UK Biobank Showcase (for field definitions): http://biobank.ndph.ox.ac.uk/showcase/schema.cgi
- Read code mappings: https://biobank.ctsu.ox.ac.uk/crystal/refer.cgi?id=592
