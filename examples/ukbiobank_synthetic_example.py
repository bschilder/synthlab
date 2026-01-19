#!/usr/bin/env python3
"""
Example: Download and load UK Biobank Synthetic Dataset

This example demonstrates how to:
1. List available files
2. Download specific categories
3. Load and work with the data
"""

from pathlib import Path
from synthlab import (
    list_available_files,
    download_category,
    load_tabular_data,
    load_medical_records,
    load_genetic_dictionary,
)

# Set output directory
output_dir = Path("data/ukbiobank_synthetic")

print("UK Biobank Synthetic Dataset Example")
print("=" * 60)

# 1. List available files
print("\n1. Available files by category:")
print("-" * 60)
available = list_available_files()
for category, files in available.items():
    print(f"\n{category.upper()} ({len(files)} files):")
    # Show first few files as examples
    for filename in files[:5]:
        print(f"  • {filename}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

# 2. Download tabular data (most commonly used)
print("\n\n2. Downloading tabular data...")
print("-" * 60)
tabular_dir = output_dir / "tabular"
try:
    download_category("tabular", tabular_dir, verify_md5=True)
except Exception as e:
    print(f"Error downloading: {e}")
    print("Note: This requires an internet connection and may take a while.")
    print("Skipping download for this example...")
    tabular_dir = None

# 3. Load tabular data (if downloaded)
if tabular_dir and tabular_dir.exists():
    print("\n\n3. Loading tabular data...")
    print("-" * 60)
    try:
        # Load a sample of the data (first 1000 rows for demo)
        tabular_data = load_tabular_data(tabular_dir, sample_rows=1000)
        
        print(f"\nLoaded {len(tabular_data)} tabular files:")
        for name, df in tabular_data.items():
            print(f"  • {name}: {len(df)} rows × {len(df.columns)} columns")
            print(f"    Columns: {', '.join(df.columns[:5])}...")
    except Exception as e:
        print(f"Error loading data: {e}")

# 4. Example: Download medical records
print("\n\n4. Medical records download example:")
print("-" * 60)
print("To download medical records:")
print("  download_category('medical', output_dir / 'medical')")
print("Note: Medical records are very large (~400M rows)")

# 5. Example: Download genetic data
print("\n\n5. Genetic data download example:")
print("-" * 60)
print("To download genetic data:")
print("  download_category('genetic', output_dir / 'genetic')")
print("Note: Genetic data includes compressed chromosome files")

print("\n" + "=" * 60)
print("For more information, see:")
print("  https://biobank.ndph.ox.ac.uk/synthetic_dataset/")
