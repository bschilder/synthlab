#!/usr/bin/env python3
"""
Download Synthea OMOP dataset from AWS S3.

Dataset available at: https://registry.opendata.aws/synthea-omop/
S3 bucket: s3://synthea-omop/
"""

import os
from pathlib import Path


def _detect_delimiter(file_path: str) -> tuple[str, str | None]:
    """
    Detect the delimiter of a CSV/TSV file by reading the first few lines.
    Returns (delimiter, quote_char) tuple.
    - delimiter: ',' for CSV, '\t' for TSV, or ',' as default
    - quote_char: '"' for CSV, None for TSV (TSV typically doesn't use quotes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            # Count tabs vs commas
            tab_count = first_line.count('\t')
            comma_count = first_line.count(',')
            
            if tab_count > comma_count and tab_count > 0:
                return ('\t', None)  # TSV - typically no quote character
            else:
                return (',', '"')  # CSV - typically uses double quotes
    except Exception:
        return (',', '"')  # Default to CSV if detection fails


def list_synthea_datasets():
    """
    List available datasets in the Synthea OMOP S3 bucket.
    
    Returns:
        list: List of available dataset names
        
    Raises:
        ImportError: If boto3 is not installed
        Exception: For network or other errors
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as e:
        print(f"⚠️  Error: boto3 is not installed")
        print("   Install with: pip install boto3")
        print("   Or check your internet connection")
        raise ImportError(
            "boto3 is required to list Synthea OMOP datasets. "
            "Install with: pip install boto3"
        ) from e
    
    try:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket = 'synthea-omop'
        
        print("Listing available datasets in s3://synthea-omop/...")
        print("=" * 60)
        
        # List top-level prefixes (directories)
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Delimiter='/')
        
        datasets = []
        for page in pages:
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    dataset_name = prefix['Prefix'].rstrip('/')
                    datasets.append(dataset_name)
                    print(f"  • {dataset_name}")
        
        return datasets
    except Exception as e:
        print(f"⚠️  Error listing datasets: {e}")
        print("   Make sure boto3 is installed: pip install boto3")
        print("   Or check your internet connection")
        raise

def download_dataset(dataset_name, output_dir, size_limit_mb=None, decompress_lzo=True):
    """
    Download a Synthea OMOP dataset from S3.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'synthea_1k', 'synthea_100k')
        output_dir: Local directory to save files
        size_limit_mb: Optional size limit in MB to warn before downloading
        decompress_lzo: If True, automatically decompress .lzo files to .csv (requires lzop)
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as e:
        raise ImportError(
            "boto3 is required to download Synthea OMOP datasets. "
            "Install with: pip install boto3"
        ) from e
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'synthea-omop'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading dataset: {dataset_name}")
    print(f"Output directory: {output_path}")
    print("=" * 60)
    
    # List all objects in the dataset prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{dataset_name}/")
    
    files = []
    total_size = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                # Skip directories (keys ending with /) and empty files
                if obj['Key'].endswith('/') or obj['Size'] == 0:
                    continue
                files.append(obj)
                total_size += obj['Size']
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"Found {len(files)} files ({total_size_mb:.1f} MB total)")
    
    if size_limit_mb and total_size_mb > size_limit_mb:
        response = input(f"\n⚠️  Dataset is {total_size_mb:.1f} MB. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    # Download files
    downloaded = 0
    decompressed = 0
    for obj in files:
        key = obj['Key']
        filename = os.path.basename(key)
        local_path = output_path / filename
        
        # Skip if already downloaded
        if local_path.exists():
            local_size = local_path.stat().st_size
            if local_size == obj['Size']:
                print(f"  ✓ {filename} (already exists)")
                downloaded += 1
                
                # Check if we need to decompress
                if decompress_lzo and filename.endswith('.lzo'):
                    csv_path = output_path / filename[:-4]  # Remove .lzo extension
                    if not csv_path.exists():
                        _decompress_lzo(local_path, csv_path)
                        if csv_path.exists():
                            decompressed += 1
                continue
        
        print(f"  ↓ {filename} ({obj['Size'] / (1024*1024):.1f} MB)...", end=' ', flush=True)
        s3.download_file(bucket, key, str(local_path))
        print("✓")
        downloaded += 1
        
        # Decompress .lzo files if requested
        if decompress_lzo and filename.endswith('.lzo'):
            csv_path = output_path / filename[:-4]  # Remove .lzo extension
            _decompress_lzo(local_path, csv_path)
            if csv_path.exists():
                decompressed += 1
    
    print(f"\n✓ Downloaded {downloaded}/{len(files)} files to {output_path}")
    if decompressed > 0:
        print(f"✓ Decompressed {decompressed} .lzo files to .csv")
    return output_path


def _decompress_lzo(lzo_path, csv_path):
    """
    Decompress a .lzo file to .csv using lzop command-line tool.
    
    Args:
        lzo_path: Path to .lzo file
        csv_path: Path where decompressed .csv file should be saved
    """
    import subprocess
    
    try:
        # Try using lzop command-line tool
        result = subprocess.run(
            ['lzop', '-d', '-c', str(lzo_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Write decompressed content to CSV file
        with open(csv_path, 'wb') as f:
            f.write(result.stdout)
        
        print(f"    → Decompressed to {csv_path.name}")
        return True
    except FileNotFoundError:
        print(f"\n⚠️  lzop not found. Cannot decompress {lzo_path.name}")
        print("   Install lzop:")
        print("     - Linux: sudo apt-get install lzop  (or yum install lzop)")
        print("     - macOS: brew install lzop")
        print("     - Or download from: https://www.lzop.org/")
        print("   Files will remain in .lzo format.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Error decompressing {lzo_path.name}: {e.stderr.decode()}")
        return False


def convert_csv_to_parquet(csv_dir, omop_dir=None, force=False, verbose=True):
    """
    Convert OMOP CSV files to Parquet format with robust type handling.
    
    This function handles mixed data types (e.g., LOINC codes like "34552-0" that
    look like numbers but are strings) by using robust CSV reading settings.
    Also automatically detects TSV files that are incorrectly named as CSV.
    
    Args:
        csv_dir: Path to directory containing CSV files (str or Path)
        omop_dir: Path to OMOP output directory (str or Path). If None, uses csv_dir/OMOP
        force: If True, reconvert files even if Parquet already exists
        verbose: If True, print progress messages
    
    Returns:
        dict: Summary with keys:
            - 'converted': Number of files successfully converted
            - 'skipped': Number of files skipped (already exist)
            - 'errors': Number of files that failed to convert
            - 'error_files': List of tuples (filename, error_message) for failed conversions
            - 'omop_dir': Path to OMOP directory
    """
    import polars as pl
    
    csv_dir = Path(csv_dir)
    if omop_dir is None:
        omop_dir = csv_dir / "OMOP"
    else:
        omop_dir = Path(omop_dir)
    
    # Create OMOP directory
    omop_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"CSV directory: {csv_dir}")
        print(f"OMOP directory: {omop_dir}")
        print()
    
    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))
    
    if not csv_files:
        if verbose:
            print(f"⚠️  No CSV files found in {csv_dir}")
        return {
            'converted': 0,
            'skipped': 0,
            'errors': 0,
            'omop_dir': str(omop_dir)
        }
    
    converted = 0
    skipped = 0
    errors = 0
    error_files = []
    
    for csv_file in csv_files:
        parquet_file = omop_dir / (csv_file.stem + ".parquet")
        
        # Skip if already exists and not forcing
        if parquet_file.exists() and not force:
            skipped += 1
            if verbose:
                print(f"  ⊘ Skipped {csv_file.name} (Parquet already exists)")
            continue
        
        try:
            # Detect delimiter (CSV vs TSV) - handles files incorrectly named as CSV
            delimiter, quote_char = _detect_delimiter(str(csv_file))
            file_type = "TSV" if delimiter == '\t' else "CSV"
            
            if verbose and delimiter == '\t':
                print(f"  🔍 Detected {csv_file.name} as TSV (tab-delimited), not CSV", end=" ... ")
            
            # Build read_csv arguments with robust settings
            read_kwargs: dict = {
                "separator": delimiter,
                "infer_schema_length": 10000,    # Sample more rows for type inference
                "ignore_errors": True,            # Don't fail on parsing errors
                "try_parse_dates": True,          # Auto-detect date columns
                "truncate_ragged_lines": True,    # Handle inconsistent row lengths
            }
            
            # For TSV files, disable quote parsing to avoid issues with embedded quotes
            if delimiter == '\t':
                read_kwargs["quote_char"] = None  # Disable quote parsing for TSV
            elif quote_char is not None:
                read_kwargs["quote_char"] = quote_char
            
            df = pl.read_csv(csv_file, **read_kwargs)
            df.write_parquet(parquet_file)
            converted += 1
            if verbose:
                if delimiter == '\t':
                    print(f"✓ Converted {csv_file.name} -> {parquet_file.name}")
                else:
                    print(f"  ✓ Converted {csv_file.name} -> {parquet_file.name}")
        except Exception as e:
            errors += 1
            error_files.append((csv_file.name, str(e)))
            if verbose:
                print(f"  ✗ Error converting {csv_file.name}: {e}")
    
    # Print summary
    if verbose:
        if converted > 0:
            print(f"\n✓ Converted {converted} CSV files to Parquet in OMOP directory")
        if skipped > 0:
            print(f"⊘ Skipped {skipped} files (Parquet already exists)")
        if errors > 0:
            print(f"✗ Failed to convert {errors} files")
            for filename, error in error_files:
                print(f"    - {filename}: {error}")
        
        print(f"\n✓ Files converted to: {omop_dir}")
        print(f"  (Will use Parquet if available, otherwise CSV)")
    
    return {
        'converted': converted,
        'skipped': skipped,
        'errors': errors,
        'error_files': error_files,
        'omop_dir': str(omop_dir)
    }

if __name__ == "__main__":
    import sys
    
    print("Synthea OMOP Dataset Downloader")
    print("=" * 60)
    print("Dataset: https://registry.opendata.aws/synthea-omop/")
    print()
    
    # List available datasets
    datasets = list_synthea_datasets()
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else f"data/synthea_omop_aws/{dataset_name}"
        download_dataset(dataset_name, output_dir)
    else:
        print("\nUsage:")
        print("  python download_synthea_omop.py <dataset_name> [output_dir]")
        print("\nExample:")
        print("  python download_synthea_omop.py synthea_1k data/synthea_omop_aws/1k")
        print("\nAvailable datasets:")
        for ds in datasets:
            print(f"  • {ds}")
