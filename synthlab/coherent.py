#!/usr/bin/env python3
"""
Synthea Coherent Data Set - Multimodal synthetic healthcare data.

The Coherent Data Set combines multiple synthetic data types into a unified,
patient-level electronic health record:
- EHR: FHIR records (demographics, conditions, medications, encounters, etc.)
- Imaging: MRI DICOM files (brain imaging)
- Genomics: DNA test results (CSV format, *_dna.csv files)
- Clinical Notes: SOAP-style notes
- Physiological Data: SBML models

All data types are linked via FHIR to create coherent patient records.

References:
- AWS Open Data: https://registry.opendata.aws/synthea-coherent-data/
- Paper: https://www.mdpi.com/2079-9292/11/8/1199
- S3 Bucket: s3://synthea-open-data/coherent/
"""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

# Check for optional dependencies
_has_tqdm = False
try:
    from tqdm import tqdm as _tqdm_import
    _has_tqdm = True
except ImportError:
    _tqdm_import = None


# S3 bucket information (public, no credentials required)
S3_BUCKET = "synthea-open-data"
S3_PREFIX = "coherent/"
S3_REGION = "us-east-1"

# Dataset components - paths are relative to S3_PREFIX
# Actual structure discovered from S3 bucket:
#   coherent/unzipped/fhir/   - FHIR JSON bundles
#   coherent/unzipped/dicom/  - DICOM imaging
#   coherent/unzipped/dna/    - Genomics CSV files (NOT VCF!)
#   coherent/unzipped/csv/    - Tabular CSV data
COHERENT_COMPONENTS = {
    "fhir": {
        "description": "FHIR R4 patient records (JSON bundles)",
        "path": "unzipped/fhir/",
        "alt_paths": [],
        "format": "JSON (FHIR R4)",
        "contents": "Demographics, conditions, medications, encounters, observations, etc.",
        "extensions": [".json"],
    },
    "dicom": {
        "description": "MRI brain imaging (DICOM files)",
        "path": "unzipped/dicom/",
        "alt_paths": [],
        "format": "DICOM",
        "contents": "Synthetic brain MRI scans linked to patients",
        "extensions": [".dcm"],
    },
    "genomics": {
        "description": "DNA/genomics data (CSV format with genetic variants)",
        "path": "unzipped/dna/",
        "alt_paths": [],
        "format": "CSV",
        "contents": "Synthetic genetic data for patients (DNA test results)",
        "extensions": [".csv", "_dna.csv"],
    },
    "csv": {
        "description": "Tabular data exports",
        "path": "unzipped/csv/",
        "alt_paths": [],
        "format": "CSV",
        "contents": "Structured tabular data exports",
        "extensions": [".csv"],
    },
}


def get_coherent_cache_dir() -> Path:
    """
    Get the default cache directory for Coherent Data Set.

    Returns:
        Path: Path to ~/.cache/synthlab/coherent/
    """
    cache_dir = Path.home() / ".cache" / "synthlab" / "coherent"
    return cache_dir


def _check_aws_cli() -> bool:
    """Check if AWS CLI is available."""
    try:
        result = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _check_boto3() -> bool:
    """Check if boto3 is available."""
    try:
        import boto3
        return True
    except ImportError:
        return False


def list_coherent_components() -> dict[str, Any]:
    """
    List available components in the Coherent Data Set.

    Returns:
        dict: Dictionary mapping component names to their metadata

    Example:
        >>> from synthlab.coherent import list_coherent_components
        >>> components = list_coherent_components()
        >>> for name, info in components.items():
        ...     print(f"{name}: {info['description']}")
    """
    return COHERENT_COMPONENTS.copy()


def discover_s3_structure(max_depth: int = 2) -> dict[str, Any]:
    """
    Discover the actual folder structure in the S3 bucket.

    Args:
        max_depth: Maximum depth to explore (reserved for future use)

    Returns:
        dict: Dictionary with 'folders', 'files', and 'total_size_mb' keys

    Note:
        Requires boto3 or AWS CLI.
    """
    _ = max_depth  # Reserved for future use
    if _check_boto3():
        return _discover_structure_boto3(max_depth)
    elif _check_aws_cli():
        return _discover_structure_cli(max_depth)
    else:
        raise RuntimeError(
            "Either AWS CLI or boto3 is required.\n"
            "Install with: pip install boto3"
        )


def find_files_by_extension(
    extensions: str | list[str],
    max_files: int = 100,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Search the entire S3 bucket for files with specific extensions.

    This recursively searches all folders to find files regardless of
    where they are stored.

    Args:
        extensions: File extension(s) to search for. Can be a single string
                   (e.g., '.vcf') or a list (e.g., ['.vcf', '.vcf.gz'])
        max_files: Maximum number of files to return
        verbose: If True, print progress

    Returns:
        list: List of file dictionaries with 'key', 'size', 'size_mb' keys

    Example:
        >>> from synthlab.coherent import find_files_by_extension
        >>> vcf_files = find_files_by_extension('.vcf')  # Single extension
        >>> vcf_files = find_files_by_extension(['.vcf', '.vcf.gz'])  # Multiple
        >>> for f in vcf_files:
        ...     print(f['key'])
    """
    if not _check_boto3():
        raise RuntimeError(
            "boto3 is required for recursive file search.\n"
            "Install with: pip install boto3"
        )

    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    # Convert string to list if needed
    if isinstance(extensions, str):
        ext_list = [extensions]
    else:
        ext_list = list(extensions)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=S3_REGION)

    if verbose:
        print(f"Searching s3://{S3_BUCKET}/{S3_PREFIX} for {ext_list}...")

    files: list[dict[str, Any]] = []

    # Normalize extensions to lowercase
    ext_list = [ext.lower() for ext in ext_list]

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            key_lower = key.lower()

            # Check if file matches any extension
            if any(key_lower.endswith(ext) for ext in ext_list):
                files.append({
                    "key": key,
                    "size": obj["Size"],
                    "size_mb": obj["Size"] / (1024 * 1024),
                })

                if verbose and len(files) % 10 == 0:
                    print(f"  Found {len(files)} files...")

                if len(files) >= max_files:
                    if verbose:
                        print(f"  Reached max_files limit ({max_files})")
                    return files

    if verbose:
        print(f"  Found {len(files)} files total")

    return files


def _discover_structure_boto3(max_depth: int) -> dict[str, Any]:  # noqa: ARG001
    """Discover S3 structure using boto3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=S3_REGION)

    structure = {"folders": [], "files": [], "total_size_mb": 0}

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter="/"):
        # Get folders (CommonPrefixes)
        for prefix in page.get("CommonPrefixes", []):
            folder = prefix["Prefix"][len(S3_PREFIX):]
            structure["folders"].append(folder)

        # Get files at this level
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key != S3_PREFIX:  # Skip the prefix itself
                structure["files"].append({
                    "key": key,
                    "size_mb": obj["Size"] / (1024 * 1024),
                })
                structure["total_size_mb"] += obj["Size"] / (1024 * 1024)

    return structure


def _discover_structure_cli(max_depth: int) -> dict[str, Any]:  # noqa: ARG001
    """Discover S3 structure using AWS CLI."""
    cmd = [
        "aws", "s3", "ls",
        "--no-sign-request",
        f"s3://{S3_BUCKET}/{S3_PREFIX}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AWS CLI error: {result.stderr}")

    structure = {"folders": [], "files": [], "total_size_mb": 0}

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        if line.strip().startswith("PRE "):
            # It's a folder
            folder = line.strip()[4:]
            structure["folders"].append(folder)
        else:
            # It's a file
            parts = line.split()
            if len(parts) >= 4:
                size = int(parts[2])
                key = parts[3]
                structure["files"].append({
                    "key": key,
                    "size_mb": size / (1024 * 1024),
                })
                structure["total_size_mb"] += size / (1024 * 1024)

    return structure


def list_coherent_files(
    component: Optional[str] = None,
    max_files: int = 100,
) -> list[dict]:
    """
    List files available in the Coherent Data Set S3 bucket.

    Args:
        component: Optional component to filter by ('fhir', 'dicom', 'genomics', 'notes', 'physiological')
                   If None, lists all files.
        max_files: Maximum number of files to list

    Returns:
        list: List of file dictionaries with 'key', 'size', 'size_mb', 'component' keys

    Note:
        Requires either AWS CLI or boto3 to be installed.
    """
    if _check_boto3():
        return _list_files_boto3(component, max_files)
    elif _check_aws_cli():
        return _list_files_cli(component, max_files)
    else:
        raise RuntimeError(
            "Either AWS CLI or boto3 is required to list Coherent Data Set files.\n"
            "Install with: pip install boto3\n"
            "Or install AWS CLI: https://aws.amazon.com/cli/"
        )


def _list_files_boto3(component: Optional[str], max_files: int) -> list[dict]:
    """List files using boto3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=S3_REGION)

    files: list[dict] = []

    if component:
        if component not in COHERENT_COMPONENTS:
            raise ValueError(f"Unknown component: {component}. Must be one of: {list(COHERENT_COMPONENTS.keys())}")

        # Try the primary path and alternative paths
        comp_info = COHERENT_COMPONENTS[component]
        paths_to_try = [comp_info["path"]] + comp_info.get("alt_paths", [])

        for path in paths_to_try:
            prefix = S3_PREFIX + str(path)
            paginator = s3.get_paginator("list_objects_v2")

            try:
                for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            # Skip directories
                            if key.endswith("/") or obj["Size"] == 0:
                                continue

                            files.append({
                                "key": key,
                                "size": obj["Size"],
                                "size_mb": obj["Size"] / (1024 * 1024),
                                "component": component,
                                "last_modified": obj.get("LastModified"),
                            })

                            if len(files) >= max_files:
                                return files
            except Exception:
                continue  # Try next path

            if files:
                break  # Found files, stop trying other paths

        # If no files found at expected paths, try fallback: search by extension
        if not files:
            ext_list = comp_info.get("extensions", [])
            if ext_list and isinstance(ext_list, list):
                print(f"  Paths not found, searching by extension {ext_list}...")
                found_files = find_files_by_extension(ext_list, max_files=max_files, verbose=False)
                for f in found_files:
                    f["component"] = component
                    files.append(f)
    else:
        # List all files from root
        prefix = S3_PREFIX
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith("/") or obj["Size"] == 0:
                        continue

                    # Determine component from path
                    comp = None
                    for comp_name, comp_info in COHERENT_COMPONENTS.items():
                        if comp_info["path"] in key:
                            comp = comp_name
                            break
                        for alt_path in comp_info.get("alt_paths", []):
                            if alt_path in key:
                                comp = comp_name
                                break
                        if comp:
                            break

                    files.append({
                        "key": key,
                        "size": obj["Size"],
                        "size_mb": obj["Size"] / (1024 * 1024),
                        "component": comp,
                        "last_modified": obj.get("LastModified"),
                    })

                    if len(files) >= max_files:
                        return files

    return files


def _list_files_cli(component: Optional[str], max_files: int) -> list[dict]:
    """List files using AWS CLI."""
    prefix = S3_PREFIX
    if component:
        if component not in COHERENT_COMPONENTS:
            raise ValueError(f"Unknown component: {component}. Must be one of: {list(COHERENT_COMPONENTS.keys())}")
        prefix = S3_PREFIX + COHERENT_COMPONENTS[component]["path"]

    cmd = [
        "aws", "s3", "ls",
        "--no-sign-request",
        "--recursive",
        f"s3://{S3_BUCKET}/{prefix}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AWS CLI error: {result.stderr}")

    files = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        # Parse AWS CLI output: "2022-04-01 12:00:00    12345 path/to/file"
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[2])
            key = parts[3]

            # Determine component
            comp = None
            for comp_name, comp_info in COHERENT_COMPONENTS.items():
                if comp_info["path"] in key:
                    comp = comp_name
                    break

            files.append({
                "key": key,
                "size": size,
                "size_mb": size / (1024 * 1024),
                "component": comp,
            })

            if len(files) >= max_files:
                break

    return files


def download_coherent_dataset(
    output_dir: Optional[Path] = None,
    components: Optional[list[str]] = None,
    max_patients: Optional[int] = None,
    max_files: Optional[int] = None,
    overwrite: bool = False,
    parallel: int = 10,
    verbose: bool = True,
) -> Path:
    """
    Download the Synthea Coherent Data Set from AWS S3.

    Uses parallel downloads for fast bulk transfers.

    Args:
        output_dir: Directory to save files. Defaults to ~/.cache/synthlab/coherent/
        components: List of components to download. If None, downloads all.
                   Options: 'fhir', 'dicom', 'genomics', 'notes', 'physiological'
        max_patients: Maximum number of patients to download (None = all)
        max_files: Maximum number of files to download per component (None = all)
        overwrite: If True, overwrite existing files
        parallel: Number of parallel download threads (default: 10)
        verbose: If True, print progress

    Returns:
        Path: Path to output directory

    Example:
        >>> import synthlab as sl
        >>> # Download FHIR and genomics with parallel downloads
        >>> data_dir = sl.download_coherent_dataset(components=['fhir', 'genomics'])
        >>> # Download with limited files for testing
        >>> data_dir = sl.download_coherent_dataset(components=['fhir'], max_files=100)
        >>> # Download everything (several GB)
        >>> data_dir = sl.download_coherent_dataset()

    Note:
        The full dataset is several GB. Consider downloading specific components
        or limiting max_files for initial exploration.
    """
    if output_dir is None:
        output_dir = get_coherent_cache_dir()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate components
    if components is None:
        components = list(COHERENT_COMPONENTS.keys())
    else:
        for comp in components:
            if comp not in COHERENT_COMPONENTS:
                raise ValueError(
                    f"Unknown component: {comp}. "
                    f"Must be one of: {list(COHERENT_COMPONENTS.keys())}"
                )

    if verbose:
        print("\nSynthea Coherent Data Set Download")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Components: {', '.join(components)}")
        print(f"Parallel downloads: {parallel}")
        if max_patients:
            print(f"Max patients: {max_patients}")
        if max_files:
            print(f"Max files per component: {max_files}")
        print()

    # Use boto3 if available, otherwise AWS CLI
    if _check_boto3():
        return _download_boto3_parallel(output_dir, components, max_patients, max_files, overwrite, parallel, verbose)
    elif _check_aws_cli():
        return _download_cli(output_dir, components, max_patients, overwrite, verbose)
    else:
        raise RuntimeError(
            "Either AWS CLI or boto3 is required to download the Coherent Data Set.\n"
            "Install with: pip install boto3\n"
            "Or install AWS CLI: https://aws.amazon.com/cli/"
        )


def _download_boto3_parallel(
    output_dir: Path,
    components: list[str],
    max_patients: Optional[int],
    max_files: Optional[int],
    overwrite: bool,
    parallel: int,
    verbose: bool,
) -> Path:
    """Download using boto3 with parallel transfers."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=S3_REGION)

    for component in components:
        comp_info = COHERENT_COMPONENTS[component]
        comp_dir = output_dir / component
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Try primary path and alternatives
        primary_path = str(comp_info["path"])
        alt_paths = comp_info.get("alt_paths", [])
        paths_to_try: list[str] = [primary_path] + (alt_paths if isinstance(alt_paths, list) else [])

        if verbose:
            print(f"\nDownloading {component}: {comp_info['description']}")
            print(f"  Destination: {comp_dir}")

        # Collect files to download
        files_to_download: list[tuple[str, Path, int]] = []  # (s3_key, local_path, size)
        patients_seen: set[str] = set()

        for path in paths_to_try:
            prefix = S3_PREFIX + path
            if verbose:
                print(f"  Scanning: s3://{S3_BUCKET}/{prefix}")

            paginator = s3.get_paginator("list_objects_v2")
            try:
                for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                    if "Contents" not in page:
                        continue

                    for obj in page["Contents"]:
                        key = obj["Key"]
                        size = obj["Size"]

                        # Skip directories
                        if key.endswith("/") or size == 0:
                            continue

                        # Check max_files limit
                        if max_files and len(files_to_download) >= max_files:
                            break

                        # Extract patient ID for max_patients limit
                        patient_id = _extract_patient_id(key)

                        if max_patients and patient_id:
                            if patient_id not in patients_seen:
                                if len(patients_seen) >= max_patients:
                                    continue  # Skip files from new patients
                                patients_seen.add(patient_id)

                        # Determine local path
                        relative_path = key[len(prefix):]
                        if not relative_path:
                            relative_path = key.split("/")[-1]
                        local_path = comp_dir / relative_path

                        # Skip if exists and not overwriting
                        if local_path.exists() and not overwrite:
                            continue

                        files_to_download.append((key, local_path, size))

                    if max_files and len(files_to_download) >= max_files:
                        break

            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not access {prefix}: {e}")
                continue

            if files_to_download:
                break  # Found files, stop trying alternatives

        if not files_to_download:
            if verbose:
                print(f"  No files found for {component}")
            continue

        total_size_mb = sum(f[2] for f in files_to_download) / (1024 * 1024)
        if verbose:
            print(f"  Found {len(files_to_download)} files ({total_size_mb:.1f} MB)")
            if patients_seen:
                print(f"  Patients: {len(patients_seen)}")

        # Download files in parallel
        def download_one(item: tuple[str, Path, int]) -> tuple[str, bool, str]:
            """Download a single file. Returns (key, success, error_msg)."""
            key, local_path, size = item
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(S3_BUCKET, key, str(local_path))
                return (key, True, "")
            except Exception as e:
                return (key, False, str(e))

        downloaded = 0
        failed = 0

        # Use tqdm for progress bar if available
        total_files = len(files_to_download)

        if _has_tqdm and _tqdm_import is not None and verbose:
            pbar = _tqdm_import(total=total_files, desc=f"  {component}", unit="file")
        else:
            pbar = None
            if verbose:
                print(f"  Downloading {total_files} files...")

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_one, f): f for f in files_to_download}
            for i, future in enumerate(as_completed(futures)):
                key, success, _ = future.result()
                if success:
                    downloaded += 1
                else:
                    failed += 1

                if pbar is not None:
                    pbar.update(1)
                elif verbose:
                    # Text-based progress bar when tqdm not available
                    progress = (i + 1) / total_files
                    bar_width = 40
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"\r  [{bar}] {i+1}/{total_files}", end="", flush=True)

        if pbar is not None:
            pbar.close()
        elif verbose:
            print()  # New line after progress bar

        if verbose:
            print(f"  Downloaded: {downloaded}, Failed: {failed}")

    if verbose:
        print(f"\n✓ Download complete: {output_dir}")

    return output_dir


def _extract_patient_id(key: str) -> Optional[str]:
    """Extract patient ID (UUID) from S3 key."""
    path_parts = key.split("/")
    for part in path_parts:
        # Check for UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        if len(part) >= 36 and part[:36].count("-") == 4:
            candidate = part[:36]
            # Verify it looks like a UUID
            segments = candidate.split("-")
            if len(segments) == 5 and all(len(s) in [8, 4, 4, 4, 12] for s in segments):
                return candidate
        # Also check filename without extension
        name_part = part.split(".")[0]
        if len(name_part) == 36 and name_part.count("-") == 4:
            segments = name_part.split("-")
            if len(segments) == 5:
                return name_part
    return None


def _download_cli(
    output_dir: Path,
    components: list[str],
    max_patients: Optional[int],  # noqa: ARG001  Not used in CLI mode
    overwrite: bool,
    verbose: bool,
) -> Path:
    """Download using AWS CLI."""
    for component in components:
        comp_info = COHERENT_COMPONENTS[component]
        prefix = S3_PREFIX + str(comp_info["path"])
        comp_dir = output_dir / component

        if verbose:
            print(f"\nDownloading {component}: {comp_info['description']}")

        # Build AWS CLI command
        cmd = [
            "aws", "s3", "sync",
            "--no-sign-request",
            f"s3://{S3_BUCKET}/{prefix}",
            str(comp_dir),
        ]

        if not overwrite:
            cmd.append("--no-clobber")  # Skip existing files

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not verbose, text=True)

        if result.returncode != 0:
            if verbose:
                print(f"  ✗ Error downloading {component}")
            else:
                raise RuntimeError(f"AWS CLI error: {result.stderr}")

    if verbose:
        print(f"\n✓ Download complete: {output_dir}")

    return output_dir


def extract_patient_id(bundle: dict[str, Any]) -> Optional[str]:
    """
    Extract patient UUID from a FHIR Bundle.

    Args:
        bundle: FHIR Bundle dictionary

    Returns:
        Patient UUID string, or None if not found

    Example:
        >>> import synthlab as sl
        >>> patients = sl.load_fhir_patients(max_patients=5)
        >>> patient_id = sl.extract_patient_id(patients[0])
        >>> print(patient_id)
        'c7a5b8d4-1234-5678-abcd-ef0123456789'
    """
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            return resource.get("id")
    return None


def find_dna_csv_for_patient(
    patient_id: str,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Find DNA CSV file for a patient UUID.

    The Coherent Data Set stores genomics data as CSV files with the naming
    convention: {FirstName}_{LastName}_{PatientUUID}_dna.csv

    Args:
        patient_id: Patient UUID to search for
        cache_dir: Cache directory. Defaults to ~/.cache/synthlab/coherent/

    Returns:
        Path to DNA CSV file, or None if not found

    Example:
        >>> import synthlab as sl
        >>> dna_path = sl.find_dna_csv_for_patient("c7a5b8d4-1234-5678-abcd-ef0123456789")
        >>> if dna_path:
        ...     import polars as pl
        ...     dna_data = pl.read_csv(dna_path)
    """
    if cache_dir is None:
        cache_dir = get_coherent_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    genomics_dir = cache_dir / "genomics"
    if not genomics_dir.exists():
        return None

    matches = list(genomics_dir.rglob(f"*{patient_id}*_dna.csv"))
    return matches[0] if matches else None


def find_dicoms_for_patient(
    patient_id: str,
    cache_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Find DICOM files for a patient UUID.

    The Coherent Data Set stores imaging data as DICOM files linked
    via patient UUID in the file path or PatientID DICOM tag.

    Args:
        patient_id: Patient UUID to search for
        cache_dir: Cache directory. Defaults to ~/.cache/synthlab/coherent/

    Returns:
        List of paths to DICOM files (may be empty)

    Example:
        >>> import synthlab as sl
        >>> dicom_paths = sl.find_dicoms_for_patient("c7a5b8d4-1234-5678-abcd-ef0123456789")
        >>> print(f"Found {len(dicom_paths)} DICOM files")
    """
    if cache_dir is None:
        cache_dir = get_coherent_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    dicom_dir = cache_dir / "dicom"
    if not dicom_dir.exists():
        return []

    return list(dicom_dir.rglob(f"*{patient_id}*.dcm"))


def load_dicom_series(
    dicom_files: list[Path],
    max_files: int = 5,
) -> list[Any]:
    """
    Load a DICOM series for inspection.

    Args:
        dicom_files: List of paths to DICOM files
        max_files: Maximum number of files to load (default: 5)

    Returns:
        List of pydicom.Dataset objects

    Raises:
        ImportError: If pydicom is not installed

    Example:
        >>> import synthlab as sl
        >>> dicom_paths = sl.find_dicoms_for_patient(patient_id)
        >>> datasets = sl.load_dicom_series(dicom_paths, max_files=3)
        >>> for ds in datasets:
        ...     print(ds.PatientName, ds.Modality)
    """
    if not dicom_files:
        return []

    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required to load DICOM files: pip install pydicom")

    datasets = []
    for dcm_path in dicom_files[:max_files]:
        try:
            datasets.append(pydicom.dcmread(dcm_path))
        except Exception:
            continue

    return datasets


def load_fhir_patients(
    data_dir: Optional[Path] = None,
    max_patients: Optional[int] = None,
) -> list[dict]:
    """
    Load FHIR patient bundles from the Coherent Data Set.

    Args:
        data_dir: Directory containing Coherent Data Set.
                  Defaults to ~/.cache/synthlab/coherent/
        max_patients: Maximum number of patients to load

    Returns:
        list: List of FHIR Bundle dictionaries

    Example:
        >>> from synthlab.coherent import load_fhir_patients
        >>> patients = load_fhir_patients(max_patients=10)
        >>> for p in patients:
        ...     # Extract patient resource
        ...     patient_resource = next(
        ...         e['resource'] for e in p['entry']
        ...         if e['resource']['resourceType'] == 'Patient'
        ...     )
        ...     print(patient_resource.get('name'))
    """
    if data_dir is None:
        data_dir = get_coherent_cache_dir()
    else:
        data_dir = Path(data_dir)

    fhir_dir = data_dir / "fhir"

    if not fhir_dir.exists():
        raise FileNotFoundError(
            f"FHIR directory not found: {fhir_dir}\n"
            f"Download first with: download_coherent_dataset(components=['fhir'])"
        )

    # Find all JSON files
    json_files = list(fhir_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No FHIR JSON files found in {fhir_dir}")

    patients = []
    for json_file in json_files:
        if max_patients and len(patients) >= max_patients:
            break

        try:
            with open(json_file, "r") as f:
                bundle = json.load(f)
                if bundle.get("resourceType") == "Bundle":
                    patients.append(bundle)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    print(f"✓ Loaded {len(patients)} FHIR patient bundles")
    return patients


def get_coherent_info() -> dict:
    """
    Get information about the Coherent Data Set.

    Returns:
        dict: Dataset information and capabilities
    """
    aws_cli = _check_aws_cli()
    boto3_available = _check_boto3()

    return {
        "name": "Synthea Coherent Data Set",
        "description": "Multimodal synthetic healthcare data combining EHR, imaging, genomics, notes, and physiological data",
        "source": f"s3://{S3_BUCKET}/{S3_PREFIX}",
        "license": "CC BY 4.0",
        "paper": "https://www.mdpi.com/2079-9292/11/8/1199",
        "aws_registry": "https://registry.opendata.aws/synthea-coherent-data/",
        "components": COHERENT_COMPONENTS,
        "tools_available": {
            "aws_cli": aws_cli,
            "boto3": boto3_available,
        },
        "functions": [
            "list_coherent_components() - List dataset components",
            "list_coherent_files(component=None) - List files in S3",
            "download_coherent_dataset(components=None) - Download data",
            "load_fhir_patients(max_patients=None) - Load FHIR records",
        ],
    }


def print_s3_structure():
    """
    Print the actual folder structure in the S3 bucket.

    This is useful for discovering the exact paths when the default
    component paths don't work.
    """
    print("\nDiscovering S3 bucket structure...")
    print(f"Bucket: s3://{S3_BUCKET}/{S3_PREFIX}")
    print("=" * 60)

    try:
        structure = discover_s3_structure()

        if structure["folders"]:
            print("\nFolders:")
            for folder in sorted(structure["folders"]):
                print(f"  📁 {folder}")

        if structure["files"]:
            print(f"\nFiles at root ({len(structure['files'])}):")
            for f in structure["files"][:10]:
                print(f"  📄 {f['key']} ({f['size_mb']:.2f} MB)")
            if len(structure["files"]) > 10:
                print(f"  ... and {len(structure['files']) - 10} more")

        print(f"\nTotal size at root level: {structure['total_size_mb']:.2f} MB")

    except RuntimeError as e:
        print(f"Error: {e}")


def print_coherent_info():
    """Print formatted information about the Coherent Data Set."""
    info = get_coherent_info()

    print("\n" + "=" * 70)
    print("Synthea Coherent Data Set")
    print("=" * 70)
    print(f"\n{info['description']}")
    print(f"\nSource: {info['source']}")
    print(f"License: {info['license']}")
    print(f"Paper: {info['paper']}")

    print("\n" + "-" * 70)
    print("Components:")
    print("-" * 70)
    for name, comp in info["components"].items():
        print(f"\n  {name.upper()}")
        print(f"    {comp['description']}")
        print(f"    Format: {comp['format']}")
        print(f"    Contents: {comp['contents']}")

    print("\n" + "-" * 70)
    print("Requirements:")
    print("-" * 70)
    print(f"  AWS CLI: {'✓ Available' if info['tools_available']['aws_cli'] else '✗ Not found'}")
    print(f"  boto3:   {'✓ Available' if info['tools_available']['boto3'] else '✗ Not found'}")

    if not info["tools_available"]["aws_cli"] and not info["tools_available"]["boto3"]:
        print("\n  ⚠️  Install boto3 to download: pip install boto3")

    print("\n" + "-" * 70)
    print("Quick Start:")
    print("-" * 70)
    print("""
  from synthlab.coherent import download_coherent_dataset, load_fhir_patients

  # Download FHIR and genomics data
  download_coherent_dataset(components=['fhir', 'genomics'])

  # Load patient records
  patients = load_fhir_patients(max_patients=10)
""")


# =============================================================================
# Multimodal Patient Data Classes
# =============================================================================


@dataclass
class Patient:
    """
    A single patient with optional multimodal data from the Coherent Data Set.

    This class provides a unified interface for accessing patient data across
    multiple modalities (EHR, imaging, genomics, etc.). Not all modalities
    need to be present for each patient.

    Attributes:
        patient_id: Unique patient identifier (UUID)
        name: Patient name (if available)
        fhir_path: Path to FHIR JSON file (lazy loaded via .fhir property)
        dicom_paths: List of paths to DICOM files
        genomics_path: Path to DNA CSV file (lazy loaded via .genomics property)

    Example:
        >>> from synthlab.coherent import MultimodalDataset
        >>> dataset = MultimodalDataset.from_cache(max_patients=10)
        >>> patient = dataset[0]
        >>> print(patient.modalities)
        ['fhir', 'genomics']
        >>> demographics = patient.get_demographics()
        >>> conditions = patient.get_conditions()
    """

    patient_id: str
    name: Optional[str] = None
    fhir_path: Optional[Path] = None  # Path for lazy loading
    dicom_paths: list[Path] = field(default_factory=list)
    genomics_path: Optional[Path] = None
    notes_paths: list[Path] = field(default_factory=list)
    _fhir_bundle: Optional[dict] = field(default=None, repr=False)  # Cached FHIR
    _genomics_df: Optional[Any] = field(default=None, repr=False)

    @property
    def fhir(self) -> Optional[dict]:
        """
        Load and return FHIR bundle (lazy loaded).

        Returns None if FHIR data is not available.
        Caches the result for subsequent calls.
        """
        if self.fhir_path is None:
            return None

        if self._fhir_bundle is None:
            try:
                with open(self.fhir_path) as f:
                    self._fhir_bundle = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load FHIR bundle: {e}")
                return None

        return self._fhir_bundle

    @property
    def modalities(self) -> list[str]:
        """Return list of available modalities for this patient."""
        available = []
        if self.fhir_path is not None:
            available.append("fhir")
        if self.dicom_paths:
            available.append("dicom")
        if self.genomics_path is not None:
            available.append("genomics")
        if self.notes_paths:
            available.append("notes")
        return available

    def has_modality(self, modality: str) -> bool:
        """Check if patient has a specific modality."""
        return modality in self.modalities

    def has_all_modalities(self, modalities: list[str]) -> bool:
        """Check if patient has all specified modalities."""
        return all(self.has_modality(m) for m in modalities)

    def has_any_modality(self, modalities: list[str]) -> bool:
        """Check if patient has any of the specified modalities."""
        return any(self.has_modality(m) for m in modalities)

    @property
    def genomics(self) -> Optional[Any]:
        """
        Load and return genomics data as a Polars DataFrame.

        Returns None if genomics data is not available.
        Caches the result for subsequent calls.
        """
        if self.genomics_path is None:
            return None

        if self._genomics_df is None:
            try:
                import polars as pl
                self._genomics_df = pl.read_csv(self.genomics_path)
            except ImportError:
                raise ImportError("polars is required to load genomics data: pip install polars")

        return self._genomics_df

    def get_demographics(self) -> Optional[dict]:
        """
        Extract patient demographics from FHIR bundle.

        Returns:
            dict with keys: id, name, given_name, family_name, gender,
                           birth_date, address, phone, marital_status
            None if FHIR data not available
        """
        if self.fhir is None:
            return None

        patient_resource = self._get_fhir_resource("Patient")
        if not patient_resource:
            return None

        # Extract name
        names = patient_resource.get("name", [{}])
        name_info = names[0] if names else {}
        given = " ".join(name_info.get("given", []))
        family = name_info.get("family", "")

        # Extract address
        addresses = patient_resource.get("address", [{}])
        addr = addresses[0] if addresses else {}
        address_str = ", ".join(filter(None, [
            " ".join(addr.get("line", [])),
            addr.get("city", ""),
            addr.get("state", ""),
            addr.get("postalCode", ""),
            addr.get("country", ""),
        ]))

        # Extract phone
        telecoms = patient_resource.get("telecom", [])
        phone = next((t.get("value") for t in telecoms if t.get("system") == "phone"), None)

        return {
            "id": patient_resource.get("id"),
            "name": f"{given} {family}".strip(),
            "given_name": given,
            "family_name": family,
            "gender": patient_resource.get("gender"),
            "birth_date": patient_resource.get("birthDate"),
            "address": address_str or None,
            "phone": phone,
            "marital_status": patient_resource.get("maritalStatus", {}).get("text"),
        }

    def get_conditions(self) -> list[dict]:
        """
        Extract conditions/diagnoses from FHIR bundle.

        Returns:
            List of dicts with keys: code, display, system, onset_date, clinical_status
        """
        if self.fhir is None:
            return []

        conditions = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Condition":
                continue

            coding = resource.get("code", {}).get("coding", [{}])[0]
            conditions.append({
                "code": coding.get("code"),
                "display": coding.get("display"),
                "system": coding.get("system"),
                "onset_date": resource.get("onsetDateTime"),
                "clinical_status": resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code"),
            })

        return conditions

    def get_medications(self) -> list[dict]:
        """
        Extract medications from FHIR bundle.

        Returns:
            List of dicts with keys: code, display, system, status, authored_on
        """
        if self.fhir is None:
            return []

        medications = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "MedicationRequest":
                continue

            # Get medication code from medicationCodeableConcept
            med_concept = resource.get("medicationCodeableConcept", {})
            coding = med_concept.get("coding", [{}])[0]

            medications.append({
                "code": coding.get("code"),
                "display": coding.get("display"),
                "system": coding.get("system"),
                "status": resource.get("status"),
                "authored_on": resource.get("authoredOn"),
            })

        return medications

    def get_encounters(self) -> list[dict]:
        """
        Extract encounters/visits from FHIR bundle.

        Returns:
            List of dicts with keys: id, type, class, status, start, end, reason
        """
        if self.fhir is None:
            return []

        encounters = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Encounter":
                continue

            # Get encounter type
            enc_types = resource.get("type", [{}])
            enc_type = enc_types[0].get("coding", [{}])[0] if enc_types else {}

            # Get period
            period = resource.get("period", {})

            # Get reason
            reasons = resource.get("reasonCode", [{}])
            reason = reasons[0].get("coding", [{}])[0].get("display") if reasons else None

            encounters.append({
                "id": resource.get("id"),
                "type": enc_type.get("display"),
                "class": resource.get("class", {}).get("code"),
                "status": resource.get("status"),
                "start": period.get("start"),
                "end": period.get("end"),
                "reason": reason,
            })

        return encounters

    def get_observations(self, category: Optional[str] = None) -> list[dict]:
        """
        Extract observations (labs, vitals, etc.) from FHIR bundle.

        Args:
            category: Optional filter by category ('vital-signs', 'laboratory', etc.)

        Returns:
            List of dicts with keys: code, display, value, unit, date, category
        """
        if self.fhir is None:
            return []

        observations = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Observation":
                continue

            # Get category
            categories = resource.get("category", [{}])
            cat_coding = categories[0].get("coding", [{}])[0] if categories else {}
            obs_category = cat_coding.get("code")

            if category and obs_category != category:
                continue

            # Get code
            coding = resource.get("code", {}).get("coding", [{}])[0]

            # Get value (handle different value types)
            value = None
            unit = None
            if "valueQuantity" in resource:
                vq = resource["valueQuantity"]
                value = vq.get("value")
                unit = vq.get("unit")
            elif "valueCodeableConcept" in resource:
                value = resource["valueCodeableConcept"].get("text")
            elif "valueString" in resource:
                value = resource["valueString"]

            observations.append({
                "code": coding.get("code"),
                "display": coding.get("display"),
                "value": value,
                "unit": unit,
                "date": resource.get("effectiveDateTime"),
                "category": obs_category,
            })

        return observations

    def get_procedures(self) -> list[dict]:
        """
        Extract procedures from FHIR bundle.

        Returns:
            List of dicts with keys: code, display, system, status, performed_date
        """
        if self.fhir is None:
            return []

        procedures = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Procedure":
                continue

            coding = resource.get("code", {}).get("coding", [{}])[0]
            procedures.append({
                "code": coding.get("code"),
                "display": coding.get("display"),
                "system": coding.get("system"),
                "status": resource.get("status"),
                "performed_date": resource.get("performedDateTime") or resource.get("performedPeriod", {}).get("start"),
            })

        return procedures

    def get_fhir_resources(self, resource_type: str) -> list[dict]:
        """
        Get all FHIR resources of a specific type.

        Args:
            resource_type: FHIR resource type (e.g., 'Condition', 'Observation')

        Returns:
            List of FHIR resource dictionaries
        """
        if self.fhir is None:
            return []

        resources = []
        for entry in self.fhir.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == resource_type:
                resources.append(resource)
        return resources

    def _get_fhir_resource(self, resource_type: str) -> Optional[dict]:
        """Get first FHIR resource of a specific type."""
        resources = self.get_fhir_resources(resource_type)
        return resources[0] if resources else None

    def load_dicom(self, index: int = 0) -> Any:
        """
        Load a DICOM file using pydicom.

        Args:
            index: Index of DICOM file to load (default: 0)

        Returns:
            pydicom.Dataset object

        Raises:
            ImportError: If pydicom not installed
            IndexError: If index out of range
            FileNotFoundError: If no DICOM files available
        """
        if not self.dicom_paths:
            raise FileNotFoundError(f"No DICOM files available for patient {self.patient_id}")

        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom is required to load DICOM files: pip install pydicom")

        return pydicom.dcmread(self.dicom_paths[index])

    def summary(self) -> dict:
        """
        Get a summary of available data for this patient.

        Returns:
            dict with modality counts and basic info
        """
        demographics = self.get_demographics()

        return {
            "patient_id": self.patient_id,
            "name": self.name or (demographics.get("name") if demographics else None),
            "modalities": self.modalities,
            "n_dicom_files": len(self.dicom_paths),
            "n_conditions": len(self.get_conditions()) if self.fhir else 0,
            "n_medications": len(self.get_medications()) if self.fhir else 0,
            "n_encounters": len(self.get_encounters()) if self.fhir else 0,
            "n_observations": len(self.get_observations()) if self.fhir else 0,
            "has_genomics": self.genomics_path is not None,
        }

    def __repr__(self) -> str:
        return f"Patient(id={self.patient_id[:8]}..., name={self.name}, modalities={self.modalities})"


@dataclass
class MultimodalDataset:
    """
    A collection of patients with multimodal data from the Coherent Data Set.

    This class provides methods to load, query, filter, and iterate over
    patients with their associated multimodal data.

    Example:
        >>> from synthlab.coherent import MultimodalDataset
        >>>
        >>> # Load from cache directory
        >>> dataset = MultimodalDataset.from_cache(max_patients=100)
        >>> print(f"Loaded {len(dataset)} patients")
        >>>
        >>> # Filter to patients with both FHIR and genomics
        >>> multimodal = dataset.with_modalities(['fhir', 'genomics'])
        >>>
        >>> # Query by condition
        >>> diabetics = dataset.with_condition('diabetes')
        >>>
        >>> # Iterate over patients
        >>> for patient in dataset:
        ...     print(patient.get_demographics())
    """

    patients: list[Patient] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Patient, "MultimodalDataset"]:
        if isinstance(key, int):
            return self.patients[key]
        elif isinstance(key, slice):
            return MultimodalDataset(patients=self.patients[key])
        elif isinstance(key, str):
            # Lookup by patient_id
            for p in self.patients:
                if p.patient_id == key or p.patient_id.startswith(key):
                    return p
            raise KeyError(f"Patient not found: {key}")
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self) -> Iterator[Patient]:
        return iter(self.patients)

    def __contains__(self, patient_id: str) -> bool:
        return any(p.patient_id == patient_id for p in self.patients)

    @classmethod
    def from_cache(
        cls,
        cache_dir: Optional[Path] = None,
        max_patients: Optional[int] = None,
        modalities: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> "MultimodalDataset":
        """
        Load dataset from the cache directory.

        Args:
            cache_dir: Path to cache directory. Defaults to ~/.cache/synthlab/coherent/
            max_patients: Maximum number of patients to load
            modalities: List of modalities to load. If None, loads all available.
                       Options: 'fhir', 'dicom', 'genomics', 'notes'
            verbose: Print loading progress

        Returns:
            MultimodalDataset with loaded patients
        """
        if cache_dir is None:
            cache_dir = get_coherent_cache_dir()
        else:
            cache_dir = Path(cache_dir)

        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {cache_dir}\n"
                f"Download data first with: download_coherent_dataset()"
            )

        if modalities is None:
            modalities = ["fhir", "dicom", "genomics", "notes"]

        # Collect all patient IDs from available data
        patient_data: dict[str, dict] = {}

        # Helper to create progress bar or return iterator
        def _iter_with_progress(items: list, desc: str, unit: str = "file"):
            if not verbose:
                return items
            if _has_tqdm and _tqdm_import is not None:
                return _tqdm_import(items, desc=desc, unit=unit)
            else:
                print(f"{desc}: {len(items)} {unit}s")
                return items

        # Index FHIR paths (lazy loading - JSON loaded on access)
        if "fhir" in modalities:
            fhir_dir = cache_dir / "fhir"
            if fhir_dir.exists():
                json_files = list(fhir_dir.rglob("*.json"))
                for json_file in _iter_with_progress(json_files, "Indexing FHIR"):
                    patient_id = _extract_patient_id_from_filename(json_file.name)
                    if patient_id:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {"name": _extract_name_from_filename(json_file.name)}
                        patient_data[patient_id]["fhir_path"] = json_file

        # Load DICOM paths
        if "dicom" in modalities:
            dicom_dir = cache_dir / "dicom"
            if dicom_dir.exists():
                dcm_files = list(dicom_dir.rglob("*.dcm"))
                for dcm_file in _iter_with_progress(dcm_files, "Indexing DICOM"):
                    patient_id = _extract_patient_id_from_filename(dcm_file.name)
                    if patient_id:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {"name": _extract_name_from_filename(dcm_file.name)}
                        if "dicom_paths" not in patient_data[patient_id]:
                            patient_data[patient_id]["dicom_paths"] = []
                        patient_data[patient_id]["dicom_paths"].append(dcm_file)

        # Load genomics paths
        if "genomics" in modalities:
            genomics_dir = cache_dir / "genomics"
            if genomics_dir.exists():
                csv_files = list(genomics_dir.rglob("*_dna.csv"))
                for csv_file in _iter_with_progress(csv_files, "Indexing genomics"):
                    patient_id = _extract_patient_id_from_filename(csv_file.name)
                    if patient_id:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {"name": _extract_name_from_filename(csv_file.name)}
                        patient_data[patient_id]["genomics_path"] = csv_file

        # Load notes paths
        if "notes" in modalities:
            notes_dir = cache_dir / "notes"
            if notes_dir.exists():
                note_files = [f for f in notes_dir.rglob("*") if f.is_file()]
                for note_file in _iter_with_progress(note_files, "Indexing notes"):
                    patient_id = _extract_patient_id_from_filename(note_file.name)
                    if patient_id:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {"name": _extract_name_from_filename(note_file.name)}
                        if "notes_paths" not in patient_data[patient_id]:
                            patient_data[patient_id]["notes_paths"] = []
                        patient_data[patient_id]["notes_paths"].append(note_file)

        # Create Patient objects
        patients = []
        for patient_id, data in patient_data.items():
            if max_patients and len(patients) >= max_patients:
                break

            patient = Patient(
                patient_id=patient_id,
                name=data.get("name"),
                fhir_path=data.get("fhir_path"),
                dicom_paths=data.get("dicom_paths", []),
                genomics_path=data.get("genomics_path"),
                notes_paths=data.get("notes_paths", []),
            )
            patients.append(patient)

        if verbose:
            print(f"✓ Loaded {len(patients)} patients")

        return cls(patients=patients)

    def filter(self, predicate: Callable[[Patient], bool]) -> "MultimodalDataset":
        """
        Filter patients using a custom predicate function.

        Args:
            predicate: Function that takes a Patient and returns True to keep

        Returns:
            New MultimodalDataset with filtered patients
        """
        return MultimodalDataset(patients=[p for p in self.patients if predicate(p)])

    def with_modality(self, modality: str) -> "MultimodalDataset":
        """
        Get subset of patients that have a specific modality.

        Args:
            modality: Modality name ('fhir', 'dicom', 'genomics', 'notes')

        Returns:
            New MultimodalDataset with filtered patients
        """
        return self.filter(lambda p: p.has_modality(modality))

    def with_modalities(self, modalities: list[str], require_all: bool = True) -> "MultimodalDataset":
        """
        Get subset of patients that have specified modalities.

        Args:
            modalities: List of modality names
            require_all: If True, patient must have ALL modalities.
                        If False, patient must have ANY of the modalities.

        Returns:
            New MultimodalDataset with filtered patients
        """
        if require_all:
            return self.filter(lambda p: p.has_all_modalities(modalities))
        else:
            return self.filter(lambda p: p.has_any_modality(modalities))

    def with_condition(self, condition: str, case_sensitive: bool = False) -> "MultimodalDataset":
        """
        Get subset of patients that have a specific condition.

        Args:
            condition: Condition name or code to search for
            case_sensitive: Whether to match case-sensitively

        Returns:
            New MultimodalDataset with filtered patients
        """
        def has_condition(patient: Patient) -> bool:
            for cond in patient.get_conditions():
                display = cond.get("display") or ""
                code = cond.get("code") or ""

                if case_sensitive:
                    if condition in display or condition == code:
                        return True
                else:
                    if condition.lower() in display.lower() or condition == code:
                        return True
            return False

        return self.filter(has_condition)

    def with_medication(self, medication: str, case_sensitive: bool = False) -> "MultimodalDataset":
        """
        Get subset of patients that have a specific medication.

        Args:
            medication: Medication name or code to search for
            case_sensitive: Whether to match case-sensitively

        Returns:
            New MultimodalDataset with filtered patients
        """
        def has_medication(patient: Patient) -> bool:
            for med in patient.get_medications():
                display = med.get("display") or ""
                code = med.get("code") or ""

                if case_sensitive:
                    if medication in display or medication == code:
                        return True
                else:
                    if medication.lower() in display.lower() or medication == code:
                        return True
            return False

        return self.filter(has_medication)

    def with_gender(self, gender: str) -> "MultimodalDataset":
        """
        Get subset of patients with a specific gender.

        Args:
            gender: Gender to filter by ('male', 'female', 'other', 'unknown')

        Returns:
            New MultimodalDataset with filtered patients
        """
        def matches_gender(patient: Patient) -> bool:
            demographics = patient.get_demographics()
            if demographics:
                return demographics.get("gender", "").lower() == gender.lower()
            return False

        return self.filter(matches_gender)

    def summary(self) -> dict:
        """
        Get summary statistics for the dataset.

        Returns:
            dict with counts and statistics
        """
        modality_counts = {"fhir": 0, "dicom": 0, "genomics": 0, "notes": 0}
        total_dicom = 0
        total_conditions = 0
        total_medications = 0

        for patient in self.patients:
            for mod in patient.modalities:
                modality_counts[mod] = modality_counts.get(mod, 0) + 1
            total_dicom += len(patient.dicom_paths)
            total_conditions += len(patient.get_conditions())
            total_medications += len(patient.get_medications())

        return {
            "n_patients": len(self.patients),
            "modality_coverage": modality_counts,
            "total_dicom_files": total_dicom,
            "total_conditions": total_conditions,
            "total_medications": total_medications,
            "patients_with_all_modalities": len(self.with_modalities(["fhir", "dicom", "genomics"])),
        }

    def to_dataframe(self, include_demographics: bool = True) -> Any:
        """
        Convert patient summary data to a Polars DataFrame.

        Args:
            include_demographics: Include demographic information

        Returns:
            Polars DataFrame with one row per patient
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required: pip install polars")

        rows = []
        for patient in self.patients:
            row = {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "has_fhir": patient.has_modality("fhir"),
                "has_dicom": patient.has_modality("dicom"),
                "has_genomics": patient.has_modality("genomics"),
                "n_dicom_files": len(patient.dicom_paths),
                "n_conditions": len(patient.get_conditions()),
                "n_medications": len(patient.get_medications()),
            }

            if include_demographics:
                demographics = patient.get_demographics()
                if demographics:
                    row.update({
                        "gender": demographics.get("gender"),
                        "birth_date": demographics.get("birth_date"),
                    })

            rows.append(row)

        return pl.DataFrame(rows)

    def print_summary(self) -> None:
        """Print a formatted summary of the dataset."""
        summary = self.summary()

        print("\n" + "=" * 60)
        print("Multimodal Dataset Summary")
        print("=" * 60)
        print(f"\nTotal patients: {summary['n_patients']}")
        print(f"\nModality coverage:")
        for mod, count in summary["modality_coverage"].items():
            pct = (count / summary["n_patients"] * 100) if summary["n_patients"] > 0 else 0
            print(f"  {mod}: {count} patients ({pct:.1f}%)")
        print(f"\nPatients with all modalities: {summary['patients_with_all_modalities']}")
        print(f"Total DICOM files: {summary['total_dicom_files']}")
        print(f"Total conditions: {summary['total_conditions']}")
        print(f"Total medications: {summary['total_medications']}")

    # =========================================================================
    # Iteration methods for events across all patients
    # =========================================================================

    def iter_conditions(self) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all conditions across all patients.

        Yields:
            Tuples of (patient_id, condition_dict)

        Example:
            >>> for patient_id, condition in dataset.iter_conditions():
            ...     print(f"{patient_id}: {condition['display']}")
        """
        for patient in self.patients:
            for condition in patient.get_conditions():
                yield (patient.patient_id, condition)

    def iter_medications(self) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all medications across all patients.

        Yields:
            Tuples of (patient_id, medication_dict)
        """
        for patient in self.patients:
            for medication in patient.get_medications():
                yield (patient.patient_id, medication)

    def iter_encounters(self) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all encounters across all patients.

        Yields:
            Tuples of (patient_id, encounter_dict)
        """
        for patient in self.patients:
            for encounter in patient.get_encounters():
                yield (patient.patient_id, encounter)

    def iter_observations(self, category: Optional[str] = None) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all observations across all patients.

        Args:
            category: Optional filter by category ('vital-signs', 'laboratory', etc.)

        Yields:
            Tuples of (patient_id, observation_dict)
        """
        for patient in self.patients:
            for observation in patient.get_observations(category=category):
                yield (patient.patient_id, observation)

    def iter_procedures(self) -> Iterator[tuple[str, dict]]:
        """
        Iterate over all procedures across all patients.

        Yields:
            Tuples of (patient_id, procedure_dict)
        """
        for patient in self.patients:
            for procedure in patient.get_procedures():
                yield (patient.patient_id, procedure)

    # =========================================================================
    # Aggregation methods
    # =========================================================================

    def all_conditions(self) -> list[tuple[str, dict]]:
        """Get all conditions across all patients as a list."""
        return list(self.iter_conditions())

    def all_medications(self) -> list[tuple[str, dict]]:
        """Get all medications across all patients as a list."""
        return list(self.iter_medications())

    def all_encounters(self) -> list[tuple[str, dict]]:
        """Get all encounters across all patients as a list."""
        return list(self.iter_encounters())

    def all_observations(self, category: Optional[str] = None) -> list[tuple[str, dict]]:
        """Get all observations across all patients as a list."""
        return list(self.iter_observations(category=category))

    def conditions_dataframe(self) -> Any:
        """
        Get all conditions as a Polars DataFrame.

        Returns:
            DataFrame with columns: patient_id, code, display, system, onset_date, clinical_status
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required: pip install polars")

        rows = []
        for patient_id, cond in self.iter_conditions():
            rows.append({
                "patient_id": patient_id,
                "code": cond.get("code"),
                "display": cond.get("display"),
                "system": cond.get("system"),
                "onset_date": cond.get("onset_date"),
                "clinical_status": cond.get("clinical_status"),
            })
        return pl.DataFrame(rows)

    def medications_dataframe(self) -> Any:
        """
        Get all medications as a Polars DataFrame.

        Returns:
            DataFrame with columns: patient_id, code, display, system, status, authored_on
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required: pip install polars")

        rows = []
        for patient_id, med in self.iter_medications():
            rows.append({
                "patient_id": patient_id,
                "code": med.get("code"),
                "display": med.get("display"),
                "system": med.get("system"),
                "status": med.get("status"),
                "authored_on": med.get("authored_on"),
            })
        return pl.DataFrame(rows)

    def observations_dataframe(self, category: Optional[str] = None) -> Any:
        """
        Get all observations as a Polars DataFrame.

        Args:
            category: Optional filter by category

        Returns:
            DataFrame with columns: patient_id, code, display, value, unit, date, category
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("polars is required: pip install polars")

        rows = []
        for patient_id, obs in self.iter_observations(category=category):
            rows.append({
                "patient_id": patient_id,
                "code": obs.get("code"),
                "display": obs.get("display"),
                "value": obs.get("value"),
                "unit": obs.get("unit"),
                "date": obs.get("date"),
                "category": obs.get("category"),
            })
        return pl.DataFrame(rows)

    # =========================================================================
    # Search methods
    # =========================================================================

    def search(self, query: str, case_sensitive: bool = False) -> "MultimodalDataset":
        """
        Search for patients matching a query across conditions, medications, and names.

        Args:
            query: Search string to match
            case_sensitive: Whether to match case-sensitively

        Returns:
            New MultimodalDataset with matching patients
        """
        def matches(patient: Patient) -> bool:
            q = query if case_sensitive else query.lower()

            # Search in name
            if patient.name:
                name = patient.name if case_sensitive else patient.name.lower()
                if q in name:
                    return True

            # Search in conditions
            for cond in patient.get_conditions():
                display = cond.get("display") or ""
                if not case_sensitive:
                    display = display.lower()
                if q in display:
                    return True

            # Search in medications
            for med in patient.get_medications():
                display = med.get("display") or ""
                if not case_sensitive:
                    display = display.lower()
                if q in display:
                    return True

            return False

        return self.filter(matches)

    def find_patients_with_observation(
        self,
        code: Optional[str] = None,
        display: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "MultimodalDataset":
        """
        Find patients with observations matching criteria.

        Args:
            code: Observation code to match
            display: Text to search in observation display (case-insensitive)
            min_value: Minimum observation value
            max_value: Maximum observation value

        Returns:
            New MultimodalDataset with matching patients
        """
        def matches(patient: Patient) -> bool:
            for obs in patient.get_observations():
                # Check code
                if code and obs.get("code") != code:
                    continue

                # Check display
                if display:
                    obs_display = (obs.get("display") or "").lower()
                    if display.lower() not in obs_display:
                        continue

                # Check value range
                obs_value = obs.get("value")
                if obs_value is not None and isinstance(obs_value, (int, float)):
                    if min_value is not None and obs_value < min_value:
                        continue
                    if max_value is not None and obs_value > max_value:
                        continue

                return True
            return False

        return self.filter(matches)

    @property
    def patient_ids(self) -> list[str]:
        """Get list of all patient IDs."""
        return [p.patient_id for p in self.patients]

    def sample(self, n: int, seed: Optional[int] = None) -> "MultimodalDataset":
        """
        Get a random sample of patients.

        Args:
            n: Number of patients to sample
            seed: Random seed for reproducibility

        Returns:
            New MultimodalDataset with sampled patients
        """
        import random
        if seed is not None:
            random.seed(seed)

        n = min(n, len(self.patients))
        sampled = random.sample(self.patients, n)
        return MultimodalDataset(patients=sampled)

    def __repr__(self) -> str:
        return f"MultimodalDataset(n_patients={len(self.patients)})"


def load_multimodal_dataset(
    max_patients: Optional[int] = None,
    modalities: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    download: bool = False,
    download_components: Optional[list[str]] = None,
    verbose: bool = True,
) -> MultimodalDataset:
    """
    Load a multimodal dataset from the Coherent Data Set.

    This is the main entry point for working with multimodal patient data.
    It automatically loads all available modalities and creates Patient objects
    with lazy-loaded data.

    Args:
        max_patients: Maximum number of patients to load (None = all)
        modalities: List of modalities to load. If None, loads all available.
                   Options: 'fhir', 'dicom', 'genomics', 'notes'
        cache_dir: Path to cache directory. Defaults to ~/.cache/synthlab/coherent/
        download: If True and data not found, download it first
        download_components: Components to download if download=True.
                            Defaults to ['fhir', 'genomics']
        verbose: Print loading progress

    Returns:
        MultimodalDataset with loaded patients

    Example:
        >>> import synthlab as sl
        >>>
        >>> # Load all available data
        >>> dataset = sl.load_multimodal_dataset(max_patients=100)
        >>>
        >>> # Iterate over patients
        >>> for patient in dataset:
        ...     print(patient.name, patient.modalities)
        >>>
        >>> # Filter to patients with specific modalities
        >>> ehr_genomics = dataset.with_modalities(['fhir', 'genomics'])
        >>>
        >>> # Search for diabetic patients
        >>> diabetics = dataset.with_condition('diabetes')
        >>>
        >>> # Get all conditions as a DataFrame
        >>> conditions_df = dataset.conditions_dataframe()
    """
    if cache_dir is None:
        cache_dir = get_coherent_cache_dir()
    else:
        cache_dir = Path(cache_dir)

    # Download data if requested and not found
    if download and not cache_dir.exists():
        if verbose:
            print("Data not found, downloading...")
        components = download_components or ["fhir", "genomics"]
        download_coherent_dataset(
            output_dir=cache_dir,
            components=components,
            max_patients=max_patients,
            verbose=verbose,
        )

    # Load from cache
    return MultimodalDataset.from_cache(
        cache_dir=cache_dir,
        max_patients=max_patients,
        modalities=modalities,
        verbose=verbose,
    )


def _extract_patient_id_from_filename(filename: str) -> Optional[str]:
    """Extract patient UUID from a filename."""
    # Pattern: {FirstName}_{LastName}_{UUID}... or just {UUID}...
    # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    uuid_pattern = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
    match = re.search(uuid_pattern, filename, re.IGNORECASE)
    return match.group(0).lower() if match else None


def _extract_name_from_filename(filename: str) -> Optional[str]:
    """Extract patient name from a filename."""
    # Pattern: {FirstName}_{LastName}_{UUID}...
    # Names typically have numbers appended (e.g., Abe604_Frami345)

    name_pattern = r"^([A-Z][a-z]+\d*)_([A-Z][a-z]+\d*)_"
    match = re.match(name_pattern, filename)
    if match:
        first = re.sub(r"\d+$", "", match.group(1))  # Remove trailing numbers
        last = re.sub(r"\d+$", "", match.group(2))
        return f"{first} {last}"
    return None
