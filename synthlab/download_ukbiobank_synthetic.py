#!/usr/bin/env python3
"""
Download UK Biobank Synthetic Dataset.

The UK Biobank Synthetic Dataset is a large-scale synthetic dataset designed for system testing.
It contains four main categories:
1. Tabular Records: Phenotype data (~27K columns × 600K rows) - split into 23 TSV files
2. Medical Records: GP clinical records (~400M rows) - split into 6 text files
3. Genetic Records: SNP genotype data (~600K samples × 840K SNPs) - dictionary + 26 chromosome files
4. Bulk Files: ~6M files in zip archives (for system testing, not analysis)

Dataset URL: https://biobank.ndph.ox.ac.uk/synthetic_dataset/
"""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


BASE_URL = "https://biobank.ndph.ox.ac.uk/synthetic_dataset/"

# File categories with their metadata
TABULAR_FILES = {
    "41257_HES_SimDates.tsv": "a68feb44e037397bc3cb43a6c0c86ef9",
    "41260_HES_SimDates.tsv": "4ff448b195ad417c3ae1324312782c30",
    "41262_HES_SimDates.tsv": "46aced37adea430907b81b8370f4718b",
    "41263_HES_SimDates.tsv": "5fc75c1d4d221d4e8366d4ce7920e7f8",
    "41280_HES_SimDates.tsv": "60007421300548e3a03c317e3392e5d1",
    "41281_HES_SimDates.tsv": "3b5a706c475050c5a64ad4359d224309",
    "41282_HES_SimDates.tsv": "7592c86dbb8502ca0630a763aa85be47",
    "41283_HES_SimDates.tsv": "5c35335d9e91f1eb4c0dca92213f6cb9",
    "bulk_strings.tsv": "7e7ec9ba895eaf465cb766cddcf29a72",
    "dates_death.tsv": "44af5c5d7bf4c4a6ca8fcdaa5329c9ec",
    "datetime_fields_2.tsv": "0f50afe342c6dca8c9a23ee41df0c8e3",
    "datetime_fields.tsv": "708dff0bf4989c50cad25f7ffd15623b",
    "fo_fields_trimmed.tsv": "ff0689f3629da3cd46097199f59db826",
    "integer_arrays_part1.tsv": "47e4214a945327914d5a82189cf0d560",
    "integer_arrays_part2.tsv": "1630aa738230ea4d5a28cf91f3c66f6d",
    "integer_diet_quest_fields.tsv": "86944f36c7ea72b397e5740f7ee6453a",
    "integer_no_arrays.tsv": "d57178f580c9bd90ba7f33a9c371a894",
    "integer_other_quest_fields.tsv": "e0793fe5cb9e87f28f3af77cbd14e0e6",
    "oaa_fields.tsv": "7701c46303680aa3f06c4e85b2babf35",
    "real_fields1.tsv": "f63dd2423b242f6c00ebee665190265b",
    "real_fields2.tsv": "6a32a2d4341d8abedd31303ec25915e9",
    "string_fields1.tsv": "b327f5edbfc7693b1a53379f1bd899e7",
    "string_fields2.tsv": "3220e8fbdeffd86ebe56356b1d53fbae",
    "tabular.md5": "dc5fafa749070aafcff6f1b87d889836",
}

MEDICAL_FILES = {
    "set3a1.txt": "a355712357f968daa1b47d56a4d6d39b",
    "set3a2a.txt": "3403f6b997fabbb761d08b2bc33b9a9d",
    "set3a3.txt": "0f7a1bb0a20e4576fa4aeebb8394f94b",
    "set3a4.txt": "87446b78921c412b1417e514f91cc7e7",
    "set3b.txt": "9d70c29ad9343528f6e4dd2a07a89bcc",
    "set3c.txt": "fd5a2a07e8b3b93c7b11ec01dbb45536",
    "medrec.md5": "58348146c9e59f8591f90985d92ed125",
}

GENETIC_FILES = {
    "gene_dic.dat": "4693cf9eb7f91317d7daac37e988014f",
    "rand_chr1.dat.gz": "7ca58214640718632dc4a01429513f1d",
    "rand_chr2.dat.gz": "58b4813f57abb674b07c0b7e41dbdab0",
    "rand_chr3.dat.gz": "071e91174af1cb4073fc8f1e8feab1e8",
    "rand_chr4.dat.gz": "657896f8a50a46ed974e744e1b94ffc7",
    "rand_chr5.dat.gz": "b916295d01dc7587ecb2947633a44af8",
    "rand_chr6.dat.gz": "d92673bd9762167bfae4001e058e8375",
    "rand_chr7.dat.gz": "69b0a4c003f21aa14e78d4e5de5fe5a9",
    "rand_chr8.dat.gz": "7bba0d96e922f3145896688067fa2161",
    "rand_chr9.dat.gz": "fba274d9c0ba39d02baa3a9bc568f974",
    "rand_chr10.dat.gz": "4d7b40fe2eb6c826202775ae41722bf5",
    "rand_chr11.dat.gz": "3727eeab271981f3da2896a931b04c31",
    "rand_chr12.dat.gz": "a9b24a033c4934ed43cfeb0d897183e0",
    "rand_chr13.dat.gz": "904fda8a5e2b1ef15c518b8e3beccdbb",
    "rand_chr14.dat.gz": "6d08650c317be1cbb8ff6ac8aed86de7",
    "rand_chr15.dat.gz": "6675d8d3f751db0c0c9d781b547ce17a",
    "rand_chr16.dat.gz": "bedda0edd3cffefa0520057fa3c1a428",
    "rand_chr17.dat.gz": "afe53d77480342b0a393a95ed76c5b32",
    "rand_chr18.dat.gz": "ffb67ae3a56a68344f502eb14b04847c",
    "rand_chr19.dat.gz": "5df0ee5dbe05e1c0f00c88fc99282bf1",
    "rand_chr20.dat.gz": "f40dc53eea55a7de31d85b63202c94a2",
    "rand_chr21.dat.gz": "0ddecef031166409db2c3ff3926c05a7",
    "rand_chr22.dat.gz": "ff5df3309294b36af9996ad32a4776e8",
    "rand_chrmt.dat.gz": "fbce95deada157bce69befd104d92ac7",
    "rand_chrx.dat.gz": "b8e81f5ab174c60f08180145a4b0cf38",
    "rand_chrxy.dat.gz": "712fcee16d4208b60c11fe94e9e01f64",
    "rand_chry.dat.gz": "70e527b49fff8333de13f2c2fe02c6cf",
    "genotype.md5": "3bf65e4d0b943523fa8fcf7a9cb5cafb",
    "rand_chr.md5": "08f8cd13d36b65e701ec4fa56f5a6f29",
}

BULK_FILES = {
    "bulk_20158.zip": "22b0e0395489636e70d68e49c11f02e8",
    "bulk_20203.zip": "19fc94a0c71673fd43de461c087a84c3",
    "bulk_20205.zip": "56f75f157596329e5a8fd13f80486d95",
    "bulk_20206.zip": "4a50584a01788027c2b3d9070c11da15",
    "bulk_20220.zip": "4986c4b3544eb788fa19fe4f0e926c16",
    "bulk_20221.zip": "813add11bf33b8954559961aa04e64e5",
    "bulk_20222.zip": "e131f6265b472892b34d2fde660b0e6d",
    "bulk_20223.zip": "5bbfd22d65626b3e063811f5ce654c19",
    "bulk_20224.zip": "bc8993e333a32458c8f5f67a21df0584",
    "bulk_20225.zip": "59c5fd760a1a9a2de3240c0a3ae3712d",
    "bulk_20226.zip": "4f97913068281f4f44de785a101b3941",
    "bulk_20227.zip": "cd9b71e3f000b5f61b8c99678d0cff6f",
    "bulk_20249.zip": "1ab8ad5e5f8b1782cec09e9abaacf8ca",
    "bulk_20250.zip": "895af470d160a44fa57a5d1683f61356",
    "bulk_20251.zip": "fab0e60572f2d4bbe3a827e045e63efa",
    "bulk_20252.zip": "51663fe25e4b657587683a2ba35ca255",
    "bulk_20253.zip": "4cb1c4d09b420df949f7e51d6eed3a9d",
    "bulk_20254.zip": "b3219e5ae8defa2b000c85e5fb2fecf2",
    "bulk_20259.zip": "eb6a158a98c6edda134b9dd406ac98dc",
    "bulk_20260.zip": "c4fc581515d09d9a7e371b230df602c2",
    "bulk_21017.zip": "46ee0b12f137908bceb1c22c31fd66dd",
    "bulk_22002.zip": "67941f04ec3e061883820f547f124dfb",
    "bulk_23164.zip": "02d2d0aeb29189e656b97268c1c7c5b8",
    "bulk_23184.zip": "7baebecb68a4b05f16bde7d592e8bc64",
    "bulk_25747.zip": "e01f74af130f72b950a1e3e2eae2e3e7",
    "bulk_25748.zip": "6c43c9e2c354388c130ceef57c188f34",
    "bulk_25749.zip": "7e2307bfea721001fcf228c727b51912",
    "bulk_25750.zip": "0a6fed624a693ef78894ce2757918705",
    "bulk_25751.zip": "e389ea73d386d6dbe6e2db4df90b5b3b",
    "bulk_25752.zip": "bd5c23490b6d1b1f630d93c199cab97f",
    "bulk_25753.zip": "90c911d989f040bd94c51ad6898026fd",
    "bulk_25754.zip": "83f1f096a0841488b2d9e0c959d68ae1",
    "bulk_25755.zip": "ace2c18709c985d1a989898e885a3cfd",
    "bulk_6025.zip": "e7c606f980c37273bd824048cd2a529d",
    "bulk_90001.zip": "8ae7f063988737fbf5325d6781408458",
    "bulk_90004.zip": "9868a3be6429e8d9de690ca5fbce3233",
    "bulk.md5": "37f62129f1287949d10f2ba765053b4b",
}

ALL_CATEGORIES = {
    "tabular": TABULAR_FILES,
    "medical": MEDICAL_FILES,
    "genetic": GENETIC_FILES,
    "bulk": BULK_FILES,
}


def get_cache_dir() -> Path:
    """
    Get the default cache directory for UK Biobank Synthetic Dataset.
    
    Returns:
        Path: Path to ~/.cache/synthlab/ukbiobank_synthetic/
    """
    cache_dir = Path.home() / ".cache" / "synthlab" / "ukbiobank_synthetic"
    return cache_dir


def _check_requests():
    """Check if requests is installed."""
    if requests is None:
        raise ImportError(
            "requests is required to download UK Biobank Synthetic Dataset. "
            "Install with: pip install requests"
        )


def _calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file."""
    if not file_path.exists():
        return False
    actual_md5 = _calculate_md5(file_path)
    return actual_md5.lower() == expected_md5.lower()


def _check_aria2c() -> bool:
    """Check if aria2c is available."""
    try:
        result = subprocess.run(
            ["aria2c", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _check_wget() -> bool:
    """Check if wget is available."""
    try:
        result = subprocess.run(
            ["wget", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _download_with_aria2c(
    url: str,
    output_path: Path,
    connections: int = 8,
    quiet: bool = False,
) -> bool:
    """
    Download a file using aria2c with multiple connections.

    aria2c can download different parts of a file simultaneously,
    which dramatically speeds up downloads from slow servers.
    """
    cmd = [
        "aria2c",
        "-x", str(connections),  # Max connections per server
        "-s", str(connections),  # Split file into N parts
        "-k", "1M",              # Min split size
        "-d", str(output_path.parent),
        "-o", output_path.name,
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
    ]
    if quiet:
        cmd.append("--quiet")
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=quiet, text=True)
    return result.returncode == 0


def _download_with_wget(
    url: str,
    output_path: Path,
    quiet: bool = False,
) -> bool:
    """Download a file using wget with resume support."""
    cmd = [
        "wget",
        "-c",  # Continue/resume
        "-O", str(output_path),
    ]
    if quiet:
        cmd.append("-q")
    else:
        cmd.append("--show-progress")
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=quiet, text=True)
    return result.returncode == 0


def list_available_files(category: Optional[str] = None):
    """
    List available files in the UK Biobank Synthetic Dataset.
    
    Args:
        category: Optional category to filter by ('tabular', 'medical', 'genetic', 'bulk')
                 If None, lists all categories.
    
    Returns:
        dict: Dictionary mapping category names to file lists
    """
    if category:
        if category not in ALL_CATEGORIES:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Must be one of: {list(ALL_CATEGORIES.keys())}"
            )
        return {category: list(ALL_CATEGORIES[category].keys())}
    
    return {cat: list(files.keys()) for cat, files in ALL_CATEGORIES.items()}


def download_file(
    filename: str,
    output_dir: Optional[Path] = None,
    category: Optional[str] = None,
    verify_md5: bool = True,
    overwrite: bool = False,
    file_index: Optional[int] = None,
    total_files: Optional[int] = None,
    show_progress: bool = True,
    _quiet: bool = False,
    backend: str = "auto",
    aria2c_connections: int = 8,
) -> Path:
    """
    Download a single file from the UK Biobank Synthetic Dataset.

    Args:
        filename: Name of the file to download
        output_dir: Directory to save the file. If None, uses ~/.cache/synthlab/ukbiobank_synthetic/{category}/
        category: Optional category hint ('tabular', 'medical', 'genetic', 'bulk')
                 If None, searches all categories.
        verify_md5: If True, verify MD5 checksum after download
        overwrite: If True, overwrite existing files
        file_index: Optional file index for progress display (e.g., 5)
        total_files: Optional total number of files for progress display (e.g., 23)
        show_progress: If True and tqdm is installed, show download progress bar
        _quiet: Internal flag for parallel downloads - suppresses print output
        backend: Download backend to use:
            - "auto": Try aria2c, then wget, then requests (default)
            - "aria2c": Use aria2c (fastest, uses multiple connections per file)
            - "wget": Use wget (good resume support)
            - "requests": Use Python requests library
        aria2c_connections: Number of connections per file for aria2c (default: 8)

    Returns:
        Path: Path to downloaded file
    """
    _check_requests()
    
    if output_dir is None:
        # Determine category if not provided
        if category is None:
            for cat, files in ALL_CATEGORIES.items():
                if filename in files:
                    category = cat
                    break
            if category is None:
                raise ValueError(f"File '{filename}' not found in any category. Please specify category.")
        output_dir = get_cache_dir() / category
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the file and its expected MD5
    expected_md5 = None
    if category:
        if category not in ALL_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        if filename not in ALL_CATEGORIES[category]:
            raise ValueError(f"File '{filename}' not found in category '{category}'")
        expected_md5 = ALL_CATEGORIES[category][filename]
    else:
        # Search all categories
        for cat, files in ALL_CATEGORIES.items():
            if filename in files:
                expected_md5 = files[filename]
                category = cat
                break
        
        if expected_md5 is None:
            raise ValueError(f"File '{filename}' not found in any category")
    
    file_path = output_dir / filename
    
    # Check if file already exists and is valid
    if file_path.exists() and not overwrite:
        if verify_md5:
            if _verify_md5(file_path, expected_md5):
                if not _quiet:
                    counter_str = f"[{file_index}/{total_files}] " if file_index is not None and total_files is not None else ""
                    print(f"  {counter_str}✓ {filename} (already exists and verified)")
                return file_path
            else:
                if not _quiet:
                    counter_str = f"[{file_index}/{total_files}] " if file_index is not None and total_files is not None else ""
                    print(f"  {counter_str}⚠ {filename} (exists but MD5 mismatch, re-downloading)")
        else:
            if not _quiet:
                counter_str = f"[{file_index}/{total_files}] " if file_index is not None and total_files is not None else ""
                print(f"  {counter_str}✓ {filename} (already exists)")
            return file_path
    
    # Download the file
    # Files are organized in subdirectories by category
    # Map category names to their subdirectory names on the server
    category_subdirs = {
        "tabular": "tabular",
        "medical": "medrec",
        "genetic": "genotype",  # Note: server uses "genotype" not "genetic"
        "bulk": "bulk",
    }
    
    if category and category in category_subdirs:
        subdir = category_subdirs[category]
        url = urljoin(BASE_URL, f"{subdir}/{filename}")
    else:
        # Fallback: try without subdirectory (for backwards compatibility)
        url = urljoin(BASE_URL, filename)
    
    # Display progress counter if provided
    counter_str = f"[{file_index}/{total_files}] " if file_index is not None and total_files is not None else ""

    # Determine which backend to use
    use_backend = backend
    if backend == "auto":
        if _check_aria2c():
            use_backend = "aria2c"
        elif _check_wget():
            use_backend = "wget"
        else:
            use_backend = "requests"

    download_success = False

    if use_backend == "aria2c":
        if not _quiet:
            print(f"  {counter_str}↓ {filename} (aria2c, {aria2c_connections} connections)...")
        download_success = _download_with_aria2c(url, file_path, aria2c_connections, quiet=_quiet)
        if not download_success:
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(f"aria2c failed to download {filename}")

    elif use_backend == "wget":
        if not _quiet:
            print(f"  {counter_str}↓ {filename} (wget)...")
        download_success = _download_with_wget(url, file_path, quiet=_quiet)
        if not download_success:
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(f"wget failed to download {filename}")

    else:  # requests
        _check_requests()
        try:
            response = requests.get(url, stream=True, timeout=(10, 300))  # type: ignore[union-attr]
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Use tqdm for progress bar if available and not in quiet mode
            use_progress_bar = show_progress and HAS_TQDM and total_size > 0 and not _quiet

            pbar = None
            if use_progress_bar:
                pbar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"{counter_str}{filename}",
                    leave=True,
                )
            elif not _quiet:
                size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                print(f"  {counter_str}↓ {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))

            if pbar is not None:
                pbar.close()
            download_success = True

        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(f"Failed to download {filename}: {e}") from e

    # Verify MD5 if requested
    if verify_md5:
        if _verify_md5(file_path, expected_md5):
            if not _quiet and use_backend == "requests":
                print("✓ (verified)")
            elif not _quiet:
                print(f"  {counter_str}✓ {filename} (verified)")
        else:
            actual_md5 = _calculate_md5(file_path)
            file_path.unlink()
            if not _quiet:
                print(f"  {counter_str}✗ {filename} (MD5 mismatch!)")
            raise ValueError(
                f"MD5 checksum mismatch for {filename}. "
                f"Expected: {expected_md5}, Got: {actual_md5}"
            )
    else:
        if not _quiet and use_backend == "requests":
            print("✓")

    return file_path


def download_category(
    category: str,
    output_dir: Optional[Path] = None,
    verify_md5: bool = True,
    overwrite: bool = False,
    skip_md5_files: bool = True,
    max_workers: int = 4,
    backend: str = "auto",
    aria2c_connections: int = 8,
) -> Path:
    """
    Download all files in a category.

    Args:
        category: Category to download ('tabular', 'medical', 'genetic', 'bulk')
        output_dir: Directory to save files. If None, uses ~/.cache/synthlab/ukbiobank_synthetic/{category}/
        verify_md5: If True, verify MD5 checksums after download
        overwrite: If True, overwrite existing files
        skip_md5_files: If True, skip downloading MD5 checksum files
        max_workers: Number of parallel download threads (default: 4). Set to 1 for sequential.
        backend: Download backend - "auto", "aria2c", "wget", or "requests"
        aria2c_connections: Number of connections per file for aria2c (default: 8)

    Returns:
        Path: Path to output directory
    """
    if category not in ALL_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Must be one of: {list(ALL_CATEGORIES.keys())}"
        )

    if output_dir is None:
        output_dir = get_cache_dir() / category
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = ALL_CATEGORIES[category]
    file_list = [f for f in files.keys() if not (skip_md5_files and f.endswith(".md5"))]
    total_files = len(file_list)

    # Determine actual backend that will be used
    actual_backend = backend
    if backend == "auto":
        if _check_aria2c():
            actual_backend = "aria2c"
        elif _check_wget():
            actual_backend = "wget"
        else:
            actual_backend = "requests"

    print(f"\nDownloading {category} category ({total_files} files)")
    print(f"Output directory: {output_dir}")
    print(f"Download backend: {actual_backend}" + (f" ({aria2c_connections} connections/file)" if actual_backend == "aria2c" else ""))
    if max_workers > 1 and actual_backend == "requests":
        print(f"Parallel downloads: {max_workers} threads")
    print("=" * 60)

    if max_workers == 1:
        # Sequential download (original behavior)
        for idx, filename in enumerate(file_list, start=1):
            try:
                download_file(
                    filename,
                    output_dir,
                    category=category,
                    verify_md5=verify_md5,
                    overwrite=overwrite,
                    file_index=idx,
                    total_files=total_files,
                    backend=backend,
                    aria2c_connections=aria2c_connections,
                )
            except Exception as e:
                counter_str = f"[{idx}/{total_files}] " if total_files > 0 else ""
                print(f"  {counter_str}✗ Failed to download {filename}: {e}")
                raise
    else:
        # Parallel downloads
        completed = 0
        failed: list[tuple[str, str]] = []

        def _download_one(filename: str) -> tuple[str, bool, str]:
            """Download a single file, return (filename, success, error_msg)."""
            try:
                download_file(
                    filename,
                    output_dir,
                    category=category,
                    verify_md5=verify_md5,
                    overwrite=overwrite,
                    show_progress=False,  # Disable per-file progress in parallel mode
                    _quiet=True,
                    backend=backend,
                    aria2c_connections=aria2c_connections,
                )
                return (filename, True, "")
            except Exception as e:
                return (filename, False, str(e))

        # Use tqdm for overall progress if available
        if HAS_TQDM:
            with tqdm(total=total_files, desc="Downloading", unit="file") as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_download_one, f): f for f in file_list}
                    for future in as_completed(futures):
                        filename, success, error_msg = future.result()
                        if success:
                            completed += 1
                        else:
                            failed.append((filename, error_msg))
                        pbar.update(1)
                        pbar.set_postfix({"done": completed, "failed": len(failed)})
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download_one, f): f for f in file_list}
                for future in as_completed(futures):
                    filename, success, error_msg = future.result()
                    if success:
                        completed += 1
                        print(f"  [{completed}/{total_files}] ✓ {filename}")
                    else:
                        failed.append((filename, error_msg))
                        print(f"  ✗ {filename}: {error_msg}")

        if failed:
            print(f"\n⚠ {len(failed)} files failed to download:")
            for filename, error_msg in failed:
                print(f"    - {filename}: {error_msg}")
            raise RuntimeError(f"Failed to download {len(failed)} files")

    print(f"\n✓ Downloaded {len(file_list)} files to {output_dir}")
    return output_dir


def load_tabular_data(
    data_dir: Optional[Path] = None,
    file_pattern: Optional[str] = None,
    sample_rows: Optional[int] = None,
) -> dict:
    """
    Load tabular phenotype data from TSV files.
    
    Args:
        data_dir: Directory containing tabular TSV files. If None, uses ~/.cache/synthlab/ukbiobank_synthetic/tabular/
        file_pattern: Optional pattern to filter files (e.g., "integer_*.tsv")
        sample_rows: Optional number of rows to sample (for testing)
    
    Returns:
        dict: Dictionary mapping filename (without extension) to DataFrame
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required to load tabular data. Install with: pip install polars"
        )
    
    if data_dir is None:
        data_dir = get_cache_dir() / "tabular"
    else:
        data_dir = Path(data_dir)
    tsv_files = list(data_dir.glob("*.tsv"))
    
    if file_pattern:
        tsv_files = [f for f in tsv_files if file_pattern in f.name]
    
    # Exclude MD5 files
    tsv_files = [f for f in tsv_files if not f.name.endswith(".md5")]
    
    if not tsv_files:
        raise ValueError(f"No TSV files found in {data_dir}")
    
    data = {}
    for tsv_file in tsv_files:
        print(f"  Loading {tsv_file.name}...", end=" ", flush=True)
        try:
            df = pl.read_csv(
                tsv_file,
                separator="\t",
                infer_schema_length=10000,
                n_rows=sample_rows,
            )
            key = tsv_file.stem
            data[key] = df
            print(f"✓ ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    return data


def load_medical_records(
    data_dir: Optional[Path] = None,
    file_pattern: Optional[str] = None,
    sample_rows: Optional[int] = None,
):
    """
    Load medical records (GP clinical data).
    
    Columns: EID, data_provider, event_date, read2, read3, value1, value2, value3
    
    Args:
        data_dir: Directory containing medical record text files. If None, uses ~/.cache/synthlab/ukbiobank_synthetic/medical/
        file_pattern: Optional pattern to filter files (e.g., "set3a*.txt")
        sample_rows: Optional number of rows to sample (for testing)
    
    Returns:
        DataFrame: Combined medical records
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required to load medical records. Install with: pip install polars"
        )
    
    if data_dir is None:
        data_dir = get_cache_dir() / "medical"
    else:
        data_dir = Path(data_dir)
    txt_files = list(data_dir.glob("*.txt"))
    
    if file_pattern:
        txt_files = [f for f in txt_files if file_pattern in f.name]
    
    # Exclude MD5 files
    txt_files = [f for f in txt_files if not f.name.endswith(".md5")]
    
    if not txt_files:
        raise ValueError(f"No medical record files found in {data_dir}")
    
    dfs = []
    for txt_file in txt_files:
        print(f"  Loading {txt_file.name}...", end=" ", flush=True)
        try:
            df = pl.read_csv(
                txt_file,
                separator="\t",
                has_header=False,
                new_columns=["EID", "data_provider", "event_date", "read2", "read3", "value1", "value2", "value3"],
                n_rows=sample_rows,
            )
            dfs.append(df)
            print(f"✓ ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    if dfs:
        combined = pl.concat(dfs)
        print(f"\n✓ Combined {len(combined)} total medical records")
        return combined
    else:
        return pl.DataFrame()


def load_genetic_dictionary(data_dir: Optional[Path] = None):
    """
    Load the genetic dictionary file.
    
    Args:
        data_dir: Directory containing genetic data files. If None, uses ~/.cache/synthlab/ukbiobank_synthetic/genetic/
    
    Returns:
        DataFrame: Dictionary with columns: affy_id, chromosome, index, variant0, variant1, variant2, variant3
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required to load genetic data. Install with: pip install polars"
        )
    
    if data_dir is None:
        data_dir = get_cache_dir() / "genetic"
    else:
        data_dir = Path(data_dir)
    dict_file = data_dir / "gene_dic.dat"
    
    if not dict_file.exists():
        raise FileNotFoundError(f"Genetic dictionary not found: {dict_file}")
    
    print(f"  Loading {dict_file.name}...", end=" ", flush=True)
    df = pl.read_csv(
        dict_file,
        separator="\t",
        has_header=False,
        new_columns=["affy_id", "chromosome", "index", "variant0", "variant1", "variant2", "variant3"],
    )
    print(f"✓ ({len(df)} SNPs)")
    return df


if __name__ == "__main__":
    import sys
    
    print("UK Biobank Synthetic Dataset Downloader")
    print("=" * 60)
    print(f"Dataset: {BASE_URL}")
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python download_ukbiobank_synthetic.py <category> [output_dir]")
        print("\nCategories:")
        print("  tabular  - Phenotype data (23 TSV files, ~27K columns × 600K rows)")
        print("  medical  - GP clinical records (6 text files, ~400M rows)")
        print("  genetic  - SNP genotype data (dictionary + 26 chromosome files)")
        print("  bulk     - Bulk file repository (~6M files in zip archives)")
        print("\nExample:")
        print("  python download_ukbiobank_synthetic.py tabular")
        print("  # Downloads to ~/.cache/synthlab/ukbiobank_synthetic/tabular/")
        print("\n  python download_ukbiobank_synthetic.py tabular /custom/path")
        print("  # Downloads to custom path")
        sys.exit(1)
    
    category = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = None  # Use default cache directory
    
    try:
        download_category(category, output_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
