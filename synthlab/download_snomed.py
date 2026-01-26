#!/usr/bin/env python3
"""
Download SNOMED CT vocabulary data for SynthLab.

The CONCEPT.csv file is hosted on GitHub releases and downloaded on first use.
For private repos, authentication is handled via:
1. GitHub CLI (gh) if available and authenticated
2. GITHUB_TOKEN environment variable
3. GH_TOKEN environment variable
"""

import gzip
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

# GitHub repository and release info
GITHUB_REPO = "bschilder/synthlab"
GITHUB_RELEASE_TAG = "vocab-v1"
GITHUB_ASSET_NAME = "CONCEPT.csv.gz"

# Direct URL (works for public repos)
SNOMED_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{GITHUB_ASSET_NAME}"

# Default cache location
DEFAULT_SNOMED_DATA_DIR = Path.home() / ".cache" / "synthlab" / "snomed_data"


def get_snomed_data_dir() -> Path:
    """Get the SNOMED data directory, creating if needed."""
    DEFAULT_SNOMED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_SNOMED_DATA_DIR


def get_concept_csv_path() -> Path:
    """Get the path to CONCEPT.csv, downloading if needed."""
    data_dir = get_snomed_data_dir()
    concept_path = data_dir / "CONCEPT.csv"

    if not concept_path.exists():
        download_snomed_vocabulary(verbose=True)

    return concept_path


def _get_github_token() -> str | None:
    """Get GitHub token from environment variables."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _find_gh_cli() -> str | None:
    """Find the gh CLI executable path."""
    import sys

    # Check common locations
    candidates = [
        "gh",  # In PATH
        Path(sys.executable).parent / "gh",  # Same dir as Python (conda env)
    ]

    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return str(candidate)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue

    return None


def _gh_cli_available() -> bool:
    """Check if GitHub CLI is available and authenticated."""
    gh_path = _find_gh_cli()
    if not gh_path:
        return False

    try:
        result = subprocess.run(
            [gh_path, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _download_with_gh_cli(output_path: Path, verbose: bool = True) -> bool:
    """
    Download release asset using GitHub CLI.

    Returns True if successful, False otherwise.
    """
    gh_path = _find_gh_cli()
    if not gh_path:
        return False

    try:
        if verbose:
            print("  Using GitHub CLI for authenticated download...")

        result = subprocess.run(
            [
                gh_path, "release", "download", GITHUB_RELEASE_TAG,
                "--repo", GITHUB_REPO,
                "--pattern", GITHUB_ASSET_NAME,
                "--dir", str(output_path.parent),
                "--clobber",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large files
        )

        if result.returncode == 0:
            return True

        if verbose:
            print(f"  gh CLI failed: {result.stderr.strip()}")
        return False

    except (subprocess.SubprocessError, FileNotFoundError) as e:
        if verbose:
            print(f"  gh CLI error: {e}")
        return False


def _download_with_token(url: str, output_path: Path, token: str, verbose: bool = True) -> bool:
    """
    Download using GitHub token authentication.

    Returns True if successful, False otherwise.
    """
    try:
        if verbose:
            print("  Using token authentication...")

        # For GitHub release assets, we need to use the API
        # First get the asset URL, then download with token
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{GITHUB_RELEASE_TAG}"

        request = urllib.request.Request(api_url)
        request.add_header("Authorization", f"token {token}")
        request.add_header("Accept", "application/vnd.github.v3+json")

        with urllib.request.urlopen(request, timeout=30) as response:
            import json
            release_data = json.loads(response.read().decode())

        # Find the asset
        asset_url = None
        for asset in release_data.get("assets", []):
            if asset["name"] == GITHUB_ASSET_NAME:
                asset_url = asset["url"]
                break

        if not asset_url:
            if verbose:
                print(f"  Asset {GITHUB_ASSET_NAME} not found in release")
            return False

        # Download the asset
        request = urllib.request.Request(asset_url)
        request.add_header("Authorization", f"token {token}")
        request.add_header("Accept", "application/octet-stream")

        with urllib.request.urlopen(request, timeout=300) as response:
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response, f)

        return True

    except Exception as e:
        if verbose:
            print(f"  Token auth failed: {e}")
        return False


def _download_public(url: str, output_path: Path, verbose: bool = True) -> bool:
    """
    Download from public URL (no authentication).

    Returns True if successful, False otherwise.
    """
    try:
        if verbose:
            print("  Attempting public download...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except urllib.error.HTTPError as e:
        if verbose:
            print(f"  Public download failed: HTTP {e.code}")
        return False
    except Exception as e:
        if verbose:
            print(f"  Public download failed: {e}")
        return False


def download_snomed_vocabulary(
    output_dir: Path | str | None = None,
    url: str = SNOMED_RELEASE_URL,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """
    Download SNOMED CT CONCEPT.csv from GitHub releases.

    For private repositories, authentication is handled automatically via:
    1. GitHub CLI (gh) if available and authenticated
    2. GITHUB_TOKEN or GH_TOKEN environment variable

    Args:
        output_dir: Directory to save CONCEPT.csv. Defaults to ~/.cache/synthlab/snomed_data/
        url: URL to download from (defaults to GitHub releases)
        verbose: Print progress messages
        force: Re-download even if file exists

    Returns:
        Path to the downloaded CONCEPT.csv file
    """
    if output_dir is None:
        output_dir = get_snomed_data_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    concept_path = output_dir / "CONCEPT.csv"
    compressed_path = output_dir / "CONCEPT.csv.gz"

    # Check if already downloaded
    if concept_path.exists():
        if verbose:
            print(f"SNOMED CONCEPT.csv already exists at {concept_path}")
            if force:
                print("Skipping download (delete the file to re-download).")
        return concept_path

    if verbose:
        print(f"Downloading SNOMED vocabulary from GitHub releases...")
        print(f"  Repository: {GITHUB_REPO}")
        print(f"  Release: {GITHUB_RELEASE_TAG}")
        print(f"  Destination: {concept_path}")

    try:
        # Try download methods in order of preference
        download_success = False

        # 1. Try GitHub CLI (best for private repos)
        if _gh_cli_available():
            download_success = _download_with_gh_cli(compressed_path, verbose)

        # 2. Try token authentication
        if not download_success:
            token = _get_github_token()
            if token:
                download_success = _download_with_token(url, compressed_path, token, verbose)

        # 3. Try public download (works if repo is public)
        if not download_success:
            download_success = _download_public(url, compressed_path, verbose)

        if not download_success:
            raise RuntimeError(
                "Failed to download SNOMED vocabulary.\n\n"
                "For private repositories, ensure one of:\n"
                "  1. GitHub CLI is installed and authenticated: gh auth login\n"
                "  2. GITHUB_TOKEN environment variable is set\n"
                "  3. GH_TOKEN environment variable is set\n\n"
                f"Repository: {GITHUB_REPO}\n"
                f"Release: {GITHUB_RELEASE_TAG}"
            )

        if verbose:
            size_mb = compressed_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded ({size_mb:.1f} MB)")

        # Decompress
        if verbose:
            print("  Decompressing...", end=" ", flush=True)

        with gzip.open(compressed_path, 'rb') as f_in:
            with open(concept_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if verbose:
            size_mb = concept_path.stat().st_size / (1024 * 1024)
            print(f"done ({size_mb:.1f} MB)")

        # Remove compressed file
        compressed_path.unlink()

        if verbose:
            print(f"  SNOMED CONCEPT.csv saved to {concept_path}")

        return concept_path

    except Exception as e:
        # Clean up partial downloads
        if compressed_path.exists():
            compressed_path.unlink()
        if concept_path.exists():
            concept_path.unlink()
        raise


def is_snomed_available() -> bool:
    """Check if SNOMED CONCEPT.csv is available locally."""
    concept_path = get_snomed_data_dir() / "CONCEPT.csv"
    return concept_path.exists()


if __name__ == "__main__":
    print("SNOMED Vocabulary Downloader")
    print("=" * 60)
    download_snomed_vocabulary(verbose=True)
