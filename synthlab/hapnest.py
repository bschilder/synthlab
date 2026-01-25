#!/usr/bin/env python3
"""
HAPNEST: Realistic Synthetic Genotype Generation and Download.

HAPNEST (Haplotype-based Next-generation Synthetic genotypes) generates
realistic synthetic genotype data with:
- Linkage disequilibrium (LD) structure
- Population stratification across 6 ancestry groups
- Configurable sample sizes and chromosomes

This module provides:
- **Download pre-generated data**: 1M+ individuals, 6.8M variants
- **Generate new data**: Using HAPNEST via Singularity/Docker container

References:
    Paper: https://academic.oup.com/bioinformatics/article/39/9/btad535/7255913
    Data: https://www.ebi.ac.uk/biostudies/studies/S-BSST936
    Code: https://github.com/intervene-EU-H2020/synthetic_data

Example:
    Download pre-generated data::

        import synthlab as sl
        data_dir = sl.download_hapnest()

    Generate new data (requires Singularity/Docker)::

        from synthlab import HAPNESTRunner, HAPNESTConfig

        runner = HAPNESTRunner()
        config = HAPNESTConfig(n_samples=1000, chromosome=22, seed=42)
        result = runner.run(config)
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, List, Any
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


# =============================================================================
# Constants and Metadata
# =============================================================================

# URLs
HAPNEST_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/intervene/"
HAPNEST_BIOSTUDIES_URL = "https://www.ebi.ac.uk/biostudies/studies/S-BSST936"
HAPNEST_PAPER_URL = "https://academic.oup.com/bioinformatics/article/39/9/btad535/7255913"
HAPNEST_GITHUB_URL = "https://github.com/intervene-EU-H2020/synthetic_data"
HAPNEST_DOCKER_IMAGE = "sophiewharrie/intervene-synthetic-data"

# Pre-generated dataset files (available at EBI FTP)
HAPNEST_FILES = {
    "hapnest.pgen": "Genotype data in PLINK 2 format",
    "hapnest.pvar": "Variant information",
    "hapnest.psam": "Sample information",
}

# Ancestry groups
HAPNEST_ANCESTRIES = {
    "AFR": "African",
    "AMR": "Admixed American",
    "CSA": "Central/South Asian",
    "EAS": "East Asian",
    "EUR": "European",
    "MID": "Middle Eastern",
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_hapnest_cache_dir() -> Path:
    """
    Get the default cache directory for HAPNEST data.

    Returns:
        Path: Path to ~/.cache/synthlab/hapnest/
    """
    cache_dir = Path.home() / ".cache" / "synthlab" / "hapnest"
    return cache_dir


def _check_requests():
    """Check if requests is installed."""
    if requests is None:
        raise ImportError(
            "requests is required to download HAPNEST data. "
            "Install with: pip install requests"
        )


def _check_plink2() -> bool:
    """Check if PLINK 2 is available."""
    try:
        result = subprocess.run(
            ["plink2", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# =============================================================================
# Download Pre-generated Data
# =============================================================================

def list_hapnest_files() -> dict:
    """
    List available HAPNEST dataset files and information.

    Returns:
        dict: Dictionary with file categories, descriptions, and metadata

    Example:
        >>> info = sl.list_hapnest_files()
        >>> print(info['ancestries'])
        {'AFR': 'African', 'EUR': 'European', ...}
    """
    return {
        "dataset": {
            "description": "HAPNEST synthetic genotype dataset with realistic LD",
            "files": HAPNEST_FILES,
            "format": "PLINK 2 (.pgen, .pvar, .psam)",
            "url": HAPNEST_BASE_URL,
            "download": "sl.download_hapnest()",
        },
        "full_dataset_info": {
            "description": "Full HAPNEST dataset (1,008,000 individuals, 6.8M variants)",
            "individuals": "1,008,000",
            "variants": "6,800,000",
            "format": "PLINK 1/2 format",
            "url": HAPNEST_BIOSTUDIES_URL,
            "note": "Very large (TB-scale). Download from BioStudies directly.",
        },
        "ancestries": HAPNEST_ANCESTRIES,
        "references": {
            "paper": HAPNEST_PAPER_URL,
            "data": HAPNEST_BIOSTUDIES_URL,
            "code": HAPNEST_GITHUB_URL,
        },
    }


def download_hapnest(
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Download the pre-generated HAPNEST synthetic genotype dataset.

    Downloads realistic synthetic genotype data with linkage disequilibrium
    structure and population stratification. Data is in PLINK 2 format
    (.pgen, .pvar, .psam).

    This downloads PRE-GENERATED data from EBI. To generate NEW data with
    custom parameters, use HAPNESTRunner instead.

    Args:
        output_dir: Directory to save files. Defaults to ~/.cache/synthlab/hapnest/
        overwrite: If True, overwrite existing files
        show_progress: If True, show download progress bar

    Returns:
        Path: Path to output directory containing the dataset

    Example:
        >>> data_dir = sl.download_hapnest()
        >>> # Load with PLINK 2:
        >>> # plink2 --pfile ~/.cache/synthlab/hapnest/hapnest --freq
    """
    _check_requests()

    if output_dir is None:
        output_dir = get_hapnest_cache_dir()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading HAPNEST dataset (pre-generated)")
    print(f"Source: {HAPNEST_BIOSTUDIES_URL}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for filename, description in HAPNEST_FILES.items():
        file_path = output_dir / filename

        # Skip if exists and not overwriting
        if file_path.exists() and not overwrite:
            print(f"  ✓ {filename} (already exists)")
            continue

        url = urljoin(HAPNEST_BASE_URL, filename)

        try:
            response = requests.get(url, stream=True, timeout=(10, 300))  # type: ignore
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            if show_progress and HAS_TQDM and total_size > 0:
                pbar = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                )
            else:
                pbar = None
                size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                print(f"  ↓ {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))

            if pbar:
                pbar.close()
            else:
                print("✓")

        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(f"Failed to download {filename}: {e}") from e

    print(f"\n✓ Downloaded HAPNEST dataset to {output_dir}")
    print(f"\nDataset contains:")
    print(f"  • Realistic synthetic genotypes with LD structure")
    print(f"  • 6 ancestry groups: {', '.join(HAPNEST_ANCESTRIES.keys())}")
    print(f"  • PLINK 2 format (.pgen, .pvar, .psam)")
    print(f"\nTo load with PLINK 2:")
    print(f"  plink2 --pfile {output_dir / 'hapnest'} --freq")

    return output_dir


# Backward-compatible alias
download_hapnest_small = download_hapnest


def load_hapnest_variants(
    data_dir: Optional[Path] = None,
) -> "pl.DataFrame":  # type: ignore
    """
    Load HAPNEST variant information (.pvar file).

    Args:
        data_dir: Directory containing HAPNEST files.
                  Defaults to ~/.cache/synthlab/hapnest/

    Returns:
        polars.DataFrame: Variant information with columns:
            - CHROM: Chromosome
            - POS: Position
            - ID: Variant ID (rsID)
            - REF: Reference allele
            - ALT: Alternate allele
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required to load HAPNEST data. Install with: pip install polars"
        )

    if data_dir is None:
        data_dir = get_hapnest_cache_dir()
    else:
        data_dir = Path(data_dir)

    pvar_file = data_dir / "hapnest.pvar"

    if not pvar_file.exists():
        raise FileNotFoundError(
            f"HAPNEST variant file not found: {pvar_file}\n"
            f"Download first with: sl.download_hapnest()"
        )

    print(f"Loading variants from {pvar_file}...")

    # PVAR format: #CHROM POS ID REF ALT [QUAL FILTER INFO ...]
    df = pl.read_csv(
        pvar_file,
        separator="\t",
        comment_prefix="##",  # Skip header comments
        has_header=True,
    )

    # Rename #CHROM to CHROM if present
    if "#CHROM" in df.columns:
        df = df.rename({"#CHROM": "CHROM"})

    print(f"✓ Loaded {len(df):,} variants")
    return df


def load_hapnest_samples(
    data_dir: Optional[Path] = None,
) -> "pl.DataFrame":  # type: ignore
    """
    Load HAPNEST sample information (.psam file).

    Args:
        data_dir: Directory containing HAPNEST files.
                  Defaults to ~/.cache/synthlab/hapnest/

    Returns:
        polars.DataFrame: Sample information with columns:
            - IID: Individual ID
            - SEX: Sex (1=male, 2=female, 0=unknown)
            - Additional phenotype columns if present
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required to load HAPNEST data. Install with: pip install polars"
        )

    if data_dir is None:
        data_dir = get_hapnest_cache_dir()
    else:
        data_dir = Path(data_dir)

    psam_file = data_dir / "hapnest.psam"

    if not psam_file.exists():
        raise FileNotFoundError(
            f"HAPNEST sample file not found: {psam_file}\n"
            f"Download first with: sl.download_hapnest()"
        )

    print(f"Loading samples from {psam_file}...")

    # PSAM format: #IID [SID] SEX [phenotypes...]
    df = pl.read_csv(
        psam_file,
        separator="\t",
        comment_prefix="##",
        has_header=True,
    )

    # Rename #IID to IID if present
    if "#IID" in df.columns:
        df = df.rename({"#IID": "IID"})

    print(f"✓ Loaded {len(df):,} samples")
    return df


# =============================================================================
# HAPNEST Generation (via Container)
# =============================================================================

@dataclass
class HAPNESTConfig:
    """
    Configuration for HAPNEST synthetic genotype generation.

    HAPNEST generates realistic synthetic genotypes with linkage disequilibrium
    structure and population stratification using a Singularity/Docker container.

    Attributes:
        n_samples: Number of synthetic individuals to generate
        chromosome: Chromosome to simulate (1-22, or "all")
        superpopulation: Ancestry group (AFR, AMR, CSA, EAS, EUR, MID, or "none" for all)
        output_dir: Directory for output files
        output_prefix: Prefix for output filenames
        seed: Random seed for reproducibility
        memory_mb: Available memory in MB
        batch_size: Samples per write batch

    Example:
        >>> config = HAPNESTConfig(
        ...     n_samples=1000,
        ...     chromosome=22,
        ...     superpopulation="EUR",
        ...     seed=42,
        ... )
    """

    n_samples: int = 1000
    chromosome: int | str = 22  # 1-22 or "all"
    superpopulation: str = "EUR"  # AFR, AMR, CSA, EAS, EUR, MID, or "none"
    output_dir: Optional[Path] = None
    output_prefix: str = "hapnest_synthetic"
    seed: int = 42
    memory_mb: int = 8000
    batch_size: int = 100

    # Advanced phenotype settings (optional)
    n_traits: int = 0  # 0 = no phenotypes
    heritability: float = 0.5
    polygenicity: float = 0.01
    prevalence: Optional[float] = None  # If set, generates binary phenotype

    # Custom population structure (advanced)
    custom_populations: Optional[List[Dict[str, Any]]] = None

    def to_yaml(self) -> str:
        """Convert config to HAPNEST YAML format."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to generate HAPNEST config. "
                "Install with: pip install pyyaml"
            )

        config: dict = {
            "global_parameters": {
                "random_seed": self.seed,
                "chromosome": self.chromosome if self.chromosome != "all" else "all",
                "superpopulation": self.superpopulation,
                "memory": self.memory_mb,
                "batchsize": self.batch_size,
            },
            "output": {
                "output_dir": str(self.output_dir or "/data/outputs"),
                "output_prefix": self.output_prefix,
            },
            "genotype_data": {
                "nsamples": self.n_samples,
                "samples": {"default": True},
            },
        }

        # Add custom population structure if specified
        if self.custom_populations:
            config["genotype_data"]["samples"] = {
                "default": False,
                "custom": self.custom_populations,
            }

        # Add phenotype settings if requested
        if self.n_traits > 0:
            config["phenotype_data"] = {
                "nTrait": self.n_traits,
                "nPopulation": 1,
                "Polygenicity": [[self.polygenicity]],
                "ProportionGeno": [[self.heritability]],
            }
            if self.prevalence is not None:
                config["phenotype_data"]["Prevalence"] = [[self.prevalence]]

        return yaml.dump(config, default_flow_style=False)


class HAPNESTRunner:
    """
    Run HAPNEST to generate realistic synthetic genotype data.

    HAPNEST creates synthetic genotypes with realistic linkage disequilibrium (LD)
    structure and population stratification. It requires either Singularity or
    Docker to be installed.

    The generated data includes:
    - Genotype files in PLINK format (.bed, .bim, .fam)
    - Optional phenotype files
    - Quality metrics and validation results

    References:
        Paper: https://academic.oup.com/bioinformatics/article/39/9/btad535/7255913
        Code: https://github.com/intervene-EU-H2020/synthetic_data
        Data: https://www.ebi.ac.uk/biostudies/studies/S-BSST936

    Example:
        >>> from synthlab import HAPNESTRunner, HAPNESTConfig
        >>> runner = HAPNESTRunner()
        >>> config = HAPNESTConfig(n_samples=1000, chromosome=22, seed=42)
        >>> result = runner.run(config)
        >>> print(f"Output: {result['output_dir']}")
    """

    def __init__(
        self,
        container_path: Optional[Path] = None,
        container_type: Literal["singularity", "docker", "auto"] = "auto",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the HAPNEST runner.

        Args:
            container_path: Path to Singularity .sif file (optional, will download if needed)
            container_type: "singularity", "docker", or "auto" to detect
            cache_dir: Directory for caching container and reference data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_hapnest_cache_dir() / "runner"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect container runtime
        if container_type == "auto":
            self.container_type = self._detect_container_runtime()
        else:
            self.container_type = container_type

        # Set container path
        if container_path:
            self.container_path = Path(container_path)
        else:
            self.container_path = self.cache_dir / "intervene-synthetic-data_latest.sif"

        self._initialized = False

    def _detect_container_runtime(self) -> str:
        """Detect available container runtime."""
        # Check for Singularity first (preferred for HPC)
        try:
            result = subprocess.run(
                ["singularity", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return "singularity"
        except FileNotFoundError:
            pass

        # Check for Docker
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return "docker"
        except FileNotFoundError:
            pass

        raise RuntimeError(
            "No container runtime found. HAPNEST requires Singularity or Docker.\n"
            "Install Singularity: https://sylabs.io/guides/3.0/user-guide/installation.html\n"
            "Or Docker: https://docs.docker.com/get-docker/"
        )

    def _pull_container(self) -> None:
        """Pull the HAPNEST container image."""
        if self.container_path.exists():
            print(f"Using existing container: {self.container_path}")
            return

        print(f"Pulling HAPNEST container ({self.container_type})...")
        print(f"Image: {HAPNEST_DOCKER_IMAGE}")

        if self.container_type == "singularity":
            cmd = [
                "singularity", "pull",
                str(self.container_path),
                f"docker://{HAPNEST_DOCKER_IMAGE}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull container: {result.stderr}")
            print(f"Container saved to: {self.container_path}")
        else:
            # Docker pulls on demand, just verify image is accessible
            cmd = ["docker", "pull", HAPNEST_DOCKER_IMAGE]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull Docker image: {result.stderr}")
            print("Docker image pulled successfully")

    def _run_container_cmd(
        self,
        command: str,
        data_dir: Path,
        verbose: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command inside the container."""
        if self.container_type == "singularity":
            cmd = [
                "singularity", "exec",
                "--bind", f"{data_dir}:/data/",
                str(self.container_path),
                "bash", "-c", command,
            ]
        else:
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{data_dir}:/data/",
                HAPNEST_DOCKER_IMAGE,
                "bash", "-c", command,
            ]

        if verbose:
            print(f"$ {command}")

        return subprocess.run(cmd, capture_output=True, text=True)

    def initialize(self, force: bool = False) -> None:
        """
        Initialize HAPNEST (pull container, fetch reference data).

        This downloads the container image and reference genetic data needed
        for generation. Only needs to be run once.

        Args:
            force: If True, re-download even if already initialized
        """
        if self._initialized and not force:
            print("HAPNEST already initialized")
            return

        print("Initializing HAPNEST...")
        print("=" * 60)

        # Pull container
        self._pull_container()

        # Create data directory structure
        data_dir = self.cache_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Initialize dependencies
        print("\nInitializing dependencies...")
        result = self._run_container_cmd("init", data_dir)
        if result.returncode != 0:
            print(f"Warning: init returned non-zero: {result.stderr}")

        # Fetch reference data
        print("\nFetching reference data (this may take a while)...")
        result = self._run_container_cmd("fetch", data_dir)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to fetch reference data: {result.stderr}")

        self._initialized = True
        print("\n✓ HAPNEST initialized successfully")

    def run(
        self,
        config: HAPNESTConfig,
        n_threads: int = 1,
        generate_phenotypes: bool = False,
        validate: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Generate synthetic genotype data using HAPNEST.

        Args:
            config: HAPNESTConfig with generation parameters
            n_threads: Number of threads for parallel processing
            generate_phenotypes: If True, also generate phenotypes
            validate: If True, run validation metrics after generation
            verbose: Print progress information

        Returns:
            dict: Result containing:
                - returncode: 0 on success
                - output_dir: Path to output directory
                - output_files: List of generated files
                - config: The config used
        """
        # Ensure initialized
        if not self._initialized:
            self.initialize()

        # Set up output directory
        if config.output_dir is None:
            config.output_dir = self.cache_dir / "outputs" / f"run_{config.seed}"
        config.output_dir = Path(config.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory with config
        data_dir = self.cache_dir / "data"
        config_file = data_dir / "config.yaml"

        # Write config
        with open(config_file, "w") as f:
            f.write(config.to_yaml())

        if verbose:
            print(f"\nGenerating HAPNEST synthetic genotypes")
            print(f"  Samples: {config.n_samples:,}")
            print(f"  Chromosome: {config.chromosome}")
            print(f"  Ancestry: {config.superpopulation}")
            print(f"  Seed: {config.seed}")
            print(f"  Output: {config.output_dir}")
            print("=" * 60)

        # Generate genotypes
        if verbose:
            print("\nGenerating genotypes...")
        result = self._run_container_cmd(
            f"generate_geno {n_threads} /data/config.yaml",
            data_dir,
            verbose=verbose,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Genotype generation failed: {result.stderr}")
        if verbose:
            print("  ✓ Genotypes generated")

        # Generate phenotypes if requested
        if generate_phenotypes and config.n_traits > 0:
            if verbose:
                print("\nGenerating phenotypes...")
            result = self._run_container_cmd(
                "generate_pheno /data/config.yaml",
                data_dir,
                verbose=verbose,
            )
            if result.returncode != 0:
                print(f"Warning: Phenotype generation failed: {result.stderr}")

        # Validate if requested
        if validate:
            if verbose:
                print("\nValidating synthetic data...")
            result = self._run_container_cmd(
                "validate /data/config.yaml",
                data_dir,
                verbose=verbose,
            )
            if result.returncode != 0:
                print(f"Warning: Validation failed: {result.stderr}")

        # Find output files
        output_files = list(config.output_dir.glob("*"))

        if verbose:
            print(f"\n✓ HAPNEST generation complete")
            print(f"\nOutput files:")
            for f in output_files:
                size = f.stat().st_size / 1024
                print(f"  • {f.name} ({size:.1f} KB)")

        return {
            "returncode": 0,
            "output_dir": config.output_dir,
            "output_files": output_files,
            "config": config,
        }

    @staticmethod
    def get_info() -> dict:
        """Get information about HAPNEST and requirements."""
        return {
            "description": "Generate realistic synthetic genotypes with LD structure",
            "paper": HAPNEST_PAPER_URL,
            "code": HAPNEST_GITHUB_URL,
            "data": HAPNEST_BIOSTUDIES_URL,
            "requirements": {
                "container": "Singularity or Docker",
                "disk": "~10 GB for reference data",
                "memory": "8+ GB recommended",
            },
            "ancestries": HAPNEST_ANCESTRIES,
            "output_format": "PLINK (.bed, .bim, .fam)",
        }


# =============================================================================
# Information Functions
# =============================================================================

def get_hapnest_info() -> dict:
    """
    Get comprehensive information about HAPNEST capabilities.

    Returns:
        dict: Information about HAPNEST datasets, generation, and tools
    """
    # Check for container runtime
    singularity_available = False
    docker_available = False
    try:
        result = subprocess.run(["singularity", "--version"], capture_output=True)
        singularity_available = result.returncode == 0
    except FileNotFoundError:
        pass
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True)
        docker_available = result.returncode == 0
    except FileNotFoundError:
        pass

    plink2_available = _check_plink2()

    return {
        "download_pregenerated": {
            "description": "Download pre-generated HAPNEST data from EBI",
            "individuals": "1,008,000",
            "variants": "6,800,000",
            "ancestries": HAPNEST_ANCESTRIES,
            "format": "PLINK 2 (.pgen, .pvar, .psam)",
            "usage": "sl.download_hapnest()",
        },
        "generate_new": {
            "description": "Generate NEW realistic synthetic genotypes",
            "requires": "Singularity or Docker container",
            "usage": "runner = sl.HAPNESTRunner(); runner.run(config)",
            "configurable": ["n_samples", "chromosome", "ancestry", "phenotypes"],
        },
        "references": {
            "paper": HAPNEST_PAPER_URL,
            "data": HAPNEST_BIOSTUDIES_URL,
            "code": HAPNEST_GITHUB_URL,
        },
        "tools": {
            "plink2": {
                "available": plink2_available,
                "url": "https://www.cog-genomics.org/plink/2.0/",
            },
            "singularity": {
                "available": singularity_available,
                "note": "Preferred for HPC environments",
            },
            "docker": {
                "available": docker_available,
                "note": "Alternative to Singularity",
            },
        },
    }
