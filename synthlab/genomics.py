#!/usr/bin/env python3
"""
Genomics data utilities for SynthLab.

This module provides:
- Random genotype generation for quick testing (NO LD structure)
- Re-exports from synthlab.hapnest for backward compatibility

For realistic synthetic genotypes with LD structure, use the hapnest module:
    from synthlab import download_hapnest, HAPNESTRunner, HAPNESTConfig

References:
    HAPNEST Paper: https://academic.oup.com/bioinformatics/article/39/9/btad535/7255913
    HAPNEST Data: https://www.ebi.ac.uk/biostudies/studies/S-BSST936
    HAPNEST Code: https://github.com/intervene-EU-H2020/synthetic_data
"""

import subprocess
from pathlib import Path
from typing import Optional

# Re-export HAPNEST functions for backward compatibility
from synthlab.hapnest import (
    # Constants
    HAPNEST_BASE_URL,
    HAPNEST_BIOSTUDIES_URL,
    HAPNEST_PAPER_URL,
    HAPNEST_GITHUB_URL,
    HAPNEST_DOCKER_IMAGE,
    HAPNEST_FILES,
    HAPNEST_ANCESTRIES,
    # Download functions
    download_hapnest,
    download_hapnest_small,
    load_hapnest_variants,
    load_hapnest_samples,
    list_hapnest_files,
    get_hapnest_cache_dir,
    get_hapnest_info,
    # Generation classes
    HAPNESTConfig,
    HAPNESTRunner,
)


def get_genomics_cache_dir() -> Path:
    """
    Get the default cache directory for genomics data.

    Returns:
        Path: Path to ~/.cache/synthlab/genomics/
    """
    cache_dir = Path.home() / ".cache" / "synthlab" / "genomics"
    return cache_dir


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
# Random Genotype Generation (Testing Only)
# =============================================================================

def generate_random_genotypes(
    n_samples: int = 1000,
    n_variants: int = 10000,
    output_dir: Optional[Path] = None,
    output_prefix: str = "demo_synthetic",
    seed: Optional[int] = None,
    maf_range: tuple = (0.01, 0.5),
) -> Path:
    """
    Generate random genotype data for quick testing.

    WARNING: This creates RANDOM genotypes with NO linkage disequilibrium (LD).
    The data follows Hardy-Weinberg equilibrium but variants are independent.

    This is suitable for:
    - Testing data pipelines and file I/O
    - Checking code runs without errors
    - Quick prototyping

    This is NOT suitable for:
    - GWAS (genome-wide association studies)
    - PRS (polygenic risk scores)
    - Fine-mapping
    - Any method that relies on LD structure

    For REALISTIC synthetic genotypes with LD, use one of:
    - download_hapnest(): Download pre-generated HAPNEST data
    - HAPNESTRunner: Generate new data with HAPNEST (requires container)

    Args:
        n_samples: Number of individuals
        n_variants: Number of genetic variants
        output_dir: Output directory. Defaults to ~/.cache/synthlab/genomics/demo/
        output_prefix: Prefix for output files (default: "demo_synthetic")
        seed: Random seed for reproducibility
        maf_range: Minor allele frequency range (min, max)

    Returns:
        Path: Path to output directory containing .pvar, .psam, and genotype files

    Example:
        >>> from synthlab import generate_random_genotypes
        >>> data_dir = generate_random_genotypes(n_samples=100, n_variants=1000, seed=42)
    """
    try:
        import numpy as np
        import polars as pl
    except ImportError:
        raise ImportError(
            "numpy and polars are required. Install with: pip install numpy polars"
        )

    if output_dir is None:
        output_dir = get_genomics_cache_dir() / "demo"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    print(f"\nGenerating random genotype data (no LD structure)")
    print(f"  Samples: {n_samples:,}")
    print(f"  Variants: {n_variants:,}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Generate random MAFs
    mafs = np.random.uniform(maf_range[0], maf_range[1], n_variants)

    # Generate genotypes (0, 1, 2) based on MAF under HWE
    print("  Generating genotypes...", end=" ", flush=True)
    genotypes = np.zeros((n_samples, n_variants), dtype=np.int8)
    for i, maf in enumerate(mafs):
        p = 1 - maf  # Major allele frequency
        q = maf      # Minor allele frequency
        # Genotype probabilities under HWE: p^2, 2pq, q^2
        probs = [p**2, 2*p*q, q**2]
        genotypes[:, i] = np.random.choice([0, 1, 2], size=n_samples, p=probs)
    print("✓")

    # Create variant info (BIM-like format)
    print("  Creating variant annotations...", end=" ", flush=True)
    chroms = np.random.choice(list(range(1, 23)), n_variants)
    positions = np.sort(np.random.randint(1000, 100000000, n_variants))
    variant_ids = [f"rs{i+1}" for i in range(n_variants)]
    ref_alleles = np.random.choice(["A", "C", "G", "T"], n_variants)
    alt_alleles = np.random.choice(["A", "C", "G", "T"], n_variants)

    variants_df = pl.DataFrame({
        "CHROM": chroms,
        "POS": positions,
        "ID": variant_ids,
        "REF": ref_alleles,
        "ALT": alt_alleles,
        "MAF": mafs,
    })
    print("✓")

    # Create sample info (FAM-like format)
    print("  Creating sample annotations...", end=" ", flush=True)
    sample_ids = [f"SAMPLE_{i+1:06d}" for i in range(n_samples)]
    sexes = np.random.choice([1, 2], n_samples)  # 1=male, 2=female

    samples_df = pl.DataFrame({
        "IID": sample_ids,
        "SEX": sexes,
    })
    print("✓")

    # Save files
    print("  Saving files...", end=" ", flush=True)

    # Save variant info (.pvar format)
    pvar_file = output_dir / f"{output_prefix}.pvar"
    with open(pvar_file, "w") as f:
        f.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for row in variants_df.iter_rows():
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\n")

    # Save sample info (.psam format)
    psam_file = output_dir / f"{output_prefix}.psam"
    with open(psam_file, "w") as f:
        f.write("#IID\tSEX\n")
        for row in samples_df.iter_rows():
            f.write(f"{row[0]}\t{row[1]}\n")

    # Save genotypes as compressed CSV (simple format for testing)
    # Note: For real use, convert to PLINK binary format
    geno_file = output_dir / f"{output_prefix}_genotypes.csv.gz"
    geno_df = pl.DataFrame(genotypes, schema=[f"V{i+1}" for i in range(n_variants)])
    geno_df = geno_df.with_columns(pl.Series("IID", sample_ids).alias("IID"))
    # Reorder columns to put IID first
    geno_df = geno_df.select(["IID"] + [f"V{i+1}" for i in range(n_variants)])
    geno_df.write_csv(geno_file)
    print("✓")

    print(f"\n✓ Generated random genotype data:")
    print(f"  • Variants: {pvar_file}")
    print(f"  • Samples: {psam_file}")
    print(f"  • Genotypes: {geno_file}")
    print(f"\n⚠️  WARNING: This is RANDOM data with NO LD structure!")
    print(f"   - Good for: testing pipelines, checking code runs")
    print(f"   - Bad for: GWAS, PRS, fine-mapping, any LD-dependent method")
    print(f"\nFor REALISTIC synthetic data with LD structure:")
    print(f"  • Download pre-generated: sl.download_hapnest()")
    print(f"  • Generate new data: sl.HAPNESTRunner().run(config)")

    return output_dir


# Backward-compatible alias
generate_synthetic_genotypes = generate_random_genotypes


# =============================================================================
# Information Functions
# =============================================================================

def get_genomics_info() -> dict:
    """
    Get information about available genomics datasets and tools.

    Returns:
        dict: Information about genomics capabilities
    """
    plink2_available = _check_plink2()

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

    return {
        "download_pregenerated": {
            "hapnest": {
                "description": "HAPNEST synthetic genotypes with REALISTIC LD structure",
                "individuals": "1,008,000",
                "variants": "6,800,000",
                "ancestries": 6,
                "format": "PLINK 2 (.pgen, .pvar, .psam)",
                "download": "sl.download_hapnest()",
                "paper": HAPNEST_PAPER_URL,
                "data": HAPNEST_BIOSTUDIES_URL,
            },
            "ukb_synthetic_genetic": {
                "description": "UK Biobank Synthetic genetic data",
                "individuals": "~600,000",
                "variants": "~840,000",
                "format": "Custom compressed",
                "download": "sl.download_category('genetic')",
            },
        },
        "generate_new": {
            "hapnest_runner": {
                "description": "Generate NEW realistic synthetic genotypes with HAPNEST",
                "has_ld": True,
                "requires": "Singularity or Docker",
                "usage": "runner = sl.HAPNESTRunner(); runner.run(config)",
                "code": HAPNEST_GITHUB_URL,
            },
            "random_genotypes": {
                "description": "Generate random genotypes for TESTING ONLY (no LD!)",
                "has_ld": False,
                "warning": "NOT suitable for GWAS, PRS, or LD-dependent methods",
                "usage": "sl.generate_random_genotypes(n_samples=100, n_variants=1000)",
            },
        },
        "tools": {
            "plink2": {
                "available": plink2_available,
                "url": "https://www.cog-genomics.org/plink/2.0/",
                "note": "Required for working with .pgen/.pvar/.psam files",
            },
            "singularity": {
                "available": singularity_available,
                "note": "Required for HAPNESTRunner (preferred for HPC)",
            },
            "docker": {
                "available": docker_available,
                "note": "Alternative to Singularity for HAPNESTRunner",
            },
        },
        "functions": [
            "# Download pre-generated data (realistic LD):",
            "download_hapnest() - Download HAPNEST dataset from EBI",
            "load_hapnest_variants() - Load variant information (.pvar)",
            "load_hapnest_samples() - Load sample information (.psam)",
            "",
            "# Generate new data:",
            "HAPNESTRunner - Generate realistic genotypes (requires container)",
            "generate_random_genotypes() - Quick random genotypes (NO LD, testing only)",
        ],
    }
