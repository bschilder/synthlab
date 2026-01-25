#!/usr/bin/env python3
"""
Medical imaging dataset utilities for SynthLab.

This module provides tools for discovering and downloading publicly available
medical imaging datasets, including:
- Synthetic imaging datasets
- De-identified clinical imaging archives (TCIA, Stanford AIMI)
- Histopathology datasets (MHIST, PanNuke, CAMELYON)

Note: Most medical imaging datasets are large (GB to TB scale) and may require
registration or data use agreements.

References:
- TCIA: https://www.cancerimagingarchive.net/
- Stanford AIMI: https://aimi.stanford.edu/shared-datasets
- MHIST: https://bmirds.github.io/MHIST/
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
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


@dataclass
class ImagingDataset:
    """Metadata for a medical imaging dataset."""
    name: str
    description: str
    modality: str  # e.g., "CT", "MRI", "Histopathology", "X-ray"
    size_gb: Optional[float] = None
    n_images: Optional[int] = None
    n_patients: Optional[int] = None
    access_type: str = "open"  # "open", "registration", "dua" (data use agreement)
    download_url: Optional[str] = None
    homepage_url: Optional[str] = None
    license: Optional[str] = None
    format: str = "DICOM"  # "DICOM", "NIfTI", "PNG", "TIFF", etc.
    tags: list = field(default_factory=list)


# Catalog of publicly available medical imaging datasets
IMAGING_DATASETS = {
    # =========================================================================
    # Histopathology Datasets
    # =========================================================================
    "mhist": ImagingDataset(
        name="MHIST",
        description="Minimalist Histopathology Image Analysis Dataset - H&E stained colorectal polyps",
        modality="Histopathology",
        n_images=3152,
        format="PNG",
        size_gb=0.35,
        access_type="open",
        download_url="https://bmirds.github.io/MHIST/",
        homepage_url="https://bmirds.github.io/MHIST/",
        license="CC BY-NC-SA 4.0",
        tags=["histopathology", "colorectal", "polyps", "classification"],
    ),
    "pannuke": ImagingDataset(
        name="PanNuke",
        description="Pan-cancer histology dataset for nuclei instance segmentation",
        modality="Histopathology",
        n_images=7753,  # Fold 1+2+3
        format="PNG/NPY",
        size_gb=2.0,
        access_type="open",
        download_url="https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke",
        homepage_url="https://jgamper.github.io/PanNukeDataset/",
        license="CC BY-NC-SA 4.0",
        tags=["histopathology", "nuclei", "segmentation", "pan-cancer"],
    ),
    "camelyon16": ImagingDataset(
        name="CAMELYON16",
        description="Detection of cancer metastases in lymph node WSI",
        modality="Histopathology",
        n_images=400,  # Whole slide images
        n_patients=None,
        format="TIFF (WSI)",
        size_gb=700,  # Approximately
        access_type="registration",
        download_url="https://camelyon16.grand-challenge.org/Data/",
        homepage_url="https://camelyon16.grand-challenge.org/",
        license="CC0 1.0",
        tags=["histopathology", "breast_cancer", "metastasis", "wsi"],
    ),
    "camelyon17": ImagingDataset(
        name="CAMELYON17",
        description="Automated detection and classification of breast cancer metastases",
        modality="Histopathology",
        n_images=1000,  # Whole slide images
        n_patients=200,
        format="TIFF (WSI)",
        size_gb=2500,  # Approximately
        access_type="registration",
        download_url="https://camelyon17.grand-challenge.org/Data/",
        homepage_url="https://camelyon17.grand-challenge.org/",
        license="CC0 1.0",
        tags=["histopathology", "breast_cancer", "metastasis", "wsi"],
    ),
    "bcss": ImagingDataset(
        name="BCSS",
        description="Breast Cancer Semantic Segmentation dataset",
        modality="Histopathology",
        n_images=151,
        format="PNG",
        size_gb=1.5,
        access_type="open",
        download_url="https://bcsegmentation.grand-challenge.org/",
        homepage_url="https://bcsegmentation.grand-challenge.org/",
        tags=["histopathology", "breast_cancer", "segmentation"],
    ),

    # =========================================================================
    # Radiology Datasets (CT, MRI, X-ray)
    # =========================================================================
    "tcia_lung_ct": ImagingDataset(
        name="LIDC-IDRI",
        description="Lung Image Database Consortium - CT scans with lung nodule annotations",
        modality="CT",
        n_images=1018,  # CT scans
        n_patients=1010,
        format="DICOM",
        size_gb=125,
        access_type="open",
        download_url="https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
        homepage_url="https://www.cancerimagingarchive.net/collection/lidc-idri/",
        license="CC BY 3.0",
        tags=["ct", "lung", "nodule", "cancer_screening"],
    ),
    "chestxray14": ImagingDataset(
        name="ChestX-ray14",
        description="NIH Chest X-ray dataset with 14 disease labels",
        modality="X-ray",
        n_images=112120,
        n_patients=30805,
        format="PNG",
        size_gb=42,
        access_type="open",
        download_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        homepage_url="https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community",
        license="CC0 1.0",
        tags=["xray", "chest", "multi_label", "classification"],
    ),
    "mimic_cxr": ImagingDataset(
        name="MIMIC-CXR",
        description="Large publicly available chest X-ray dataset with free-text radiology reports",
        modality="X-ray",
        n_images=377110,
        n_patients=65379,
        format="DICOM/JPG",
        size_gb=4700,  # Full DICOM version
        access_type="dua",  # Requires PhysioNet credentialing
        download_url="https://physionet.org/content/mimic-cxr/2.0.0/",
        homepage_url="https://physionet.org/content/mimic-cxr/2.0.0/",
        license="PhysioNet Credentialed Health Data License",
        tags=["xray", "chest", "reports", "multi_label"],
    ),
    "rsna_pneumonia": ImagingDataset(
        name="RSNA Pneumonia Detection",
        description="RSNA challenge dataset for pneumonia detection in chest X-rays",
        modality="X-ray",
        n_images=30000,
        format="DICOM",
        size_gb=3,
        access_type="registration",
        download_url="https://www.kaggle.com/c/rsna-pneumonia-detection-challenge",
        homepage_url="https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018",
        tags=["xray", "chest", "pneumonia", "detection"],
    ),
    "brats": ImagingDataset(
        name="BraTS",
        description="Brain Tumor Segmentation Challenge - Multi-institutional MRI data",
        modality="MRI",
        n_images=2000,  # Varies by year
        format="NIfTI",
        size_gb=50,
        access_type="registration",
        download_url="https://www.synapse.org/#!Synapse:syn27046444/wiki/616571",
        homepage_url="https://www.med.upenn.edu/cbica/brats/",
        license="CC BY-SA 4.0",
        tags=["mri", "brain", "tumor", "segmentation", "glioma"],
    ),
    "oasis": ImagingDataset(
        name="OASIS",
        description="Open Access Series of Imaging Studies - Brain MRI data",
        modality="MRI",
        n_images=2000,
        n_patients=1664,
        format="NIfTI",
        size_gb=100,
        access_type="registration",
        download_url="https://www.oasis-brains.org/",
        homepage_url="https://www.oasis-brains.org/",
        license="Custom (free for research)",
        tags=["mri", "brain", "alzheimers", "longitudinal"],
    ),
    "adni": ImagingDataset(
        name="ADNI",
        description="Alzheimer's Disease Neuroimaging Initiative",
        modality="MRI/PET",
        n_patients=2000,
        format="DICOM/NIfTI",
        size_gb=1000,
        access_type="dua",
        download_url="https://adni.loni.usc.edu/data-samples/access-data/",
        homepage_url="https://adni.loni.usc.edu/",
        tags=["mri", "pet", "brain", "alzheimers"],
    ),

    # =========================================================================
    # Synthetic Medical Imaging Datasets
    # =========================================================================
    "snow_synthetic": ImagingDataset(
        name="SNOW",
        description="Synthetic Nuclei and annOtation Wizard - Synthetic pathology images",
        modality="Histopathology (Synthetic)",
        format="PNG",
        access_type="open",
        download_url="https://www.nature.com/articles/s41597-023-02125-y",
        homepage_url="https://www.nature.com/articles/s41597-023-02125-y",
        license="CC BY 4.0",
        tags=["histopathology", "synthetic", "nuclei", "segmentation"],
    ),

    # =========================================================================
    # Multi-modal / Other
    # =========================================================================
    "stanford_aimi_chexpert": ImagingDataset(
        name="CheXpert",
        description="Large chest X-ray dataset with uncertainty labels",
        modality="X-ray",
        n_images=224316,
        n_patients=65240,
        format="JPG",
        size_gb=11,
        access_type="registration",
        download_url="https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2",
        homepage_url="https://stanfordmlgroup.github.io/competitions/chexpert/",
        license="Stanford AIMI License",
        tags=["xray", "chest", "multi_label", "uncertainty"],
    ),
}


def get_imaging_cache_dir() -> Path:
    """
    Get the default cache directory for imaging data.

    Returns:
        Path: Path to ~/.cache/synthlab/imaging/
    """
    cache_dir = Path.home() / ".cache" / "synthlab" / "imaging"
    return cache_dir


def list_imaging_datasets(
    modality: Optional[str] = None,
    access_type: Optional[str] = None,
    tag: Optional[str] = None,
) -> dict[str, ImagingDataset]:
    """
    List available medical imaging datasets.

    Args:
        modality: Filter by modality (e.g., "CT", "MRI", "Histopathology", "X-ray")
        access_type: Filter by access type ("open", "registration", "dua")
        tag: Filter by tag (e.g., "lung", "brain", "synthetic")

    Returns:
        dict: Filtered dictionary of datasets

    Example:
        >>> from synthlab.imaging import list_imaging_datasets
        >>> # List all histopathology datasets
        >>> histo = list_imaging_datasets(modality="Histopathology")
        >>> # List open access datasets
        >>> open_datasets = list_imaging_datasets(access_type="open")
    """
    result = {}

    for key, dataset in IMAGING_DATASETS.items():
        # Apply filters
        if modality and modality.lower() not in dataset.modality.lower():
            continue
        if access_type and access_type.lower() != dataset.access_type.lower():
            continue
        if tag and tag.lower() not in [t.lower() for t in dataset.tags]:
            continue

        result[key] = dataset

    return result


def get_dataset_info(dataset_name: str) -> ImagingDataset:
    """
    Get detailed information about a specific dataset.

    Args:
        dataset_name: Name of the dataset (key in IMAGING_DATASETS)

    Returns:
        ImagingDataset: Dataset metadata

    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in IMAGING_DATASETS:
        available = ", ".join(IMAGING_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available: {available}"
        )
    return IMAGING_DATASETS[dataset_name]


def print_dataset_catalog(
    modality: Optional[str] = None,
    access_type: Optional[str] = None,
):
    """
    Print a formatted catalog of available imaging datasets.

    Args:
        modality: Filter by modality
        access_type: Filter by access type
    """
    datasets = list_imaging_datasets(modality=modality, access_type=access_type)

    print("\n" + "=" * 80)
    print("Medical Imaging Dataset Catalog")
    print("=" * 80)

    if modality:
        print(f"Filtered by modality: {modality}")
    if access_type:
        print(f"Filtered by access type: {access_type}")

    # Group by modality
    modalities: dict[str, list] = {}
    for key, ds in datasets.items():
        mod = ds.modality.split()[0]  # Get base modality
        if mod not in modalities:
            modalities[mod] = []
        modalities[mod].append((key, ds))

    for mod, ds_list in sorted(modalities.items()):
        print(f"\n{mod}")
        print("-" * 40)

        for key, ds in ds_list:
            size_str = f"{ds.size_gb:.1f} GB" if ds.size_gb else "?"
            access_icon = {"open": "🟢", "registration": "🟡", "dua": "🔴"}.get(
                ds.access_type, "⚪"
            )

            print(f"  {access_icon} {ds.name}")
            print(f"     {ds.description[:60]}...")
            print(f"     Size: {size_str} | Format: {ds.format} | Access: {ds.access_type}")
            if ds.n_images:
                print(f"     Images: {ds.n_images:,}", end="")
                if ds.n_patients:
                    print(f" | Patients: {ds.n_patients:,}")
                else:
                    print()
            print()

    print("\nAccess legend: 🟢 Open | 🟡 Registration | 🔴 Data Use Agreement")
    print(f"\nTotal datasets: {len(datasets)}")


def download_mhist(
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Download the MHIST histopathology dataset.

    MHIST contains 3,152 H&E stained images of colorectal polyps for
    binary classification (hyperplastic vs sessile serrated adenoma).

    Args:
        output_dir: Directory to save files. Defaults to ~/.cache/synthlab/imaging/mhist/
        overwrite: If True, overwrite existing files

    Returns:
        Path: Path to output directory

    Note:
        This downloads from the MHIST GitHub releases. Check the license
        (CC BY-NC-SA 4.0) before use.
    """
    if requests is None:
        raise ImportError(
            "requests is required to download imaging datasets. "
            "Install with: pip install requests"
        )

    if output_dir is None:
        output_dir = get_imaging_cache_dir() / "mhist"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MHIST is hosted on GitHub releases
    # Check their page for the current download link
    print(f"\nMHIST Dataset Download")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    print("MHIST must be downloaded from the official source:")
    print("  https://bmirds.github.io/MHIST/")
    print()
    print("Steps:")
    print("  1. Visit the website above")
    print("  2. Download the ZIP file")
    print(f"  3. Extract to: {output_dir}")
    print()
    print("Dataset info:")
    print("  • 3,152 H&E stained images (224x224 pixels)")
    print("  • Binary classification task")
    print("  • License: CC BY-NC-SA 4.0")

    return output_dir


def get_imaging_info() -> dict:
    """
    Get summary information about imaging datasets and capabilities.

    Returns:
        dict: Summary information
    """
    datasets = IMAGING_DATASETS
    modalities = set(ds.modality.split()[0] for ds in datasets.values())

    open_count = sum(1 for ds in datasets.values() if ds.access_type == "open")
    reg_count = sum(1 for ds in datasets.values() if ds.access_type == "registration")
    dua_count = sum(1 for ds in datasets.values() if ds.access_type == "dua")

    return {
        "total_datasets": len(datasets),
        "modalities": sorted(modalities),
        "access_breakdown": {
            "open": open_count,
            "registration": reg_count,
            "dua": dua_count,
        },
        "functions": [
            "list_imaging_datasets(modality=None, access_type=None, tag=None)",
            "get_dataset_info(dataset_name)",
            "print_dataset_catalog(modality=None, access_type=None)",
            "download_mhist(output_dir=None)",
        ],
        "note": (
            "Most medical imaging datasets are large and require manual download. "
            "Use print_dataset_catalog() to see available datasets and their download URLs."
        ),
    }
