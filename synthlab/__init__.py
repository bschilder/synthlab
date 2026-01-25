"""
SynthLab - Python tools for working with synthetic healthcare datasets.

SynthLab provides Python interfaces for working with major synthetic healthcare datasets:
- Synthea: A synthetic patient population simulator
- Synthea Coherent Data Set: Multimodal data (EHR + Imaging + Genomics + Notes)
- UK Biobank Synthetic Dataset: Large-scale synthetic dataset for system testing
- HAPNEST: Synthetic genomics data (genotypes and phenotypes)
- Medical Imaging: Catalog and utilities for public imaging datasets
"""

# Utils module (exposed as sl.utils)
from synthlab import utils

from synthlab.synthea import (
    SyntheaRunner,
    SyntheaConfig,
    convert_synthea_to_omop,
    add_concept_names,
    download_synthea_jar,
    SYNTHEA_JAR_URL,
)

from synthlab.download_synthea_omop import (
    list_synthea_datasets,
    download_dataset,
    convert_csv_to_parquet,
)

from synthlab.download_ukbiobank_synthetic import (
    list_available_files,
    download_file,
    download_category,
    load_tabular_data,
    load_medical_records,
    load_genetic_dictionary,
    get_cache_dir,
)

# Genomics utilities
from synthlab.genomics import (
    generate_random_genotypes,
    generate_synthetic_genotypes,  # Backward-compatible alias
    get_genomics_info,
    get_genomics_cache_dir,
)

# HAPNEST submodule (realistic synthetic genotypes with LD)
from synthlab.hapnest import (
    # Download pre-generated data
    download_hapnest,
    download_hapnest_small,  # Backward-compatible alias
    load_hapnest_variants,
    load_hapnest_samples,
    list_hapnest_files,
    get_hapnest_cache_dir,
    get_hapnest_info,
    # Generate new data (requires Singularity/Docker)
    HAPNESTRunner,
    HAPNESTConfig,
    # Constants
    HAPNEST_PAPER_URL,
    HAPNEST_GITHUB_URL,
    HAPNEST_BIOSTUDIES_URL,
    HAPNEST_ANCESTRIES,
)

from synthlab.imaging import (
    list_imaging_datasets,
    get_dataset_info,
    print_dataset_catalog,
    download_mhist,
    get_imaging_info,
    get_imaging_cache_dir,
    ImagingDataset,
    IMAGING_DATASETS,
)

from synthlab.imaging_generation import (
    ImagingGenerator,
    ImagingGeneratorConfig,
    GeneratedImage,
    generate_synthetic_image,
    get_imaging_generation_info,
    print_imaging_generation_info,
)

from synthlab.soap import (
    SOAPNote,
    SOAPNoteGenerator,
    generate_soap_note,
    FHIRFormatter,
    CausalGraph,
    CausalEdge,
    CausalNode,
    parse_causal_graph,
    NODE_TYPES,
    CAUSAL_EDGE_TYPES,
)

from synthlab.snomed import (
    SNOMEDLinker,
    SNOMEDConcept,
    SNOMEDMatch,
    LinkedEntity,
    GroundedNode,
    GroundedEdge,
    GroundedCausalGraph,
    ExtractedEntity,
    MedicalEntityExtractor,
    DocumentLinker,
    load_snomed_from_csv,
    load_snomed_from_umls,
    get_snomed_cache_dir,
    get_snomed_info,
    print_snomed_info,
    get_sample_snomed_concepts,
    setup_sample_linker,
    create_entity_pipeline,
    SAPBERT_MODEL_ID,
    SAPBERT_MODELS,
    EMBEDDING_MODELS,
)

from synthlab.coherent import (
    list_coherent_components,
    list_coherent_files,
    download_coherent_dataset,
    load_fhir_patients,
    get_coherent_info,
    print_coherent_info,
    print_s3_structure,
    discover_s3_structure,
    find_files_by_extension,
    get_coherent_cache_dir,
    COHERENT_COMPONENTS,
    # Main entry point for multimodal data
    load_multimodal_dataset,
    # Multimodal data classes
    Patient,
    MultimodalDataset,
    # Multimodal linking utilities
    extract_patient_id,
    find_dna_csv_for_patient,
    find_dicoms_for_patient,
    load_dicom_series,
)

__version__ = "0.2.0"

# ANSI color codes - Synthwave/Vaporwave palette
_RESET = "\033[0m"
_BOLD = "\033[1m"
# Hot pink to cyan gradient
_PINK = "\033[38;5;199m"      # Hot pink
_MAGENTA = "\033[38;5;165m"   # Magenta
_PURPLE = "\033[38;5;135m"    # Purple
_VIOLET = "\033[38;5;99m"     # Violet
_BLUE = "\033[38;5;75m"       # Light blue
_CYAN = "\033[38;5;51m"       # Cyan
_TEAL = "\033[38;5;49m"       # Teal
# Accents
_ORANGE = "\033[38;5;208m"    # Sunset orange
_YELLOW = "\033[38;5;227m"    # Neon yellow
_WHITE = "\033[38;5;255m"     # Bright white

_LOGO = f"""
{_PINK}  ░██████╗██╗░░░██╗███╗░░██╗████████╗██╗░░██╗  ██╗░░░░░░█████╗░██████╗░{_RESET}
{_MAGENTA}  ██╔════╝╚██╗░██╔╝████╗░██║╚══██╔══╝██║░░██║  ██║░░░░░██╔══██╗██╔══██╗{_RESET}
{_PURPLE}  ╚█████╗░░╚████╔╝░██╔██╗██║░░░██║░░░███████║  ██║░░░░░███████║██████╦╝{_RESET}
{_VIOLET}  ░╚═══██╗░░╚██╔╝░░██║╚████║░░░██║░░░██╔══██║  ██║░░░░░██╔══██║██╔══██╗{_RESET}
{_BLUE}  ██████╔╝░░░██║░░░██║░╚███║░░░██║░░░██║░░██║  ███████╗██║░░██║██████╦╝{_RESET}
{_CYAN}  ╚═════╝░░░░╚═╝░░░╚═╝░░╚══╝░░░╚═╝░░░╚═╝░░╚═╝  ╚══════╝╚═╝░░╚═╝╚═════╝░{_RESET}
{_TEAL}        ▌║█║▌│║▌│║▌║▌█║▌║█║▌│║▌│║▌║▌█║▌║█║▌│║▌│║▌║▌█║▌│║▌│║▌║▌█║{_RESET}
{_VIOLET}  ════════════════════════════════════════════════════════════════════{_RESET}
{_WHITE}  {_BOLD}Synthetic Healthcare Data Toolkit{_RESET}
{_VIOLET}  ────────────────────────────────────────────────────────────────────{_RESET}
{_PINK}  ◈{_RESET} EHR        {_WHITE}Synthetic patient records (diagnoses, meds, labs){_RESET}
{_MAGENTA}  ◈{_RESET} Genomics   {_WHITE}Synthetic genotypes with realistic LD structure{_RESET}
{_PURPLE}  ◈{_RESET} Imaging    {_WHITE}Datasets + synthetic generation (CT, MRI, X-ray){_RESET}
{_CYAN}  ◈{_RESET} Multimodal {_WHITE}Linked EHR + Imaging + Genomics per patient{_RESET}
{_TEAL}  ◈{_RESET} AI Notes   {_WHITE}SOAP notes with causal graph analysis{_RESET}
{_VIOLET}  ════════════════════════════════════════════════════════════════════{_RESET}
"""


def _print_banner():
    """Print the SynthLab banner on import."""
    import sys
    # Only print in interactive mode (notebooks, REPL)
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        print(_LOGO)
        print(f"  {_CYAN}Version:{_RESET} {_WHITE}{__version__}{_RESET}")
        print(f"  {_CYAN}Cache:{_RESET}   {_WHITE}{get_coherent_cache_dir().parent}{_RESET}")
        print()


# Print banner on import
_print_banner()

__all__ = [
    # Utils module
    "utils",
    # Synthea
    "SyntheaRunner",
    "SyntheaConfig",
    "convert_synthea_to_omop",
    "add_concept_names",
    "download_synthea_jar",
    "SYNTHEA_JAR_URL",
    # Synthea OMOP downloads
    "list_synthea_datasets",
    "download_dataset",
    "convert_csv_to_parquet",
    # UK Biobank Synthetic
    "list_available_files",
    "download_file",
    "download_category",
    "load_tabular_data",
    "load_medical_records",
    "load_genetic_dictionary",
    "get_cache_dir",
    # Genomics utilities
    "generate_random_genotypes",
    "generate_synthetic_genotypes",  # Backward-compatible alias
    "get_genomics_info",
    "get_genomics_cache_dir",
    # HAPNEST (realistic synthetic genotypes with LD)
    "download_hapnest",
    "download_hapnest_small",  # Backward-compatible alias
    "load_hapnest_variants",
    "load_hapnest_samples",
    "list_hapnest_files",
    "get_hapnest_cache_dir",
    "get_hapnest_info",
    # HAPNEST generation (requires container)
    "HAPNESTRunner",
    "HAPNESTConfig",
    "HAPNEST_PAPER_URL",
    "HAPNEST_GITHUB_URL",
    "HAPNEST_BIOSTUDIES_URL",
    "HAPNEST_ANCESTRIES",
    # Medical Imaging (catalog/download)
    "list_imaging_datasets",
    "get_dataset_info",
    "print_dataset_catalog",
    "download_mhist",
    "get_imaging_info",
    "get_imaging_cache_dir",
    "ImagingDataset",
    "IMAGING_DATASETS",
    # Medical Imaging (generation)
    "ImagingGenerator",
    "ImagingGeneratorConfig",
    "GeneratedImage",
    "generate_synthetic_image",
    "get_imaging_generation_info",
    "print_imaging_generation_info",
    # Coherent Data Set (multimodal)
    "list_coherent_components",
    "list_coherent_files",
    "download_coherent_dataset",
    "load_fhir_patients",
    "get_coherent_info",
    "print_coherent_info",
    "print_s3_structure",
    "discover_s3_structure",
    "find_files_by_extension",
    "get_coherent_cache_dir",
    "COHERENT_COMPONENTS",
    # Main entry point for multimodal data
    "load_multimodal_dataset",
    # Multimodal data classes
    "Patient",
    "MultimodalDataset",
    # Multimodal linking utilities
    "extract_patient_id",
    "find_dna_csv_for_patient",
    "find_dicoms_for_patient",
    "load_dicom_series",
    # SOAP Note Generation (MedGemma)
    "SOAPNote",
    "SOAPNoteGenerator",
    "generate_soap_note",
    "FHIRFormatter",
    # Causal Graph Analysis
    "CausalGraph",
    "CausalEdge",
    "CausalNode",
    "parse_causal_graph",
    "NODE_TYPES",
    "CAUSAL_EDGE_TYPES",
    # SNOMED Entity Linking
    "SNOMEDLinker",
    "SNOMEDConcept",
    "SNOMEDMatch",
    "LinkedEntity",
    "GroundedNode",
    "GroundedEdge",
    "GroundedCausalGraph",
    "ExtractedEntity",
    "MedicalEntityExtractor",
    "DocumentLinker",
    "load_snomed_from_csv",
    "load_snomed_from_umls",
    "get_snomed_cache_dir",
    "get_snomed_info",
    "print_snomed_info",
    "get_sample_snomed_concepts",
    "setup_sample_linker",
    "create_entity_pipeline",
    "SAPBERT_MODEL_ID",
    "SAPBERT_MODELS",
    "EMBEDDING_MODELS",
]
