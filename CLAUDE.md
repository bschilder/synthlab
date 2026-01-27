# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SynthLab is a Python toolkit for working with synthetic healthcare datasets. It provides unified interfaces to multiple data sources: EHR records (Synthea), genomics (HAPNEST), medical imaging, and clinical notes with AI-powered analysis.

## Development Commands

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific extras
pip install -e ".[snomed]"    # SNOMED entity linking (FAISS)
pip install -e ".[imaging]"   # DICOM support
pip install -e ".[genomics]"  # NumPy for genotype generation
pip install -e ".[aws]"       # S3 dataset downloads

# Alternative: Use conda environment
conda env create -f envs/environment.yml
conda activate synthlab
```

No formal test suite exists yet. Run notebooks in `notebooks/` to verify functionality.

## Architecture

### Module Structure

All modules are in `synthlab/` and exported through `__init__.py`. The package uses `import synthlab as sl` convention.

| Module | Purpose |
|--------|---------|
| `synthea.py` | Synthea patient simulator runner + OMOP CDM conversion |
| `coherent.py` | Synthea Coherent multimodal dataset (EHR + Imaging + Genomics) |
| `hapnest.py` | HAPNEST synthetic genomics with realistic LD structure |
| `genomics.py` | Simple synthetic genotype generation utilities |
| `imaging.py` | Medical imaging dataset catalog and downloads |
| `imaging_generation.py` | AI-powered synthetic image generation |
| `soap.py` | SOAP note generation using MedGemma LLM + causal graph extraction |
| `snomed.py` | SNOMED CT entity linking (SapBERT + FAISS) |
| `download_ukbiobank_synthetic.py` | UK Biobank synthetic dataset downloader |
| `download_synthea_omop.py` | Pre-generated Synthea OMOP dataset downloader |
| `utils.py` | Shared utilities (token counting, caching) |

### Key Patterns

**Lazy imports**: Heavy dependencies (torch, transformers, faiss) are imported only when needed to keep startup fast.

**Cache directory**: All downloaded data goes to `~/.cache/synthlab/<module>/`. Each module has a `get_<module>_cache_dir()` function.

**Info functions**: Most modules provide `get_<module>_info()` and `print_<module>_info()` for introspection.

**Dataclasses**: Configuration and result objects use Python dataclasses (e.g., `SyntheaConfig`, `SOAPNote`, `SNOMEDConcept`).

### SOAP Note Pipeline (soap.py + snomed.py)

The most complex subsystem generates clinical SOAP notes from FHIR patient data:

1. `SOAPNoteGenerator` takes a FHIR patient bundle
2. Uses MedGemma (Google's medical LLM) to generate structured notes
3. Extracts causal graphs showing clinical relationships
4. Optionally grounds entities to SNOMED CT via `SNOMEDLinker`
5. Optionally annotates genetic variants via BioMCP

Key classes:
- `SOAPNoteGenerator`: Main orchestrator with auto-chunking for long contexts
- `CausalGraph`, `CausalNode`, `CausalEdge`: Parsed relationship structure
- `SNOMEDLinker`: Embedding-based entity linking with FAISS index
- `GroundedCausalGraph`: Graph with SNOMED CT IDs attached

### SNOMED Entity Linking Workflow

The SNOMED linking uses a **two-phase architecture**:

**Phase 1: One-time index build** (offline, ~10-30 min for full SNOMED)
```python
# Build and cache embeddings for all SNOMED concepts
sl.build_snomed_index("/path/to/CONCEPT.csv")
```

**Phase 2: Runtime queries** (fast, loads cached index)
```python
linker = sl.load_snomed_linker()  # Loads pre-built index instantly
matches = linker.link("diabetes")  # Only embeds the query term
```

The workflow for SOAP notes:
1. Generate SOAP note and extract medical terms (potential graph nodes)
2. Embed only those extracted terms (few dozen)
3. Search against pre-computed SNOMED index (350k+ concepts)
4. Map matched terms to build grounded causal graph

### Data Loading

SNOMED concepts can be loaded from multiple sources:
- `load_snomed_from_omop()`: OMOP CDM CONCEPT.csv (recommended for full vocabulary)
- `load_snomed_from_umls()`: UMLS Metathesaurus MRCONSO.RRF
- `load_snomed_from_csv()`: Custom CSV files
- `fetch_snomed_from_browser()`: SNOMED International API (no license required)
- `get_sample_snomed_concepts()`: ~200 common terms for testing

## Configuration

MedGemma requires a Hugging Face token with access to `google/medgemma-27b-text-it`:
```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

## File Conventions

- Notebooks in `notebooks/` demonstrate each module's capabilities
- Documentation in `docs/` covers complex subsystems (entity linking, UK Biobank)
- Examples in `examples/` show basic usage patterns
