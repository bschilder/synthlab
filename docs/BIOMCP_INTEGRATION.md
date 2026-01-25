# BioMCP Integration

BioMCP (Biomedical Model Context Protocol) is an open-source toolkit that connects AI systems to authoritative biomedical data sources. SynthLab uses BioMCP to enrich genetic variant annotations with clinical significance, disease associations, and pharmacogenomic information.

## What is BioMCP?

BioMCP provides structured access to multiple biomedical databases through a unified API:

| Data Source | Information Provided |
|-------------|---------------------|
| **ClinVar** | Clinical significance, pathogenicity classifications |
| **MyVariant.info** | Variant annotations, population frequencies |
| **MyGene.info** | Gene function, pathways, expression |
| **MyDisease.info** | Disease descriptions, phenotypes, inheritance |
| **MyChem.info** | Drug information, interactions, targets |
| **PubMed/PubTator3** | Biomedical literature, gene-disease associations |
| **ClinicalTrials.gov** | Active clinical trials by condition/intervention |
| **gnomAD** | Population allele frequencies |
| **AlphaGenome** | AI-predicted variant functional effects |

**Links:**
- Website: https://biomcp.org/
- GitHub: https://github.com/genomoncology/biomcp
- Python SDK: https://biomcp.org/apis/python-sdk/

## Installation

```bash
pip install biomcp-python
```

## How SynthLab Uses BioMCP

### Current Integration

When generating SOAP notes with `use_biomcp=True`, SynthLab enriches genetic variants with clinical annotations:

```python
import synthlab as sl

# Load patient data
dataset = sl.load_multimodal_dataset(max_patients=1)
patient = dataset[0]

# Generate SOAP note with BioMCP enrichment
generator = sl.SOAPNoteGenerator(
    use_biomcp=True,  # Enable BioMCP variant annotation
    verbose=True,
)
soap_note = generator.generate(patient)
```

### What Gets Enriched

For each genetic variant (rsID) in the patient's genomic data, BioMCP provides:

```
**rs699 (AGT)**
  - Clinical significance: risk factor [reviewed by expert panel]
  - Disease associations: Essential hypertension, Coronary artery disease
  - Effect: Decreased risk of CAD
  - Drug interactions: ACE inhibitors, ARBs
  - Predictions: CADD=12.5, PolyPhen=benign
  - Population freq (gnomAD): 0.4521
```

### Pipeline Flow

```
┌─────────────────┐
│  Patient Data   │
│  (FHIR + VCF)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract rsIDs   │
│ from genomics   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│    BioMCP API   │────▶│ ClinVar, gnomAD, │
│  variant_getter │     │ MyVariant.info   │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Enriched Data:  │
│ - Clinical sig  │
│ - Diseases      │
│ - Predictions   │
│ - Frequencies   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MedGemma LLM  │
│ Interprets in   │
│ clinical context│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SOAP Note     │
│ with genetic    │
│ interpretation  │
└─────────────────┘
```

## BioMCP API Reference

### Currently Used

#### `variant_getter(variant_id)`

Retrieves detailed annotations for a single variant.

```python
from biomcp.variants.get import variant_getter

result = await variant_getter(variant_id="rs699")

# Returns:
# - clinical_significance: "risk factor"
# - conditions: ["Essential hypertension", "Preeclampsia"]
# - gene: "AGT"
# - protein_change: "M268T"
# - frequencies: {"gnomad": 0.4521}
# - predictions: {"cadd": 12.5, "polyphen": "benign"}
```

### Available for Future Use

#### `search_variants(query)`

Search for variants by gene, significance, or region.

```python
from biomcp.variants.search import search_variants, VariantQuery, ClinicalSignificance

# Find all pathogenic BRCA1 variants
results = await search_variants(
    VariantQuery(
        gene="BRCA1",
        significance=ClinicalSignificance.PATHOGENIC
    )
)
```

#### `predict_variant_effects()`

AI-powered functional effect predictions using AlphaGenome.

```python
from biomcp.variants.alphagenome import predict_variant_effects

effects = await predict_variant_effects(
    chromosome="7",
    position=140453136,
    reference="A",
    alternate="T",
    tissue_types=["heart", "liver"]
)
```

#### `get_gene(gene_id)`

Get gene function, pathways, and expression data.

```python
from biomcp.genes import get_gene

gene_info = await get_gene("AGT")
# Returns: function, pathways, GO terms, expression patterns
```

#### `get_disease(disease_id)`

Get disease information including phenotypes and inheritance.

```python
from biomcp.diseases import get_disease

disease_info = await get_disease("essential hypertension")
# Returns: description, symptoms, inheritance, associated genes
```

#### `get_drug(drug_id)`

Get drug information for pharmacogenomics.

```python
from biomcp.drugs import get_drug

drug_info = await get_drug("warfarin")
# Returns: mechanism, interactions, pharmacogenomic variants
```

#### `search_articles(request)`

Search PubMed for relevant literature.

```python
from biomcp.articles.search import search_articles, PubmedRequest

articles = await search_articles(
    PubmedRequest(
        genes=["AGT", "ACE"],
        diseases=["hypertension"],
        max_results=10
    )
)
```

#### `search_trials(query)`

Search ClinicalTrials.gov for relevant trials.

```python
from biomcp.trials.search import search_trials, TrialQuery, RecruitingStatus

trials = await search_trials(
    TrialQuery(
        condition="hypertension",
        status=RecruitingStatus.OPEN,
        phase=["PHASE3", "PHASE4"]
    )
)
```

## Example: Full Variant Annotation

```python
import asyncio
from biomcp.variants.get import variant_getter
from biomcp.genes import get_gene
from biomcp.diseases import get_disease

async def annotate_variant_fully(rsid: str):
    """Get comprehensive variant annotation."""

    # 1. Get variant info
    variant = await variant_getter(variant_id=rsid)

    # 2. Get gene info
    if variant.get("gene"):
        gene = await get_gene(variant["gene"])

    # 3. Get disease info for each association
    diseases = []
    for condition in variant.get("conditions", []):
        disease = await get_disease(condition)
        diseases.append(disease)

    return {
        "variant": variant,
        "gene": gene,
        "diseases": diseases
    }

# Usage
result = asyncio.run(annotate_variant_fully("rs699"))
```

## Configuration

### Rate Limiting

BioMCP queries external APIs that may have rate limits. SynthLab limits queries to:
- Maximum 20 variants per patient (configurable via `max_variants` parameter)
- Prioritizes pathogenic/risk factor variants

### Caching

BioMCP responses are stored in the SOAP note for debugging:

```python
soap_note = generator.generate(patient)

# Access raw BioMCP annotations
print(soap_note.biomcp_annotations)
# {'rs699': {'clinical_significance': 'risk factor', ...}, ...}
```

## Troubleshooting

### BioMCP not available

```python
# Check if BioMCP is installed
import synthlab as sl
print(sl.utils.check_biomcp_available())
```

If not installed:
```bash
pip install biomcp-python
```

### No annotations returned

Possible causes:
1. Variant not in ClinVar/databases
2. Network connectivity issues
3. Rate limiting (try again later)

### Slow performance

BioMCP makes network requests for each variant. For large numbers of variants:
- Reduce `max_variants` parameter
- Use async batch processing (future enhancement)

## Future Enhancements

Planned improvements to BioMCP integration:

1. **Gene context** - Use `get_gene()` to add pathway/function information
2. **Literature citations** - Use `search_articles()` to cite evidence
3. **Clinical trials** - Use `search_trials()` to suggest relevant trials
4. **AI predictions** - Use `predict_variant_effects()` for novel variants
5. **Batch processing** - Parallel API calls for faster annotation
6. **Caching layer** - Cache responses to reduce API calls

## References

- BioMCP Documentation: https://biomcp.org/
- MyVariant.info: https://myvariant.info/
- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- gnomAD: https://gnomad.broadinstitute.org/
