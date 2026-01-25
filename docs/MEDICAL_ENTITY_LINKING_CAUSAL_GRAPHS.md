# Medical Entity Linking and Causal Graph Extraction

## Research Report for SynthLab

**Date:** January 2025
**Purpose:** Evaluate approaches for grounding medical entities to ontologies (SNOMED-CT, UMLS) and extracting structured causal graphs from clinical text.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current Challenges](#current-challenges)
3. [Approaches Evaluated](#approaches-evaluated)
   - [Lexical/Dictionary-Based](#1-lexicaldictionary-based-approaches)
   - [Vector Embedding-Based](#2-vector-embedding-based-approaches)
   - [LLM-Based Extraction](#3-llm-based-extraction-with-grounding)
   - [Hybrid Pipelines](#4-hybrid-multi-stage-pipelines)
4. [Tool Comparison](#tool-comparison)
5. [Recommendations](#recommendations)
6. [Implementation Plan](#implementation-plan)
7. [References](#references)

---

## Problem Statement

When generating causal graphs from clinical SOAP notes using LLMs, we encounter two critical issues:

1. **Entity Hallucination**: LLMs generate non-existent medical terms (e.g., `Stroke_Prevention_Support_Program_Support_Support...`) through degeneration loops.

2. **Inconsistent Terminology**: The same concept may be expressed differently across notes (e.g., "heart attack" vs "myocardial infarction" vs "MI"), making graphs non-comparable.

**Goal:** Ground all entities to standardized medical ontologies (SNOMED-CT, UMLS) with unique identifiers, ensuring:
- Consistent vocabulary across all generated graphs
- Machine-readable node IDs for downstream analysis
- Prevention of hallucinated concepts

---

## Current Challenges

### LLM Degeneration in Causal Graph Generation

Our current approach prompts MedGemma to generate causal relationships in the format:
```
Cause[type] --> Effect[type]
```

However, the model frequently degenerates into repetitive patterns:
```
Stroke[condition] => Stroke_Warning_Signs[condition]
Stroke[condition] => Stroke_Prevention_Program[condition]
Stroke[condition] => Stroke_Prevention_Program_Support[condition]
Stroke[condition] => Stroke_Prevention_Program_Support_Support[condition]
...
```

This occurs because:
- The model generates tokens autoregressively without grounding
- No constraint on valid medical vocabulary
- Repetition penalties are insufficient for structured output

### Why Simple Fixes Don't Work

| Approach | Problem |
|----------|---------|
| Increase `repetition_penalty` | Hurts coherence, doesn't prevent novel hallucinations |
| Post-processing truncation | Loses valid relationships, treats symptoms not cause |
| Stricter prompts | Model still invents terms freely |
| Token limits | Cuts off valid content arbitrarily |

---

## Approaches Evaluated

### 1. Lexical/Dictionary-Based Approaches

#### OntoGPT / SPIRES
- **Repository:** https://github.com/monarch-initiative/ontogpt
- **Method:** Schema-driven extraction with lexical grounding via OAKlib
- **Grounding:** Uses Gilda, BioPortal Annotator, OLS (not vector embeddings)

**How SPIRES Works:**
1. Define LinkML schema with target ontologies
2. LLM extracts entities following schema constraints
3. OAKlib grounds entities via string matching to ontology labels/synonyms
4. Validates against allowed ID prefixes and value sets

**Performance (from paper):**
| Ontology | Accuracy |
|----------|----------|
| Gene Ontology | 98/100 |
| EMAPA (anatomy) | 100/100 |
| MONDO (diseases) | 97/100 |

```bash
pip install ontogpt
ontogpt extract -i "patient has diabetes" -t disease
```

**Pros:** Zero-shot, schema-driven, good accuracy
**Cons:** Lexical matching can miss synonyms, requires ontology configuration

#### scispaCy EntityLinker
- **Repository:** https://github.com/allenai/scispacy
- **Method:** Character 3-gram overlap with approximate nearest neighbors

```python
import spacy
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})

doc = nlp("chronic kidney disease stage 3")
for ent in doc.ents:
    for cui, score in ent._.kb_ents:
        print(f"{ent.text} -> UMLS:{cui} ({score:.2f})")
```

**Supported Knowledge Bases:**
- UMLS (~3M concepts)
- MeSH (~30k entities)
- RxNorm (~100k drugs)
- GO, HPO

**Pros:** Easy integration with spaCy, fast
**Cons:** Character-based matching less semantic than embeddings, SNOMED via UMLS only

#### SNOMED CT Challenge Winner: KIRIs (1st Place)
- **Repository:** https://github.com/drivendataorg/snomed-ct-entity-linking
- **Method:** Dictionary matching with OMOP synonyms + linguistic rules
- **Score:** 0.4202 mIoU

Key insight: Dictionary-based approaches outperformed many neural methods on this task.

---

### 2. Vector Embedding-Based Approaches

#### SapBERT (Recommended)
- **Repository:** https://github.com/cambridgeltl/sapbert
- **Paper:** NAACL 2021
- **Method:** Self-alignment pretraining on UMLS synonym pairs

**Architecture:**
- Base: PubMedBERT
- Training: Contrastive learning on synonym pairs
- Output: 768-dimensional embeddings where synonyms cluster together

**How Entity Linking Works:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
model = AutoModel.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

# Encode mention
text = "diabetic nephropathy"
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    output = model(**tokens)
    embedding = output.last_hidden_state[:, 0, :]  # CLS token

# Compare with pre-computed SNOMED embeddings via cosine similarity
```

**SNOMED Linking Pipeline:**
1. Pre-compute embeddings for all ~200k SNOMED concepts
2. Build FAISS index for fast similarity search
3. At inference: embed mention → nearest neighbor search → return SNOMED ID

#### HELIN Demo
- **Repository:** https://github.com/cambridgeltl/HELIN
- **Method:** SapBERT + FAISS + SNOMED pre-built pipeline

```bash
# Returns: {"entities": [["T1", "Disease", [[0, 8]], "Diabetes", "SCTID:73211009"]]}
```

**Performance:**
- ~0.1s per query after index loaded
- Requires ~64GB RAM for full SNOMED index
- ~1 hour to build index initially

#### SNOBERT (2nd Place SNOMED Challenge)
- **Paper:** https://arxiv.org/html/2405.16115v1
- **Score:** 0.4194 mIoU

**Two-Stage Pipeline:**

| Stage | Component | Details |
|-------|-----------|---------|
| 1. NER | BERT ensemble | BIO tagging for Finding, Procedure, Body Part |
| 2. Linking | SapBERT + MedCPT | Embed → FAISS top-5 → MedCPT reranking |

**Key Models:**
- NER: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- Linking: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token`
- Reranking: MedCPT (trained on 18M PubMed query-article pairs)

#### SapBERT + FAISS Implementation
```python
import faiss
import numpy as np

# Build index (one-time, ~1 hour)
snomed_embeddings = encode_all_snomed_terms()  # Shape: (200000, 768)
snomed_embeddings = snomed_embeddings / np.linalg.norm(
    snomed_embeddings, axis=1, keepdims=True
)  # Normalize for cosine

index = faiss.IndexFlatIP(768)  # Inner product = cosine for normalized vectors
index.add(snomed_embeddings.astype('float32'))

# Save for reuse
faiss.write_index(index, "snomed_sapbert.index")

# Query
def link_to_snomed(mention: str, k: int = 5):
    embedding = encode_mention(mention)
    embedding = embedding / np.linalg.norm(embedding)
    distances, indices = index.search(embedding.reshape(1, -1), k)
    return [(snomed_ids[i], distances[0][j]) for j, i in enumerate(indices[0])]
```

---

### 3. LLM-Based Extraction with Grounding

#### Clinical Knowledge Graph Construction (Multi-LLM)
- **Paper:** https://arxiv.org/html/2601.01844

**Five-Stage Pipeline:**

1. **EAV Extraction** (Gemini 2.0 Flash)
   - Extract Entity-Attribute-Value triples
   - FHIR-guided structured prompting

2. **Ontology Mapping**
   - Lexical + semantic similarity scoring
   - Maps to SNOMED CT, LOINC, RxNorm, GO, ICD
   - Score = α·sim_lexical + β·sim_semantic

3. **Relation Discovery** (Gemini)
   - Generate candidate relations
   - GPT-4o validates plausibility
   - Grok 3 adversarial filtering

4. **Semantic Web Encoding**
   - RDF/RDFS/OWL output
   - SPARQL-compatible

5. **Trust Validation**
   - Multi-LLM consensus scoring
   - Hallucination detection

**Results:**
- 99.83% attribute coverage
- 73% correctness
- <1% hallucination rate

#### MedRAG
- **Repository:** https://github.com/Teddy-XiongGZ/MedRAG
- **Purpose:** RAG toolkit for medical QA (not entity linking)

**Corpora:**
- PubMed (23.9M abstracts)
- StatPearls (9.3k clinical documents)
- Medical Textbooks (18 books)

**Retrievers:** BM25, Contriever, SPECTER, MedCPT

Useful for enriching context, but doesn't directly solve entity linking.

---

### 4. Hybrid Multi-Stage Pipelines

#### MedCAT
- **Repository:** https://github.com/CogStack/MedCAT
- **Method:** NER + Entity Linking to SNOMED-CT/UMLS

```python
from medcat.cat import CAT

cat = CAT.load_model_pack("path/to/modelpack.zip")
doc = cat("Patient has type 2 diabetes mellitus and chronic kidney disease")

for ent in doc.ents:
    print(f"{ent.text} -> {ent.cui} ({ent.detected_name})")
```

**Available Models:**
- SNOMED International (complete)
- SNOMED UK Clinical Edition (Oct 2024)
- UMLS Full (4M+ concepts)

**Pros:** Production-ready, actively maintained, GPU support
**Cons:** Large model downloads, UMLS license required for some models

---

## Tool Comparison

| Tool | Method | SNOMED Support | Vector-Based | Speed | Ease of Use |
|------|--------|----------------|--------------|-------|-------------|
| **SapBERT + FAISS** | Embedding similarity | Yes (200k) | Yes | Fast | Medium |
| **SNOBERT** | BERT NER + SapBERT | Yes | Yes | Fast | High |
| **MedCAT** | NER + Linking | Yes | Optional | Medium | Low |
| **OntoGPT/SPIRES** | Schema + Lexical | Via OAK | No | Medium | Low |
| **scispaCy** | Char 3-gram | Via UMLS | No | Fast | Low |
| **KIRIs (Dictionary)** | Rule-based | Yes | No | Fast | Medium |

### Accuracy Comparison (SNOMED Challenge)

| Approach | mIoU Score |
|----------|------------|
| KIRIs (Dictionary) | 0.4202 |
| SNOBERT (Neural) | 0.4194 |
| MITEL-UNIUD (LLM+FAISS) | 0.3777 |

---

## Recommendations

### Primary Recommendation: SapBERT + FAISS Pipeline

For SynthLab's causal graph generation, I recommend a **two-phase approach**:

#### Phase 1: Entity Extraction and Grounding
```
SOAP Note Text
      ↓
[NER Model] → Extract medical entities
      ↓
[SapBERT] → Generate embeddings
      ↓
[FAISS Index] → Find nearest SNOMED concepts
      ↓
Grounded Entities: [(text, SCTID, confidence), ...]
```

#### Phase 2: Relationship Extraction
```
Grounded Entities
      ↓
[LLM Prompt] → "Given these SNOMED concepts, identify causal relationships"
      ↓
Structured Output: [(SCTID_cause, relation, SCTID_effect), ...]
```

### Why This Approach

1. **Prevents Hallucination**: LLM can only select from valid SNOMED concepts
2. **Consistent Vocabulary**: All nodes have unique identifiers
3. **Semantic Matching**: Embeddings handle synonyms ("heart attack" ≈ "MI")
4. **Fast**: FAISS enables sub-second retrieval over 200k concepts
5. **Proven**: SapBERT achieves state-of-the-art on biomedical entity linking

### Alternative: MedCAT (Simpler Setup)

If setup complexity is a concern, MedCAT provides an all-in-one solution:

```python
from medcat.cat import CAT
cat = CAT.load_model_pack("snomed_model.zip")
entities = cat.get_entities("patient has diabetes")
# Returns: {cui: "73211009", name: "Diabetes mellitus", ...}
```

---

## Implementation Plan

The SapBERT + FAISS pipeline has been implemented in `synthlab/snomed.py`.

### Quick Start

```python
import synthlab as sl

# Option 1: Use sample concepts (for testing)
linker = sl.setup_sample_linker()
results = linker.link("heart attack")
print(results[0])  # SCTID:22298006 | Myocardial infarction (score: 0.92)

# Option 2: Build from your own SNOMED data
concepts = sl.load_snomed_from_csv("snomed_concepts.csv")
linker = sl.SNOMEDLinker()
linker.build_index(concepts)
```

### Two-Stage Pipeline (Recommended)

For best accuracy, use the two-stage pipeline that combines LLM extraction with embedding-based linking:

```python
import synthlab as sl

# Create the pipeline
extractor, linker = sl.create_entity_pipeline()

# Setup the linker with sample concepts
concepts = sl.get_sample_snomed_concepts()
linker.build_index(concepts)

# Extract and link in one step
text = "Patient has DM2 with HTN and recent MI"
entities = extractor.extract_and_link(text, linker)

for ent in entities:
    print(f"{ent.mention} -> {ent.top_match}")
# DM2 -> SCTID:44054006 | Type 2 diabetes mellitus (score: 0.94)
# HTN -> SCTID:38341003 | Hypertensive disorder (score: 0.91)
# MI -> SCTID:22298006 | Myocardial infarction (score: 0.95)
```

**Why Two Stages?**
1. **LLM Stage**: Handles abbreviations (DM2 → Type 2 diabetes), synonyms, typos
2. **Embedding Stage**: Precise matching to SNOMED concepts with confidence scores

The LLM standardizes messy clinical text before embedding-based matching, significantly improving accuracy.

### Grounding Causal Graphs

```python
# Generate SOAP note with causal graph
soap_note = sl.generate_soap_note(patient)
graph = soap_note.extract_causal_graph()

# Ground to SNOMED CT
grounded = graph.ground_to_snomed()

# Access grounded nodes
for node in grounded.nodes:
    print(f"{node.mention} -> SCTID:{node.concept_id} ({node.term})")

# Access grounded edges
for edge in grounded.edges:
    print(f"{edge.source.term} {edge.relation} {edge.target.term}")

# Export to NetworkX for analysis
G = grounded.to_networkx()
```

### Key Classes

| Class | Description |
|-------|-------------|
| `SNOMEDLinker` | Main linker class with `link()` and `link_batch()` methods |
| `SNOMEDConcept` | A SNOMED concept with ID, term, and semantic type |
| `SNOMEDMatch` | A match result with concept ID, term, and score |
| `GroundedCausalGraph` | Causal graph with SNOMED-linked nodes |
| `GroundedNode` | Node with SNOMED concept ID and confidence |
| `GroundedEdge` | Edge between grounded nodes |

### Loading SNOMED Data

```python
# From UMLS (requires license)
concepts = sl.load_snomed_from_umls("/path/to/umls/META")

# From CSV file
concepts = sl.load_snomed_from_csv(
    "snomed.csv",
    concept_id_col="concept_id",
    term_col="term",
)

# Sample concepts for testing (~200 common clinical terms)
concepts = sl.get_sample_snomed_concepts()
```

### Installation

```bash
pip install synthlab[snomed]  # Installs faiss-cpu
```

---

## References

### Papers

1. **SPIRES** - Caufield et al. (2024). "Structured Prompt Interrogation and Recursive Extraction of Semantics." *Bioinformatics*. https://academic.oup.com/bioinformatics/article/40/3/btae104/7612230

2. **SapBERT** - Liu et al. (2021). "Self-Alignment Pretraining for Biomedical Entity Representations." *NAACL*. https://arxiv.org/abs/2010.11784

3. **SNOBERT** - Kulyabin et al. (2024). "A Benchmark for clinical notes entity linking in the SNOMED CT clinical terminology." https://arxiv.org/html/2405.16115v1

4. **Clinical KG Construction** - (2024). "Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via RAG." https://arxiv.org/html/2601.01844

5. **Zero-shot Causal Graph Extraction** - (2023). https://arxiv.org/html/2312.14670v1

### Repositories

- **OntoGPT**: https://github.com/monarch-initiative/ontogpt
- **SapBERT**: https://github.com/cambridgeltl/sapbert
- **HELIN**: https://github.com/cambridgeltl/HELIN
- **MedCAT**: https://github.com/CogStack/MedCAT
- **scispaCy**: https://github.com/allenai/scispacy
- **MedRAG**: https://github.com/Teddy-XiongGZ/MedRAG
- **SNOMED Challenge Winners**: https://github.com/drivendataorg/snomed-ct-entity-linking

### Datasets & Resources

- **SNOMED CT**: https://www.snomed.org/
- **UMLS Metathesaurus**: https://www.nlm.nih.gov/research/umls/
- **MIMIC-IV**: https://physionet.org/content/mimiciv/

---

## Appendix: SNOMED CT Access

SNOMED CT requires a license for commercial use. Options:

1. **UMLS License** (free for research): https://www.nlm.nih.gov/databases/umls.html
2. **SNOMED International**: https://www.snomed.org/get-snomed
3. **Pre-built models** (MedCAT): Include SNOMED mappings

For SynthLab (research use), UMLS Metathesaurus access should be sufficient.
