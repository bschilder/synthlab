# Literature Review: Causal Graph Generation from Free Text

## Research Report for SynthLab

**Date:** January 2025
**Purpose:** Comprehensive literature review on methods for extracting and constructing causal graphs from unstructured text, with emphasis on medical and biomedical applications.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Foundational Concepts](#foundational-concepts)
3. [Traditional Approaches](#traditional-approaches)
   - [Knowledge-Based Methods](#knowledge-based-methods)
   - [Statistical Machine Learning](#statistical-machine-learning)
4. [Deep Learning Approaches](#deep-learning-approaches)
   - [Recurrent Neural Networks](#recurrent-neural-networks)
   - [Convolutional Neural Networks](#convolutional-neural-networks)
   - [Transformer-Based Models](#transformer-based-models)
   - [Graph Neural Networks](#graph-neural-networks)
5. [Large Language Models for Causal Discovery](#large-language-models-for-causal-discovery)
   - [LLMs as Meta-Experts](#llms-as-meta-experts)
   - [Multi-Agent Approaches](#multi-agent-approaches)
   - [RAG-Enhanced Causal Discovery](#rag-enhanced-causal-discovery)
6. [Medical and Biomedical Applications](#medical-and-biomedical-applications)
   - [Clinical Text Processing](#clinical-text-processing)
   - [Biomedical Knowledge Graphs](#biomedical-knowledge-graphs)
   - [Electronic Health Records](#electronic-health-records)
7. [Key Resources and Datasets](#key-resources-and-datasets)
8. [Benchmark Datasets](#benchmark-datasets)
9. [Evaluation Metrics and Challenges](#evaluation-metrics-and-challenges)
10. [Future Directions](#future-directions)
11. [References](#references)

---

## Introduction

Causal graph generation from free text is a fundamental task in natural language processing (NLP) that aims to automatically extract cause-effect relationships from unstructured text and represent them as directed graphs. This capability is critical for:

- Building scientific knowledge bases
- Supporting clinical decision-making
- Enabling hypothesis generation and literature-based discovery
- Powering predictive analytics in healthcare and other domains

As noted by [Xu et al. (2020)](https://www.sciencedirect.com/science/article/pii/S1532046421001490): "The identification of causal relationships between events or entities within biomedical texts is of great importance for creating scientific knowledge bases and is also a fundamental natural language processing (NLP) task."

This review synthesizes the current state of research across three main paradigms: traditional methods, deep learning approaches, and large language model-based techniques.

---

## Foundational Concepts

### What is a Causal Relation?

A causal (cause-effect) relation is defined as an association between two events where:
- The first event (cause) must occur before the second (effect)
- The cause is at least partially responsible for the effect
- The relationship implies directionality and temporal ordering

### Explicit vs. Implicit Causality

Causal relationships in text can be:

| Type | Description | Example |
|------|-------------|---------|
| **Explicit** | Marked by causal connectives | "Smoking *causes* lung cancer" |
| **Implicit** | No explicit markers present | "He smoked for 30 years. He developed lung cancer." |

Implicit causality is particularly challenging as it requires world knowledge and inference capabilities.

### The Causal Extraction Pipeline

Typical causality extraction involves three stages:

1. **Causal Sentence Detection**: Identify sentences containing causal relations
2. **Entity Recognition**: Extract cause and effect entities/events
3. **Relation Extraction**: Classify the causal relationship type and direction

---

## Traditional Approaches

### Knowledge-Based Methods

Early approaches relied on manually crafted rules and lexical patterns.

**Pattern Matching:**
- Use of causal connectives (because, therefore, causes, leads to)
- Syntactic patterns in dependency parses
- Domain-specific gazetteers and dictionaries

**Strengths:** Interpretable, high precision for explicit causality
**Weaknesses:** Poor recall, require extensive manual effort, limited cross-domain applicability

According to [Asghar (2016)](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/survey-of-the-extraction-and-applications-of-causal-relations/8B43AC51BE1F0B53B82DA99997DBC7E6), pattern-based systems struggled with implicit causal expressions and required significant domain expertise to develop.

### Statistical Machine Learning

Feature-engineered approaches using traditional ML algorithms:

- **Feature Types:** Lexical (n-grams, POS tags), syntactic (dependency paths), semantic (WordNet relations)
- **Classifiers:** SVM, Random Forest, Maximum Entropy
- **Representation:** Bag-of-words, TF-IDF weighted features

**Limitations:**
- Labor-intensive feature engineering
- Error propagation from NLP preprocessing pipelines
- Feature sparsity for rare causal expressions

---

## Deep Learning Approaches

The advent of deep learning transformed causal relation extraction by enabling automatic feature learning from raw text.

### Recurrent Neural Networks

**Bidirectional LSTMs (BiLSTM):**
- Capture long-range dependencies in both directions
- Often combined with attention mechanisms
- BiLSTM-CRF for sequence labeling of cause/effect spans

[Dasgupta et al. (2018)](https://arxiv.org/abs/2012.05453) demonstrated that attention-based BiLSTM models achieved significant improvements over traditional feature-based methods on biomedical causal extraction tasks.

### Convolutional Neural Networks

**Applications in Causal Extraction:**
- Multiview CNNs (MVC) for capturing different semantic perspectives
- Character-level and word-level convolutions
- Efficient parallel processing for sentence-level classification

### Transformer-Based Models

The introduction of BERT and its variants marked a paradigm shift.

#### BERT-Based Models

| Model | Description | Performance |
|-------|-------------|-------------|
| **BERT** | General-purpose pretrained model | Baseline for fine-tuning |
| **BioBERT** | Pretrained on biomedical literature | Best for medical text |
| **ClinicalBERT** | Pretrained on clinical notes (MIMIC-III) | Optimal for EHR data |
| **PubMedBERT** | Pretrained on PubMed abstracts | Strong for research text |

#### Causal BERT

[Li et al. (2020)](https://arxiv.org/abs/2012.05453) introduced Causal BERT, which investigates language model capabilities for causal association among events expressed in natural language. The model uses sentence context combined with event information, achieving state-of-the-art performance across three different data distributions.

#### CausalBERT (Knowledge Injection)

[Li & Ding (2021)](https://www.semanticscholar.org/paper/CausalBERT:-Injecting-Causal-Knowledge-Into-Models-Li-Ding/ff2f48fe6438adcaf860aac0f41c584568beafb5) proposed CausalBERT which injects causal knowledge into pre-trained models with minimal supervision. This approach captures rich causal knowledge and outperforms state-of-the-art methods on causal inference benchmarks.

#### DepBERT

[Recent work (2025)](https://arxiv.org/html/2507.09925v1) introduced DepBERT, which extends transformer-based models by incorporating the dependency tree structure within the model framework. On the CausalGPT dataset, DepBERT achieves 10.3% higher exact matching accuracy compared to baseline methods.

### Graph Neural Networks

[Job et al. (2025)](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024) provide a comprehensive review of graph neural networks for causal learning:

- **Graph Convolutional Networks (GCNs):** Model entity relationships in knowledge graphs
- **Graph Attention Networks (GATs):** Learn weighted importance of neighboring nodes
- **Hierarchical Graph Networks:** Capture multi-level causal structures

[Yang et al. (2023)](https://dl.acm.org/doi/abs/10.1007/s00500-023-08882-7) developed a knowledge-guided hierarchical graph network specifically for biomedical event causal relation extraction, leveraging domain knowledge to improve relation classification.

---

## Large Language Models for Causal Discovery

### LLMs as Meta-Experts

[Kıcıman et al. (2024)](https://arxiv.org/abs/2402.11068) provide a comprehensive survey on LLMs for causal discovery, identifying three primary integration paradigms:

1. **Direct Causal Extraction:** LLMs infer causal graphs directly from natural language descriptions
2. **Posterior Correction:** LLMs validate and refine causal relationships identified by statistical methods
3. **Prior Information Sources:** LLMs provide domain knowledge and constraints for traditional algorithms

As noted in their survey: "Large Language Models offer a transformative solution to the challenges of causal discovery, potentially acting as scalable and generalized 'meta-experts.' Their ability to process and synthesize massive amounts of text—effectively distilling knowledge from countless documents, research papers, and expert opinions—makes them powerful tools for automating expert-level reasoning."

### LLM-CD Framework

[Ban et al. (2024)](https://dl.acm.org/doi/10.1145/3711896.3736874) introduced LLM-CD, a novel causal modeling framework that:

- Integrates metadata-based reasoning capabilities of LLMs with data-driven modeling
- Couples LLM reasoning at various stages of traditional causal discovery algorithms
- Quantifies uncertainty using evidence-based deep learning theory to address hallucination

### Multi-Agent Approaches

[Huang et al. (2024)](https://arxiv.org/html/2407.15073v1) explored multi-agent LLM capabilities for causal discovery:

| Model Type | Description |
|------------|-------------|
| **Meta Agents Model** | Relies on reasoning and discussions among LLM agents |
| **Coding Agents Model** | Leverages code execution and statistical libraries |
| **Hybrid Model** | Integrates both meta reasoning and statistical methods |

These frameworks effectively utilize LLMs' expert knowledge, reasoning capabilities, and multi-agent cooperation combined with statistical causal methods.

### RAG-Enhanced Causal Discovery

#### CausalRAG

[Chen et al. (2025)](https://arxiv.org/html/2503.19878v2) proposed CausalRAG, which integrates causal graphs into retrieval-augmented generation. The system's path-based expansion integrates both entities and their causal explanations, yielding retrieval sets that are broader in coverage and more tightly aligned with causal queries.

#### RAG-Based Literature Mining

[Ban et al. (2024)](https://arxiv.org/html/2402.15301v1) developed methods using RAG-based LLMs to:

1. Retrieve relevant text chunks from aggregated scientific literature
2. Identify and label potential causal associations between factors
3. Construct causal graphs from extracted relationships

### Causal Inference Augmented LLMs

[Guo et al. (2025)](https://aclanthology.org/2025.findings-naacl.327.pdf) provide a comprehensive survey on causal inference with large language models, covering:

- Causal reasoning capabilities of LLMs
- Integration of causal inference principles into LLM architectures
- Evaluation benchmarks including CLadder (NeurIPS 2023) and CausalBench

---

## Medical and Biomedical Applications

### Clinical Text Processing

#### Challenges with Medical Text

Medical text presents unique challenges for causal extraction:

- Dense abbreviations and acronyms
- Domain-specific terminology
- Implicit temporal relationships
- Complex multi-factorial causation
- Privacy and access constraints

#### Causality Extraction from Clinical Practice Guidelines

[Groza et al. (2024)](https://www.mdpi.com/2078-2489/16/1/13) explored causality extraction from medical text using LLMs, specifically focusing on clinical practice guidelines (CPGs). Key findings:

- BioBERT outperformed other models including GPT-4 with an average F1-score of 0.72
- GPT-4 showed issues with hallucination for token-level predictions
- Fine-tuned smaller models often outperformed larger zero-shot LLMs

#### GatorTron

[Yang et al. (2022)](https://www.nature.com/articles/s41746-022-00742-2) developed GatorTron, a large language model for electronic health records trained on:

- Over 90 billion words of text
- Over 82 billion words of de-identified clinical text

The model was evaluated on five clinical NLP tasks including clinical concept extraction, medical relation extraction, and semantic textual similarity.

#### Specialized Medical NLP Tools

| Tool | Training Data | Best For |
|------|---------------|----------|
| **scispaCy** | Scientific/biomedical text | NER and entity linking |
| **BioBERT** | PubMed + PMC | Biomedical literature |
| **ClinicalBERT** | MIMIC-III clinical notes | EHR processing |
| **Med7** | Electronic health records | Clinical concept extraction |
| **GatorTron** | 90B+ words clinical text | Clinical NLP tasks |

### Biomedical Knowledge Graphs

#### SemMedDB

[Kilicoglu et al. (2012)](https://academic.oup.com/bioinformatics/article/28/23/3158/195282) created SemMedDB, a PubMed-scale repository of biomedical semantic predications:

- Contains ~98 million predications from 29+ million PubMed abstracts
- Extracts subject-predicate-object triples using SemRep
- Concepts mapped to UMLS Metathesaurus
- Predicates include CAUSES, TREATS, PREDISPOSES, etc.

#### SemRep

[Rindflesch & Fiszman (2003)](https://link.springer.com/article/10.1186/s12859-020-3517-7) developed SemRep, the underlying NLP system for SemMedDB:

- Rule-based semantic relation extraction
- Uses UMLS domain knowledge for normalization
- Performance on CDR benchmark: Precision 0.69, Recall 0.42, F1 0.52
- Enhanced versions using PubMedBERT achieve F1 0.70

#### iKraph

[Zhang et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10760044/) constructed iKraph, a large-scale biomedical knowledge graph using:

- NLP pipeline applied to all PubMed abstracts
- Causal knowledge graph with probabilistic semantic reasoning (PSR)
- Manual verification confirmed human annotator-level accuracy

#### Causal Knowledge Graphs for Clinical Decision Support

[Wang et al. (2023)](https://www.sciencedirect.com/science/article/pii/S1532046423000199) constructed a causal knowledge graph for diabetic nephropathy clinical decision support:

- Extracted causal triples from SemMedDB, UpToDate, and Churchill's Pocketbook
- Combined rule-based and language model methods
- Result: 153,289 concepts and 1,719,968 causal triples

### Electronic Health Records

#### Challenges with EHR Data

[Shen et al. (2021)](https://www.nature.com/articles/s41598-021-99990-7) identified key challenges for causal discovery from EHR:

- Latent confounding and implicit selection biases
- Systematic and random noise
- Longitudinal sparsity and incompleteness
- Data collected for clinical care, not research

#### Notable Applications

**Type 2 Diabetes Mellitus:**
[Shen et al. (2021)](https://www.nature.com/articles/s41598-021-99990-7) demonstrated causal structure discovery methodology using T2D as a clinical example, leveraging the extensive clinical trial knowledge base for validation.

**Heart Failure Prediction:**
[Recent work (2025)](https://arxiv.org/html/2506.03068v1) applied causal discovery to heart failure prediction, finding:
- Age as the most causal factor
- Body weight and serum glucose as secondary factors
- Blood urea nitrogen most affected by heart failure

**Acute Kidney Injury:**
[Bhatraju et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29589567/) used causal discovery methods (McDSL) to infer causal relationships between EHR features and Stage-3 AKI risk.

#### CausalMedLM

[Li et al. (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125021562) introduced CausalMedLM, which:

- Constructs symptom-disease causal graphs via causal discovery
- Translates structured medical knowledge into sequential representations
- Uses prompting frameworks to guide LLMs in causal reasoning

---

## Key Resources and Datasets

### Major Biomedical Resources

| Resource | Description | Access |
|----------|-------------|--------|
| [SemMedDB](https://ii.nlm.nih.gov/SemRep_SemMedDB_SKR/index.shtml_123119) | PubMed-scale semantic predications | UMLS license required |
| [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/) | Unified medical vocabulary | UMLS license required |
| [SNOMED CT](https://www.snomed.org/) | Clinical terminology standard | SNOMED license required |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Biomedical literature database | Free access |

### Code Repositories

| Repository | Purpose | Link |
|------------|---------|------|
| **CausalNLP_Papers** | Reading list for causality in NLP | [GitHub](https://github.com/zhijing-jin/CausalNLP_Papers) |
| **Awesome-Causal-LLM** | LLM + causality resources | [GitHub](https://github.com/anpwu/Awesome-Causal-LLM) |
| **SapBERT** | Biomedical entity embeddings | [GitHub](https://github.com/cambridgeltl/sapbert) |
| **MedCAT** | Clinical NER and entity linking | [GitHub](https://github.com/CogStack/MedCAT) |
| **OntoGPT** | Schema-driven extraction | [GitHub](https://github.com/monarch-initiative/ontogpt) |

---

## Benchmark Datasets

### General Causal Extraction

| Dataset | Domain | Size | Task |
|---------|--------|------|------|
| **SemEval-2010 Task 8** | General | 10,717 sentences | Multi-way relation classification |
| **BECAUSE Corpus 2.0** | General | ~1000 sentences | Causal and overlapping relations |
| **CausalGPT** | General | Varies | Cause-effect pair extraction |
| **CLadder** | General | Multi-rung | Causal reasoning evaluation |
| **CausalBench** | General | Comprehensive | LLM causal reasoning |

### Biomedical Specific

| Dataset | Domain | Description |
|---------|--------|-------------|
| [MediCause](https://ceur-ws.org/Vol-3184/TEXT2KG_Paper_1.pdf) | Medical | 1,202 causal sentences from medical publications |
| **BioCreative-V Track 4** | Biomedical | BEL statement extraction |
| **CDR (Chemical-Disease Relations)** | Biomedical | Standard benchmark for chemical-disease causality |
| **SCITE** | Scientific | Citation-based causal claims |

### MediCause Dataset

[Groza et al. (2022)](https://ceur-ws.org/Vol-3184/TEXT2KG_Paper_1.pdf) created MediCause, providing:

- Ontological model for causal entities in medical text
- 1,202 annotated causal sentences
- BioBERT-large achieved best F1-score of 0.844 for entity recognition

---

## Evaluation Metrics and Challenges

### Common Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Precision** | Correct extractions / Total extractions | Quality of predictions |
| **Recall** | Correct extractions / Total true relations | Coverage of true relations |
| **F1-Score** | Harmonic mean of P and R | Balanced performance |
| **mIoU** | Mean Intersection over Union | Entity span matching |
| **Exact Match** | Perfect span + relation match | Strict evaluation |

### Key Challenges

1. **Distinguishing Correlation from Causation**
   - Neural networks may learn spurious correlations
   - Requires domain knowledge for validation

2. **Implicit Causality**
   - No explicit markers in text
   - Requires commonsense reasoning

3. **LLM Hallucination**
   - Models may generate non-existent causal relationships
   - Particularly problematic in medical domains

4. **Data Scarcity**
   - Limited annotated causal relation datasets
   - Expensive expert annotation required

5. **Cross-Domain Generalization**
   - Models trained on one domain may fail on others
   - Domain adaptation remains challenging

---

## Future Directions

### Emerging Research Areas

1. **Hybrid Neuro-Symbolic Approaches**
   - Combining neural extraction with symbolic reasoning
   - Grounding extracted relations in formal ontologies

2. **Causal LLM Agents**
   - Multi-agent systems for causal discovery
   - Tool-augmented LLMs for causal analysis

3. **Multimodal Causal Discovery**
   - Integrating text with images, structured data
   - Cross-modal causal inference

4. **Uncertainty Quantification**
   - Confidence estimation for extracted relations
   - Calibrated predictions for clinical use

5. **Interactive Causal Graph Construction**
   - Human-in-the-loop refinement
   - Expert validation workflows

### Research Gaps

- Limited work on implicit causality extraction
- Need for larger, diverse medical causal datasets
- Insufficient evaluation of real-world clinical impact
- Lack of temporal causal modeling in text

---

## References

### Survey Papers

1. **Yang et al. (2022).** "A Survey on Extraction of Causal Relations from Natural Language Text." *Knowledge and Information Systems*. [Springer](https://link.springer.com/article/10.1007/s10115-022-01665-w) | [arXiv](https://arxiv.org/abs/2101.06426)

2. **Xu et al. (2021).** "Causal relationship extraction from biomedical text using deep neural models: A comprehensive survey." *Journal of Biomedical Informatics*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046421001490) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34044157/)

3. **Kıcıman et al. (2024).** "Large Language Models for Causal Discovery: Current Landscape and Future Directions." *IJCAI 2025*. [arXiv](https://arxiv.org/abs/2402.11068) | [PDF](https://www.ijcai.org/proceedings/2025/1186.pdf)

4. **Jin (2024).** "Causality for Natural Language Processing." [arXiv](https://arxiv.org/pdf/2504.14530)

5. **Guo et al. (2025).** "Causal Inference with Large Language Model: A Survey." *NAACL Findings*. [ACL Anthology](https://aclanthology.org/2025.findings-naacl.327.pdf)

### Key Research Papers

6. **Li et al. (2020).** "Causal BERT: Language Models for Causality Detection Between Events Expressed in Text." *LNCS*. [Springer](https://link.springer.com/chapter/10.1007/978-3-030-80119-9_64) | [arXiv](https://arxiv.org/abs/2012.05453)

7. **Li & Ding (2021).** "CausalBERT: Injecting Causal Knowledge Into Pre-trained Models with Minimal Supervision." [Semantic Scholar](https://www.semanticscholar.org/paper/CausalBERT:-Injecting-Causal-Knowledge-Into-Models-Li-Ding/ff2f48fe6438adcaf860aac0f41c584568beafb5)

8. **Chen et al. (2025).** "CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation." *ACL Findings*. [arXiv](https://arxiv.org/html/2503.19878v2) | [ACL Anthology](https://aclanthology.org/2025.findings-acl.1165.pdf)

9. **Ban et al. (2024).** "Causal Discovery through Synergizing Large Language Model and Data-Driven Reasoning." *KDD*. [ACM](https://dl.acm.org/doi/10.1145/3711896.3736874)

10. **Huang et al. (2024).** "Multi-Agent Causal Discovery Using Large Language Models." [arXiv](https://arxiv.org/html/2407.15073v1)

### Medical/Biomedical Applications

11. **Groza et al. (2024).** "Causality Extraction from Medical Text Using Large Language Models (LLMs)." *Information*. [MDPI](https://www.mdpi.com/2078-2489/16/1/13) | [arXiv](https://arxiv.org/html/2407.10020v1)

12. **Kilicoglu et al. (2012).** "SemMedDB: a PubMed-scale repository of biomedical semantic predications." *Bioinformatics*. [Oxford Academic](https://academic.oup.com/bioinformatics/article/28/23/3158/195282) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3509487/)

13. **Rindflesch & Fiszman (2020).** "Broad-coverage biomedical relation extraction with SemRep." *BMC Bioinformatics*. [Springer](https://link.springer.com/article/10.1186/s12859-020-3517-7) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7222583/)

14. **Yang et al. (2022).** "A large language model for electronic health records." *npj Digital Medicine*. [Nature](https://www.nature.com/articles/s41746-022-00742-2)

15. **Wang et al. (2023).** "Causal knowledge graph construction and evaluation for clinical decision support of diabetic nephropathy." *JBI*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046423000199) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36731730/)

16. **Zhang et al. (2023).** "A comprehensive large scale biomedical knowledge graph for AI powered data driven biomedical research." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10760044/)

17. **Shen et al. (2021).** "A novel method for causal structure discovery from EHR data and its application to type-2 diabetes mellitus." *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-021-99990-7) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34697394/)

### Knowledge Graphs and Causal Inference

18. **McSparron et al. (2025).** "Causal knowledge graph analysis identifies adverse drug effects." *Bioinformatics*. [Oxford Academic](https://academic.oup.com/bioinformatics/article/42/1/btaf661/8378293)

19. **Shen et al. (2023).** "Causal feature selection using a knowledge graph combining structured knowledge from the biomedical literature and ontologies." *JBI*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1532046423000898) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10355339/)

20. **Job et al. (2025).** "Exploring Causal Learning Through Graph Neural Networks: An In-Depth Review." *WIREs Data Mining and Knowledge Discovery*. [Wiley](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70024)

### Datasets and Benchmarks

21. **Groza et al. (2022).** "MediCause: Causal Relation Modelling and Extraction from Medical Publications." *TEXT2KG Workshop*. [CEUR-WS](https://ceur-ws.org/Vol-3184/TEXT2KG_Paper_1.pdf)

22. **Zhang et al. (2021).** "Extraction of causal relations based on SBEL and BERT model." *Database*. [Oxford Academic](https://academic.oup.com/database/article/doi/10.1093/database/baab005/6133143) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7904051/)

23. **Rodrigues et al. (2024).** "An Empirical Study of Causal Relation Extraction Transfer: Design and Data." [arXiv](https://arxiv.org/html/2503.06076)

### Additional Resources

24. **Asghar (2016).** "A survey of the extraction and applications of causal relations." *Natural Language Engineering*. [Cambridge](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/survey-of-the-extraction-and-applications-of-causal-relations/8B43AC51BE1F0B53B82DA99997DBC7E6)

25. **Yang et al. (2023).** "Biomedical event causal relation extraction based on a knowledge-guided hierarchical graph network." *Soft Computing*. [Springer](https://dl.acm.org/doi/abs/10.1007/s00500-023-08882-7)

26. **Moradi et al. (2024).** "Evaluating the ChatGPT family of models for biomedical reasoning and classification." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10990500/)

---

## Appendix: Curated Reading Lists

### Essential Papers for Beginners

1. [Yang et al. (2022)](https://arxiv.org/abs/2101.06426) - Comprehensive survey on causal extraction from text
2. [Xu et al. (2021)](https://pubmed.ncbi.nlm.nih.gov/34044157/) - Deep learning for biomedical causal extraction
3. [Kıcıman et al. (2024)](https://arxiv.org/abs/2402.11068) - LLMs for causal discovery survey

### For Medical/Clinical Applications

1. [Groza et al. (2024)](https://www.mdpi.com/2078-2489/16/1/13) - LLMs for medical causality extraction
2. [Kilicoglu et al. (2012)](https://academic.oup.com/bioinformatics/article/28/23/3158/195282) - SemMedDB introduction
3. [Wang et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36731730/) - Clinical causal knowledge graphs

### For LLM Practitioners

1. [Chen et al. (2025)](https://arxiv.org/html/2503.19878v2) - CausalRAG framework
2. [Huang et al. (2024)](https://arxiv.org/html/2407.15073v1) - Multi-agent causal discovery
3. [Ban et al. (2024)](https://dl.acm.org/doi/10.1145/3711896.3736874) - LLM-CD framework

---

*Last updated: January 2025*
