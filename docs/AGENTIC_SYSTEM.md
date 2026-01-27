# Agentic System Design (SynthLab + MedGemma)

This document defines a modular, agentic pipeline that uses SynthLab for data access/evaluation and MedGemma for plan generation. The system is offline-safe by default and treats BioMCP as a validator and evidence supplier.

## Repository roles

### SynthLab (data + evaluation)
- Synthetic data generation and loaders (clinical notes, genomics, imaging, EHR timeline)
- Standard schemas for outputs/inputs
- BioMCP integration as external evidence/validation (no diagnosis claims)
- Evaluation harness using timestamped phenotype timelines

### medgemma_cup (agents + models)
- Agent orchestration and model prompts (MedGemma)
- Multi-modal fusion logic (text/genomics/imaging)
- Safety gates, re-ranking, and verification
- User-facing output assembly (SOAP + simulated EHR + diagnostic priority)

## Boundary contract
Shared schema lives at:
- `/home/daweilin/medagent/schemas/medagent_schema.py`
- `/home/daweilin/medagent/schemas/medagent_schema.json`

Key payloads:
- `PatientContext` (notes + genotype card + imaging summary + optional timeline)
- `PlanOutput` (SOAP + EHR summary + diagnostic steps + evidence + provenance)

## Pipeline stages
1) **Ingest + Normalize**
   - Pull SynthLab notes, genomic card, imaging summary
   - Snapshot provenance (patient id, timestamps, dataset source)

2) **Signal Extraction (modular)**
   - Notes agent: symptoms, red flags, constraints
   - Genomics agent: variant hints (supporting evidence only)
   - Imaging agent: key findings (text summary or caption)

3) **Planner (MedGemma)**
   - Generate SOAP + simulated EHR summary
   - Propose candidate diagnostic steps (ranked)

4) **Evidence / Validation (BioMCP)**
   - Fetch guideline/literature snippets
   - Attach evidence to plan items
   - If evidence missing, mark as hypothesis-only

5) **Safety Gate + Re-ranker**
   - Remove contraindicated steps
   - Re-rank based on evidence coverage + constraints

6) **Verifier**
   - Enforce uncertainty language
   - Ensure each recommendation has rationale + evidence or is flagged

7) **Final Assembly**
   - PlanOutput with audit trail and provenance

## Evaluation strategy
Use SynthLab timestamped phenotype timelines to score:
- Recall@K for expected next tests
- Time-to-recommendation vs phenotype onset
- Evidence coverage ratio (evidence-backed items / total)
- Safety gate accuracy on contraindication cases

## Guardrails
- No direct diagnosis claims; decision support only
- Genomics used as supporting evidence, never primary diagnosis
- Explicit missing-info questions required
- Evidence required for genotype-driven suggestions

## Suggested engineering patterns
- Keep modules swappable via small interfaces
- Make BioMCP optional but enforce evidence gating when present
- Maintain offline-safe tests (no data downloads in repo)
