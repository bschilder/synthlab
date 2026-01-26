#!/usr/bin/env python3
"""
SOAP Note Generation using MedGemma.

This module provides an agentic workflow for generating SOAP notes from
FHIR patient records using Google's MedGemma multimodal medical AI model.

Features:
- Hierarchical summarization for long patient histories
- Multimodal support (FHIR + DICOM images)
- Structured SOAP note output
- Future-oriented clinical insights

SOAP Note Structure:
- Subjective: Patient-reported symptoms, complaints, history
- Objective: Clinical findings, vitals, labs, imaging
- Assessment: Diagnoses, clinical reasoning, risk factors
- Plan: Treatment, follow-up, preventive care, future considerations
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from synthlab.snomed import SNOMEDLinker, GroundedCausalGraph

# Import causal graph types from dedicated module
from synthlab.causal_graph import (
    CAUSAL_EDGE_TYPES,
    CausalNode,
    CausalEdge,
    CausalGraph,
    parse_causal_graph,
)

# Import count_tokens from utils module
from synthlab.utils import count_tokens

# Optional imports - checked at runtime
_transformers_available = False
_torch_available = False
_PIL_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    torch = None

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
    _transformers_available = True
except ImportError:
    AutoProcessor = None
    AutoModelForImageTextToText = None
    pipeline = None

try:
    from PIL import Image
    _PIL_available = True
except ImportError:
    Image = None

# Progress bar support
_tqdm_available = False
_tqdm = None
try:
    from tqdm.auto import tqdm as _tqdm
    _tqdm_available = True
except ImportError:
    pass

# BioMCP support for variant annotation (optional)
# Install with: pip install biomcp-python
# See: https://biomcp.org/apis/python-sdk/
_biomcp_available = False
_biomcp_variant_getter = None
_biomcp_search_variants = None
try:
    # New API path (biomcp-python >= 0.2)
    from biomcp.variants.getter import get_variant as _biomcp_get_variant
    _biomcp_available = True
except ImportError:
    try:
        # Old API path (biomcp-python < 0.2)
        from biomcp.variants.get import variant_getter as _biomcp_get_variant
        _biomcp_available = True
    except ImportError:
        _biomcp_get_variant = None
        _biomcp_available = False


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChunkSummary:
    """A single chunk/time-period summary from hierarchical processing."""
    period: str  # e.g., "2015-2020"
    start_year: str
    end_year: str
    prompt: str  # The prompt used to generate this chunk
    summary: str  # The model's response
    input_tokens: int = 0
    output_tokens: int = 0
    generation_seconds: float = 0.0


@dataclass
class SOAPNote:
    """
    Structured SOAP note output.

    Attributes:
        patient_id: Patient identifier
        patient_name: Patient name
        generated_at: Timestamp of generation
        patient_story: Narrative summary of patient's health journey
        subjective: Patient-reported information
        objective: Clinical findings and measurements
        assessment: Clinical reasoning and diagnoses (includes causal analysis)
        plan: Treatment and follow-up recommendations
        future_considerations: Predictive insights for preventive care
        summary: Brief executive summary
        raw_response: Full model response (for debugging)
        images_analyzed: Number of images included in analysis
        time_periods_summarized: Number of time chunks processed
        chunk_summaries: List of chunk-level summaries (for hierarchical mode)
        prompts_used: Dict of all prompts used during generation
        causal_graph_raw: Raw causal graph text (if separate_causal_graph enabled)
        biomcp_annotations: Raw BioMCP variant annotations (before LLM summarization)
    """

    patient_id: str
    patient_name: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Patient narrative
    patient_story: str = ""

    # SOAP components
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""

    # Extended components
    future_considerations: str = ""
    summary: str = ""
    causal_graph: str = ""  # Causal graph section
    genetic_summary: str = ""  # Genetic interpretation (generated separately to avoid truncation)

    # Metadata
    raw_response: str = field(default="", repr=False)
    images_analyzed: int = 0
    time_periods_summarized: int = 0
    model_used: str = ""
    token_stats: dict[str, Any] = field(default_factory=dict)

    # Intermediate outputs (new)
    chunk_summaries: list[ChunkSummary] = field(default_factory=list)
    prompts_used: dict[str, str] = field(default_factory=dict)
    causal_graph_raw: str = ""  # Raw causal graph if generated separately
    biomcp_annotations: dict[str, Any] = field(default_factory=dict)  # Raw BioMCP results
    grounded_causal_graph: Optional["GroundedCausalGraph"] = None  # SNOMED-grounded graph

    def __str__(self) -> str:
        """Format as readable SOAP note using markdown."""
        lines = [
            f"# SOAP Note: {self.patient_name or self.patient_id}",
            f"Generated: {self.generated_at}",
            "",
        ]

        # Check if any sections have content
        has_content = any([
            self.summary, self.patient_story, self.subjective,
            self.objective, self.assessment, self.plan
        ])

        if has_content:
            if self.summary:
                lines.extend(["## Summary", self.summary, ""])

            if self.patient_story:
                lines.extend(["## Patient Story", self.patient_story, ""])

            lines.extend([
                "## Subjective",
                self.subjective or "(No data)",
                "",
                "## Objective",
                self.objective or "(No data)",
                "",
                "## Assessment",
                self.assessment or "(No data)",
                "",
                "## Plan",
                self.plan or "(No data)",
                "",
            ])

            if self.future_considerations:
                lines.extend(["## Future Considerations", self.future_considerations, ""])

            if self.genetic_summary:
                lines.extend(["## Genetic Interpretation", self.genetic_summary, ""])

            if self.grounded_causal_graph:
                # Show SNOMED-grounded causal graph with concept IDs
                lines.append("## Causal Graph (SNOMED Grounded)")
                lines.append("")
                lines.append("### Nodes")
                for node in self.grounded_causal_graph.nodes:
                    # Format: Term [type] (SCTID: concept_id) - confidence
                    lines.append(f"- {node.term} [{node.node_type}] (SCTID: {node.concept_id}) [{node.confidence:.2f}]")
                lines.append("")
                lines.append("### Edges")
                for edge in self.grounded_causal_graph.edges:
                    # Format: Source RELATION Target
                    if edge.is_interaction:
                        op = " && " if edge.interaction == "and" else " || "
                        source_str = op.join(s.term for s in edge.sources)
                    else:
                        source_str = edge.source.term if edge.source else "?"
                    lines.append(f"- {source_str} {edge.relation} {edge.target.term}")
                lines.append("")
            elif self.causal_graph:
                lines.extend(["## Causal Graph", self.causal_graph, ""])
        else:
            # No sections parsed - show raw response
            lines.extend([
                "## Raw Model Response",
                "(Sections could not be parsed from model output)",
                "",
                self.raw_response[:2000] if self.raw_response else "(Empty response)",
                "",
            ])

        # Compact metadata line
        meta = f"[images: {self.images_analyzed}, periods: {self.time_periods_summarized}, model: {self.model_used}]"
        lines.append(meta)

        return "\n".join(lines)

    def show_raw(self) -> str:
        """Return the raw model response for debugging."""
        return self.raw_response

    def token_summary_df(self):
        """
        Return token_stats as a pandas DataFrame.

        Returns a DataFrame with columns: type, id, input_tokens, output_tokens, seconds
        - type: "chunk", "soap", "causal_graph", "total_chunks"
        - id: chunk period (e.g., "1969-1973") or None
        - input_tokens: input token count
        - output_tokens: output token count
        - seconds: generation time
        """
        stats = self.token_stats or {}
        if not stats:
            raise ValueError("Token stats are empty; generate a SOAP note first.")

        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required for token_summary_df()") from e

        rows = []

        # Process chunk stats if present (hierarchical mode)
        if "chunk_stats" in stats:
            for chunk in stats["chunk_stats"]:
                rows.append({
                    "type": "chunk",
                    "id": chunk.get("period", "unknown"),
                    "input_tokens": chunk.get("input_tokens"),
                    "output_tokens": chunk.get("summary_tokens"),
                    "seconds": chunk.get("summary_seconds"),
                })

            # Add chunk totals
            rows.append({
                "type": "total_chunks",
                "id": None,
                "input_tokens": stats.get("total_chunk_input_tokens"),
                "output_tokens": stats.get("total_chunk_summary_tokens"),
                "seconds": stats.get("total_chunk_seconds"),
            })

        # SOAP generation stats
        rows.append({
            "type": "soap",
            "id": None,
            "input_tokens": stats.get("soap_input_tokens", stats.get("input_tokens")),
            "output_tokens": stats.get("soap_output_tokens", stats.get("output_tokens")),
            "seconds": stats.get("soap_generation_seconds", stats.get("aggregate_seconds")),
        })

        # Causal graph stats if present (uses same input as SOAP)
        if stats.get("causal_graph_tokens"):
            rows.append({
                "type": "causal_graph",
                "id": None,
                "input_tokens": stats.get("soap_input_tokens", stats.get("input_tokens")),
                "output_tokens": stats.get("causal_graph_tokens"),
                "seconds": stats.get("causal_graph_seconds"),
            })

        # BioMCP stats if present
        if stats.get("biomcp_seconds"):
            rows.append({
                "type": "biomcp",
                "id": None,
                "input_tokens": None,
                "output_tokens": None,
                "seconds": stats.get("biomcp_seconds"),
            })

        return pd.DataFrame(rows)

    def extract_causal_graph(self) -> "CausalGraph":
        """
        Extract and parse the causal graph from the causal_graph section.

        Returns:
            CausalGraph object with parsed causal relationships

        Example:
            >>> soap_note = generator.generate(patient)
            >>> graph = soap_note.extract_causal_graph()
            >>> print(f"Found {len(graph)} causal relationships")
            >>>
            >>> # Find all risk factors for diabetes
            >>> for edge in graph.risk_factors("diabetes"):
            ...     print(f"  {edge.source} {edge.edge_type} diabetes")
            >>>
            >>> # Trace causal chains from obesity
            >>> chains = graph.causal_chains("obesity")
            >>> for chain in chains:
            ...     print(" -> ".join(e.target_name for e in chain))
        """
        # Parse from causal_graph section (or assessment for backwards compatibility)
        text = self.causal_graph or self.assessment
        return parse_causal_graph(text)

    def extract_grounded_causal_graph(
        self,
        linker: Optional["SNOMEDLinker"] = None,
        embedding_model: Optional[str] = None,
        threshold: float = 0.5,
        verbose: Union[bool, int] = True,
    ) -> "GroundedCausalGraph":
        """
        Extract and ground the causal graph to SNOMED CT concepts.

        This is a convenience method that extracts the causal graph and
        links each node to SNOMED CT using semantic embeddings.

        Args:
            linker: SNOMEDLinker instance. If None, creates one using embedding_model.
            embedding_model: Embedding model to use if linker is None. Options:
                - "sapbert" (default): Biomedical-specific
                - "qwen3-0.6b": Efficient Qwen3 embeddings
                - See synthlab.snomed.EMBEDDING_MODELS for full list
            threshold: Minimum similarity score for matches (0-1)
            verbose: Print progress messages

        Returns:
            GroundedCausalGraph with SNOMED-linked nodes

        Example:
            >>> soap_note = generator.generate(patient)
            >>> grounded = soap_note.extract_grounded_causal_graph()
            >>> for node in grounded.nodes:
            ...     print(f"{node.mention} -> SCTID:{node.concept_id} ({node.term})")
            >>>
            >>> # Use efficient Qwen3 embeddings
            >>> grounded = soap_note.extract_grounded_causal_graph(
            ...     embedding_model="qwen3-0.6b"
            ... )
        """
        # Return cached grounded graph if available
        if self.grounded_causal_graph is not None:
            return self.grounded_causal_graph

        # Import here to avoid circular dependency
        from synthlab.snomed import SNOMEDLinker, EMBEDDING_MODELS

        # Create linker if not provided
        if linker is None:
            model_id = EMBEDDING_MODELS.get(embedding_model) if embedding_model else None
            linker = SNOMEDLinker(
                model_id=model_id if model_id else "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                verbose=verbose,
            )

        # Extract and ground the graph
        graph = self.extract_causal_graph()
        grounded = graph.ground_to_snomed(linker=linker, threshold=threshold, verbose=verbose)

        # Cache for reuse
        self.grounded_causal_graph = grounded

        return grounded

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        # Convert ChunkSummary objects to dicts
        chunk_summaries_dicts = []
        for cs in self.chunk_summaries:
            chunk_summaries_dicts.append({
                "period": cs.period,
                "start_year": cs.start_year,
                "end_year": cs.end_year,
                "prompt": cs.prompt,
                "summary": cs.summary,
                "input_tokens": cs.input_tokens,
                "output_tokens": cs.output_tokens,
                "generation_seconds": cs.generation_seconds,
            })

        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "generated_at": self.generated_at,
            "patient_story": self.patient_story,
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "future_considerations": self.future_considerations,
            "summary": self.summary,
            "causal_graph": self.causal_graph,
            "genetic_summary": self.genetic_summary,
            "images_analyzed": self.images_analyzed,
            "time_periods_summarized": self.time_periods_summarized,
            "model_used": self.model_used,
            "token_stats": self.token_stats,
            # Intermediate outputs
            "chunk_summaries": chunk_summaries_dicts,
            "prompts_used": self.prompts_used,
            "causal_graph_raw": self.causal_graph_raw,
            "biomcp_annotations": self.biomcp_annotations,
            # SNOMED grounded graph (if available)
            "grounded_causal_graph": self.grounded_causal_graph.to_dict() if self.grounded_causal_graph else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TimePeriodSummary:
    """Summary of a specific time period in patient history."""

    start_date: str
    end_date: str
    summary: str
    key_events: list[str] = field(default_factory=list)
    conditions_active: list[str] = field(default_factory=list)
    medications_active: list[str] = field(default_factory=list)



# =============================================================================
# BioMCP Variant Annotation (Optional)
# =============================================================================


async def _annotate_variant_biomcp(rsid: str, verbose: bool = False) -> Optional[dict]:
    """
    Annotate a single variant using BioMCP.

    BioMCP returns data from MyVariant.info with a deeply nested structure containing
    data from multiple sources (ClinVar, gnomAD, dbNSFP, CADD, etc.). This function
    extracts relevant fields from this nested structure.

    Args:
        rsid: rsID of the variant (e.g., "rs121913529")
        verbose: If True, print debug information

    Returns:
        Dict with variant annotations or None if not found/error
    """
    if not _biomcp_available:
        return None

    try:
        import json
        # Try to get variant info - this is an async call
        # Use the module-level import (handles both old and new API paths)
        # Use output_json=True to get structured data and include_external=True for comprehensive annotations
        result_str = await _biomcp_get_variant(variant_id=rsid, output_json=True, include_external=True)
        if verbose:
            print(f"      [DEBUG] BioMCP get_variant({rsid}) returned: {type(result_str).__name__}, length: {len(result_str) if result_str else 0}")
            if result_str:
                print(f"      [DEBUG]   Preview: {result_str[:500]}...")

        if not result_str:
            return None

        # Parse JSON response
        try:
            parsed = json.loads(result_str) if isinstance(result_str, str) else result_str
        except json.JSONDecodeError:
            if verbose:
                print(f"      [DEBUG]   Failed to parse JSON, using raw response")
            return {"rsid": rsid, "raw": result_str}

        # BioMCP returns a list of variants - take the first one
        if isinstance(parsed, list):
            if not parsed:
                return None
            result = parsed[0]
        else:
            result = parsed

        if verbose:
            print(f"      [DEBUG]   Parsed result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            # Show samples from key nested objects for debugging
            if isinstance(result, dict):
                for key in ["clinvar", "dbnsfp", "cadd", "gnomad_exome", "gnomad_genome", "dbsnp", "snpeff", "pharmgkb"]:
                    if key in result:
                        val = result[key]
                        if isinstance(val, dict):
                            print(f"      [DEBUG]   {key} keys: {list(val.keys())[:15]}")
                        else:
                            print(f"      [DEBUG]   {key}: {type(val).__name__}")
                # If verbose >= 2, dump the full result for debugging
                print(f"      [DEBUG]   Full result (first 2000 chars):")
                import json as _json
                print(f"      {_json.dumps(result, indent=2, default=str)[:2000]}")

        if not result or not isinstance(result, dict):
            return None

        # Helper to safely get nested values
        def _get_nested(obj, *keys, default=None):
            """Get a nested value from a dict, handling missing keys gracefully."""
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list) and current:
                    # Take first element if it's a list
                    current = current[0] if len(current) > 0 else None
                    if isinstance(current, dict):
                        current = current.get(key)
                    else:
                        return default
                else:
                    return default
                if current is None:
                    return default
            return current if current is not None else default

        # Extract gene symbol from multiple possible sources
        gene_symbol = (
            _get_nested(result, "cadd", "gene", "genename") or
            _get_nested(result, "cadd", "gene", "gene_id") or
            _get_nested(result, "dbnsfp", "genename") or
            _get_nested(result, "dbsnp", "gene", "symbol") or
            _get_nested(result, "snpeff", "gene_name")
        )
        # Handle case where genename is a list
        if isinstance(gene_symbol, list):
            gene_symbol = gene_symbol[0] if gene_symbol else None

        # Extract clinical significance from ClinVar
        clinical_significance = (
            _get_nested(result, "clinvar", "clinical_significance") or
            _get_nested(result, "clinvar", "rcv", "clinical_significance")
        )
        # ClinVar can return a list of significances
        if isinstance(clinical_significance, list):
            clinical_significance = ", ".join(clinical_significance)

        # Extract review status from ClinVar
        review_status = _get_nested(result, "clinvar", "review_status")
        if isinstance(review_status, list):
            review_status = review_status[0] if review_status else None

        # Extract conditions/phenotypes from ClinVar
        conditions = []
        rcv_data = _get_nested(result, "clinvar", "rcv")
        if rcv_data:
            if isinstance(rcv_data, list):
                for rcv in rcv_data:
                    if isinstance(rcv, dict):
                        cond = rcv.get("conditions")
                        if cond:
                            if isinstance(cond, list):
                                conditions.extend(cond)
                            else:
                                conditions.append(cond)
            elif isinstance(rcv_data, dict):
                cond = rcv_data.get("conditions")
                if cond:
                    if isinstance(cond, list):
                        conditions.extend(cond)
                    else:
                        conditions.append(cond)
        # Also check clinvar.gene.conditions
        gene_conditions = _get_nested(result, "clinvar", "gene", "conditions")
        if gene_conditions:
            if isinstance(gene_conditions, list):
                conditions.extend(gene_conditions)
            else:
                conditions.append(gene_conditions)

        # Extract condition names from dict objects and deduplicate
        condition_names = []
        seen_conditions = set()
        for cond in conditions:
            if isinstance(cond, dict):
                # ClinVar condition objects have 'name' or 'preferred_name' field
                cond_name = cond.get("name") or cond.get("preferred_name") or cond.get("trait") or str(cond)
            else:
                cond_name = str(cond)
            if cond_name and cond_name not in seen_conditions:
                condition_names.append(cond_name)
                seen_conditions.add(cond_name)
        conditions = condition_names

        # Extract protein change
        protein_change = (
            _get_nested(result, "dbnsfp", "aa_change") or
            _get_nested(result, "cadd", "aa_change") or
            _get_nested(result, "snpeff", "hgvs_p")
        )
        if isinstance(protein_change, list):
            protein_change = protein_change[0] if protein_change else None

        # Extract population frequencies
        frequencies = {}
        # gnomAD exome
        gnomad_exome_af = _get_nested(result, "gnomad_exome", "af", "af")
        if gnomad_exome_af is None:
            gnomad_exome_af = _get_nested(result, "gnomad_exome", "af")
        if gnomad_exome_af is not None:
            frequencies["gnomad_exome"] = gnomad_exome_af
        # gnomAD genome
        gnomad_genome_af = _get_nested(result, "gnomad_genome", "af", "af")
        if gnomad_genome_af is None:
            gnomad_genome_af = _get_nested(result, "gnomad_genome", "af")
        if gnomad_genome_af is not None:
            frequencies["gnomad_genome"] = gnomad_genome_af
        # dbSNP
        dbsnp_af = _get_nested(result, "dbsnp", "allele_origin")
        if dbsnp_af is not None:
            frequencies["dbsnp"] = dbsnp_af
        # ExAC
        exac_af = _get_nested(result, "exac", "af")
        if exac_af is not None:
            frequencies["exac"] = exac_af
        # 1000 Genomes
        tg_af = _get_nested(result, "1000g", "af") or _get_nested(result, "1000genomes", "af")
        if tg_af is not None:
            frequencies["1000genomes"] = tg_af

        # Extract computational predictions
        predictions = {}
        # CADD
        cadd_phred = _get_nested(result, "cadd", "phred")
        if cadd_phred is not None:
            predictions["cadd_phred"] = cadd_phred
        cadd_raw = _get_nested(result, "cadd", "raw_score")
        if cadd_raw is not None:
            predictions["cadd_raw"] = cadd_raw
        # SIFT
        sift_pred = _get_nested(result, "dbnsfp", "sift", "pred")
        if sift_pred is None:
            sift_pred = _get_nested(result, "dbnsfp", "sift_pred")
        if sift_pred is not None:
            predictions["sift"] = sift_pred[0] if isinstance(sift_pred, list) else sift_pred
        sift_score = _get_nested(result, "dbnsfp", "sift", "score")
        if sift_score is not None:
            predictions["sift_score"] = sift_score[0] if isinstance(sift_score, list) else sift_score
        # PolyPhen-2
        polyphen_pred = _get_nested(result, "dbnsfp", "polyphen2", "hvar", "pred")
        if polyphen_pred is None:
            polyphen_pred = _get_nested(result, "dbnsfp", "polyphen2_hvar_pred")
        if polyphen_pred is not None:
            predictions["polyphen2"] = polyphen_pred[0] if isinstance(polyphen_pred, list) else polyphen_pred
        polyphen_score = _get_nested(result, "dbnsfp", "polyphen2", "hvar", "score")
        if polyphen_score is not None:
            predictions["polyphen2_score"] = polyphen_score[0] if isinstance(polyphen_score, list) else polyphen_score

        # Extract drug associations (pharmacogenomics)
        drug_associations = []
        pharmgkb = _get_nested(result, "pharmgkb")
        if pharmgkb:
            if isinstance(pharmgkb, dict):
                drugs = pharmgkb.get("drugs") or pharmgkb.get("drug")
                if drugs:
                    if isinstance(drugs, list):
                        drug_associations.extend(drugs)
                    else:
                        drug_associations.append(drugs)

        # Construct the normalized annotation dict
        annotation = {
            "rsid": rsid,
            "clinical_significance": clinical_significance,
            "conditions": conditions,
            "phenotypes": conditions,  # Alias for backward compatibility
            "gene": gene_symbol,
            "protein_change": protein_change,
            "frequencies": frequencies if frequencies else {},
            "predictions": predictions if predictions else {},
            "drug_associations": drug_associations,
            "actionability": None,  # Not directly available in MyVariant.info
            "review_status": review_status,
            "effect_type": None,  # Would need to infer from clinical_significance
        }

        if verbose:
            non_empty = {k: v for k, v in annotation.items() if v and v != {}}
            print(f"      [DEBUG]   Extracted annotation: {non_empty}")

        return annotation

    except Exception as e:
        if verbose:
            import traceback
            print(f"      [DEBUG] BioMCP variant_getter({rsid}) exception: {e}")
            traceback.print_exc()
        # Don't silently swallow - return None but log the error
    return None


def test_biomcp_response(rsid: str = "rs699") -> dict:
    """
    Test function to debug what BioMCP actually returns for a variant.

    Usage:
        from synthlab.soap import test_biomcp_response
        result = test_biomcp_response("rs699")
        print(result)
    """
    if not _biomcp_available:
        return {"error": "biomcp-python not installed"}

    import asyncio
    import json

    async def _fetch():
        result_str = await _biomcp_get_variant(variant_id=rsid, output_json=True, include_external=True)
        return result_str

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result_str = loop.run_until_complete(_fetch())

    if not result_str:
        return {"error": "No response from BioMCP"}

    try:
        parsed = json.loads(result_str) if isinstance(result_str, str) else result_str
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw": result_str[:1000]}

    # Return raw structure for inspection
    return {
        "raw_type": type(parsed).__name__,
        "is_list": isinstance(parsed, list),
        "length": len(parsed) if isinstance(parsed, list) else None,
        "first_item_keys": list(parsed[0].keys()) if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) else None,
        "top_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
        "raw_preview": str(parsed)[:2000],
    }


def annotate_variants_biomcp(rsids: list[str], max_variants: int = 20, verbose: bool = False) -> dict[str, dict]:
    """
    Annotate multiple variants using BioMCP (synchronous wrapper).

    This function queries BioMCP for detailed variant annotations including
    clinical significance, associated conditions, population frequencies,
    and functional predictions.

    Args:
        rsids: List of rsIDs to annotate
        max_variants: Maximum number of variants to query (to avoid rate limits)
        verbose: If True, print debug information

    Returns:
        Dict mapping rsID to annotation dict

    Example:
        >>> annotations = annotate_variants_biomcp(["rs121913529", "rs6025"])
        >>> print(annotations["rs121913529"]["clinical_significance"])
        "Pathogenic"
    """
    if not _biomcp_available:
        if verbose:
            print(f"    [DEBUG] BioMCP not available (biomcp-python not installed)")
        return {}

    if verbose:
        print(f"    [DEBUG] annotate_variants_biomcp called with {len(rsids)} rsIDs, max={max_variants}")
        if rsids:
            print(f"    [DEBUG] First few rsIDs: {rsids[:5]}")

    import asyncio

    async def _annotate_batch():
        results = {}
        for i, rsid in enumerate(rsids[:max_variants]):
            if verbose and i == 0:
                print(f"    [DEBUG] Querying first variant for detailed debug...")
            annotation = await _annotate_variant_biomcp(rsid, verbose=(verbose and i == 0))
            if annotation:
                results[rsid] = annotation
            elif verbose and i == 0:
                print(f"    [DEBUG] First variant {rsid} returned None (no data or error)")
        return results

    try:
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _annotate_batch())
                result = future.result()
                if verbose:
                    print(f"    [DEBUG] BioMCP returned {len(result)} annotations (async context)")
                return result
        else:
            result = asyncio.run(_annotate_batch())
            if verbose:
                print(f"    [DEBUG] BioMCP returned {len(result)} annotations")
            return result
    except Exception as e:
        # Re-raise with more context instead of silently returning empty
        raise RuntimeError(
            f"BioMCP annotation failed for {len(rsids)} rsIDs: {e}. "
            f"First few rsIDs: {rsids[:5]}"
        ) from e


# =============================================================================
# FHIR Data Extraction and Formatting
# =============================================================================


class FHIRFormatter:
    """
    Extracts and formats FHIR data for LLM input.

    Converts FHIR bundles into structured text that MedGemma can process.
    """

    @staticmethod
    def format_patient_for_llm(
        patient: Any,
        max_tokens: int = 50000,
        use_biomcp: bool = False,
        include_genomics: bool = True,
        verbose: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """
        Format patient data as structured text for LLM input.

        Args:
            patient: Patient object from MultimodalDataset
            max_tokens: Approximate max tokens (characters / 4)
            use_biomcp: If True, use BioMCP to enrich genetic variant annotations.
                        Requires: pip install biomcp-python
            include_genomics: If True, include genomics section in output text.
                        Set to False when generating genetic summary separately
                        to avoid truncation in main SOAP note.
            verbose: If True, print debug information

        Returns:
            Tuple of (formatted_text, biomcp_annotations)
            - formatted_text: Formatted string for LLM input
            - biomcp_annotations: Raw BioMCP annotations dict (empty if not used)
        """
        sections = []
        biomcp_annotations = {}  # Store raw BioMCP annotations

        # Demographics
        demographics = patient.get_demographics()
        if demographics:
            sections.append(FHIRFormatter._format_demographics(demographics))

        # Conditions
        conditions = patient.get_conditions()
        if conditions:
            sections.append(FHIRFormatter._format_conditions(conditions))

        # Medications
        medications = patient.get_medications()
        if medications:
            sections.append(FHIRFormatter._format_medications(medications))

        # Encounters
        encounters = patient.get_encounters()
        if encounters:
            sections.append(FHIRFormatter._format_encounters(encounters))

        # Observations (vitals, labs)
        observations = patient.get_observations()
        if observations:
            sections.append(FHIRFormatter._format_observations(observations))

        # Procedures
        procedures = patient.get_procedures()
        if procedures:
            sections.append(FHIRFormatter._format_procedures(procedures))

        # Genomics (if available and requested)
        if include_genomics and hasattr(patient, 'genomics') and patient.genomics is not None:
            if verbose:
                print(f"    [DEBUG] format_patient_for_llm: include_genomics=True, calling _format_genomics")
            try:
                genomics_text, genomics_biomcp = FHIRFormatter._format_genomics(
                    patient.genomics,
                    use_biomcp=use_biomcp,
                    verbose=verbose,
                )
                sections.append(genomics_text)
                biomcp_annotations = genomics_biomcp
            except Exception as e:
                if verbose:
                    import traceback
                    print(f"    [DEBUG] Genomics formatting failed (include=True): {e}")
                    traceback.print_exc()
        elif not include_genomics and hasattr(patient, 'genomics') and patient.genomics is not None:
            # Still collect BioMCP annotations even if not including genomics text
            if verbose:
                print(f"    [DEBUG] format_patient_for_llm: include_genomics=False, calling _format_genomics for BioMCP only")
            try:
                _, genomics_biomcp = FHIRFormatter._format_genomics(
                    patient.genomics,
                    use_biomcp=use_biomcp,
                    verbose=verbose,
                )
                biomcp_annotations = genomics_biomcp
                # Add placeholder noting genetic data exists
                sections.append("## GENETIC / GENOMIC FINDINGS\n[Genetic interpretation generated separately - see Genetic Interpretation section]")
            except Exception as e:
                import traceback
                print(f"    [DEBUG] Genomics formatting failed: {e}")
                traceback.print_exc()

        # Combine and truncate if needed
        full_text = "\n\n".join(sections)

        # Rough token estimate (1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[... truncated for length ...]"

        return full_text, biomcp_annotations

    @staticmethod
    def _format_demographics(demographics: dict) -> str:
        """Format demographics section."""
        lines = ["## PATIENT DEMOGRAPHICS"]
        if demographics.get("name"):
            lines.append(f"Name: {demographics['name']}")
        if demographics.get("gender"):
            lines.append(f"Gender: {demographics['gender']}")
        if demographics.get("birth_date"):
            lines.append(f"Date of Birth: {demographics['birth_date']}")
        if demographics.get("address"):
            lines.append(f"Address: {demographics['address']}")
        return "\n".join(lines)

    @staticmethod
    def _format_conditions(conditions: list[dict]) -> str:
        """Format conditions section."""
        lines = ["## CONDITIONS / DIAGNOSES"]

        # Sort by onset date
        sorted_conds = sorted(
            conditions,
            key=lambda x: x.get("onset_date") or "9999",
            reverse=True
        )

        for cond in sorted_conds[:50]:  # Limit to 50 most recent
            display = cond.get("display", "Unknown condition")
            onset = cond.get("onset_date", "Unknown date")
            if onset and len(onset) > 10:
                onset = onset[:10]
            status = cond.get("clinical_status", "")
            status_str = f" [{status}]" if status else ""
            lines.append(f"- {onset}: {display}{status_str}")

        if len(conditions) > 50:
            lines.append(f"  ... and {len(conditions) - 50} more conditions")

        return "\n".join(lines)

    @staticmethod
    def _format_medications(medications: list[dict]) -> str:
        """Format medications section."""
        lines = ["## MEDICATIONS"]

        # Sort by date
        sorted_meds = sorted(
            medications,
            key=lambda x: x.get("authored_on") or "9999",
            reverse=True
        )

        for med in sorted_meds[:30]:  # Limit to 30 most recent
            display = med.get("display", "Unknown medication")
            date = med.get("authored_on", "")
            if date and len(date) > 10:
                date = date[:10]
            status = med.get("status", "")
            date_str = f" ({date})" if date else ""
            status_str = f" [{status}]" if status else ""
            lines.append(f"- {display}{date_str}{status_str}")

        if len(medications) > 30:
            lines.append(f"  ... and {len(medications) - 30} more medications")

        return "\n".join(lines)

    @staticmethod
    def _format_encounters(encounters: list[dict]) -> str:
        """Format encounters section."""
        lines = ["## ENCOUNTERS / VISITS"]

        # Sort by date
        sorted_enc = sorted(
            encounters,
            key=lambda x: x.get("start") or "9999",
            reverse=True
        )

        for enc in sorted_enc[:20]:  # Limit to 20 most recent
            enc_type = enc.get("type", "Unknown")
            enc_class = enc.get("class", "")
            start = enc.get("start", "Unknown date")
            if start and len(start) > 10:
                start = start[:10]
            reason = enc.get("reason", "")
            class_str = f" ({enc_class})" if enc_class else ""
            reason_str = f" - {reason}" if reason else ""
            lines.append(f"- {start}: {enc_type}{class_str}{reason_str}")

        if len(encounters) > 20:
            lines.append(f"  ... and {len(encounters) - 20} more encounters")

        return "\n".join(lines)

    @staticmethod
    def _format_observations(observations: list[dict]) -> str:
        """Format observations section (vitals, labs)."""
        lines = ["## OBSERVATIONS (Labs, Vitals)"]

        # Group by category
        vitals = [o for o in observations if o.get("category") == "vital-signs"]
        labs = [o for o in observations if o.get("category") == "laboratory"]
        other = [o for o in observations if o.get("category") not in ["vital-signs", "laboratory"]]

        if vitals:
            lines.append("\n### Vital Signs (Recent)")
            sorted_vitals = sorted(vitals, key=lambda x: x.get("date") or "0", reverse=True)
            for obs in sorted_vitals[:20]:
                display = obs.get("display", "Unknown")
                value = obs.get("value", "")
                unit = obs.get("unit", "")
                date = obs.get("date", "")
                if date and len(date) > 10:
                    date = date[:10]
                val_str = f": {value} {unit}".strip() if value else ""
                lines.append(f"- {date}: {display}{val_str}")

        if labs:
            lines.append("\n### Laboratory Results (Recent)")
            sorted_labs = sorted(labs, key=lambda x: x.get("date") or "0", reverse=True)
            for obs in sorted_labs[:30]:
                display = obs.get("display", "Unknown")
                value = obs.get("value", "")
                unit = obs.get("unit", "")
                date = obs.get("date", "")
                if date and len(date) > 10:
                    date = date[:10]
                val_str = f": {value} {unit}".strip() if value else ""
                lines.append(f"- {date}: {display}{val_str}")

        return "\n".join(lines)

    @staticmethod
    def _format_procedures(procedures: list[dict]) -> str:
        """Format procedures section."""
        lines = ["## PROCEDURES"]

        sorted_procs = sorted(
            procedures,
            key=lambda x: x.get("performed_date") or "9999",
            reverse=True
        )

        for proc in sorted_procs[:20]:
            display = proc.get("display", "Unknown procedure")
            date = proc.get("performed_date", "")
            if date and len(date) > 10:
                date = date[:10]
            date_str = f" ({date})" if date else ""
            lines.append(f"- {display}{date_str}")

        if len(procedures) > 20:
            lines.append(f"  ... and {len(procedures) - 20} more procedures")

        return "\n".join(lines)

    @staticmethod
    def _format_genomics(
        genomics_df: Any,
        use_biomcp: bool = False,
        verbose: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """
        Format genomics data for LLM input.

        Filters to clinically significant variants (Pathogenic, Likely Pathogenic,
        Risk Factor) and formats them for clinical interpretation.

        Args:
            genomics_df: Polars DataFrame with genomics data
            use_biomcp: If True, use BioMCP to enrich variant annotations with
                        clinical significance, associated conditions, and functional
                        predictions. Requires biomcp-python package.
                        See: https://biomcp.org/apis/python-sdk/
            verbose: If True, print debug information

        Returns:
            Tuple of (formatted_text, biomcp_annotations_dict)
            - formatted_text: Formatted string with genetic findings
            - biomcp_annotations_dict: Raw BioMCP annotations (empty if not used)
        """
        biomcp_raw = {}  # Store raw bioMCP annotations
        lines = ["## GENETIC / GENOMIC FINDINGS"]

        if verbose:
            print(f"    [DEBUG] _format_genomics called: use_biomcp={use_biomcp}, df_type={type(genomics_df)}")
            if genomics_df is not None:
                print(f"    [DEBUG] genomics_df has {len(genomics_df)} rows")

        if genomics_df is None or len(genomics_df) == 0:
            lines.append("No genetic data available.")
            return "\n".join(lines), biomcp_raw

        # Filter to variants the patient has (VARIANT == true)
        try:
            patient_variants = genomics_df.filter(genomics_df["VARIANT"] == True)
            if verbose:
                print(f"    [DEBUG] Filtered VARIANT==True: {len(patient_variants)} variants")
        except Exception as e:
            # Try string comparison if boolean doesn't work
            if verbose:
                print(f"    [DEBUG] VARIANT==True filter failed: {e}, trying string")
            try:
                patient_variants = genomics_df.filter(genomics_df["VARIANT"] == "true")
                if verbose:
                    print(f"    [DEBUG] Filtered VARIANT=='true': {len(patient_variants)} variants")
            except Exception as e2:
                if verbose:
                    print(f"    [DEBUG] VARIANT=='true' filter also failed: {e2}, using all rows")
                patient_variants = genomics_df

        if len(patient_variants) == 0:
            if verbose:
                print(f"    [DEBUG] No patient variants after filtering, returning early")
            lines.append("No clinically significant genetic variants detected.")
            return "\n".join(lines), biomcp_raw

        # Categorize by clinical significance - use actual values from data, not hard-coded list
        # Priority order for display (higher priority = shown first)
        # Based on clinical actionability, not a fixed list
        significance_priority = {
            "pathogenic": 1,
            "likely pathogenic": 2,
            "risk factor": 3,
            "drug response": 4,
            "protective": 5,
            "association": 6,
            "affects": 7,
            "uncertain significance": 8,
            "likely benign": 9,
            "benign": 10,
            "conflicting": 11,
        }

        def get_priority(sig_str: str) -> int:
            """Get priority for a significance string (lower = more important)."""
            sig_lower = str(sig_str).lower()
            for key, priority in significance_priority.items():
                if key in sig_lower:
                    return priority
            return 100  # Unknown categories go last

        # Get unique variants (dedupe by rsID), group by actual significance
        seen_variants = set()
        categorized: dict[str, list] = {}

        for row in patient_variants.iter_rows(named=True):
            rsid = row.get("INDEX_PREFIX", row.get("INDEX", "Unknown"))
            if rsid in seen_variants:
                continue
            seen_variants.add(rsid)

            gene = row.get("GENE", "Unknown")
            significance = str(row.get("CLINICAL_SIGNIFICANCE", "Unknown")).strip()
            chrom = row.get("CHROMOSOME", "?")
            allele = row.get("ALLELE", "")
            ancestral = row.get("ANCESTRAL_ALLELE", "")

            # Determine if it's a variant vs reference
            is_variant = allele != ancestral if ancestral else True

            if not is_variant:
                continue

            # Group by actual significance value from the data
            if significance not in categorized:
                categorized[significance] = []
            categorized[significance].append({
                "rsid": rsid,
                "gene": gene,
                "chrom": chrom,
                "significance": significance,
                "allele": allele,
            })

        # Sort categories by clinical priority and format output
        sorted_categories = sorted(categorized.keys(), key=get_priority)

        if verbose:
            total_variants = sum(len(v) for v in categorized.values())
            print(f"    [DEBUG] Categorized {total_variants} unique variants into {len(categorized)} categories")
            for sig, variants in categorized.items():
                n_with_rs = sum(1 for v in variants if v['rsid'].startswith('rs'))
                print(f"    [DEBUG]   {sig}: {len(variants)} variants ({n_with_rs} with rs prefix)")

        for sig in sorted_categories:
            variants = categorized[sig]
            # Skip benign variants unless there's nothing else
            sig_lower = sig.lower()
            if "benign" in sig_lower and len(sorted_categories) > 1:
                continue
            if variants:
                lines.append(f"\n### {sig} Variants")
                for v in variants[:10]:  # Limit per category
                    gene_str = f" ({v['gene']})" if v['gene'] != "Unlisted" else ""
                    lines.append(f"- {v['rsid']}{gene_str} - Chr{v['chrom']}")

        # Summary of uncertain variants
        uncertain = categorized.get("Uncertain", [])
        if uncertain:
            # Group by gene for uncertain variants
            genes_with_uncertain = set(v['gene'] for v in uncertain if v['gene'] != "Unlisted")
            if genes_with_uncertain:
                lines.append(f"\n### Variants of Uncertain Significance")
                lines.append(f"Found {len(uncertain)} VUS across genes: {', '.join(sorted(genes_with_uncertain)[:10])}")

        # BioMCP enrichment provides real clinical annotations from ClinVar, PharmGKB, etc.
        # This replaces any hard-coded gene lists with actual database-backed information
        if use_biomcp and _biomcp_available:
            if verbose:
                print(f"    [DEBUG] BioMCP available, processing variants")
                print(f"    [DEBUG] categorized has {len(categorized)} categories: {list(categorized.keys())}")
            # Get rsIDs of ALL variants for enrichment (not just pathogenic)
            all_rsids = []
            for category_variants in categorized.values():
                for v in category_variants:
                    if v['rsid'].startswith("rs"):
                        all_rsids.append(v['rsid'])

            if verbose:
                print(f"    [DEBUG] Found {len(all_rsids)} rsIDs starting with 'rs'")

            if all_rsids:
                # Sort variants by clinical priority (pathogenic first, then risk factors, etc.)
                # Use the same priority function defined earlier
                all_variants_with_priority = []
                for sig, variants in categorized.items():
                    priority = get_priority(sig)
                    for v in variants:
                        if v['rsid'].startswith("rs"):
                            all_variants_with_priority.append((priority, v['rsid']))

                # Sort by priority and dedupe
                all_variants_with_priority.sort(key=lambda x: x[0])
                prioritized_rsids = []
                seen = set()
                for _, rsid in all_variants_with_priority:
                    if rsid not in seen:
                        prioritized_rsids.append(rsid)
                        seen.add(rsid)

                if verbose:
                    print(f"    [DEBUG] Calling annotate_variants_biomcp with {len(prioritized_rsids)} prioritized rsIDs")

                biomcp_annotations = annotate_variants_biomcp(prioritized_rsids, max_variants=20, verbose=verbose)
                if not biomcp_annotations:
                    # BioMCP returned nothing - warn but continue (API might have issues)
                    if verbose:
                        print(f"    [WARNING] BioMCP returned 0 annotations for {len(prioritized_rsids)} rsIDs")
                        print(f"    [WARNING] rsIDs sent: {prioritized_rsids[:5]}{'...' if len(prioritized_rsids) > 5 else ''}")
                        print(f"    [WARNING] Continuing without BioMCP enrichment")
                if biomcp_annotations:
                    # Store raw annotations before summarizing
                    biomcp_raw = biomcp_annotations

                    lines.append("\n### Clinical Variant Annotations (BioMCP)")
                    lines.append("*Curated disease associations and clinical interpretations:*\n")

                    for rsid, ann in biomcp_annotations.items():
                        gene = ann.get("gene", "")
                        gene_str = f" ({gene})" if gene else ""
                        ann_lines = [f"**{rsid}{gene_str}**"]
                        has_data = False  # Track if we found any clinical data

                        # Clinical significance with review status
                        clin_sig = ann.get("clinical_significance")
                        if clin_sig:
                            has_data = True
                            review = ann.get("review_status", "")
                            review_str = f" [{review}]" if review else ""
                            ann_lines.append(f"  - Clinical significance: {clin_sig}{review_str}")

                        # Disease/phenotype associations - THIS IS KEY
                        conditions = ann.get("conditions") or ann.get("phenotypes") or []
                        if conditions:
                            has_data = True
                            if len(conditions) <= 3:
                                conditions_str = ", ".join(str(c) for c in conditions)
                            else:
                                conditions_str = ", ".join(str(c) for c in conditions[:3]) + f" (+{len(conditions)-3} more)"
                            ann_lines.append(f"  - Disease associations: {conditions_str}")

                        # Effect type (protective vs risk) if available
                        effect = ann.get("effect_type")
                        if effect:
                            has_data = True
                            ann_lines.append(f"  - Effect: {effect}")

                        # Drug associations (pharmacogenomics)
                        drugs = ann.get("drug_associations") or []
                        if drugs:
                            has_data = True
                            drugs_str = ", ".join(str(d) for d in drugs[:3])
                            ann_lines.append(f"  - Drug interactions: {drugs_str}")

                        # Actionability
                        if ann.get("actionability"):
                            has_data = True
                            ann_lines.append(f"  - Actionability: {ann['actionability']}")

                        # Functional predictions (handle both old and new field names)
                        predictions = ann.get("predictions", {})
                        pred_parts = []
                        # CADD score
                        cadd = predictions.get("cadd_phred") or predictions.get("cadd")
                        if cadd:
                            pred_parts.append(f"CADD={cadd}")
                        # PolyPhen-2 prediction
                        polyphen = predictions.get("polyphen2") or predictions.get("polyphen")
                        if polyphen:
                            pred_parts.append(f"PolyPhen={polyphen}")
                        # SIFT prediction
                        sift = predictions.get("sift")
                        if sift:
                            pred_parts.append(f"SIFT={sift}")
                        if pred_parts:
                            has_data = True
                            ann_lines.append(f"  - Predictions: {', '.join(pred_parts)}")

                        # Protein change
                        if ann.get("protein_change"):
                            has_data = True
                            ann_lines.append(f"  - Protein change: {ann['protein_change']}")

                        # Population frequency (check multiple sources)
                        frequencies = ann.get("frequencies", {})
                        freq = (
                            frequencies.get("gnomad_exome") or
                            frequencies.get("gnomad_genome") or
                            frequencies.get("gnomad") or
                            frequencies.get("exac") or
                            frequencies.get("1000genomes")
                        )
                        if freq:
                            has_data = True
                            try:
                                ann_lines.append(f"  - Population freq (gnomAD): {float(freq):.4f}")
                            except (ValueError, TypeError):
                                ann_lines.append(f"  - Population freq: {freq}")

                        # If no clinical data found, add a note
                        if not has_data:
                            ann_lines.append("  - No clinical annotations available in databases")

                        lines.extend(ann_lines)
                        lines.append("")  # Blank line between variants

        elif use_biomcp and not _biomcp_available:
            if verbose:
                print(f"    [DEBUG] BioMCP requested but not available (biomcp-python not installed)")
            lines.append("\n*Note: BioMCP enrichment requested but biomcp-python not installed.*")
            lines.append("*Install with: pip install biomcp-python*")

        return "\n".join(lines), biomcp_raw

    @staticmethod
    def chunk_by_time_period(
        patient: Any,
        period_years: int = 5
    ) -> list[tuple[str, str, str]]:
        """
        Chunk patient data by time periods for hierarchical summarization.

        Args:
            patient: Patient object
            period_years: Number of years per chunk

        Returns:
            List of (start_date, end_date, formatted_text) tuples
        """
        # Get all events with dates
        events = []

        for cond in patient.get_conditions():
            if cond.get("onset_date"):
                events.append(("condition", cond["onset_date"], cond))

        for med in patient.get_medications():
            if med.get("authored_on"):
                events.append(("medication", med["authored_on"], med))

        for enc in patient.get_encounters():
            if enc.get("start"):
                events.append(("encounter", enc["start"], enc))

        for obs in patient.get_observations():
            if obs.get("date"):
                events.append(("observation", obs["date"], obs))

        for proc in patient.get_procedures():
            if proc.get("performed_date"):
                events.append(("procedure", proc["performed_date"], proc))

        if not events:
            return []

        # Sort by date
        events.sort(key=lambda x: x[1])

        # Find date range
        min_date = events[0][1][:4]  # Year
        max_date = events[-1][1][:4]

        chunks = []
        start_year = int(min_date)
        end_year = int(max_date)

        current_start = start_year
        while current_start <= end_year:
            current_end = min(current_start + period_years - 1, end_year)

            # Filter events in this period
            period_events = [
                e for e in events
                if current_start <= int(e[1][:4]) <= current_end
            ]

            if period_events:
                # Format this chunk
                text = FHIRFormatter._format_period_chunk(
                    patient, period_events, current_start, current_end
                )
                chunks.append((str(current_start), str(current_end), text))

            current_start = current_end + 1

        return chunks

    @staticmethod
    def _format_period_chunk(
        patient: Any,
        events: list[tuple],
        start_year: int,
        end_year: int
    ) -> str:
        """Format a single time period chunk."""
        lines = [f"## PERIOD: {start_year} - {end_year}"]

        # Group events by type
        conditions = [e[2] for e in events if e[0] == "condition"]
        medications = [e[2] for e in events if e[0] == "medication"]
        encounters = [e[2] for e in events if e[0] == "encounter"]
        observations = [e[2] for e in events if e[0] == "observation"]
        procedures = [e[2] for e in events if e[0] == "procedure"]

        if conditions:
            lines.append("\nConditions diagnosed:")
            for c in conditions[:10]:
                lines.append(f"  - {c.get('display', 'Unknown')}")

        if medications:
            lines.append("\nMedications prescribed:")
            for m in medications[:10]:
                lines.append(f"  - {m.get('display', 'Unknown')}")

        if encounters:
            lines.append(f"\nEncounters: {len(encounters)} visits")

        if procedures:
            lines.append("\nProcedures:")
            for p in procedures[:5]:
                lines.append(f"  - {p.get('display', 'Unknown')}")

        if observations:
            lines.append(f"\nObservations recorded: {len(observations)}")

        return "\n".join(lines)


# =============================================================================
# SOAP Note Generator
# =============================================================================


class SOAPNoteGenerator:
    """
    Generate SOAP notes from patient data using MedGemma.

    This class provides a high-level interface for generating clinical SOAP notes
    from FHIR patient records. It supports:
    - Direct generation for short histories
    - Hierarchical summarization for long histories
    - Multimodal input (FHIR + DICOM images)

    Example:
        >>> from synthlab.soap import SOAPNoteGenerator
        >>> import synthlab as sl
        >>>
        >>> # Load patient data
        >>> dataset = sl.load_multimodal_dataset(max_patients=5)
        >>> patient = dataset[0]
        >>>
        >>> # Generate SOAP note
        >>> generator = SOAPNoteGenerator()
        >>> soap_note = generator.generate(patient)
        >>> print(soap_note)
    """

    # Default prompts
    SOAP_SYSTEM_PROMPT = """You are an expert clinical assistant helping to summarize patient medical records.
Your task is to generate a comprehensive SOAP note that captures the patient's complete medical history.
Focus on clinically significant information that would help inform future care decisions.
Be thorough but concise. Highlight patterns, risk factors, and areas requiring attention."""


    # ==========================================================================
    # UNIFIED PROMPT TEMPLATE
    # ==========================================================================
    # Single base template with conditional sections for images and causal graph.
    # Use _build_chunk_prompt(), _build_aggregate_prompt(), or _build_direct_prompt().

    # Conditional section: imaging instructions (inserted into Objective)
    _IMAGING_SECTION = """
- **Imaging Findings**: For each attached image, describe:
  - Modality (CT, MRI, X-ray, ultrasound)
  - Anatomical region
  - Key findings (normal or abnormal)
  - Clinical significance"""

    # Conditional section: causal graph (dedicated section at end of note)
    _CAUSAL_GRAPH_SECTION = """

# Causal Graph
DIRECT causal relationships observed in this patient, one per line.

Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Type_2_Diabetes)
- Types: condition, medication, procedure, lifestyle, symptom, finding, genetic
- Arrows:
  - ++> strongly increases risk
  - +> increases risk
  - --> strongly protects/reduces
  - -> protects/reduces
  - => directly causes

Joint/Interaction Effects (when BOTH factors required together):
- A && B => C  means A AND B together cause C (neither alone is sufficient)
- A || B => C  means A OR B can cause C (either alone is sufficient)

Examples:
Obesity[lifestyle] ++> Type_2_Diabetes[condition]
Type_2_Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Type_2_Diabetes[condition] +> Cardiovascular_Disease[condition]
Smoking[lifestyle] ++> COPD[condition]
Smoking[lifestyle] ++> Lung_Cancer[condition]
Hypertension[condition] ++> Stroke[condition]
Hypertension[condition] ++> Chronic_Kidney_Disease[condition]
Metformin[medication] --> Blood_Glucose[finding]
Metformin[medication] -> Cardiovascular_Risk[finding]
Statin[medication] --> LDL_Cholesterol[finding]
ACE_Inhibitor[medication] --> Blood_Pressure[finding]
ACE_Inhibitor[medication] --> Proteinuria[finding]
Appendicitis[condition] => Appendectomy[procedure]
BRCA1_Mutation[genetic] ++> Breast_Cancer[condition]
Exercise[lifestyle] --> Insulin_Resistance[finding]
Smoking[lifestyle] && Asbestos_Exposure[lifestyle] ++> Lung_Cancer[condition]
Obesity[lifestyle] && Sedentary_Lifestyle[lifestyle] ++> Type_2_Diabetes[condition]
BRCA1_Mutation[genetic] || BRCA2_Mutation[genetic] || PALB2_Mutation[genetic] ++> Breast_Cancer[condition]
Warfarin[medication] && NSAIDs[medication] ++> GI_Bleeding[condition]

Write relationships for THIS patient based on their actual conditions and treatments."""

    # Conditional section: genetics assessment (inserted into Assessment when separate_genetics=False)
    _GENETICS_ASSESSMENT_SECTION = """
4. **Genetic Factors** - How the patient's genetic variants influence their health:
   - Variants that increase risk for conditions in their history
   - Pharmacogenomic variants affecting medication response
   - Protective variants and their clinical implications
   - Recommended genetic-informed interventions"""

    def prompt(
        self,
        mode: str = "direct",
        include_images: bool = False,
        num_images: int = 1,
        print_prompt: bool = True,
    ) -> str:
        """Preview the assembled prompt based on current configuration.

        Useful for debugging and understanding what prompt will be sent to the model.

        Args:
            mode: Prompt type:
                - "direct": Main SOAP note generation (single pass)
                - "aggregate": Aggregate from time period summaries
                - "chunk": Time period chunk summary
                - "causal_graph": Separate causal graph generation
                - "entity_extraction": Entity extraction for grounded mode
                - "grounded_relationship": Relationship identification for grounded mode
            include_images: Whether to include imaging section
            num_images: Number of images (for display in prompt)
            print_prompt: If True, print the prompt. If False, just return it.

        Returns:
            The assembled prompt string

        Example:
            >>> generator = SOAPNoteGenerator(ground_snomed=True)
            >>> generator.prompt()  # Shows direct prompt without causal graph (grounded mode)
            >>> generator.prompt(mode="aggregate", include_images=True)
            >>> generator.prompt(mode="grounded_relationship")  # Shows the grounded graph prompt
        """
        # Determine if inline causal graph should be included based on config
        skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)
        include_causal_graph = not skip_inline_graph

        # For chunks, also check chunk_causal_graph setting
        include_chunk_graph = False  # Default for non-chunk modes

        # Handle different prompt modes
        if mode == "grounded_relationship":
            # Sample concept list with SNOMED IDs for demonstration
            sample_concepts = """Type_2_Diabetes_Mellitus[SNOMED:44054006]
Diabetic_Nephropathy[SNOMED:236499007]
Metformin[SNOMED:372567009]
Hypertension[SNOMED:38341003]
ACE_Inhibitor[SNOMED:41549009]
Chronic_Kidney_Disease[SNOMED:709044004]
Obesity[SNOMED:414916001]
Elevated_HbA1c[SNOMED:444275009]"""
            prompt = self.GROUNDED_RELATIONSHIP_PROMPT.format(
                soap_note="[SOAP NOTE TEXT WOULD APPEAR HERE - patient's clinical history]",
                concept_list=sample_concepts,
            )

        elif mode == "entity_extraction":
            prompt = self.ENTITY_EXTRACTION_PROMPT.format(
                soap_note="[SOAP NOTE TEXT WOULD APPEAR HERE]"
            )

        elif mode == "causal_graph":
            prompt = self.CAUSAL_GRAPH_PROMPT.format(
                soap_note="[SOAP NOTE TEXT WOULD APPEAR HERE]"
            )
            prompt += self._get_admission_exclusion_instruction()

        elif mode == "chunk":
            skip_chunk_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)
            include_chunk_graph = self.chunk_causal_graph and not skip_chunk_graph
            prompt = self._build_chunk_prompt(
                chunk_data="[PATIENT DATA FOR TIME PERIOD WOULD APPEAR HERE]",
                time_period="2020-2024",
                include_causal_graph=include_chunk_graph,
            )
            if include_chunk_graph:
                prompt += self._get_admission_exclusion_instruction()

        elif mode == "aggregate":
            prompt = self._build_aggregate_prompt(
                patient_name="[PATIENT NAME]",
                summaries_text="[TIME PERIOD SUMMARIES WOULD APPEAR HERE]",
                include_images=include_images,
                num_images=num_images,
                include_causal_graph=include_causal_graph,
            )
            if include_causal_graph:
                prompt += self._get_admission_exclusion_instruction()

        else:  # direct
            prompt = self._build_direct_prompt(
                patient_data="[PATIENT DATA WOULD APPEAR HERE]",
                include_images=include_images,
                num_images=num_images,
                include_causal_graph=include_causal_graph,
            )
            if include_causal_graph:
                prompt += self._get_admission_exclusion_instruction()

        # Build info header
        mode_descriptions = {
            "direct": "Main SOAP note generation (single pass)",
            "aggregate": "Aggregate SOAP from time period summaries",
            "chunk": "Time period chunk summary",
            "causal_graph": "Separate causal graph generation (when separate_causal_graph=True)",
            "entity_extraction": "Entity extraction (grounded mode stage 1)",
            "grounded_relationship": "Relationship identification (grounded mode stage 2)",
        }

        info_lines = [
            "=" * 70,
            f"PROMPT PREVIEW: {mode}",
            f"  {mode_descriptions.get(mode, mode)}",
            "=" * 70,
            f"Configuration:",
            f"  - separate_causal_graph: {self.separate_causal_graph}",
            f"  - ground_snomed: {self.ground_snomed}",
            f"  - grounded_graph_mode: {self.grounded_graph_mode}",
            f"  - chunk_causal_graph: {self.chunk_causal_graph}",
            f"  - include_admissions: {self.include_admissions}",
        ]

        if mode in ("direct", "aggregate", "chunk"):
            info_lines.extend([
                f"",
                f"Prompt includes:",
                f"  - Imaging section: {include_images}",
                f"  - Inline causal graph: {include_causal_graph if mode != 'chunk' else include_chunk_graph}",
            ])

        info_lines.extend(["=" * 70, ""])
        header = "\n".join(info_lines)

        if print_prompt:
            print(header)
            print(prompt)
            print("\n" + "=" * 70)

        return prompt

    def _build_chunk_prompt(
        self,
        chunk_data: str,
        time_period: str = "",
        include_causal_graph: bool = False,
    ) -> str:
        """Build chunk summary prompt with same structure as final SOAP.

        Args:
            chunk_data: Patient data for this time period
            time_period: Description of the time period (e.g., "2018-2020")
            include_causal_graph: Whether to include causal graph (default False for chunks)
        """
        causal_section = self._CAUSAL_GRAPH_SECTION if include_causal_graph else ""
        period_label = f" ({time_period})" if time_period else ""

        return f"""Summarize this time period{period_label} from a patient's medical history.

=== PATIENT DATA FOR THIS TIME PERIOD ===
{chunk_data}
=== END OF TIME PERIOD DATA ===

Generate a focused summary using the markdown format below.

# Subjective
Patient-reported symptoms and complaints during this period.

# Objective
Clinical findings:
- Vital signs and trends
- Laboratory results (highlight abnormal values)
- Procedures performed

# Assessment
Clinical reasoning:
1. **Diagnoses** - Conditions identified or managed
2. **Disease progression** - How conditions evolved
3. **Risk factors** - New or ongoing concerns

# Plan
Treatment during this period:
1. Medications started/changed
2. Monitoring performed
3. Referrals made

# Summary
2-3 sentences on key events and health trajectory during this period.{causal_section}"""

    def _build_aggregate_prompt(
        self,
        patient_name: str,
        summaries_text: str,
        include_images: bool = False,
        num_images: int = 0,
        include_causal_graph: bool = True,
    ) -> str:
        """Build aggregate prompt with conditional sections."""
        image_note = f" {num_images} medical image(s) attached - analyze them." if include_images else ""
        imaging_section = self._IMAGING_SECTION if include_images else ""
        causal_section = self._CAUSAL_GRAPH_SECTION if include_causal_graph else ""
        # Include genetics in assessment when genomics are inline (separate_genetics=False)
        genetics_section = self._GENETICS_ASSESSMENT_SECTION if (self.use_biomcp and not self.separate_genetics) else ""

        return f"""You are synthesizing a SOAP note from multiple time period summaries.{image_note}

=== PATIENT ===
{patient_name}

=== TIME PERIOD SUMMARIES ===
{summaries_text}

=== END OF INPUT DATA ===

Generate a comprehensive SOAP note using the exact markdown format below.

# Patient Story
1-2 paragraph narrative synthesizing the patient's health journey across all time periods.

# Subjective
Consolidated patient-reported symptoms and history across time periods.

# Objective
Synthesized clinical findings:
- Most recent vital signs with trends over time
- Key laboratory values and changes{imaging_section}

# Assessment
Integrated clinical reasoning:
1. **Active Problem List** - Current diagnoses by clinical priority
2. **Disease Trajectories** - How conditions have progressed
3. **Risk Stratification** - Risk factors and prognosis{genetics_section}

# Plan
Treatment strategy:
1. Medications - current regimen and changes
2. Monitoring - labs, imaging needed
3. Lifestyle interventions
4. Referrals
5. Follow-up schedule

# Future Considerations
Conditions likely to progress, preventive interventions, screening needs.

# Summary
2-3 sentences on critical findings and priorities.{causal_section}"""

    def _build_direct_prompt(
        self,
        patient_data: str,
        include_images: bool = False,
        num_images: int = 0,
        include_causal_graph: bool = True,
    ) -> str:
        """Build direct (non-hierarchical) prompt with conditional sections."""
        # Simplified user prompt - detailed template is in system prompt
        image_note = f"\n\n{num_images} medical image(s) are attached after this text. Include imaging findings in the Objective section." if include_images else ""
        causal_note = "\n\nInclude a Causal Graph section at the end." if include_causal_graph else ""
        genetics_note = "\n\nInclude genetic risk assessment in the Assessment section." if (self.use_biomcp and not self.separate_genetics) else ""

        return f"""Generate a SOAP note for this patient. Output the SOAP note directly. Do NOT describe your approach or thinking process - just write the note.

=== PATIENT DATA ===
{patient_data}
=== END ==={image_note}{genetics_note}{causal_note}"""

    CAUSAL_GRAPH_PROMPT = """Generate causal relationships for this patient.

=== SOAP NOTE ===
{soap_note}
=== END SOAP NOTE ===

Output relationships for THIS patient's actual conditions and treatments:"""

    # Two-stage grounded causal graph prompts
    ENTITY_EXTRACTION_PROMPT = """Extract the most clinically significant medical entities from this SOAP note.

=== SOAP NOTE ===
{soap_note}
=== END SOAP NOTE ===

RULES:
1. Extract each unique entity ONLY ONCE (no duplicates)
2. Focus on the 30-50 MOST IMPORTANT entities for understanding this patient's health
3. INCLUDE: diagnosed conditions, active medications, key procedures, significant symptoms, abnormal lab findings, relevant lifestyle factors, genetic variants
4. EXCLUDE: normal findings, routine procedures, administrative terms, devices, general concepts
5. Use standard medical terminology (e.g., "Type 2 diabetes mellitus" not "sugar problem")
6. If no medical content, output "NO_ENTITIES"

Extract the key medical entities (one per line):"""

    GROUNDED_RELATIONSHIP_PROMPT = """Identify causal relationships between the medical concepts for this patient.

=== PATIENT CLINICAL SUMMARY ===
{soap_note}
=== END SUMMARY ===

=== MEDICAL CONCEPTS (grounded to SNOMED CT) ===
{concept_list}
=== END CONCEPTS ===

Output causal relationships that explain THIS patient's disease progression, treatment effects, and risk factors.
Focus on relationships that are clinically meaningful for this specific patient.
Output one relationship per line using EXACTLY the concept labels shown above:"""

    # Genetic interpretation prompt (separate agent to avoid output truncation)
    GENETIC_SUMMARY_PROMPT = """You are a clinical geneticist interpreting genetic test results for a patient.

PATIENT CONTEXT (from SOAP note):
{soap_summary}

GENETIC VARIANTS AND ANNOTATIONS:
{genetic_annotations}

YOUR TASK: Write a clinical genetic interpretation that:

1. **Risk Assessment**: For each clinically significant variant, explain what it means for THIS patient given their medical history
2. **Disease Connections**: Connect genetic findings to the patient's existing conditions (e.g., "The rs699 variant in AGT, associated with decreased CAD risk, is relevant given the patient's hypertension history")
3. **Pharmacogenomics**: Note any drug-gene interactions relevant to current or potential medications
4. **Actionable Recommendations**: Suggest genetic counseling, additional testing, or lifestyle modifications based on findings
5. **Risk Modifiers**: Explain how genetic risk factors interact with environmental/lifestyle factors already documented

Write in clinical prose suitable for a physician. Be specific about risk directions (increased vs decreased) and confidence levels.

GENETIC INTERPRETATION:"""

    def __init__(
        self,
        model_id: str = "google/medgemma-1.5-4b-it",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        max_tokens_per_chunk: Optional[int] = None,
        chunk_period_years: Optional[int | str] = "auto",
        max_new_tokens_chunk: int = 8192,
        max_new_tokens_final: int = 8192,
        verbose: Union[bool, int] = True,
        approximate_tokens: bool = False,
        multi_gpu: bool = False,
        gpu_memory_fraction: Optional[float] = None,
        batch_size: Optional[int] = None,
        include_imaging: bool = True,
        max_images: int = 3,
        use_biomcp: bool = False,
        separate_genetics: bool = False,
        separate_causal_graph: bool = False,
        chunk_causal_graph: bool = False,
        include_admissions: bool = False,
        ground_snomed: bool = False,
        snomed_embedding_model: Optional[str] = None,
        snomed_index_name: Optional[str] = None,
        grounded_graph_mode: bool = True,
        entity_extraction_model: Optional[str] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """
        Initialize the SOAP note generator.

        Args:
            model_id: HuggingFace model ID for MedGemma
            device: Device to use ('auto', 'cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
            torch_dtype: Torch dtype ('bfloat16', 'float16', 'float32')
            quantization: Quantization mode for faster inference ('4bit', '8bit', or None)
            max_tokens_per_chunk: Token threshold for triggering hierarchical processing.
                                  If patient data exceeds this, it will be split into chunks
                                  and summarized before generating the final SOAP note.
                                  None = never trigger hierarchical mode based on token count
                                  (hierarchical mode can still be triggered by multiple time chunks).
                                  Default is 128,000 (MedGemma's context limit) when chunk_period_years="auto".
            chunk_period_years: Years per chunk for time-based splitting (default: "auto"):
                                  - "auto" (default) = automatically determine optimal chunk size
                                    based on token count and patient history span. Uses hierarchical
                                    mode only when needed (similar to Claude Code's auto-compaction).
                                  - int (e.g., 5) = fixed 5-year chunks
                                  - None = no chunking, always process all data at once (direct mode)
            max_new_tokens_chunk: Max tokens to generate per chunk summary (default: 8192)
            max_new_tokens_final: Max tokens to generate for final SOAP note (default: 8192)
            verbose: Print progress messages and token statistics
            approximate_tokens: Use fast approximate token counting (chars/4)
            multi_gpu: If True, distribute model across all available GPUs
            gpu_memory_fraction: Fraction of GPU memory to use per device (0.0-1.0).
                                 If None, uses all available memory.
            batch_size: Batch size for generate_batch(). If None (default), automatically
                       optimizes based on available GPU memory. Higher values are faster
                       but use more memory.
            include_imaging: Include DICOM images in analysis (default: True)
            max_images: Maximum number of images to include per patient (default: 3)
            use_biomcp: Use BioMCP to enrich genetic variant annotations with
                        clinical significance, associated conditions, and functional
                        predictions. Requires: pip install biomcp-python
                        See: https://biomcp.org/
            separate_genetics: If True, generate genetic summary in a separate LLM
                        call after the SOAP note (avoids truncation for long outputs).
                        If False (default), include BioMCP annotations directly in
                        the input to the SOAP generator, resulting in integrated
                        genetic interpretation within the SOAP note itself.
            separate_causal_graph: If True, generate the causal graph in a separate
                        prompt after the main SOAP note. This avoids hitting token
                        generation limits. (default: False)
            chunk_causal_graph: If True, include causal graph generation in chunk
                        summaries (for hierarchical mode). By default (False), only
                        the final aggregated SOAP note includes a causal graph.
                        Enable this for detailed per-period causal analysis.
            include_admissions: If True, include Hospital Admissions in the causal
                        graph. If False (default), admissions are excluded from
                        the causal graph to focus on condition-to-condition and
                        condition-to-finding relationships.
            ground_snomed: If True, automatically ground causal graph nodes to
                        SNOMED CT concepts using semantic embeddings. This adds
                        standardized concept IDs to each node. (default: False)
            snomed_embedding_model: Embedding model for SNOMED linking. Options:
                        - "sapbert" (default): Biomedical-specific SapBERT
                        - "qwen3-0.6b": Efficient Qwen3 embeddings
                        - See synthlab.snomed.EMBEDDING_MODELS for full list
                        Only used if ground_snomed=True.
            snomed_index_name: Name of pre-built SNOMED index to use. If set, loads
                        from cache (built with sl.build_snomed_index()). If None,
                        falls back to sample concepts (~300 terms). For full SNOMED
                        (350k concepts), first run:
                            sl.build_snomed_index("/path/to/CONCEPT.csv", index_name="snomed_full")
                        Then set snomed_index_name="snomed_full".
            grounded_graph_mode: If True (default when ground_snomed=True), use a
                        two-stage approach: first extract and ground entities to SNOMED,
                        then identify relationships between grounded concepts. This ensures
                        100% grounding by construction. If False, generates free-form
                        causal graph then attempts retrospective SNOMED mapping.
            entity_extraction_model: Model for extracting medical entities from SOAP notes.
                        If None (default), uses the main MedGemma model (local).
                        If specified, uses MedicalEntityExtractor with an external LLM API:
                        - "gemini-2.0-flash" (recommended, fast and accurate)
                        - "gpt-4o-mini", "gpt-4o" (OpenAI)
                        - "claude-3-haiku-20240307" (Anthropic)
                        Requires appropriate API key in environment (GOOGLE_API_KEY,
                        OPENAI_API_KEY, or ANTHROPIC_API_KEY).
            do_sample: If True, use sampling for text generation with temperature
                        and top_p. If False, use greedy decoding. If None (default), use
                        model's default (greedy for MedGemma).
            repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage
                        repetition. If None (default), use model's default.
            temperature: Sampling temperature. Higher = more creative/random, lower = more
                        focused/deterministic. Only used when do_sample=True. If None, use
                        model's default.
            top_p: Nucleus sampling threshold. Only consider tokens with cumulative
                        probability >= top_p. Only used when do_sample=True. If None, use
                        model's default.
        """
        # Load .env file if present (for API keys like GOOGLE_API_KEY, ONCOKB_TOKEN)
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            # Try loading from current directory first
            load_dotenv()
            # Also try loading from project root (where synthlab is installed)
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not installed, skip

        # Validate and normalize entity extraction model (fail fast)
        if entity_extraction_model:
            # Check if it's the same as the main model or a MedGemma variant -> use local
            is_local_model = (
                entity_extraction_model == model_id
                or "medgemma" in entity_extraction_model.lower()
            )
            if is_local_model:
                entity_extraction_model = None  # Use local model

        # Validate external API models
        if entity_extraction_model:
            model_lower = entity_extraction_model.lower()

            # Normalize model name for litellm (google/gemini-* -> gemini/gemini-*)
            if entity_extraction_model.startswith("google/gemini"):
                entity_extraction_model = entity_extraction_model.replace("google/", "gemini/")

            # Validate API key
            if "gemini" in model_lower and not os.environ.get("GOOGLE_API_KEY"):
                raise ValueError(
                    f"entity_extraction_model='{entity_extraction_model}' requires GOOGLE_API_KEY.\n"
                    "Set it via:\n"
                    "  1. Environment variable: export GOOGLE_API_KEY='your-key'\n"
                    "  2. .env file in project root: GOOGLE_API_KEY=your-key\n"
                    "Get a key at: https://aistudio.google.com/apikey"
                )
            elif "gpt" in model_lower and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    f"entity_extraction_model='{entity_extraction_model}' requires OPENAI_API_KEY.\n"
                    "Set it via:\n"
                    "  1. Environment variable: export OPENAI_API_KEY='your-key'\n"
                    "  2. .env file in project root: OPENAI_API_KEY=your-key"
                )
            elif "claude" in model_lower and not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    f"entity_extraction_model='{entity_extraction_model}' requires ANTHROPIC_API_KEY.\n"
                    "Set it via:\n"
                    "  1. Environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                    "  2. .env file in project root: ANTHROPIC_API_KEY=your-key"
                )

            # Validate LLM client is available and model is recognized
            _has_litellm = False
            try:
                import litellm
                _has_litellm = True

                # Validate model name using litellm's model registry
                try:
                    litellm.get_model_info(entity_extraction_model)
                except Exception:
                    # Model not in litellm's registry - provide helpful error
                    raise ValueError(
                        f"Model '{entity_extraction_model}' not found in litellm's model registry.\n"
                        "Check available models at: https://docs.litellm.ai/docs/providers\n"
                        "Common models:\n"
                        "  - gemini/gemini-2.0-flash, gemini/gemini-1.5-pro\n"
                        "  - gpt-4o-mini, gpt-4o\n"
                        "  - claude-3-haiku-20240307, claude-3-5-sonnet-20241022"
                    )
            except ImportError:
                pass

            # Fallback to google.generativeai for Gemini if litellm not available
            if not _has_litellm and "gemini" in model_lower:
                try:
                    import google.generativeai  # noqa: F401
                except ImportError:
                    raise ImportError(
                        f"entity_extraction_model='{entity_extraction_model}' requires an LLM client.\n"
                        "Install one of:\n"
                        "  pip install litellm            # Recommended: unified API for all models\n"
                        "  pip install google-generativeai  # For Gemini models only"
                    )
            elif not _has_litellm:
                raise ImportError(
                    f"entity_extraction_model='{entity_extraction_model}' requires litellm.\n"
                    "Install with: pip install litellm"
                )

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.quantization = quantization

        # Handle auto-chunking mode
        if chunk_period_years == "auto":
            self.chunk_period_years = "auto"
            # Set default max_tokens_per_chunk if not specified (MedGemma's 128K context limit)
            self.max_tokens_per_chunk = max_tokens_per_chunk if max_tokens_per_chunk is not None else 128_000
        elif chunk_period_years is None or isinstance(chunk_period_years, int):
            self.chunk_period_years = chunk_period_years
            self.max_tokens_per_chunk = max_tokens_per_chunk
        else:
            raise ValueError(
                f"chunk_period_years must be None, int, or 'auto', got: {chunk_period_years!r}"
            )
        self.max_new_tokens_chunk = max_new_tokens_chunk
        self.max_new_tokens_final = max_new_tokens_final
        self.verbose = verbose
        self.approximate_tokens = approximate_tokens
        self.multi_gpu = multi_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.batch_size = batch_size
        self.include_imaging = include_imaging
        self.max_images = max_images
        self.use_biomcp = use_biomcp
        self.separate_genetics = separate_genetics
        self.separate_causal_graph = separate_causal_graph
        self.chunk_causal_graph = chunk_causal_graph
        self.include_admissions = include_admissions
        self.ground_snomed = ground_snomed
        self.snomed_embedding_model = snomed_embedding_model
        self.snomed_index_name = snomed_index_name
        self.grounded_graph_mode = grounded_graph_mode
        self.entity_extraction_model = entity_extraction_model
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p

        # Validate SNOMED grounding dependencies early (fail fast)
        if self.ground_snomed:
            self._validate_snomed_dependencies()

        self._model = None
        self._processor = None
        self._pipe = None
        self._tokenizer = None
        self._optimal_batch_size = None  # Cached after first calculation
        self._snomed_linker = None  # Lazy-loaded SNOMEDLinker for grounding
        self._entity_extractor = None  # Lazy-loaded MedicalEntityExtractor

        # Print configuration summary
        if self.verbose:
            self._print_config()

    def _print_config(self):
        """Print configuration summary."""
        print("SOAPNoteGenerator initialized:")
        print(f"  Model: {self.model_id}")
        print(f"  Device: {self.device}", end="")
        if self.multi_gpu:
            print(" (multi-GPU)")
        else:
            print()
        if self.quantization:
            print(f"  Quantization: {self.quantization}")
        print(f"  Imaging: {'enabled' if self.include_imaging else 'disabled'}", end="")
        if self.include_imaging:
            print(f" (max {self.max_images} images)")
        else:
            print()
        print(f"  Chunking: ", end="")
        if self.chunk_period_years is None:
            print("disabled (direct mode only)")
        elif self.chunk_period_years == "auto":
            print(f"auto (adapts to context, max {self.max_tokens_per_chunk:,} tokens/chunk)")
        else:
            print(f"{self.chunk_period_years} years per chunk")
        print(f"  Generation limits:")
        print(f"    max_new_tokens_chunk: {self.max_new_tokens_chunk:,}")
        print(f"    max_new_tokens_final: {self.max_new_tokens_final:,}")
        sampling_parts = []
        if self.do_sample is not None:
            sampling_parts.append(f"do_sample={self.do_sample}")
        if self.repetition_penalty is not None:
            sampling_parts.append(f"repetition_penalty={self.repetition_penalty}")
        if self.do_sample and self.temperature is not None:
            sampling_parts.append(f"temperature={self.temperature}")
        if self.do_sample and self.top_p is not None:
            sampling_parts.append(f"top_p={self.top_p}")
        if sampling_parts:
            print(f"  Sampling: {', '.join(sampling_parts)}")
        else:
            print("  Sampling: using model defaults")
        if self.use_biomcp:
            print("  BioMCP: enabled")
        if self.separate_causal_graph:
            print("  Separate causal graph: enabled")
        if self.include_admissions:
            print("  Causal graph: including admissions")
        else:
            print("  Causal graph: excluding admissions (default)")
        if self.ground_snomed:
            model_name = self.snomed_embedding_model or "sapbert"
            mode = "two-stage" if self.grounded_graph_mode else "retrospective"
            print(f"  SNOMED grounding: enabled ({model_name}, {mode} mode)")
        if self.entity_extraction_model:
            print(f"  Entity extraction: {self.entity_extraction_model}")

    def _validate_snomed_dependencies(self):
        """Validate that SNOMED grounding dependencies are available.

        Raises ImportError immediately if ground_snomed=True but required
        packages (faiss, torch, transformers, numpy) are not installed.
        """
        # Check faiss availability
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "SNOMED grounding requires faiss but it is not installed.\n"
                "Install with: pip install faiss-cpu\n"
                "Or for GPU support: pip install faiss-gpu\n"
                "Or disable SNOMED grounding: SOAPNoteGenerator(..., ground_snomed=False)"
            )

        # Check other required dependencies
        missing = []
        try:
            import torch
        except ImportError:
            missing.append("torch")

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            missing.append("transformers")

        try:
            import numpy
        except ImportError:
            missing.append("numpy")

        if missing:
            raise ImportError(
                f"SNOMED grounding requires additional packages: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}\n"
                "Or disable SNOMED grounding: SOAPNoteGenerator(..., ground_snomed=False)"
            )

    def _get_admission_exclusion_instruction(self) -> str:
        """Get instruction to exclude admissions from causal graph if needed."""
        if self.include_admissions:
            return ""
        return """

Note: Exclude "Hospital_Admissions" as a node in the causal graph. Do not create edges pointing to Hospital_Admissions. All other medical nodes (conditions, procedures, medications, findings, etc.) should still be included."""

    def _compute_auto_chunk_period(
        self,
        patient: Any,
        total_tokens: int,
    ) -> tuple[int | None, str]:
        """
        Automatically compute optimal chunk period based on patient data.

        This implements Claude Code-style auto-compaction: uses direct mode when
        data fits, otherwise automatically determines chunk size to stay within
        token limits.

        Args:
            patient: Patient object
            total_tokens: Pre-computed total token count of patient data

        Returns:
            (period_years, reason) tuple where:
            - period_years: Optimal years per chunk, or None for direct mode
            - reason: Human-readable explanation of the decision
        """
        # If data fits within limits, use direct mode
        if self.max_tokens_per_chunk is not None and total_tokens <= self.max_tokens_per_chunk:
            return None, f"direct (data fits: {total_tokens:,} <= {self.max_tokens_per_chunk:,} tokens)"

        # Get the date range from patient events
        events = []
        for cond in patient.get_conditions():
            if cond.get("onset_date"):
                events.append(cond["onset_date"][:4])
        for med in patient.get_medications():
            if med.get("authored_on"):
                events.append(med["authored_on"][:4])
        for enc in patient.get_encounters():
            if enc.get("start"):
                events.append(enc["start"][:4])
        for obs in patient.get_observations():
            if obs.get("date"):
                events.append(obs["date"][:4])
        for proc in patient.get_procedures():
            if proc.get("performed_date"):
                events.append(proc["performed_date"][:4])

        if not events:
            return None, "direct (no dated events)"

        # Calculate history span
        years = [int(y) for y in events if y.isdigit()]
        if not years:
            return None, "direct (no valid years)"

        min_year = min(years)
        max_year = max(years)
        history_span = max(max_year - min_year + 1, 1)

        # Calculate tokens per year
        tokens_per_year = total_tokens / history_span

        # Target chunk size: aim for 70% of max to leave headroom
        target_tokens = int(self.max_tokens_per_chunk * 0.7)

        # Calculate optimal period
        if tokens_per_year <= 0:
            optimal_period = history_span
        else:
            optimal_period = max(1, int(target_tokens / tokens_per_year))

        # Clamp to reasonable bounds
        optimal_period = min(optimal_period, history_span)  # Don't exceed history span
        optimal_period = max(1, optimal_period)  # At least 1 year

        # If optimal period >= history span, use direct mode
        if optimal_period >= history_span:
            return None, f"direct ({history_span}-year history fits in one chunk)"

        # Calculate expected number of chunks
        expected_chunks = (history_span + optimal_period - 1) // optimal_period

        reason = (
            f"auto: {optimal_period}y chunks "
            f"({history_span}y history, ~{tokens_per_year:.0f} tok/y, "
            f"~{expected_chunks} chunks)"
        )

        return optimal_period, reason

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        missing = []
        if not _torch_available:
            missing.append("torch")
        if not _transformers_available:
            missing.append("transformers>=4.50.0")
        if not _PIL_available:
            missing.append("Pillow")

        if missing:
            raise ImportError(
                f"Missing required dependencies: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

    def _load_model(self):
        """Load the model and processor."""
        if self._pipe is not None:
            return

        self._check_dependencies()

        if self.verbose:
            print(f"Loading model: {self.model_id}")
            if self.quantization:
                print(f"  Quantization: {self.quantization}")
            if self.multi_gpu:
                print("  Multi-GPU: enabled")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        # Build model kwargs
        model_kwargs = {}

        # Add quantization config if requested
        if self.quantization:
            try:
                from transformers import BitsAndBytesConfig

                if self.quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    if self.verbose:
                        print("  Using 4-bit quantization (faster inference)")
                elif self.quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    if self.verbose:
                        print("  Using 8-bit quantization (faster inference)")
            except ImportError:
                if self.verbose:
                    print("  Warning: bitsandbytes not installed, skipping quantization")
                    print("  Install with: pip install bitsandbytes")

        # Configure device mapping for multi-GPU
        device_map = None
        max_memory = None

        if self.multi_gpu:
            # Use all available GPUs
            device_map = "auto"
            if self.gpu_memory_fraction is not None and torch.cuda.is_available():
                # Set memory limit per GPU
                n_gpus = torch.cuda.device_count()
                max_memory = {}
                for i in range(n_gpus):
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    max_mem = int(total_mem * self.gpu_memory_fraction)
                    max_memory[i] = f"{max_mem // (1024**3)}GiB"
                if self.verbose:
                    print(f"  GPUs detected: {n_gpus}")
                    for gpu_id, mem in max_memory.items():
                        print(f"    GPU {gpu_id}: max {mem}")
        elif self.device == "auto":
            # Use single GPU (cuda:0) when multi_gpu is False
            device_map = {"": 0} if torch.cuda.is_available() else "auto"

        try:
            # Match MedGemma docs: https://huggingface.co/google/medgemma-1.5-4b-it
            # Note: torch_dtype is deprecated in newer transformers, use dtype
            pipeline_kwargs = {
                "task": "image-text-to-text",
                "model": self.model_id,
                "dtype": dtype,
                "use_fast": True,  # Use fast tokenizer and processor
            }

            # Try to load processor with use_fast=True to avoid deprecation warning
            try:
                processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
                pipeline_kwargs["processor"] = processor
            except Exception:
                pass  # Fall back to pipeline's default processor loading

            # Add device configuration
            if device_map:
                pipeline_kwargs["device_map"] = device_map
                if max_memory:
                    pipeline_kwargs["max_memory"] = max_memory
            elif self.device and self.device != "auto":
                pipeline_kwargs["device"] = self.device

            # Enable flash attention 2 if available (significant speedup)
            if "attn_implementation" not in model_kwargs:
                try:
                    import flash_attn  # noqa: F401
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass  # flash_attn not installed, use default attention

            if model_kwargs:
                pipeline_kwargs["model_kwargs"] = model_kwargs

            self._pipe = pipeline(**pipeline_kwargs)

            # Enable torch.compile for faster inference (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile') and self._pipe.model is not None:
                    # Use reduce-overhead mode for inference
                    self._pipe.model = torch.compile(
                        self._pipe.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    if self.verbose:
                        print("  torch.compile: enabled (reduce-overhead mode)")
            except Exception as compile_err:
                if self.verbose:
                    print(f"  torch.compile: disabled ({compile_err})")

            # Get tokenizer for token counting
            self._tokenizer = self._pipe.tokenizer

            # Configure generation settings if provided (otherwise use model defaults)
            if self._pipe.model is not None and hasattr(self._pipe.model, 'generation_config'):
                gen_config = self._pipe.model.generation_config
                config_msgs = []
                if self.do_sample is not None:
                    gen_config.do_sample = self.do_sample
                    config_msgs.append(f"do_sample={self.do_sample}")
                if self.repetition_penalty is not None:
                    gen_config.repetition_penalty = self.repetition_penalty
                    config_msgs.append(f"repetition_penalty={self.repetition_penalty}")
                if self.do_sample and self.temperature is not None:
                    gen_config.temperature = self.temperature
                    config_msgs.append(f"temperature={self.temperature}")
                if self.do_sample and self.top_p is not None:
                    gen_config.top_p = self.top_p
                    config_msgs.append(f"top_p={self.top_p}")
                if self.verbose and config_msgs:
                    print(f"  Generation config: {', '.join(config_msgs)}")

            if self.verbose:
                # Get device info
                model = self._pipe.model
                if hasattr(model, 'hf_device_map'):
                    devices = set(model.hf_device_map.values())
                    # Format as cuda0, cuda1, etc.
                    formatted = []
                    for d in sorted(devices, key=lambda x: (str(type(x)), str(x))):
                        if isinstance(d, int):
                            formatted.append(f"cuda{d}")
                        elif str(d).isdigit():
                            formatted.append(f"cuda{d}")
                        else:
                            formatted.append(str(d))
                    device_info = ", ".join(formatted)
                    print(f"✓ Model distributed across: {device_info}")
                elif hasattr(model, 'device'):
                    device_info = str(model.device)
                    # Format cuda:0 as cuda0
                    device_info = device_info.replace("cuda:", "cuda")
                    print(f"✓ Model loaded on: {device_info}")
                else:
                    print("✓ Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer for accurate counts."""
        if getattr(self, "approximate_tokens", False):
            return count_tokens(text, approximate=True)

        # Ensure model is loaded so we have the tokenizer
        if self._tokenizer is None:
            self._load_model()

        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return count_tokens(text, approximate=True)

    def _get_optimal_batch_size(self, include_imaging: bool = False) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        The calculation considers:
        - Total available GPU memory across all devices
        - Model memory footprint (estimated from model size)
        - Per-request memory overhead (KV cache, activations)
        - Whether images are included (increases memory per request)

        Returns:
            Optimal batch size (minimum 1, maximum 32)
        """
        # Return cached value if available
        if self._optimal_batch_size is not None:
            return self._optimal_batch_size

        # If user specified batch_size, use that
        if self.batch_size is not None:
            self._optimal_batch_size = self.batch_size
            return self.batch_size

        # Default for CPU
        if not _torch_available or not torch.cuda.is_available():
            self._optimal_batch_size = 1
            return 1

        try:
            # Get total GPU memory
            n_gpus = torch.cuda.device_count()
            total_memory_gb = 0

            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_mem_gb = props.total_memory / (1024 ** 3)

                # Apply memory fraction if specified
                if self.gpu_memory_fraction is not None:
                    gpu_mem_gb *= self.gpu_memory_fraction

                total_memory_gb += gpu_mem_gb

            # Estimate model memory footprint based on model ID
            # MedGemma 4B ~ 8GB in bfloat16, ~4GB in 4bit, ~6GB in 8bit
            model_size_map = {
                "4b": 8.0,
                "8b": 16.0,
                "2b": 4.0,
            }
            model_mem_gb = 8.0  # Default estimate
            model_id_lower = self.model_id.lower()
            for size_key, mem in model_size_map.items():
                if size_key in model_id_lower:
                    model_mem_gb = mem
                    break

            # Adjust for quantization
            if self.quantization == "4bit":
                model_mem_gb *= 0.5
            elif self.quantization == "8bit":
                model_mem_gb *= 0.75

            # Available memory after model is loaded
            available_gb = total_memory_gb - model_mem_gb

            # Per-request memory overhead (KV cache + activations)
            # These are conservative estimates - actual usage depends on sequence length
            # For short sequences (~2K tokens): ~0.1-0.2GB per request
            # For medium sequences (~8K tokens): ~0.3-0.5GB per request
            # With images: add ~0.5GB per image
            per_request_gb = 0.8 if include_imaging else 0.25

            # Calculate batch size
            batch_size = max(1, int(available_gb / per_request_gb))

            # Dynamic max based on GPU memory class
            # High-memory GPUs (A100, H100) can handle larger batches
            if total_memory_gb > 100:  # Multi-GPU with lots of memory
                max_batch = 128
            elif total_memory_gb > 40:  # Single A100 or similar
                max_batch = 64
            else:
                max_batch = 32

            batch_size = min(max_batch, max(1, batch_size))

            if self.verbose:
                print(f"  Auto batch size: {batch_size} "
                      f"(GPUs: {n_gpus}, total mem: {total_memory_gb:.1f}GB, "
                      f"available: {available_gb:.1f}GB, max: {max_batch})")

            self._optimal_batch_size = batch_size
            return batch_size

        except Exception:
            # Fallback to conservative default
            self._optimal_batch_size = 2
            return 2

    def _clean_response(self, text: str) -> str:
        """Remove thinking tokens, artifacts, and repetitions from model output."""
        import re

        # Remove thinking blocks with proper closing tags (e.g., <unused94>thought ... </unused94>)
        text = re.sub(r'<unused\d+>thought.*?</unused\d+>', '', text, flags=re.DOTALL)

        # Remove <thinking> blocks with proper closing tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

        # If response still starts with thinking token without closing tag,
        # try to find where actual content begins (look for section headers)
        if text.strip().startswith('<unused') and 'thought' in text[:50].lower():
            # Look for common section markers that indicate start of actual content
            section_markers = [
                r'\n#\s',           # Markdown header
                r'\n\*\*[A-Z]',    # Bold header
                r'\nSUBJECTIVE',   # SOAP sections
                r'\nOBJECTIVE',
                r'\nASSESSMENT',
                r'\nPLAN',
                r'\nPATIENT STORY',
            ]
            for marker in section_markers:
                match = re.search(marker, text, re.IGNORECASE)
                if match:
                    # Found actual content - extract from this point
                    text = text[match.start():]
                    break
            else:
                # No section markers found - the entire response may be thinking
                # Just remove the thinking prefix and see what's left
                text = re.sub(r'^<unused\d+>thought\s*', '', text)

        # Remove any remaining standalone <unusedXX> tokens
        text = re.sub(r'</?unused\d+>', '', text)

        # Detect and remove repetition loops
        text = self._remove_repetition_loops(text)

        return text.strip()

    def _remove_repetition_loops(self, text: str) -> str:
        """Detect and truncate repetition loops in model output.

        When models get stuck in repetition loops, they often repeat
        the same phrase many times. This detects such patterns and
        truncates to just the first occurrence.
        """
        import re

        # Split into lines
        lines = text.split('\n')

        # Detect repeated lines (same line appearing 3+ times consecutively)
        cleaned_lines = []
        prev_line = None
        repeat_count = 0
        max_repeats = 2  # Allow at most 2 identical consecutive lines

        for line in lines:
            line_stripped = line.strip()
            if line_stripped == prev_line and line_stripped:
                repeat_count += 1
                if repeat_count < max_repeats:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
                repeat_count = 0
                prev_line = line_stripped

        text = '\n'.join(cleaned_lines)

        # Detect repeated phrases within content (20+ chars appearing 5+ times)
        # This catches "Patient reports no recent falls. Patient reports no recent falls."
        phrase_pattern = r'(.{20,}?)\1{4,}'
        match = re.search(phrase_pattern, text)
        if match:
            # Found a repeated phrase - keep only the first occurrence
            repeated_phrase = match.group(1)
            # Replace multiple occurrences with single
            text = re.sub(re.escape(repeated_phrase) + r'(' + re.escape(repeated_phrase) + r')+',
                         repeated_phrase, text)

        return text

    def _generate_text(
        self,
        prompt: str,
        images: Optional[list] = None,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the model.

        Args:
            prompt: The user prompt
            images: Optional list of images for multimodal input
            max_new_tokens: Max tokens to generate
            system_prompt: Optional system prompt for structured output tasks
        """
        self._load_model()

        # Build messages list
        messages = []

        # Add system prompt if provided (helps with structured output)
        # Must use list format for content to match processor expectations
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Build user message content - always use list format for MedGemma
        # Text FIRST, then images - helps model retain context from clinical data
        content = []
        content.append({"type": "text", "text": prompt})
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        messages.append({"role": "user", "content": content})

        # Build generate kwargs - only include non-None values to let model use defaults
        generate_kwargs = {}
        if self.do_sample is not None:
            generate_kwargs["do_sample"] = self.do_sample
        if self.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.do_sample and self.temperature is not None:
            generate_kwargs["temperature"] = self.temperature
        if self.do_sample and self.top_p is not None:
            generate_kwargs["top_p"] = self.top_p
        # Always set max_new_tokens - use default if not specified
        generate_kwargs["max_new_tokens"] = max_new_tokens if max_new_tokens is not None else self.max_new_tokens_final

        try:
            if self.verbose:
                print(f"    [DEBUG] Pipeline input messages: {repr(messages)[:500]}")
                print(f"    [DEBUG] Generate kwargs: {generate_kwargs}")

            output = self._pipe(text=messages, **generate_kwargs)

            if self.verbose:
                print(f"    [DEBUG] Pipeline output type: {type(output)}")
                print(f"    [DEBUG] Pipeline output: {repr(output)[:1000]}")

            # image-text-to-text pipeline returns: [{"generated_text": [...messages...]}]
            # Extract response via: output[0]["generated_text"][-1]["content"]
            # See: https://huggingface.co/google/medgemma-1.5-4b-it
            if not isinstance(output, list) or len(output) == 0:
                raise RuntimeError(f"Unexpected output type: {type(output)}, expected list")

            item = output[0]
            if not isinstance(item, dict) or "generated_text" not in item:
                raise RuntimeError(f"Unexpected output format. Keys: {item.keys() if isinstance(item, dict) else 'N/A'}")

            gen_text = item["generated_text"]

            # Per MedGemma docs: output[0]["generated_text"][-1]["content"]
            # generated_text is a list of messages, last one is assistant's response
            if not isinstance(gen_text, list) or len(gen_text) == 0:
                raise RuntimeError(f"Expected generated_text to be non-empty list, got: {type(gen_text)}")

            last_msg = gen_text[-1]

            if self.verbose:
                print(f"    [DEBUG] generated_text has {len(gen_text)} messages")
                print(f"    [DEBUG] last_msg type: {type(last_msg)}")
                print(f"    [DEBUG] last_msg: {repr(last_msg)[:500]}")

            # Extract content from last message
            if isinstance(last_msg, dict) and "content" in last_msg:
                response = last_msg["content"]
                # Content should be a string per MedGemma docs
                if not isinstance(response, str):
                    raise RuntimeError(f"Expected content to be string, got {type(response)}: {repr(response)[:200]}")
            elif isinstance(last_msg, str):
                # Fallback: last_msg is directly the response string
                response = last_msg
            else:
                raise RuntimeError(f"Cannot extract content from last message: {repr(last_msg)[:500]}")

            # Fail if response is empty
            if not response or not response.strip():
                raise RuntimeError(
                    f"Model generated empty output.\n"
                    f"  max_new_tokens: {generate_kwargs.get('max_new_tokens')}\n"
                    f"  generated_text: {repr(gen_text)[:200]}"
                )

            return self._clean_response(response)
        except Exception as e:
            if self.verbose:
                print(f"    [DEBUG] Generation exception: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Generation failed: {e}")

    def _generate_text_batch(
        self,
        prompts: list[str],
        images_list: Optional[list[Optional[list]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """Generate text for multiple prompts in a true GPU batch.

        Args:
            prompts: List of prompt strings
            images_list: Optional list of image lists (one per prompt). If provided,
                         falls back to sequential processing since multimodal batching
                         is more complex.
            max_new_tokens: Maximum tokens to generate per response
        """
        self._load_model()

        # If images are provided, fall back to sequential processing
        if images_list is not None and any(images_list):
            results = []
            for prompt, images in zip(prompts, images_list):
                response = self._generate_text(prompt, images, max_new_tokens)
                results.append(response)
            return results

        model = self._pipe.model
        tokenizer = self._tokenizer

        # Ensure pad token is set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Format prompts as chat messages and apply chat template
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)

        # Tokenize all prompts together with left padding for batch generation
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
        )

        # Move to appropriate device (handle multi-GPU case)
        if hasattr(model, "hf_device_map"):
            # For multi-GPU, inputs go to the first device in the map
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                first_device = f"cuda:{first_device}"
            inputs = inputs.to(first_device)
        elif hasattr(model, "device"):
            inputs = inputs.to(model.device)
        else:
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # Build generate kwargs
        generate_kwargs = {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        try:
            # Generate in a true batch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            # Decode outputs, removing the input prompt, and clean
            results = []
            for i, output in enumerate(outputs):
                # Get only the newly generated tokens
                input_len = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_len:]
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                results.append(self._clean_response(text))

            return results
        except Exception as e:
            raise RuntimeError(f"Batch generation failed: {e}")

    def _get_progress_bar(self, iterable, desc: str, total: int = None):
        """Get a progress bar or plain iterator."""
        if _tqdm_available and self.verbose:
            return _tqdm(iterable, desc=desc, total=total)
        return iterable

    def generate(
        self,
        patient: Any,
        force_hierarchical: bool = False,
        include_imaging: Optional[bool] = None,
        max_images: Optional[int] = None,
        use_biomcp: Optional[bool] = None,
        ground_snomed: Optional[bool] = None,
        snomed_embedding_model: Optional[str] = None,
        snomed_index_name: Optional[str] = None,
    ) -> SOAPNote:
        """
        Generate a SOAP note for a patient.

        Args:
            patient: Patient object from MultimodalDataset
            force_hierarchical: Force hierarchical summarization even for short histories
            include_imaging: Include DICOM images in analysis (overrides init setting)
            max_images: Maximum number of images to include (overrides init setting)
            use_biomcp: Use BioMCP for variant enrichment (overrides init setting)
            ground_snomed: Ground causal graph to SNOMED CT (overrides init setting)
            snomed_embedding_model: Embedding model for SNOMED linking (overrides init setting).
                Options: "sapbert", "qwen3-0.6b", "gte-qwen2", etc.
            snomed_index_name: Name of pre-built SNOMED index (overrides init setting).
                Use "snomed_full" for full vocabulary after building with build_snomed_index().

        Returns:
            SOAPNote object with structured output
        """
        # Use instance defaults if not overridden
        include_imaging = include_imaging if include_imaging is not None else self.include_imaging
        max_images = max_images if max_images is not None else self.max_images
        use_biomcp = use_biomcp if use_biomcp is not None else self.use_biomcp
        ground_snomed = ground_snomed if ground_snomed is not None else self.ground_snomed
        snomed_embedding_model = snomed_embedding_model if snomed_embedding_model is not None else self.snomed_embedding_model
        # Temporarily set index name if overridden for this call
        if snomed_index_name is not None:
            old_index_name = self.snomed_index_name
            self.snomed_index_name = snomed_index_name

        patient_name = patient.name or patient.patient_id[:8]

        # Calculate total steps for progress bar
        # Steps: [biomcp] + [images] + soap_generation + [causal_graph] + [snomed_grounding] + [genetic_summary] + parsing
        total_steps = 1  # SOAP generation
        total_steps += 1  # Parsing
        if use_biomcp:
            total_steps += 1  # BioMCP enrichment
            if self.separate_genetics:
                total_steps += 1  # Genetic interpretation summary (separate LLM call)
        if include_imaging and hasattr(patient, 'dicom_paths') and patient.dicom_paths:
            total_steps += 1  # Image loading
        if self.separate_causal_graph:
            total_steps += 1  # Causal graph generation
        if ground_snomed:
            total_steps += 1  # SNOMED grounding

        # Create progress bar early
        pbar = None
        if _tqdm_available and self.verbose:
            pbar = _tqdm(total=total_steps, desc=f"SOAP: {patient_name[:12]}")
            pbar.set_postfix_str("Formatting patient data")

        # Get patient data (with optional BioMCP timing)
        biomcp_seconds = 0.0
        if use_biomcp:
            if pbar:
                pbar.set_postfix_str("BioMCP: enriching variants")
            elif self.verbose:
                print(f"  BioMCP: enriching genetic variants...")

            biomcp_start = time.perf_counter()
            # When separate_genetics=True: exclude genomics from main text, generate separately
            # When separate_genetics=False: include genomics inline in the SOAP generation
            debug_verbose = isinstance(self.verbose, int) and self.verbose >= 2
            include_genomics_inline = not self.separate_genetics
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(
                patient, use_biomcp=True, include_genomics=include_genomics_inline, verbose=debug_verbose
            )
            biomcp_seconds = time.perf_counter() - biomcp_start

            if pbar:
                pbar.update(1)
                n_variants = len(biomcp_annotations) if biomcp_annotations else 0
                pbar.set_postfix_str(f"BioMCP: {n_variants} variants ({biomcp_seconds:.1f}s)")
            elif self.verbose:
                print(f"  BioMCP enrichment: {biomcp_seconds:.2f}s")
                if debug_verbose:
                    print(f"    [DEBUG] biomcp_annotations has {len(biomcp_annotations)} entries")
        else:
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(patient, use_biomcp=False)
        text_tokens = len(patient_text) // 4  # Rough estimate

        # Load images if requested
        images = []
        if include_imaging and patient.dicom_paths:
            if pbar:
                pbar.set_postfix_str(f"Loading {len(patient.dicom_paths)} DICOM images")
            images = self._load_patient_images(patient, max_images)
            if pbar:
                pbar.update(1)

        # Determine chunk period (handles "auto" mode)
        auto_reason = None
        if self.chunk_period_years == "auto":
            # Auto-compute optimal chunk period based on data size
            effective_period, auto_reason = self._compute_auto_chunk_period(patient, text_tokens)
        elif isinstance(self.chunk_period_years, int):
            effective_period = self.chunk_period_years
        else:
            effective_period = None

        # Pre-compute time chunks to decide on approach (only if chunking is enabled)
        if effective_period is not None:
            chunks = FHIRFormatter.chunk_by_time_period(patient, effective_period)
            n_chunks = len(chunks)
        else:
            chunks = []
            n_chunks = 1  # Treat all data as one chunk for display

        # Decide on approach:
        # - Use hierarchical if forced, or token count exceeds threshold, or multiple chunks exist
        # - Skip hierarchical entirely if effective_period is None
        use_hierarchical = (
            effective_period is not None and (
                force_hierarchical or
                (self.max_tokens_per_chunk is not None and text_tokens > self.max_tokens_per_chunk) or
                n_chunks > 2  # Use hierarchical for histories spanning more than 2 time periods
            )
        )

        # Report generation mode (only if no progress bar)
        if self.verbose and not pbar:
            print(f"Generating SOAP note for: {patient_name}")
            print(f"  Input: ~{text_tokens:,} tokens", end="")
            if effective_period is not None:
                print(f" | {n_chunks} time chunk(s)", end="")
            if images:
                print(f" | {len(images)} image(s)")
            else:
                print()
            # Show auto-chunking decision if in auto mode
            if auto_reason:
                print(f"  Chunking: {auto_reason}")
            if use_hierarchical:
                reasons = []
                if force_hierarchical:
                    reasons.append("forced")
                if self.max_tokens_per_chunk is not None and text_tokens > self.max_tokens_per_chunk:
                    reasons.append(f"tokens > {self.max_tokens_per_chunk:,}")
                if n_chunks > 2:
                    reasons.append(f"{n_chunks} time chunks")
                print(f"  Processing: HIERARCHICAL ({', '.join(reasons)})")
            else:
                print("  Processing: DIRECT")

        # Generate SOAP note
        if use_hierarchical:
            if pbar:
                pbar.set_postfix_str(f"Generating SOAP (hierarchical, {n_chunks} chunks)")
            result = self._generate_hierarchical(patient, images, biomcp_annotations, chunks, biomcp_seconds, pbar=pbar)
        else:
            if pbar:
                pbar.set_postfix_str("Generating SOAP note")
            result = self._generate_direct(patient, patient_text, images, biomcp_annotations, biomcp_seconds, pbar=pbar)

        # Ground causal graph to SNOMED CT if enabled
        if ground_snomed:
            if pbar:
                pbar.set_postfix_str("SNOMED: grounding causal graph")

            if self.grounded_graph_mode:
                # Two-stage approach: extract entities, ground, then build relationships
                result = self._generate_grounded_causal_graph(result, embedding_model=snomed_embedding_model, pbar=pbar)
            else:
                # Retrospective grounding of free-form graph
                result = self._ground_causal_graph(result, embedding_model=snomed_embedding_model)

            if pbar:
                pbar.update(1)

        # Generate genetic interpretation summary if separate_genetics=True and BioMCP data is available
        # When separate_genetics=False, genetics were already included inline in the SOAP generation
        debug_verbose = isinstance(self.verbose, int) and self.verbose >= 2
        has_genetic_data = hasattr(patient, 'genomics') and patient.genomics is not None and len(patient.genomics) > 0

        if debug_verbose:
            print(f"    [DEBUG] Genetic summary check: separate_genetics={self.separate_genetics}, biomcp_annotations={bool(biomcp_annotations)} ({len(biomcp_annotations) if biomcp_annotations else 0} entries), use_biomcp={self.use_biomcp}, has_genetic_data={has_genetic_data}")

        # Warn if patient has genetic data but we got no BioMCP annotations
        if has_genetic_data and use_biomcp and not biomcp_annotations:
            if self.verbose:
                print(f"  [WARNING] Patient has genetic data ({len(patient.genomics)} variants) but BioMCP returned 0 annotations")
                print(f"  [WARNING] Genetic interpretation may be incomplete. Check BioMCP installation/API.")

        # Only generate separate genetic summary if separate_genetics=True
        if self.separate_genetics and biomcp_annotations and self.use_biomcp:
            genetic_summary = self._generate_genetic_summary(result, biomcp_annotations, pbar=pbar)

            # Error if genetic summary generation failed
            if not genetic_summary or not genetic_summary.strip():
                raise RuntimeError(
                    f"Genetic summary generation returned empty result. "
                    f"BioMCP provided {len(biomcp_annotations)} annotations but LLM returned no summary."
                )

            result.genetic_summary = genetic_summary
            # Append to raw_response so it's included in full output
            if genetic_summary and result.raw_response:
                result.raw_response = result.raw_response.rstrip() + "\n\n# Genetic Interpretation\n" + genetic_summary
        elif debug_verbose:
            if not self.separate_genetics:
                print(f"    [DEBUG] Skipping separate genetic summary: separate_genetics=False (genetics included inline)")
            elif not self.use_biomcp:
                print(f"    [DEBUG] Skipping genetic summary: use_biomcp is False")
            elif not biomcp_annotations:
                print(f"    [DEBUG] Skipping genetic summary: biomcp_annotations is empty")

        # Close progress bar
        if pbar:
            pbar.set_postfix_str("Done")
            pbar.close()

        # Restore original index name if we temporarily overrode it
        if snomed_index_name is not None:
            self.snomed_index_name = old_index_name

        return result

    def _load_snomed_linker(self, embedding_model: Optional[str] = None) -> "SNOMEDLinker":
        """Load or return cached SNOMEDLinker.

        Args:
            embedding_model: Override the embedding model (if different from cached linker,
                           a new linker will be created)
        """
        from synthlab.snomed import (
            SNOMEDLinker, EMBEDDING_MODELS, get_sample_snomed_concepts,
            load_snomed_linker, list_snomed_indices, build_snomed_index,
            load_snomed_from_omop,
        )
        from synthlab.download_snomed import get_concept_csv_path, is_snomed_available

        # Determine which model to use
        model_name = embedding_model or self.snomed_embedding_model or "sapbert"
        model_id = EMBEDDING_MODELS.get(model_name, model_name)

        # Check if we need to create a new linker (different model or not cached)
        if self._snomed_linker is not None:
            # Return cached linker if same model
            if self._snomed_linker.model_id == model_id:
                return self._snomed_linker
            # Otherwise fall through to create new linker

        if self.verbose:
            print(f"  Loading SNOMED linker ({model_name})...")

        # If index_name is specified, try to load from cache
        if self.snomed_index_name:
            try:
                self._snomed_linker = load_snomed_linker(
                    index_name=self.snomed_index_name,
                    model_id=model_id,
                    verbose=self.verbose,
                )
                return self._snomed_linker
            except FileNotFoundError:
                if self.verbose:
                    print(f"  WARNING: Index '{self.snomed_index_name}' not found!")
                    print(f"           Build it first with: sl.build_snomed_index(omop_path, index_name='{self.snomed_index_name}')")
                    print(f"           Falling back to sample concepts...")

        # Try to load any existing full index matching the current model
        model_suffix = SNOMEDLinker._get_model_cache_suffix(model_id)
        indices = list_snomed_indices(verbose=False)
        for idx_name, info in indices.items():
            # Only consider indices built with the same model
            if not idx_name.endswith(f"_{model_suffix}"):
                continue
            if info.get('n_concepts', 0) > 1000:  # Likely a full index
                if self.verbose:
                    print(f"  Found existing index '{idx_name}' ({info['n_concepts']:,} concepts)")
                try:
                    # Strip the model suffix to get the base index name
                    base_name = idx_name.rsplit(f"_{model_suffix}", 1)[0]
                    self._snomed_linker = load_snomed_linker(
                        index_name=base_name,
                        model_id=model_id,
                        verbose=self.verbose,
                    )
                    return self._snomed_linker
                except Exception:
                    pass  # Try next or fall back

        # Fall back: download SNOMED vocabulary and build full index
        if self.verbose:
            print(f"  No SNOMED index found. Building full index...")

        # Download CONCEPT.csv if not available
        concept_csv_path = get_concept_csv_path()  # Auto-downloads if needed

        # Load concepts and build index
        if self.verbose:
            print(f"  Loading SNOMED concepts from {concept_csv_path.name}...")
        concepts = load_snomed_from_omop(str(concept_csv_path), verbose=self.verbose)

        if self.verbose:
            print(f"  Building SNOMED index with {len(concepts):,} concepts...")
            print(f"  (This is a one-time operation, index will be cached)")

        self._snomed_linker = SNOMEDLinker(
            model_id=model_id,
            verbose=self.verbose,
        )
        self._snomed_linker.build_index(concepts)

        return self._snomed_linker

    def _ground_causal_graph(
        self,
        soap_note: SOAPNote,
        embedding_model: Optional[str] = None,
    ) -> SOAPNote:
        """Ground the causal graph nodes to SNOMED CT concepts.

        Args:
            soap_note: The SOAP note to ground
            embedding_model: Embedding model to use for linking
        """
        # Get the causal graph
        try:
            graph = soap_note.extract_causal_graph()
            if not graph.edges:
                if self.verbose:
                    print("  Causal graph is empty, skipping SNOMED grounding")
                return soap_note

            # Get or load the linker
            linker = self._load_snomed_linker(embedding_model=embedding_model)

            # Ground the graph
            if self.verbose:
                print(f"  Grounding {len(graph.get_nodes())} causal graph nodes to SNOMED CT...")

            grounded = graph.ground_to_snomed(linker=linker, verbose=self.verbose)

            # Store on the SOAP note
            soap_note.grounded_causal_graph = grounded

            if self.verbose:
                matched = len([n for n in grounded.nodes if n.confidence > 0])
                print(f"  Grounded {matched}/{len(grounded.nodes)} nodes to SNOMED CT")

        except Exception as e:
            if self.verbose:
                import traceback
                error_msg = str(e) if str(e) else repr(e)
                print(f"  Warning: SNOMED grounding failed: {type(e).__name__}: {error_msg}")
                traceback.print_exc()

        return soap_note

    def _generate_grounded_causal_graph(
        self,
        soap_note: SOAPNote,
        embedding_model: Optional[str] = None,
        pbar: Optional[Any] = None,
    ) -> SOAPNote:
        """Generate causal graph using two-stage grounded approach.

        This method:
        1. Extracts medical entities from the SOAP note text
        2. Grounds each entity to SNOMED CT
        3. Asks the LLM to identify relationships between grounded concepts

        This ensures 100% grounding by construction (all nodes are valid SNOMED concepts).

        Args:
            soap_note: The SOAP note containing generated text
            embedding_model: Embedding model for SNOMED linking

        Returns:
            SOAPNote with grounded_causal_graph populated
        """
        from synthlab.snomed import (
            GroundedNode,
            GroundedEdge,
            GroundedCausalGraph,
        )

        try:
            # Get the linker
            if pbar:
                pbar.set_postfix_str("SNOMED: loading linker")
            linker = self._load_snomed_linker(embedding_model=embedding_model)

            # Stage 1: Extract entities from SOAP note
            if pbar:
                pbar.set_postfix_str("SNOMED: extracting entities")
            elif self.verbose:
                print("  Stage 1: Extracting medical entities...")

            # Use cleaned SOAP text for entity extraction
            # If the SOAP note has garbage/repetition, the entity extraction will also fail
            soap_text = str(soap_note)

            # Additional cleaning for entity extraction - remove any remaining repetition
            soap_text = self._remove_repetition_loops(soap_text)

            entities = []
            entity_types = {}

            # Use external LLM for entity extraction if specified
            if self.entity_extraction_model:
                from synthlab.snomed import MedicalEntityExtractor
                # Lazy-load and cache the extractor
                if self._entity_extractor is None:
                    self._entity_extractor = MedicalEntityExtractor(
                        model=self.entity_extraction_model,
                        verbose=self.verbose,
                    )
                    if self.verbose:
                        print(f"    Entity extraction model: {self._entity_extractor.model}")
                extracted = self._entity_extractor.extract(soap_text)
                for ent in extracted:
                    # Use standardized form for better SNOMED matching
                    entity_name = ent.standardized or ent.raw_text
                    if entity_name and len(entity_name) > 1:
                        entities.append(entity_name)
                        entity_types[entity_name] = ent.entity_type
            else:
                # Use local MedGemma model
                entity_prompt = self.ENTITY_EXTRACTION_PROMPT.format(soap_note=soap_text)
                entity_system = """You output ONLY a plain text list of medical entities, one per line.
No JSON, no markdown, no explanations, no commentary.

EXAMPLE OUTPUT:
Type 2 diabetes mellitus
Hypertension
Metformin
Colonoscopy
Chest pain
Elevated HbA1c
Obesity
BRCA1 mutation"""
                entity_response = self._generate_text(
                    entity_prompt,
                    max_new_tokens=2000,
                    system_prompt=entity_system,
                )

                # Check for empty/no-content responses
                response_stripped = entity_response.strip()
                if not response_stripped or response_stripped.upper() == "NO_ENTITIES":
                    if self.verbose:
                        print("    No entities found in SOAP note")
                    return soap_note

                # Parse extracted entities - one per line
                for line in response_stripped.split("\n"):
                    line = line.strip()
                    # Skip empty, comment, or bullet lines
                    if not line or line.startswith("#"):
                        continue
                    # Skip JSON artifacts (model returned JSON instead of plain text)
                    if line in ['```json', '```', '[', ']', '{', '}'] or line.startswith('```'):
                        continue
                    if line.startswith('{') and line.endswith('}'):
                        # Try to extract entity from JSON object like {"entity": "value"}
                        import re
                        match = re.search(r'"entity"\s*:\s*"([^"]+)"', line)
                        if match:
                            line = match.group(1)
                        else:
                            continue
                    # Clean up common prefixes (bullets, numbers, dashes)
                    line = line.lstrip("-•*0123456789.) ").strip()
                    if not line:
                        continue
                    # Skip lines that look like reasoning/instructions
                    line_lower = line.lower()
                    if any(marker in line_lower for marker in [
                        "i ", "the user", "want", "need", "identify", "extract",
                        "should", "will", "let me", "here are", "following",
                        "entities", "soap note", "output", "no medical", "no entities"
                    ]):
                        continue
                    # Handle legacy format with "|" if present (backward compatibility)
                    if "|" in line:
                        line = line.split("|")[0].strip()
                    # Validate entity: reasonable length, not just punctuation
                    if line and len(line) > 2 and len(line) < 100 and any(c.isalpha() for c in line):
                        entities.append(line)
                        entity_types[line] = None  # Will use SNOMED semantic_type

            if self.verbose:
                print(f"    Extracted {len(entities)} entities")

            if not entities:
                if self.verbose:
                    print("  No entities extracted, skipping causal graph")
                return soap_note

            # Stage 2: Ground entities to SNOMED
            if pbar:
                pbar.set_postfix_str(f"SNOMED: grounding {len(entities)} entities")
            elif self.verbose:
                print("  Stage 2: Grounding to SNOMED CT...")

            # Link all entities (threshold 0.7 for higher quality matches, reduce false positives)
            results = linker.link_batch_with_cache(
                entities,
                k=1,
                threshold=0.7,
                track_gaps=True,
            )

            # Build grounded nodes (only keep high-confidence matches)
            grounded_concepts = {}  # concept_id -> GroundedNode
            entity_to_concept = {}  # entity_name -> concept_id

            # Track mapping statistics
            matched_count = 0
            unmatched_entities = []
            duplicate_mappings = 0

            # Reject obviously generic/garbage terms that indicate poor entity extraction
            # These are meta-terms, not actual medical concepts
            generic_terms_to_reject = {
                # Meta-terms
                "thought", "identified", "assessment", "observation", "event",
                "situation", "context", "unknown", "other", "unspecified", "general",
                # Technical/non-medical terms that often appear in SNOMED
                "transformer", "generator", "cleaner", "translator", "classified",
                "initial", "artificial", "linear", "calculus", "stat", "vectors",
                "matrix", "tau", "distributions", "emission", "backward", "forward",
                "differential", "gradient", "cluster", "dilution", "stabilization",
                # Devices not relevant to patient care (unless explicitly mentioned)
                "mobile phone", "wearable", "massager", "telecommunication", "server",
                # Organisms (unless infection-related)
                "clostridium", "naegleria", "bryonal", "biogroup",
                # Administrative/process terms
                "services", "service", "revision", "conversion", "processing",
                "validation", "normalization", "activation", "information",
            }

            # Also reject concepts with certain semantic types that are rarely relevant
            irrelevant_semantic_types = {
                "Physical Object", "Manufactured Object", "Geographic Area",
                "Language", "Occupation", "Organism", "Qualifier Value",
            }

            for entity_name, matches in results:
                if matches and matches[0].score >= 0.7:
                    match = matches[0]

                    # Reject generic/garbage terms (meta-terms, not medical concepts)
                    term_lower = match.term.lower()
                    if term_lower in generic_terms_to_reject:
                        unmatched_entities.append(f"{entity_name} (rejected: generic term '{match.term}')")
                        continue

                    # Reject concepts with irrelevant semantic types
                    if match.semantic_type in irrelevant_semantic_types:
                        unmatched_entities.append(f"{entity_name} (rejected: irrelevant type '{match.semantic_type}')")
                        continue

                    # Use SNOMED semantic_type for node type (e.g., "Disorder", "Procedure", etc.)
                    # Fall back to "unknown" only if SNOMED doesn't provide a type
                    node_type = match.semantic_type if match.semantic_type else "unknown"

                    matched_count += 1
                    # Avoid duplicates (same concept from different mentions)
                    if match.concept_id not in grounded_concepts:
                        grounded_concepts[match.concept_id] = GroundedNode(
                            mention=entity_name,
                            concept_id=match.concept_id,
                            term=match.term,
                            node_type=node_type,
                            confidence=match.score,
                        )
                    else:
                        duplicate_mappings += 1
                    entity_to_concept[entity_name] = match.concept_id
                else:
                    unmatched_entities.append(entity_name)

            if self.verbose:
                print(f"    Matched: {matched_count}/{len(entities)} entities ({100*matched_count/len(entities):.0f}%)")
                if duplicate_mappings > 0:
                    print(f"    Deduplicated: {duplicate_mappings} entities mapped to same concept")
                print(f"    Unique concepts: {len(grounded_concepts)}")
                if unmatched_entities and len(unmatched_entities) <= 5:
                    print(f"    Unmatched: {unmatched_entities}")
                elif unmatched_entities:
                    print(f"    Unmatched: {len(unmatched_entities)} entities (first 5: {unmatched_entities[:5]})")

            if len(grounded_concepts) < 2:
                if self.verbose:
                    print("  Too few grounded concepts for relationships")
                # Still return what we have
                soap_note.grounded_causal_graph = GroundedCausalGraph(
                    nodes=list(grounded_concepts.values()),
                    edges=[],
                )
                return soap_note

            # Stage 3: Identify relationships between grounded concepts
            if pbar:
                pbar.set_postfix_str(f"SNOMED: identifying relationships ({len(grounded_concepts)} concepts)")
            elif self.verbose:
                print("  Stage 3: Identifying causal relationships...")

            # Build concept list with SNOMED IDs (more meaningful than arbitrary labels)
            # Format: Term_With_Underscores[SNOMED:concept_id]
            concept_lines = []
            label_to_node = {}  # Map "Term[SNOMED:ID]" -> GroundedNode
            for cid, node in grounded_concepts.items():
                # Convert term to underscore format for easier LLM handling
                term_label = node.term.replace(" ", "_").replace("-", "_")
                # Create full label with SNOMED ID
                full_label = f"{term_label}[SNOMED:{cid}]"
                label_to_node[full_label] = node
                # Also map variations (uppercase, lowercase) for robust parsing
                label_to_node[full_label.upper()] = node
                label_to_node[full_label.lower()] = node
                concept_lines.append(full_label)
            concept_list = "\n".join(concept_lines)

            relationship_prompt = self.GROUNDED_RELATIONSHIP_PROMPT.format(
                soap_note=soap_text,
                concept_list=concept_list,
            )

            # Debug: show concept list being sent
            if self.verbose:
                print(f"    Concept list ({len(grounded_concepts)} concepts):")
                for line in concept_lines[:5]:
                    print(f"      {line}")
                if len(concept_lines) > 5:
                    print(f"      ... and {len(concept_lines) - 5} more")

            # System prompt to enforce structured output format with examples
            relationship_system = """You output ONLY causal relationships. No prose, no explanations, no commentary.

FORMAT: Source[SNOMED:ID] ARROW Target[SNOMED:ID]

ARROWS:
- ++> strongly increases risk
- +> increases risk
- --> treats / protects against
- => directly causes

INTERACTIONS:
- A && B => C (both required)
- A || B => C (either sufficient)

EXAMPLE OUTPUT:
Type_2_Diabetes[SNOMED:44054006] ++> Chronic_kidney_disease[SNOMED:709044004]
Hypertension[SNOMED:38341003] ++> Stroke[SNOMED:230690007]
Metformin[SNOMED:372567009] --> Type_2_Diabetes[SNOMED:44054006]
Smoking[SNOMED:77176002] && Obesity[SNOMED:414916001] ++> Coronary_artery_disease[SNOMED:53741008]
BRCA1_Mutation[SNOMED:412734009] || BRCA2_Mutation[SNOMED:412738007] || PALB2_Mutation[SNOMED:702464007] ++> Breast_Cancer[SNOMED:254837009]

Output one relationship per line. Nothing else."""

            relationship_response = self._generate_text(
                relationship_prompt,
                max_new_tokens=2000,
                system_prompt=relationship_system,
            )

            # Check if response contains any arrows (relationship format)
            has_arrows = any(arrow in relationship_response for arrow in ["++>", "+>", "-->", "->", "=>"])

            # Retry with simpler prompt if empty OR if it's prose without arrows
            if not relationship_response.strip() or not has_arrows:
                if self.verbose:
                    reason = "empty" if not relationship_response.strip() else "no relationship arrows found (prose output)"
                    print(f"    First attempt: {reason}, retrying with stricter prompt...")
                # Retry with context included
                simple_prompt = f"""Based on this patient's history, output causal relationships.

PATIENT SUMMARY:
{soap_text[:4000]}

CONCEPTS:
{concept_list}

FORMAT: Concept[SNOMED:ID] ++> Concept[SNOMED:ID]
ARROWS: ++> (risk), --> (treatment/protection)

Output one relationship per line. No other text."""
                relationship_response = self._generate_text(
                    simple_prompt,
                    max_new_tokens=1000,
                    system_prompt=relationship_system,
                )

                # Check again
                has_arrows = any(arrow in relationship_response for arrow in ["++>", "+>", "-->", "->", "=>"])
                if not has_arrows and self.verbose:
                    print(f"    WARNING: Retry also failed to produce relationship format")

            # Parse relationships
            grounded_edges = []
            # Support various arrow formats (including unicode)
            edge_type_map = {
                "++>": "risk",
                "+>": "risk",
                "?+>": "risk",
                "-->": "protective",
                "->": "protective",
                "=>": "causal",
                # Unicode arrows
                "→": "causal",
                "⟶": "causal",
                "⇒": "causal",
                "➔": "causal",
                "➜": "causal",
                # Spaced variants
                " -> ": "causal",
                " --> ": "protective",
                " => ": "causal",
                " +> ": "risk",
                " ++> ": "risk",
            }

            # Debug: show raw response
            if self.verbose:
                raw_resp = relationship_response.strip()
                if not raw_resp:
                    print(f"    WARNING: LLM returned empty response for relationships")
                else:
                    response_lines = raw_resp.split("\n")
                    print(f"    LLM response ({len(response_lines)} lines, first 10):")
                    for i, line in enumerate(response_lines[:10]):
                        # Check if line contains any arrow
                        has_arrow = any(arrow in line for arrow in edge_type_map.keys())
                        arrow_indicator = " [HAS ARROW]" if has_arrow else ""
                        print(f"      {i+1}: {line[:100]}{arrow_indicator}")
                    if len(response_lines) > 10:
                        print(f"      ... and {len(response_lines) - 10} more lines")

            parsed_count = 0
            skipped_lines = []  # Track why lines were skipped
            invalid_refs = 0

            # Helper to find node by label (handles case variations)
            def find_node(label: str):
                """Find GroundedNode by label, handling case and format variations."""
                label = label.strip()
                # Try exact match first
                if label in label_to_node:
                    return label_to_node[label]
                # Try case variations
                if label.upper() in label_to_node:
                    return label_to_node[label.upper()]
                if label.lower() in label_to_node:
                    return label_to_node[label.lower()]
                # Try extracting SNOMED ID and matching by that
                import re
                match = re.search(r'\[SNOMED:(\d+)\]', label, re.IGNORECASE)
                if match:
                    snomed_id = match.group(1)
                    if snomed_id in grounded_concepts:
                        return grounded_concepts[snomed_id]
                return None

            for line in relationship_response.strip().split("\n"):
                line = line.strip()
                # Skip empty lines, comments, and lines that look like explanations
                if not line or line.startswith("#"):
                    skipped_lines.append((line[:50], "empty/comment"))
                    continue
                # Skip lines that are too long (likely explanations) - increased limit for SNOMED labels
                if len(line) > 200:
                    skipped_lines.append((line[:50], f"too long ({len(line)} chars)"))
                    continue
                # Skip lines that look like reasoning/explanations
                line_lower = line.lower()
                if any(word in line_lower for word in ["thought", "user", "want", "task", "analyze", "means", "because", "since"]):
                    skipped_lines.append((line[:50], "explanation text"))
                    continue

                # Parse "Concept[SNOMED:ID] ARROW Concept[SNOMED:ID]" format
                # Also handles interactions: "A && B ARROW C" (AND) or "A || B ARROW C" (OR)
                for arrow, _ in edge_type_map.items():
                    if arrow in line:
                        parts = line.split(arrow)
                        if len(parts) == 2:
                            # Clean labels: strip whitespace, bullets, asterisks
                            source_part = parts[0].strip().lstrip("*-•").strip()
                            target_label = parts[1].strip().lstrip("*-•").strip()
                            parsed_count += 1

                            # Check for interaction operators in source
                            # AND interaction: "A && B" (both required together)
                            # OR interaction: "A || B" (either sufficient)
                            interaction = None
                            source_labels = []

                            if ' && ' in source_part:
                                # AND interaction - split on " && "
                                source_labels = [s.strip() for s in source_part.split(' && ') if s.strip()]
                                interaction = "and" if len(source_labels) > 1 else None
                            elif ' || ' in source_part:
                                # OR interaction - split on " || "
                                source_labels = [s.strip() for s in source_part.split(' || ') if s.strip()]
                                interaction = "or" if len(source_labels) > 1 else None
                            else:
                                # Simple single source
                                source_labels = [source_part]

                            # Find all source nodes
                            source_nodes = []
                            all_sources_valid = True
                            for src_label in source_labels:
                                src_node = find_node(src_label)
                                if src_node:
                                    source_nodes.append(src_node)
                                else:
                                    all_sources_valid = False
                                    break

                            # Find target node
                            target_node = find_node(target_label)

                            # Validate all concepts exist
                            if all_sources_valid and source_nodes and target_node:
                                grounded_edges.append(GroundedEdge(
                                    sources=source_nodes,
                                    target=target_node,
                                    relation=arrow,
                                    interaction=interaction,
                                ))
                            else:
                                invalid_refs += 1
                                if self.verbose and invalid_refs <= 3:
                                    print(f"    Invalid ref: '{source_part[:50]}...' -> '{target_label[:50]}...'")
                        break

            if self.verbose:
                print(f"    Parsed {parsed_count} relationships, {len(grounded_edges)} valid, {invalid_refs} invalid refs")
                if len(grounded_edges) == 0 and parsed_count > 0:
                    print(f"    WARNING: {parsed_count} relationships were parsed but none were valid!")
                    print(f"    This usually means concept labels don't match between LLM output and concept list")
                elif len(grounded_edges) == 0 and parsed_count == 0:
                    print(f"    WARNING: No relationships parsed from LLM response!")
                    print(f"    Check that the LLM is outputting the correct format: Concept[SNOMED:ID] ++> Concept[SNOMED:ID]")
                    # Show the full response for debugging
                    print(f"    Full relationship response:")
                    for line in relationship_response.strip().split("\n")[:20]:
                        print(f"      {line}")

            # Deduplicate edges (same source(s) -> target with same relation)
            seen_edges = set()
            unique_edges = []
            for edge in grounded_edges:
                # Create a hashable key for the edge
                source_ids = tuple(sorted(s.concept_id for s in edge.sources))
                edge_key = (source_ids, edge.target.concept_id, edge.relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    unique_edges.append(edge)

            if self.verbose and len(grounded_edges) != len(unique_edges):
                print(f"    Deduplicated: {len(grounded_edges)} -> {len(unique_edges)} edges")

            grounded_edges = unique_edges

            # Build grounded causal graph
            grounded_graph = GroundedCausalGraph(
                nodes=list(grounded_concepts.values()),
                edges=grounded_edges,
            )

            soap_note.grounded_causal_graph = grounded_graph

            # Always save raw relationship response (useful for debugging)
            soap_note.causal_graph_raw = relationship_response

            # Save the prompt used for debugging
            if hasattr(soap_note, 'prompts_used') and soap_note.prompts_used:
                soap_note.prompts_used["grounded_relationships"] = relationship_prompt
            else:
                soap_note.prompts_used = {"grounded_relationships": relationship_prompt}

            # Generate text representation of grounded graph for display
            graph_lines = ["```grounded_graph"]
            for edge in grounded_edges:
                tgt = edge.target
                # Handle interaction edges (multiple sources)
                if edge.is_interaction:
                    op = " && " if edge.interaction == "and" else " || "
                    source_strs = [
                        f"{s.term.replace(' ', '_')}[{s.node_type}]"
                        for s in edge.sources
                    ]
                    source_part = op.join(source_strs)
                else:
                    src = edge.source
                    source_part = f"{src.term.replace(' ', '_')}[{src.node_type}]"
                # Format: Source(s) ARROW Target
                graph_lines.append(
                    f"{source_part} {edge.relation} "
                    f"{tgt.term.replace(' ', '_')}[{tgt.node_type}]"
                )
            graph_lines.append("```")
            graph_lines.append("")
            graph_lines.append(f"*Grounded to SNOMED CT: {len(grounded_graph.nodes)} concepts, {len(grounded_graph.edges)} relationships*")

            grounded_graph_text = "\n".join(graph_lines)

            # Append grounded graph to raw_response (only if there's actual content)
            if hasattr(soap_note, 'raw_response') and soap_note.raw_response:
                # Only append if raw_response has real content (not just whitespace/metadata)
                if soap_note.raw_response.strip():
                    soap_note.raw_response = soap_note.raw_response.rstrip() + "\n\n# Causal Graph\n" + grounded_graph_text

            # Also update the causal_graph field
            soap_note.causal_graph = grounded_graph_text

            if self.verbose:
                interaction_count = len(grounded_graph.interactions())
                msg = f"  Grounded graph: {len(grounded_graph.nodes)} nodes, {len(grounded_graph.edges)} edges"
                if interaction_count > 0:
                    msg += f" ({interaction_count} interactions)"
                print(msg)

        except Exception as e:
            if self.verbose:
                import traceback
                error_msg = str(e) if str(e) else repr(e)
                print(f"  Warning: Grounded causal graph generation failed: {type(e).__name__}: {error_msg}")
                traceback.print_exc()

        return soap_note

    def generate_batch(
        self,
        patients: list,
        batch_size: Optional[int] = None,
        force_hierarchical: bool = False,
        include_imaging: Optional[bool] = None,
        max_images: Optional[int] = None,
        use_biomcp: Optional[bool] = None,
        ground_snomed: Optional[bool] = None,
        snomed_embedding_model: Optional[str] = None,
    ) -> list[SOAPNote]:
        """
        Generate SOAP notes for multiple patients using batched inference.

        This is significantly faster than calling generate() in a loop because
        it batches multiple requests together to maximize GPU utilization.

        Args:
            patients: List of Patient objects from MultimodalDataset
            batch_size: Number of patients to process in each batch. If None,
                       automatically optimizes based on GPU memory.
            force_hierarchical: Force hierarchical summarization
            include_imaging: Include DICOM images (overrides init setting)
            max_images: Maximum images per patient (overrides init setting)
            use_biomcp: Use BioMCP for variant enrichment (overrides init setting)
            ground_snomed: Ground causal graphs to SNOMED CT (overrides init setting)
            snomed_embedding_model: Embedding model for SNOMED linking (overrides init setting)

        Returns:
            List of SOAPNote objects (same order as input patients)

        Example:
            >>> # Auto batch size based on GPU memory
            >>> notes = generator.generate_batch(patients)
            >>>
            >>> # Or specify batch size manually
            >>> notes = generator.generate_batch(patients, batch_size=8)
        """
        # Use instance defaults if not overridden
        include_imaging = include_imaging if include_imaging is not None else self.include_imaging
        max_images = max_images if max_images is not None else self.max_images
        use_biomcp = use_biomcp if use_biomcp is not None else self.use_biomcp
        ground_snomed = ground_snomed if ground_snomed is not None else self.ground_snomed
        snomed_embedding_model = snomed_embedding_model if snomed_embedding_model is not None else self.snomed_embedding_model

        self._load_model()

        # Get batch size (auto-detect if not specified)
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(include_imaging)

        results = []
        n_patients = len(patients)

        # Progress bar
        if _tqdm_available and self.verbose:
            pbar = _tqdm(total=n_patients, desc="Generating SOAP notes")
        else:
            pbar = None
            if self.verbose:
                print(f"Generating SOAP notes for {n_patients} patients (batch_size={batch_size})")

        # Separate patients by processing mode (direct vs hierarchical)
        direct_patients = []
        hierarchical_patients = []

        debug_verbose = isinstance(self.verbose, int) and self.verbose >= 2
        include_genomics_inline = not self.separate_genetics  # Include genomics inline unless generating separately
        for i, patient in enumerate(patients):
            # When separate_genetics=True: exclude genomics from main text, generate separately
            # When separate_genetics=False: include genomics inline in the SOAP generation
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(
                patient, use_biomcp=use_biomcp, include_genomics=include_genomics_inline, verbose=debug_verbose
            )
            text_tokens = len(patient_text) // 4

            # Load images
            images = []
            if include_imaging and patient.dicom_paths:
                images = self._load_patient_images(patient, max_images)

            # Determine effective chunk period (handles "auto" mode)
            if self.chunk_period_years == "auto":
                effective_period, _ = self._compute_auto_chunk_period(patient, text_tokens)
            elif isinstance(self.chunk_period_years, int):
                effective_period = self.chunk_period_years
            else:
                effective_period = None

            # Pre-compute chunks (only if chunking is enabled)
            if effective_period is not None:
                chunks = FHIRFormatter.chunk_by_time_period(patient, effective_period)
            else:
                chunks = []

            # Use hierarchical if chunking is enabled and conditions are met
            use_hierarchical = (
                effective_period is not None and (
                    force_hierarchical or
                    (self.max_tokens_per_chunk is not None and text_tokens > self.max_tokens_per_chunk) or
                    len(chunks) > 2
                )
            )

            if use_hierarchical:
                hierarchical_patients.append((i, patient, images, biomcp_annotations, chunks))
            else:
                direct_patients.append((i, patient, patient_text, images, biomcp_annotations, len(chunks)))

        # Process direct patients in batches
        if direct_patients:
            if self.verbose and pbar is None:
                print(f"  Processing {len(direct_patients)} patients with direct generation...")

            # Skip inline causal graph if grounded graph will be generated separately
            skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)

            for batch_start in range(0, len(direct_patients), batch_size):
                batch = direct_patients[batch_start:batch_start + batch_size]

                # Prepare batch inputs
                batch_prompts = []
                batch_images = []

                for idx, patient, patient_text, images, biomcp_ann, n_chunks in batch:
                    # Build direct prompt with conditional sections
                    prompt = self._build_direct_prompt(
                        patient_data=patient_text,
                        include_images=bool(images),
                        num_images=len(images) if images else 0,
                        include_causal_graph=not skip_inline_graph,
                    )
                    # Add admission exclusion instruction if inline graph is included
                    if not skip_inline_graph:
                        prompt += self._get_admission_exclusion_instruction()
                    batch_prompts.append(prompt)
                    batch_images.append(images if images else None)

                # Batch inference
                responses = self._generate_text_batch(
                    batch_prompts,
                    batch_images,
                    max_new_tokens=self.max_new_tokens_final,
                )

                # Parse responses
                for batch_idx, ((idx, patient, patient_text, images, biomcp_ann, n_chunks), response) in enumerate(zip(batch, responses)):
                    soap_note = self._parse_soap_response(
                        response=response,
                        patient=patient,
                        images_analyzed=len(images) if images else 0,
                        time_periods=n_chunks,
                    )
                    # Store biomcp annotations and prompt used
                    soap_note.biomcp_annotations = biomcp_ann
                    soap_note.prompts_used["soap_generation"] = batch_prompts[batch_idx]
                    results.append((idx, soap_note))

                    if pbar:
                        pbar.update(1)
                    elif self.verbose:
                        name = patient.name or patient.patient_id[:8]
                        print(f"    Completed: {name}")

        # Process hierarchical patients one at a time (can't easily batch these)
        if hierarchical_patients:
            if self.verbose and pbar is None:
                print(f"  Processing {len(hierarchical_patients)} patients with hierarchical generation...")

            for idx, patient, images, biomcp_ann, chunks in hierarchical_patients:
                if self.verbose and pbar is None:
                    name = patient.name or patient.patient_id[:8]
                    print(f"    Processing {name}: {len(chunks)} time chunks...")
                soap_note = self._generate_hierarchical(patient, images, biomcp_ann, chunks)
                results.append((idx, soap_note))

                if pbar:
                    pbar.update(1)
                elif self.verbose:
                    name = patient.name or patient.patient_id[:8]
                    print(f"    Completed: {name}")

        if pbar:
            pbar.close()

        # Sort results back to original order
        results.sort(key=lambda x: x[0])
        final_notes = [soap_note for idx, soap_note in results]

        # Generate separate causal graphs for batch-processed notes if enabled
        # (hierarchical notes already handle this in _generate_hierarchical)
        skip_freeform_graph = ground_snomed and self.grounded_graph_mode
        if self.separate_causal_graph and not skip_freeform_graph:
            # Find notes that don't have a causal graph yet (batch-processed direct notes)
            notes_needing_graphs = [(i, note) for i, note in enumerate(final_notes) if not note.causal_graph]
            if notes_needing_graphs:
                if self.verbose:
                    print(f"  Generating separate causal graphs for {len(notes_needing_graphs)} notes...")
                from synthlab.causal_graph import parse_causal_graph
                graph_system = """You output ONLY causal relationships. No prose, no explanations, no commentary.

FORMAT: Cause[type] ARROW Effect[type]

TYPES: condition, medication, procedure, lifestyle, symptom, finding, genetic

ARROWS:
- ++> strongly increases risk
- +> increases risk
- --> strongly protects/reduces
- -> protects/reduces
- => directly causes

INTERACTIONS:
- A && B => C (both required together)
- A || B => C (either sufficient)

EXAMPLE OUTPUT:
Obesity[lifestyle] ++> Type_2_Diabetes[condition]
Type_2_Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Hypertension[condition] ++> Chronic_Kidney_Disease[condition]
Metformin[medication] --> Blood_Glucose[finding]
Smoking[lifestyle] && Hypertension[condition] ++> Stroke[condition]
BRCA1_Mutation[genetic] || BRCA2_Mutation[genetic] || PALB2_Mutation[genetic] ++> Breast_Cancer[condition]

Output one relationship per line. Nothing else."""
                for i, note in notes_needing_graphs:
                    graph_prompt = self.CAUSAL_GRAPH_PROMPT.format(soap_note=note.raw_response or "")
                    graph_prompt += self._get_admission_exclusion_instruction()
                    causal_graph_response = self._generate_text(
                        graph_prompt,
                        max_new_tokens=2000,
                        system_prompt=graph_system,
                    )
                    # Append to raw response
                    if note.raw_response:
                        note.raw_response = note.raw_response.rstrip() + "\n\n# Causal Graph\n" + causal_graph_response.strip()
                    # Parse and store causal graph
                    note.causal_graph = parse_causal_graph(causal_graph_response)
                    note.causal_graph_raw = causal_graph_response

        # Ground causal graphs to SNOMED CT if enabled
        if ground_snomed:
            if self.verbose:
                mode = "two-stage grounded" if self.grounded_graph_mode else "retrospective"
                print(f"  Grounding {len(final_notes)} causal graphs to SNOMED CT ({mode})...")
            for i, note in enumerate(final_notes):
                if self.grounded_graph_mode:
                    final_notes[i] = self._generate_grounded_causal_graph(note, embedding_model=snomed_embedding_model)
                else:
                    final_notes[i] = self._ground_causal_graph(note, embedding_model=snomed_embedding_model)

        return final_notes

    def _generate_genetic_summary(
        self,
        soap_note: SOAPNote,
        biomcp_annotations: dict[str, Any],
        pbar: Optional[Any] = None,
    ) -> str:
        """
        Generate a focused genetic interpretation summary.

        This runs as a separate LLM call to avoid output truncation issues,
        since Google's models have large input contexts (128K) but limited
        output tokens (8192).

        Args:
            soap_note: The generated SOAP note (for context)
            biomcp_annotations: Raw BioMCP variant annotations
            pbar: Optional progress bar

        Returns:
            Genetic interpretation summary text
        """
        if not biomcp_annotations:
            return ""

        if pbar:
            pbar.set_postfix_str(f"Generating genetic interpretation ({len(biomcp_annotations)} variants)")
        elif self.verbose:
            print(f"  Generating genetic interpretation for {len(biomcp_annotations)} variants...")

        # Build a compact SOAP summary for context (avoid sending full note)
        soap_context_parts = []
        if soap_note.summary:
            soap_context_parts.append(f"Summary: {soap_note.summary}")
        if soap_note.assessment:
            # Truncate assessment if too long
            assessment = soap_note.assessment[:2000] if len(soap_note.assessment) > 2000 else soap_note.assessment
            soap_context_parts.append(f"Assessment: {assessment}")
        if soap_note.patient_story:
            story = soap_note.patient_story[:1500] if len(soap_note.patient_story) > 1500 else soap_note.patient_story
            soap_context_parts.append(f"Patient Story: {story}")

        soap_summary = "\n\n".join(soap_context_parts) if soap_context_parts else "(No SOAP context available)"

        # Format genetic annotations for the prompt
        genetic_lines = []
        for rsid, ann in biomcp_annotations.items():
            gene = ann.get("gene", "unknown")
            clin_sig = ann.get("clinical_significance", "unknown")
            conditions = ann.get("conditions", [])
            phenotypes = ann.get("phenotypes", [])
            drug_associations = ann.get("drug_associations", [])
            effect_type = ann.get("effect_type", "")
            frequencies = ann.get("frequencies", {})

            line_parts = [f"**{rsid}** ({gene})"]
            line_parts.append(f"  - Clinical significance: {clin_sig}")

            if conditions:
                line_parts.append(f"  - Disease associations: {', '.join(conditions[:5])}")
            if phenotypes:
                line_parts.append(f"  - Phenotypes: {', '.join(phenotypes[:5])}")
            if effect_type:
                line_parts.append(f"  - Effect: {effect_type}")
            if drug_associations:
                line_parts.append(f"  - Drug interactions: {', '.join(drug_associations[:5])}")
            if frequencies:
                freq_str = ", ".join(f"{k}={v:.4f}" for k, v in frequencies.items() if isinstance(v, (int, float)))
                if freq_str:
                    line_parts.append(f"  - Population frequencies: {freq_str}")

            genetic_lines.append("\n".join(line_parts))

        genetic_annotations_text = "\n\n".join(genetic_lines)

        # Build the prompt
        prompt = self.GENETIC_SUMMARY_PROMPT.format(
            soap_summary=soap_summary,
            genetic_annotations=genetic_annotations_text,
        )

        # Generate the interpretation
        start_time = time.perf_counter()
        interpretation = self._generate_text(prompt, max_new_tokens=4096)
        gen_seconds = time.perf_counter() - start_time

        if self.verbose:
            output_tokens = self._count_tokens(interpretation)
            print(f"    Genetic summary: {output_tokens:,} tokens in {gen_seconds:.1f}s")

        if pbar:
            pbar.update(1)

        return interpretation.strip()

    def _generate_direct(
        self,
        patient: Any,
        patient_text: str,
        images: list,
        biomcp_annotations: Optional[dict[str, Any]] = None,
        biomcp_seconds: float = 0.0,
        pbar: Optional[Any] = None,
    ) -> SOAPNote:
        """Generate SOAP note directly (for shorter histories)."""
        if biomcp_annotations is None:
            biomcp_annotations = {}
        # Token counting
        input_tokens = self._count_tokens(patient_text)

        # Use passed progress bar or create internal one
        internal_pbar = False
        if pbar is None and _tqdm_available and self.verbose:
            # Create internal progress bar only if not passed from generate()
            total_steps = 3 if self.separate_causal_graph else 2
            pbar = _tqdm(total=total_steps, desc="Generating SOAP note")
            pbar.set_postfix_str("Analyzing patient data" + (f" + {len(images)} images" if images else ""))
            internal_pbar = True

        if self.verbose and not pbar:
            print(f"\n  Token Statistics:")
            print(f"    Input (patient data): {input_tokens:,} tokens")

        # Select prompt based on whether images are present and graph generation mode
        # Skip inline causal graph if we'll generate grounded graph, or if separate_causal_graph is enabled
        skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)

        # Build direct prompt with conditional sections
        prompt = self._build_direct_prompt(
            patient_data=patient_text,
            include_images=bool(images),
            num_images=len(images) if images else 0,
            include_causal_graph=not skip_inline_graph,
        )

        # Add admission exclusion instruction if needed (for inline causal graph mode)
        if not skip_inline_graph:
            prompt += self._get_admission_exclusion_instruction()

        # System prompt to enforce SOAP note structure with detailed template
        # Conditionally include genetics subsection
        genetics_subsection = """
4. **Genetic factors** - If genetic/genomic data is provided, summarize relevant variants and their clinical implications""" if (self.use_biomcp and not self.separate_genetics) else ""

        soap_system = f"""You are a clinical documentation specialist. Output SOAP notes in markdown format with EXACTLY these sections.

CRITICAL: Base ALL content strictly on the provided patient data. Do NOT fabricate details about family members, occupation, living situation, or any other information not explicitly stated in the input.

# Patient Medical History Narrative
1-2 paragraph chronological summary of the patient's documented medical events, diagnoses, and treatments. Include ONLY facts from the provided data.

# Subjective
Patient-reported symptoms, complaints, and relevant history (chief complaints, HPI, social/family history) - only if documented in the input.

# Objective
Clinical findings: vital signs, physical exam, laboratory results (highlight abnormal values), imaging findings if applicable.

# Assessment
Clinical reasoning with numbered subsections:
1. **Primary diagnoses** - Active conditions
2. **Disease progression** - How conditions evolved
3. **Risk factors** - Modifiable and non-modifiable{genetics_subsection}

# Plan
Treatment and follow-up with numbered items:
1. Medications
2. Monitoring (labs, imaging)
3. Lifestyle modifications
4. Referrals
5. Follow-up timing

# Future Considerations
Conditions likely to progress, preventive interventions, screening needs.

# Summary
2-3 sentences on critical findings and recommendations.

Each section MUST start with its header on its own line. Do not skip sections."""

        soap_start = time.perf_counter()
        if self.verbose:
            print(f"    [DEBUG] Prompt length: {len(prompt)} chars")
            print(f"    [DEBUG] Prompt preview: {prompt[:500]}...")
            print(f"    [DEBUG] patient_text length: {len(patient_text)} chars")
        response = self._generate_text(
            prompt,
            images=images if images else None,
            max_new_tokens=self.max_new_tokens_final,
            system_prompt=soap_system,
        )
        soap_seconds = time.perf_counter() - soap_start

        # Fail immediately if response is empty
        if not response or not response.strip():
            raise RuntimeError(
                f"SOAP generation returned empty response.\n"
                f"  Patient: {patient.patient_id}\n"
                f"  Prompt length: {len(prompt)} chars\n"
                f"  max_new_tokens: {self.max_new_tokens_final}"
            )

        # Count output tokens
        output_tokens = self._count_tokens(response)

        # Detect truncation (output hit token limit)
        if output_tokens >= self.max_new_tokens_final - 10:  # Small margin for tokenizer differences
            if self.verbose:
                print(f"  WARNING: Output may be truncated ({output_tokens:,} tokens = model limit)")
            if pbar:
                pbar.set_postfix_str("WARNING: output truncated!")

        if self.verbose and not pbar:
            print(f"    Output (SOAP note): {output_tokens:,} tokens")

        if pbar:
            pbar.update(1)

        # Generate causal graph separately if enabled
        # Skip free-form graph if we'll be generating a grounded graph instead
        causal_graph_tokens = 0
        causal_graph_seconds = 0.0
        skip_freeform_graph = self.ground_snomed and self.grounded_graph_mode

        if self.separate_causal_graph and not skip_freeform_graph:
            if pbar:
                pbar.set_postfix_str("Generating causal graph")

            if self.verbose and not _tqdm_available:
                print(f"    Generating causal graph separately...")

            graph_prompt = self.CAUSAL_GRAPH_PROMPT.format(soap_note=response)
            # Add admission exclusion instruction if needed
            graph_prompt += self._get_admission_exclusion_instruction()
            graph_system = """You output ONLY causal relationships. No prose, no explanations, no commentary.

FORMAT: Cause[type] ARROW Effect[type]

TYPES: condition, medication, procedure, lifestyle, symptom, finding, genetic

ARROWS:
- ++> strongly increases risk
- +> increases risk
- --> strongly protects/reduces
- -> protects/reduces
- => directly causes

INTERACTIONS:
- A && B => C (both required together)
- A || B => C (either sufficient)

EXAMPLE OUTPUT:
Obesity[lifestyle] ++> Type_2_Diabetes[condition]
Type_2_Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Hypertension[condition] ++> Chronic_Kidney_Disease[condition]
Metformin[medication] --> Blood_Glucose[finding]
Smoking[lifestyle] && Hypertension[condition] ++> Stroke[condition]
BRCA1_Mutation[genetic] || BRCA2_Mutation[genetic] || PALB2_Mutation[genetic] ++> Breast_Cancer[condition]

Output one relationship per line. Nothing else."""
            graph_start = time.perf_counter()
            causal_graph_response = self._generate_text(
                graph_prompt,
                max_new_tokens=2000,
                system_prompt=graph_system,
            )
            causal_graph_seconds = time.perf_counter() - graph_start
            causal_graph_tokens = self._count_tokens(causal_graph_response)

            if self.verbose and not _tqdm_available:
                print(f"    Output (causal graph): {causal_graph_tokens:,} tokens")

            # Append causal graph as its own section at the end
            response = response.rstrip() + "\n\n# Causal Graph\n" + causal_graph_response.strip()
            output_tokens += causal_graph_tokens

            if pbar:
                pbar.update(1)
        elif self.separate_causal_graph and skip_freeform_graph:
            # Grounded graph will be added later - don't add placeholder to raw response
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.set_postfix_str("Parsing response")

        result = self._parse_soap_response(
            response=response,
            patient=patient,
            images_analyzed=len(images),
            time_periods=1,
        )

        result.token_stats = {
            "mode": "direct" + ("_separate_graph" if self.separate_causal_graph else ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "summarizer_input_tokens": input_tokens,
            "soap_input_tokens": input_tokens,
            "soap_output_tokens": output_tokens - causal_graph_tokens,
            "soap_generation_seconds": soap_seconds,
            "causal_graph_tokens": causal_graph_tokens,
            "causal_graph_seconds": causal_graph_seconds,
            "biomcp_seconds": biomcp_seconds,
        }

        # Store intermediate outputs
        result.prompts_used = {"soap_generation": prompt}
        result.biomcp_annotations = biomcp_annotations

        if self.separate_causal_graph and not skip_freeform_graph:
            result.prompts_used["causal_graph"] = graph_prompt
            result.causal_graph_raw = causal_graph_response

        if pbar:
            pbar.update(1)
            if internal_pbar:
                pbar.close()

        # Print token stats for tqdm mode too (only if using internal pbar)
        if self.verbose and internal_pbar:
            print(f"\n  Token Statistics:")
            print(f"    Input: {input_tokens:,} tokens")
            print(f"    Output: {output_tokens:,} tokens")
            if self.separate_causal_graph:
                print(f"    Causal graph: {causal_graph_tokens:,} tokens")
            print(f"    Summarizer input: {input_tokens:,} tokens")
            print(f"    Final SOAP output: {output_tokens:,} tokens")

        return result

    def _generate_hierarchical(
        self,
        patient: Any,
        images: list,
        biomcp_annotations: Optional[dict[str, Any]] = None,
        chunks: Optional[list[tuple[str, str, str]]] = None,
        biomcp_seconds: float = 0.0,
        pbar: Optional[Any] = None,
    ) -> SOAPNote:
        """Generate SOAP note using hierarchical summarization."""
        if biomcp_annotations is None:
            biomcp_annotations = {}

        # Use pre-computed chunks or compute them now
        if chunks is None:
            # For "auto" mode, use a default period of 5 years in fallback
            # (normally chunks are pre-computed in generate() with proper auto calculation)
            if self.chunk_period_years == "auto":
                fallback_period = 5
            elif isinstance(self.chunk_period_years, int):
                fallback_period = self.chunk_period_years
            else:
                fallback_period = 5  # Reasonable default
            chunks = FHIRFormatter.chunk_by_time_period(patient, fallback_period)

        if not chunks:
            # Fallback to direct if no chunks
            # When separate_genetics=True: exclude genomics from main text
            # When separate_genetics=False: include genomics inline
            debug_verbose = isinstance(self.verbose, int) and self.verbose >= 2
            include_genomics_inline = not self.separate_genetics
            patient_text, fallback_biomcp = FHIRFormatter.format_patient_for_llm(
                patient, use_biomcp=self.use_biomcp, include_genomics=include_genomics_inline, verbose=debug_verbose
            )
            # Merge any new biomcp annotations
            if fallback_biomcp:
                biomcp_annotations.update(fallback_biomcp)
            return self._generate_direct(patient, patient_text, images, biomcp_annotations)

        # Token tracking
        chunk_stats = []  # List of (period, input_tokens, output_tokens, summary_seconds)
        total_input_tokens = 0
        total_summary_tokens = 0

        # Use passed progress bar or create internal one
        internal_pbar = False
        if pbar is None and _tqdm_available and self.verbose:
            # Create internal progress bar only if not passed from generate()
            total_steps = len(chunks) + 2 + (1 if self.separate_causal_graph else 0)
            pbar = _tqdm(total=total_steps, desc="Generating SOAP note")
            internal_pbar = True

        # Summarize each chunk sequentially
        summaries = []
        chunk_summary_objects = []

        if self.verbose and not pbar:
            print(f"\n  Summarizing {len(chunks)} time chunks...")

        for i, (start, end, chunk_text) in enumerate(chunks):
            if pbar:
                pbar.set_postfix_str(f"Summarizing {start}-{end}")

            # Count input tokens
            input_tokens = self._count_tokens(chunk_text)
            total_input_tokens += input_tokens

            # Include causal graph in chunk only if enabled AND not using grounded mode
            skip_chunk_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)
            include_chunk_graph = self.chunk_causal_graph and not skip_chunk_graph

            prompt = self._build_chunk_prompt(
                chunk_data=chunk_text,
                time_period=f"{start}-{end}",
                include_causal_graph=include_chunk_graph,
            )
            # System prompt for chunk summary structure with detailed template
            chunk_system = """You are a clinical documentation specialist. Output period summaries in markdown format with EXACTLY these sections:

# Subjective
Patient-reported symptoms and complaints during this time period.

# Objective
Clinical findings:
- Vital signs and trends
- Laboratory results (highlight abnormal values)
- Procedures performed

# Assessment
Clinical reasoning:
1. **Diagnoses** - Conditions identified or managed
2. **Disease progression** - How conditions evolved
3. **Risk factors** - New or ongoing concerns

# Plan
Treatment during this period:
1. Medications started/changed
2. Monitoring performed
3. Referrals made

# Summary
2-3 sentences on key events and health trajectory during this period.

Each section MUST start with its header on its own line. Do not skip sections."""
            chunk_start_time = time.perf_counter()
            summary = self._generate_text(
                prompt,
                max_new_tokens=self.max_new_tokens_chunk,
                system_prompt=chunk_system,
            )
            chunk_seconds = time.perf_counter() - chunk_start_time

            # Count output tokens
            output_tokens = self._count_tokens(summary)
            total_summary_tokens += output_tokens

            summaries.append(f"### {start} - {end}\n{summary}")
            chunk_stats.append((f"{start}-{end}", input_tokens, output_tokens, chunk_seconds))

            chunk_summary_objects.append(ChunkSummary(
                period=f"{start}-{end}",
                start_year=start,
                end_year=end,
                prompt=prompt,
                summary=summary,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_seconds=chunk_seconds,
            ))

            if pbar:
                pbar.update(1)

        # Print summary for verbose mode
        if self.verbose and not pbar:
            total_seconds = sum(s for _, _, _, s in chunk_stats)
            print(f"    Processed {len(chunks)} chunks in {total_seconds:.2f}s")
            print(f"    Total input: {total_input_tokens:,} tokens | Total output: {total_summary_tokens:,} tokens")

        # Aggregate summaries into final SOAP note
        if pbar:
            pbar.set_postfix_str("Aggregating summaries" + (f" + {len(images)} images" if images else ""))
        elif self.verbose:
            print(f"\n  Aggregating {len(summaries)} summaries into final SOAP...")

        demographics = patient.get_demographics()
        patient_name = demographics.get("name", patient.patient_id) if demographics else patient.patient_id

        # Count tokens in aggregate input
        summaries_text = "\n\n".join(summaries)
        aggregate_input_tokens = self._count_tokens(summaries_text)

        # Select prompt based on whether images are present and graph generation mode
        # Skip inline causal graph if we'll generate grounded graph, or if separate_causal_graph is enabled
        skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)

        # Build aggregate prompt with conditional sections
        aggregate_prompt = self._build_aggregate_prompt(
            patient_name=patient_name,
            summaries_text=summaries_text,
            include_images=bool(images),
            num_images=len(images) if images else 0,
            include_causal_graph=not skip_inline_graph,
        )

        # Add admission exclusion instruction if needed (for inline causal graph mode)
        if not skip_inline_graph:
            aggregate_prompt += self._get_admission_exclusion_instruction()

        # System prompt to enforce SOAP note structure with detailed template
        soap_system = """You are a clinical documentation specialist. Output SOAP notes in markdown format with EXACTLY these sections:

# Patient Story
1-2 paragraph narrative synthesizing the patient's health journey across all time periods.

# Subjective
Consolidated patient-reported symptoms and history across time periods.

# Objective
Synthesized clinical findings:
- Most recent vital signs with trends over time
- Key laboratory values and changes
- Imaging findings if applicable

# Assessment
Integrated clinical reasoning with numbered subsections:
1. **Active Problem List** - Current diagnoses by clinical priority
2. **Disease Trajectories** - How conditions have progressed
3. **Risk Stratification** - Risk factors and prognosis

# Plan
Treatment strategy with numbered items:
1. Medications - current regimen and changes
2. Monitoring - labs, imaging needed
3. Lifestyle interventions
4. Referrals
5. Follow-up schedule

# Future Considerations
Conditions likely to progress, preventive interventions, screening needs.

# Summary
2-3 sentences on critical findings and priorities.

Each section MUST start with its header on its own line. Do not skip sections."""

        aggregate_start = time.perf_counter()
        response = self._generate_text(
            aggregate_prompt,
            images=images if images else None,
            max_new_tokens=self.max_new_tokens_final,
            system_prompt=soap_system,
        )
        aggregate_seconds = time.perf_counter() - aggregate_start

        # Fail immediately if response is empty
        if not response or not response.strip():
            raise RuntimeError(
                f"Hierarchical SOAP generation returned empty response.\n"
                f"  Patient: {patient.patient_id}\n"
                f"  Chunks summarized: {len(summaries)}\n"
                f"  max_new_tokens: {self.max_new_tokens_final}"
            )

        # Count final output tokens
        final_output_tokens = self._count_tokens(response)

        if pbar:
            pbar.update(1)

        # Generate causal graph separately if enabled
        # Skip free-form graph if we'll be generating a grounded graph instead
        causal_graph_tokens = 0
        causal_graph_seconds = 0.0
        skip_freeform_graph = self.ground_snomed and self.grounded_graph_mode

        if self.separate_causal_graph and not skip_freeform_graph:
            if pbar:
                pbar.set_postfix_str("Generating causal graph")
            elif self.verbose:
                print(f"  Generating causal graph separately...")

            graph_prompt = self.CAUSAL_GRAPH_PROMPT.format(soap_note=response)
            # Add admission exclusion instruction if needed
            graph_prompt += self._get_admission_exclusion_instruction()
            graph_system = """You output ONLY causal relationships. No prose, no explanations, no commentary.

FORMAT: Cause[type] ARROW Effect[type]

TYPES: condition, medication, procedure, lifestyle, symptom, finding, genetic

ARROWS:
- ++> strongly increases risk
- +> increases risk
- --> strongly protects/reduces
- -> protects/reduces
- => directly causes

INTERACTIONS:
- A && B => C (both required together)
- A || B => C (either sufficient)

EXAMPLE OUTPUT:
Obesity[lifestyle] ++> Type_2_Diabetes[condition]
Type_2_Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Hypertension[condition] ++> Chronic_Kidney_Disease[condition]
Metformin[medication] --> Blood_Glucose[finding]
Smoking[lifestyle] && Hypertension[condition] ++> Stroke[condition]
BRCA1_Mutation[genetic] || BRCA2_Mutation[genetic] || PALB2_Mutation[genetic] ++> Breast_Cancer[condition]

Output one relationship per line. Nothing else."""
            graph_start = time.perf_counter()
            causal_graph_response = self._generate_text(
                graph_prompt,
                max_new_tokens=2000,
                system_prompt=graph_system,
            )
            causal_graph_seconds = time.perf_counter() - graph_start
            causal_graph_tokens = self._count_tokens(causal_graph_response)

            # Append causal graph as its own section at the end
            response = response.rstrip() + "\n\n# Causal Graph\n" + causal_graph_response.strip()
            final_output_tokens += causal_graph_tokens

            if pbar:
                pbar.update(1)
        elif self.separate_causal_graph and skip_freeform_graph:
            # Grounded graph will be added later - don't add placeholder to raw response
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.set_postfix_str("Parsing response")

        result = self._parse_soap_response(
            response=response,
            patient=patient,
            images_analyzed=len(images),
            time_periods=len(chunks),
        )

        result.token_stats = {
            "mode": "hierarchical" + ("_separate_graph" if self.separate_causal_graph else ""),
            "chunk_stats": [
                {
                    "period": period,
                    "input_tokens": inp,
                    "summary_tokens": out,
                    "summary_seconds": seconds,
                }
                for period, inp, out, seconds in chunk_stats
            ],
            "total_chunk_input_tokens": total_input_tokens,
            "total_chunk_summary_tokens": total_summary_tokens,
            "total_chunk_seconds": sum(seconds for _, _, _, seconds in chunk_stats),
            "summarizer_input_tokens": aggregate_input_tokens,
            "soap_input_tokens": aggregate_input_tokens,
            "soap_output_tokens": final_output_tokens - causal_graph_tokens,
            "aggregate_seconds": aggregate_seconds,
            "causal_graph_tokens": causal_graph_tokens,
            "causal_graph_seconds": causal_graph_seconds,
            "biomcp_seconds": biomcp_seconds,
        }

        # Store intermediate outputs
        result.chunk_summaries = chunk_summary_objects
        result.prompts_used = {"aggregate": aggregate_prompt}
        result.biomcp_annotations = biomcp_annotations

        if self.separate_causal_graph and not skip_freeform_graph:
            result.prompts_used["causal_graph"] = graph_prompt
            result.causal_graph_raw = causal_graph_response

        if pbar:
            pbar.update(1)
            if internal_pbar:
                pbar.close()

        # Print comprehensive token statistics (only if using internal pbar)
        if self.verbose and internal_pbar:
            print(f"\n  Token Statistics Summary:")
            print(f"  ══════════════════════════════════════════════")
            print(f"  Time Segments: {len(chunks)}")
            print(f"  ")
            print(f"  Per-Segment Breakdown:")
            print(f"    {'Period':<15} {'Input':>12} {'Summary':>12} {'Time(s)':>10}")
            print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
            for period, inp, out, seconds in chunk_stats:
                print(f"    {period:<15} {inp:>12,} {out:>12,} {seconds:>10.2f}")
            print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
            print(f"    {'TOTAL':<15} {total_input_tokens:>12,} {total_summary_tokens:>12,} {sum(s for _, _, _, s in chunk_stats):>10.2f}")
            print(f"  ")
            print(f"  Aggregation:")
            print(f"    Summaries input:  {aggregate_input_tokens:,} tokens")
            print(f"    Final SOAP output: {final_output_tokens - causal_graph_tokens:,} tokens")
            print(f"    Aggregation time:  {aggregate_seconds:.2f}s")
            if self.separate_causal_graph:
                print(f"  ")
                print(f"  Causal Graph (separate):")
                print(f"    Graph output: {causal_graph_tokens:,} tokens")
                print(f"    Graph time:   {causal_graph_seconds:.2f}s")
            print(f"  ")
            print(f"  Compression: {total_input_tokens:,} → {final_output_tokens:,} tokens")
            print(f"               ({100 * final_output_tokens / max(total_input_tokens, 1):.1f}% of original)")
            print(f"  ══════════════════════════════════════════════")

        return result

    def _load_patient_images(self, patient: Any, max_images: int) -> list:
        """Load DICOM images as PIL Images for multimodal analysis."""
        if not _PIL_available:
            if self.verbose:
                print("  Warning: PIL not available, skipping image analysis")
            return []

        dicom_paths = patient.dicom_paths[:max_images]
        if not dicom_paths:
            return []

        if self.verbose and not _tqdm_available:
            print(f"  Loading {len(dicom_paths)} DICOM image(s) for analysis...")

        images = []
        iterator = dicom_paths
        if _tqdm_available and self.verbose and len(dicom_paths) > 1:
            iterator = _tqdm(dicom_paths, desc="Loading DICOM images", leave=False)

        for dicom_path in iterator:
            try:
                # Load DICOM
                import pydicom
                ds = pydicom.dcmread(dicom_path)
                pixel_array = ds.pixel_array

                # Handle 3D volumes (take middle slice)
                if pixel_array.ndim == 3:
                    middle = pixel_array.shape[0] // 2
                    pixel_array = pixel_array[middle]

                # Normalize to 0-255
                pixel_array = pixel_array.astype(float)
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
                pixel_array = (pixel_array * 255).astype("uint8")

                # Convert to PIL Image
                img = Image.fromarray(pixel_array)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                images.append(img)

                if self.verbose and not _tqdm_available:
                    print(f"    Loaded: {dicom_path.name} ({img.size[0]}x{img.size[1]} pixels)")

            except Exception as e:
                if self.verbose and not _tqdm_available:
                    print(f"    Warning: Could not load {dicom_path.name}: {e}")
                continue

        if images and self.verbose and not _tqdm_available:
            print(f"  ✓ {len(images)} image(s) ready for multimodal analysis")

        return images

    def _parse_soap_response(
        self,
        response: str,
        patient: Any,
        images_analyzed: int,
        time_periods: int,
    ) -> SOAPNote:
        """Parse model response into structured SOAPNote."""
        demographics = patient.get_demographics()

        note = SOAPNote(
            patient_id=patient.patient_id,
            patient_name=demographics.get("name") if demographics else None,
            raw_response=response,
            images_analyzed=images_analyzed,
            time_periods_summarized=time_periods,
            model_used=self.model_id,
        )

        # Section name aliases (model might use different names)
        section_aliases = {
            "PATIENT STORY": ["PATIENT STORY", "PATIENT NARRATIVE", "HISTORY", "PATIENT HISTORY"],
            "SUBJECTIVE": ["SUBJECTIVE", "S:", "S.", "CHIEF COMPLAINT", "HPI"],
            "OBJECTIVE": ["OBJECTIVE", "O:", "O.", "PHYSICAL EXAM", "EXAMINATION", "FINDINGS"],
            "ASSESSMENT": ["ASSESSMENT", "A:", "A.", "DIAGNOSIS", "DIAGNOSES", "IMPRESSION"],
            "PLAN": ["PLAN", "P:", "P.", "TREATMENT", "RECOMMENDATIONS", "MANAGEMENT"],
            "FUTURE CONSIDERATIONS": ["FUTURE CONSIDERATIONS", "FUTURE", "PROGNOSIS", "FOLLOW-UP", "FOLLOWUP"],
            "SUMMARY": ["SUMMARY", "CONCLUSION", "BRIEF SUMMARY"],
            "CAUSAL_GRAPH": ["CAUSAL GRAPH", "CAUSAL_GRAPH", "CAUSAL RELATIONSHIPS"],
        }

        # Build reverse lookup
        alias_to_section = {}
        for section, aliases in section_aliases.items():
            for alias in aliases:
                alias_to_section[alias] = section

        # Parse sections from response
        sections = {name: "" for name in section_aliases.keys()}

        current_section = None
        current_content = []

        debug_parse = isinstance(self.verbose, int) and self.verbose >= 2
        if debug_parse:
            print(f"    [PARSE DEBUG] Response length: {len(response)} chars")
            print(f"    [PARSE DEBUG] Response preview: {response[:200]}...")

        for line in response.split("\n"):
            stripped = line.strip()
            line_upper = stripped.upper()

            # Skip empty lines and horizontal rules
            if not stripped or stripped in ["---", "***", "___"]:
                if current_section:
                    current_content.append(line)
                continue

            # Remove markdown formatting for header detection
            # Strip leading markdown chars (#, *, -) and trailing punctuation (*, :)
            clean_upper = line_upper.lstrip("#*- \t").rstrip(":* \t")

            # Check for section headers
            # Match if line equals alias exactly, or starts with alias followed by
            # non-alphanumeric (space, colon, dash, etc.)
            found_section = None
            for alias, section in alias_to_section.items():
                if clean_upper == alias:
                    found_section = section
                    break
                elif clean_upper.startswith(alias):
                    # Check that next char (if any) is not alphanumeric
                    # This prevents "PLANNING" from matching "PLAN"
                    next_char_idx = len(alias)
                    if next_char_idx >= len(clean_upper):
                        found_section = section
                        break
                    next_char = clean_upper[next_char_idx]
                    if not next_char.isalnum():
                        found_section = section
                        break

            if found_section:
                # Save previous section
                if current_section:
                    saved_content = "\n".join(current_content).strip()
                    sections[current_section] = saved_content
                    if debug_parse:
                        print(f"    [PARSE DEBUG] Saved {current_section}: {len(saved_content)} chars")
                current_section = found_section
                current_content = []
                if debug_parse:
                    print(f"    [PARSE DEBUG] Found section: {found_section} (from line: {stripped[:50]})")
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            saved_content = "\n".join(current_content).strip()
            sections[current_section] = saved_content
            if debug_parse:
                print(f"    [PARSE DEBUG] Saved final {current_section}: {len(saved_content)} chars")

        if debug_parse:
            for sec_name, sec_content in sections.items():
                print(f"    [PARSE DEBUG] Section {sec_name}: {len(sec_content)} chars")

        # If no sections found, try to use raw response as assessment
        if not any(sections.values()):
            # Model might have returned unstructured content
            if self.verbose:
                print("  Warning: Could not parse sections from response, using raw content")
            # Try to extract any meaningful content
            sections["ASSESSMENT"] = response.strip()

        # Assign to note
        note.patient_story = sections["PATIENT STORY"]
        note.subjective = sections["SUBJECTIVE"]
        note.objective = sections["OBJECTIVE"]
        note.assessment = sections["ASSESSMENT"]
        note.plan = sections["PLAN"]
        note.future_considerations = sections["FUTURE CONSIDERATIONS"]
        note.summary = sections["SUMMARY"]
        note.causal_graph = sections["CAUSAL_GRAPH"]

        return note


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_soap_note(
    patient: Any,
    model_id: str = "google/medgemma-1.5-4b-it",
    include_imaging: bool = True,
    quantization: Optional[str] = None,
    verbose: Union[bool, int] = True,
    approximate_tokens: bool = False,
    use_biomcp: bool = False,
    separate_genetics: bool = False,
    separate_causal_graph: bool = False,
    include_admissions: bool = False,
    chunk_period_years: Optional[int | str] = "auto",
) -> SOAPNote:
    """
    Generate a SOAP note for a patient using MedGemma.

    This is a convenience function that creates a generator and produces
    a SOAP note in one call.

    Args:
        patient: Patient object from MultimodalDataset
        model_id: HuggingFace model ID for MedGemma
        include_imaging: Include DICOM images in analysis
        quantization: Quantization for faster inference ('4bit', '8bit', or None)
        verbose: Print progress messages and token statistics
        approximate_tokens: Use fast approximate token counting (chars/4)
        use_biomcp: Use BioMCP to enrich genetic variant annotations.
                    Requires: pip install biomcp-python
                    See: https://biomcp.org/
        separate_genetics: If True, generate genetic summary in a separate LLM call.
                    If False (default), include genetics inline in the SOAP note.
        separate_causal_graph: Generate causal graph in a separate prompt to avoid
                    token limits. Only generates for final SOAP, not chunks.
        include_admissions: Include Hospital Admissions in the causal graph.
                    Default is False to focus on direct causal relationships.
        chunk_period_years: Years per chunk for time-based splitting (default: "auto"):
                    - "auto" (default) = automatically determine optimal chunk size
                    - int (e.g., 5) = fixed 5-year chunks
                    - None = no chunking, always use direct mode

    Returns:
        SOAPNote object with structured output

    Example:
        >>> import synthlab as sl
        >>> from synthlab.soap import generate_soap_note
        >>>
        >>> dataset = sl.load_multimodal_dataset(max_patients=5)
        >>> # Fast mode with 4-bit quantization
        >>> soap_note = generate_soap_note(dataset[0], quantization='4bit')
        >>> print(soap_note)
        >>>
        >>> # With separate causal graph generation (avoids token limits)
        >>> soap_note = generate_soap_note(dataset[0], separate_causal_graph=True)
    """
    generator = SOAPNoteGenerator(
        model_id=model_id,
        quantization=quantization,
        verbose=verbose,
        approximate_tokens=approximate_tokens,
        include_imaging=include_imaging,
        use_biomcp=use_biomcp,
        separate_genetics=separate_genetics,
        separate_causal_graph=separate_causal_graph,
        include_admissions=include_admissions,
        chunk_period_years=chunk_period_years,
    )
    return generator.generate(patient)
