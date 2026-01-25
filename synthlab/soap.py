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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from synthlab.snomed import SNOMEDLinker, GroundedCausalGraph


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
    from biomcp.variants.search import search_variants as _biomcp_search_variants
    from biomcp.variants.search import VariantQuery as _BioMCPVariantQuery
    from biomcp.variants.search import ClinicalSignificance as _BioMCPClinicalSignificance
    _biomcp_available = True
except ImportError:
    _BioMCPVariantQuery = None
    _BioMCPClinicalSignificance = None


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

            if self.causal_graph:
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
        verbose: bool = True,
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
# Causal Graph Representation
# =============================================================================

# Edge type definitions with their meanings and numeric weights for graph algorithms
CAUSAL_EDGE_TYPES = {
    "++>": {"name": "strongly_increases", "direction": "risk", "strength": "strong", "weight": 0.8},
    "+>": {"name": "increases", "direction": "risk", "strength": "moderate", "weight": 0.5},
    "?+>": {"name": "possibly_increases", "direction": "risk", "strength": "uncertain", "weight": 0.3},
    "-->": {"name": "strongly_decreases", "direction": "protective", "strength": "strong", "weight": -0.8},
    "->": {"name": "decreases", "direction": "protective", "strength": "moderate", "weight": -0.5},
    "?->": {"name": "possibly_decreases", "direction": "protective", "strength": "uncertain", "weight": -0.3},
    "=>": {"name": "causes", "direction": "causal", "strength": "direct", "weight": 1.0},
}

# Valid node type categories for clinical causal graphs
NODE_TYPES = {
    "condition": "Disease, diagnosis, or clinical finding",
    "medication": "Drug or therapeutic agent",
    "procedure": "Medical procedure, surgery, or intervention",
    "lifestyle": "Lifestyle factor or behavior",
    "symptom": "Patient-reported symptom or sign",
    "finding": "Lab result, imaging finding, or clinical observation",
    "outcome": "Clinical outcome or endpoint",
    "genetic": "Genetic variant, mutation, or polymorphism",
}


def _extract_node_type(node_str: str) -> tuple[str, str]:
    """
    Extract node name and type from bracket notation.

    Examples:
        "Obesity[lifestyle]" -> ("Obesity", "lifestyle")
        "T2D(2018)[condition]" -> ("T2D(2018)", "condition")
        "Smoking" -> ("Smoking", "unknown")
    """
    match = re.search(r'\[(\w+)\]\s*$', node_str)
    if match:
        node_type = match.group(1).lower()
        name = node_str[:match.start()].strip()
        # Validate type
        if node_type not in NODE_TYPES:
            node_type = "unknown"
        return name, node_type
    return node_str.strip(), "unknown"


@dataclass
class CausalNode:
    """A node in a causal graph representing a clinical entity."""
    name: str
    node_type: str = "unknown"
    time: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.name == other.name
        return self.name == other


@dataclass
class CausalEdge:
    """
    A directed edge in a causal graph, supporting interactions.

    Attributes:
        sources: List of cause/antecedent nodes (supports single or multiple)
        target: The effect/consequent node (e.g., "Type 2 Diabetes (2018)")
        edge_type: The relationship type (e.g., "++>", "+>", "->", "=>")
        interaction: Type of interaction between sources: "and", "or", or None
        direction: "risk", "protective", or "causal"
        strength: "strong", "moderate", "uncertain", or "direct"
        weight: Numeric weight for graph algorithms (-1 to 1)

    Examples:
        Simple edge: CausalEdge(sources=["Obesity"], target="Diabetes", edge_type="++>")
        AND interaction: CausalEdge(sources=["DrugA", "DrugB"], target="Toxicity", edge_type="=>", interaction="and")
        OR interaction: CausalEdge(sources=["BRCA1", "BRCA2"], target="Cancer", edge_type="++>", interaction="or")
    """
    sources: list[str]
    target: str
    edge_type: str
    interaction: Optional[str] = None  # "and", "or", or None
    direction: str = ""
    strength: str = ""
    weight: float = 0.0

    def __post_init__(self):
        # Handle legacy single source as string
        if isinstance(self.sources, str):
            self.sources = [self.sources]
        if self.edge_type in CAUSAL_EDGE_TYPES:
            info = CAUSAL_EDGE_TYPES[self.edge_type]
            self.direction = info["direction"]
            self.strength = info["strength"]
            self.weight = info["weight"]

    @property
    def source(self) -> str:
        """Get source as string (for backward compatibility). Joins with interaction operator."""
        if len(self.sources) == 1:
            return self.sources[0]
        op = " + " if self.interaction == "and" else " | " if self.interaction == "or" else ", "
        return op.join(self.sources)

    @property
    def is_interaction(self) -> bool:
        """True if this edge represents an interaction between multiple sources."""
        return len(self.sources) > 1 and self.interaction is not None

    @property
    def source_names(self) -> list[str]:
        """Extract names without temporal info for all sources."""
        return [re.sub(r'\s*\([^)]*\)\s*$', '', s).strip() for s in self.sources]

    @property
    def source_name(self) -> str:
        """Extract just the name without temporal info (first source for compatibility)."""
        return self.source_names[0] if self.sources else ""

    @property
    def target_name(self) -> str:
        """Extract just the name without temporal info."""
        return re.sub(r'\s*\([^)]*\)\s*$', '', self.target).strip()

    @property
    def source_time(self) -> Optional[str]:
        """Extract temporal info from first source if present."""
        if not self.sources:
            return None
        match = re.search(r'\(([^)]+)\)\s*$', self.sources[0])
        return match.group(1) if match else None

    @property
    def target_time(self) -> Optional[str]:
        """Extract temporal info from target if present."""
        match = re.search(r'\(([^)]+)\)\s*$', self.target)
        return match.group(1) if match else None

    @property
    def source_nodes(self) -> list[CausalNode]:
        """Get all sources as CausalNode objects with extracted types."""
        nodes = []
        for src in self.sources:
            name, node_type = _extract_node_type(src)
            clean_name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
            match = re.search(r'\(([^)]+)\)\s*$', src)
            time = match.group(1) if match else None
            nodes.append(CausalNode(name=clean_name, node_type=node_type, time=time))
        return nodes

    @property
    def source_node(self) -> CausalNode:
        """Get first source as a CausalNode (for backward compatibility)."""
        nodes = self.source_nodes
        return nodes[0] if nodes else CausalNode(name="", node_type="unknown")

    @property
    def target_node(self) -> CausalNode:
        """Get target as a CausalNode with extracted type."""
        name, node_type = _extract_node_type(self.target)
        clean_name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
        time = self.target_time
        return CausalNode(name=clean_name, node_type=node_type, time=time)

    def __str__(self) -> str:
        return f"{self.source} {self.edge_type} {self.target}"

    def __hash__(self):
        """Hash based on normalized sources, target, and edge type."""
        # Sort sources for consistent hashing regardless of order
        source_key = tuple(sorted(s.lower().strip() for s in self.sources))
        target_key = self.target.lower().strip()
        return hash((source_key, target_key, self.edge_type))

    def __eq__(self, other):
        """Two edges are equal if they have the same sources, target, and edge type."""
        if not isinstance(other, CausalEdge):
            return False
        # Normalize for comparison (case-insensitive, sorted sources)
        self_sources = sorted(s.lower().strip() for s in self.sources)
        other_sources = sorted(s.lower().strip() for s in other.sources)
        return (
            self_sources == other_sources
            and self.target.lower().strip() == other.target.lower().strip()
            and self.edge_type == other.edge_type
        )


@dataclass
class CausalGraph:
    """
    A directed graph of causal relationships extracted from clinical notes.

    Compatible with NetworkX. Tracks edge relationships and node types.

    Example:
        >>> graph = soap_note.extract_causal_graph()
        >>> print(graph.summary())
        >>>
        >>> # Query by node type
        >>> conditions = graph.nodes_by_type("condition")
        >>> medications = graph.nodes_by_type("medication")
        >>>
        >>> # Convert to NetworkX
        >>> G = graph.to_networkx()
        >>> nx.shortest_path(G, "Obesity", "Neuropathy")
    """
    edges: list[CausalEdge] = field(default_factory=list)
    _node_types: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""

    @property
    def nodes(self) -> set[str]:
        """Get all unique node names."""
        nodes = set()
        for edge in self.edges:
            # Handle multiple sources for interaction edges
            for name in edge.source_names:
                nodes.add(name)
            nodes.add(edge.target_name)
        return nodes

    def get_node(self, name: str) -> CausalNode:
        """Get a node with its type (from manual override or edge extraction)."""
        node_type = self._node_types.get(name, "unknown")
        return CausalNode(name=name, node_type=node_type)

    def get_nodes(self) -> list[CausalNode]:
        """Get all nodes as CausalNode objects with types from edges."""
        # Build type map from edges (edge extraction takes precedence)
        type_map = dict(self._node_types)
        for edge in self.edges:
            # Handle multiple source nodes for interactions
            for src in edge.source_nodes:
                if src.node_type != "unknown":
                    type_map[src.name] = src.node_type
            tgt = edge.target_node
            if tgt.node_type != "unknown":
                type_map[tgt.name] = tgt.node_type

        return [CausalNode(name=name, node_type=type_map.get(name, "unknown"))
                for name in self.nodes]

    def nodes_by_type(self, node_type: str) -> list[CausalNode]:
        """Get all nodes of a specific type (condition, medication, etc)."""
        return [n for n in self.get_nodes() if n.node_type == node_type]

    def set_node_type(self, name: str, node_type: str):
        """Manually override a node's type."""
        self._node_types[name] = node_type

    def risk_factors(self, target: str) -> list[CausalEdge]:
        """Get edges that increase probability of target."""
        target_lower = target.lower()
        return [e for e in self.edges
                if e.direction == "risk" and target_lower in e.target.lower()]

    def protective_factors(self, target: str) -> list[CausalEdge]:
        """Get edges that decrease probability of target."""
        target_lower = target.lower()
        return [e for e in self.edges
                if e.direction == "protective" and target_lower in e.target.lower()]

    def consequences_of(self, source: str) -> list[CausalEdge]:
        """Get all edges originating from source."""
        source_lower = source.lower()
        return [e for e in self.edges if source_lower in e.source.lower()]

    def causes_of(self, target: str) -> list[CausalEdge]:
        """Get all edges pointing to target."""
        target_lower = target.lower()
        return [e for e in self.edges if target_lower in e.target.lower()]

    def edges_by_direction(self, direction: str) -> list[CausalEdge]:
        """Get edges by direction ('risk', 'protective', 'causal')."""
        return [e for e in self.edges if e.direction == direction]

    def interactions(self, interaction_type: Optional[str] = None) -> list[CausalEdge]:
        """
        Get edges representing interactions (multiple sources).

        Args:
            interaction_type: Filter by "and" or "or", or None for all interactions

        Returns:
            List of CausalEdge objects with multiple sources
        """
        if interaction_type:
            return [e for e in self.edges if e.interaction == interaction_type]
        return [e for e in self.edges if e.is_interaction]

    def causal_chains(self, start: str, max_depth: int = 5) -> list[list[CausalEdge]]:
        """Find all causal chains starting from a node."""
        chains = []
        start_lower = start.lower()

        def dfs(current: str, path: list[CausalEdge], depth: int):
            if depth >= max_depth:
                return
            for edge in self.edges:
                if current.lower() in edge.source.lower() and edge not in path:
                    new_path = path + [edge]
                    chains.append(new_path)
                    dfs(edge.target_name, new_path, depth + 1)

        dfs(start, [], 0)
        return chains

    def to_dict(self) -> dict:
        """Convert to dictionary with node types and interaction info."""
        return {
            "nodes": [{"name": n.name, "type": n.node_type} for n in self.get_nodes()],
            "edges": [{
                "sources": e.sources,
                "source": e.source,  # Combined string for compatibility
                "target": e.target,
                "type": e.edge_type,
                "interaction": e.interaction,
                "direction": e.direction,
                "strength": e.strength,
                "weight": e.weight,
            } for e in self.edges],
        }

    def to_networkx(self, expand_interactions: bool = True):
        """
        Convert to NetworkX DiGraph with node and edge attributes.

        Node attributes: node_type, time, is_interaction
        Edge attributes: edge_type, direction, strength, weight, source_time, target_time, interaction

        For interaction edges (A + B => C), if expand_interactions=True (default),
        creates an intermediate "interaction node" to represent the combined effect.
        This allows standard graph algorithms to work with interactions.

        Example:
            >>> G = graph.to_networkx()
            >>> # Get condition nodes
            >>> [n for n, d in G.nodes(data=True) if d.get('node_type') == 'condition']
            >>> # Get risk edges
            >>> [(u, v) for u, v, d in G.edges(data=True) if d['direction'] == 'risk']
            >>> # Find interaction nodes
            >>> [n for n, d in G.nodes(data=True) if d.get('is_interaction')]
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install networkx")

        G = nx.DiGraph()

        # Add nodes with types
        for node in self.get_nodes():
            G.add_node(node.name, node_type=node.node_type, time=node.time, is_interaction=False)

        # Add edges with attributes
        for edge in self.edges:
            if edge.is_interaction and expand_interactions:
                # Create intermediate interaction node
                op = "AND" if edge.interaction == "and" else "OR"
                interaction_node = f"({' {op} '.join(edge.source_names)})"

                # Add interaction node
                G.add_node(interaction_node, node_type="interaction", is_interaction=True,
                           interaction_type=edge.interaction, members=edge.source_names)

                # Connect sources to interaction node
                for src_node in edge.source_nodes:
                    G.add_edge(
                        src_node.name, interaction_node,
                        edge_type="member_of", direction="interaction",
                        strength="", weight=0.0,
                    )

                # Connect interaction node to target
                G.add_edge(
                    interaction_node, edge.target_name,
                    edge_type=edge.edge_type, direction=edge.direction,
                    strength=edge.strength, weight=edge.weight,
                    target_time=edge.target_time, interaction=edge.interaction,
                )
            else:
                # Simple edge or non-expanded interaction
                for src_name in edge.source_names:
                    G.add_edge(
                        src_name, edge.target_name,
                        edge_type=edge.edge_type, direction=edge.direction,
                        strength=edge.strength, weight=edge.weight,
                        source_time=edge.source_time, target_time=edge.target_time,
                        interaction=edge.interaction,
                    )
        return G

    @classmethod
    def from_networkx(cls, G) -> "CausalGraph":
        """Create CausalGraph from NetworkX DiGraph."""
        edges = []
        node_types = {}

        for node, data in G.nodes(data=True):
            if "node_type" in data and data["node_type"] != "interaction":
                node_types[node] = data["node_type"]

        for u, v, data in G.edges(data=True):
            edge_type = data.get("edge_type", "=>")
            interaction = data.get("interaction")
            # Skip member_of edges (internal to interaction representation)
            if edge_type == "member_of":
                continue
            edges.append(CausalEdge(sources=[u], target=v, edge_type=edge_type, interaction=interaction))

        graph = cls(edges=edges)
        graph._node_types = node_types
        return graph

    def ground_to_snomed(
        self,
        linker: "SNOMEDLinker" = None,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> "GroundedCausalGraph":
        """
        Ground all nodes to SNOMED CT concepts using SapBERT + FAISS.

        This method links each node in the causal graph to its closest
        SNOMED CT concept, providing standardized concept IDs.

        Args:
            linker: SNOMEDLinker instance. If None, creates a new one.
            threshold: Minimum similarity score for matches (0-1)
            verbose: Print progress messages

        Returns:
            GroundedCausalGraph with SNOMED-linked nodes

        Example:
            >>> graph = soap_note.extract_causal_graph()
            >>> grounded = graph.ground_to_snomed()
            >>> for node in grounded.nodes:
            ...     print(f"{node.mention} -> SCTID:{node.concept_id}")
        """
        # Import here to avoid circular dependency
        from synthlab.snomed import (
            SNOMEDLinker,
            GroundedNode,
            GroundedEdge,
            GroundedCausalGraph,
        )

        if linker is None:
            linker = SNOMEDLinker(verbose=verbose)

        # Get all unique node names
        all_nodes = self.get_nodes()

        if verbose:
            print(f"Grounding {len(all_nodes)} nodes to SNOMED CT...")

        # Link all nodes in batch
        mentions = [node.name for node in all_nodes]
        results = linker.link_batch(mentions, k=1, threshold=threshold)

        # Build grounded nodes
        grounded_nodes = {}
        unmatched = []

        for node, (mention, matches) in zip(all_nodes, results):
            if matches:
                match = matches[0]
                grounded_node = GroundedNode(
                    mention=mention,
                    concept_id=match.concept_id,
                    term=match.term,
                    node_type=node.node_type,
                    confidence=match.score,
                )
                grounded_nodes[mention] = grounded_node
            else:
                unmatched.append(mention)

        if verbose and unmatched:
            print(f"  Warning: {len(unmatched)} nodes could not be matched:")
            for m in unmatched[:5]:
                print(f"    - {m}")
            if len(unmatched) > 5:
                print(f"    ... and {len(unmatched) - 5} more")

        # Build grounded edges
        grounded_edges = []
        for edge in self.edges:
            # Get source nodes (handle interactions)
            source_mentions = edge.source_names
            target_mention = edge.target_name

            # Skip if any node is unmatched
            sources_grounded = [grounded_nodes.get(m) for m in source_mentions]
            target_grounded = grounded_nodes.get(target_mention)

            if all(sources_grounded) and target_grounded:
                # For interactions, use first source for simplicity
                # (could be extended to handle multi-source edges)
                for src in sources_grounded:
                    grounded_edges.append(GroundedEdge(
                        source=src,
                        target=target_grounded,
                        relation=edge.edge_type,
                    ))

        grounded_graph = GroundedCausalGraph(
            nodes=list(grounded_nodes.values()),
            edges=grounded_edges,
        )

        if verbose:
            print(f"  Grounded: {len(grounded_nodes)}/{len(all_nodes)} nodes, "
                  f"{len(grounded_edges)}/{len(self.edges)} edges")

        return grounded_graph

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram syntax with interaction support."""
        lines = ["graph LR"]

        def clean_node(name: str) -> str:
            """Clean node name for mermaid (remove [type], replace spaces)."""
            name = re.sub(r'\[[\w]+\]', '', name).strip()
            name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
            return re.sub(r'[^\w\s]', '', name).replace(' ', '_')

        for idx, edge in enumerate(self.edges):
            tgt = clean_node(edge.target_name)

            # Edge style based on direction
            if edge.direction == "risk":
                style = "-->"
                label = "risk"
            elif edge.direction == "protective":
                style = "-.->"
                label = "protects"
            else:
                style = "==>"
                label = "causes"

            if edge.is_interaction:
                # Create descriptive interaction node
                interaction_type = "AND" if edge.interaction == "and" else "OR"
                src_names = [clean_node(n) for n in edge.source_names]
                interaction_id = f"combo_{idx}"

                # Connect sources to interaction node with descriptive label
                for src in src_names:
                    lines.append(f"    {src} --> {interaction_id}[{interaction_type}]")

                # Connect interaction node to target
                lines.append(f"    {interaction_id} {style}|{label}| {tgt}")
            else:
                src = clean_node(edge.source_name)
                lines.append(f"    {src} {style}|{label}| {tgt}")

        # Add styling
        lines.extend([
            "",
            "    classDef condition fill:#D64550,color:white",
            "    classDef medication fill:#4A90A4,color:white",
            "    classDef lifestyle fill:#E59866,color:white",
            "    classDef finding fill:#58A87C,color:white",
        ])
        return "\n".join(lines)

    def summary(self) -> str:
        """Get a summary of the graph."""
        type_counts = {}
        for n in self.get_nodes():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        dir_counts = {}
        interaction_count = 0
        for e in self.edges:
            dir_counts[e.direction] = dir_counts.get(e.direction, 0) + 1
            if e.is_interaction:
                interaction_count += 1

        summary = (f"CausalGraph: {len(self.nodes)} nodes, {len(self.edges)} edges\n"
                   f"Nodes: {', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))}\n"
                   f"Edges: {', '.join(f'{k}={v}' for k, v in sorted(dir_counts.items()))}")
        if interaction_count > 0:
            summary += f"\nInteractions: {interaction_count} (AND/OR combinations)"
        return summary

    def __str__(self) -> str:
        return self.summary()

    def __len__(self) -> int:
        return len(self.edges)

    def plot(
        self,
        figsize: tuple[float, float] = (12, 8),
        title: Optional[str] = None,
        show_legend: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 300,
        node_shape: str = "rectangle",
        label_inside: bool = True,
        node_width: float = 1.2,
        node_height: float = 0.5,
        font_size: int = 9,
        interactive: bool = False,
        notebook: bool = True,
        height: str = "600px",
        width: str = "100%",
    ):
        """
        Create a publication-quality visualization of the causal graph.

        Interactions (A + B -> C) are shown with arrows converging at a
        small junction point before reaching the target.

        Args:
            figsize: Figure size in inches (width, height).
            title: Plot title. If None, no title shown.
            show_legend: Whether to show the legend.
            save_path: If provided, save figure to this path (PNG for static, HTML for interactive).
            dpi: Resolution for saved figure (static only).
            node_shape: Shape of nodes - "rectangle" or "circle".
            label_inside: If True, labels inside nodes. If False, beside nodes.
            node_width: Width of rectangle nodes (only used if node_shape="rectangle").
            node_height: Height of rectangle nodes (only used if node_shape="rectangle").
            font_size: Font size for node labels.
            interactive: If True, create an interactive visualization using pyvis.
            notebook: If True and interactive, render inline in Jupyter notebook.
            height: Height of interactive plot (e.g., "600px").
            width: Width of interactive plot (e.g., "100%" or "800px").

        Returns:
            For static: matplotlib Figure and Axes objects (fig, ax).
            For interactive: pyvis Network object.
        """
        # Handle interactive plotting with pyvis
        if interactive:
            return self._plot_interactive(
                title=title,
                save_path=save_path,
                notebook=notebook,
                height=height,
                width=width,
            )

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import networkx as nx
        except ImportError:
            raise ImportError("matplotlib and networkx required: pip install matplotlib networkx")

        if len(self.edges) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No causal relationships found",
                    ha='center', va='center', fontsize=11, color='#888')
            ax.axis('off')
            return fig, ax

        # Color palette - nodes by type
        type_colors = {
            "condition": "#E15759",   # Red
            "medication": "#4E79A7",  # Blue
            "procedure": "#B07AA1",   # Purple
            "lifestyle": "#F28E2B",   # Orange
            "symptom": "#76B7B2",     # Teal
            "finding": "#59A14F",     # Green
            "outcome": "#555555",     # Dark gray
            "unknown": "#AAAAAA",     # Light gray
        }

        # Edge colors by interaction type (not direction)
        # AND interactions = red, OR interactions = blue, simple = black
        # Edge colors by direction (risk/protective/causal)
        edge_colors = {
            'risk': '#C44E52',       # Red - increases risk
            'protective': '#4C72B0', # Blue - decreases risk
            'causal': '#333333',     # Black - causes
        }

        def clean_name(s: str) -> str:
            """Strip [type] suffix and clean for display."""
            s = re.sub(r'\[[\w]+\]', '', s).strip()
            s = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
            return s.replace('_', ' ')

        # Build graph with only real concept nodes
        G = nx.DiGraph()
        node_types = {}

        for node in self.get_nodes():
            name = clean_name(node.name)
            if name:
                G.add_node(name)
                node_types[name] = node.node_type

        # Collect edges, tracking interactions separately
        simple_edges = []      # (src, tgt, direction)
        interaction_edges = [] # ([src1, src2, ...], tgt, direction, interaction_type)

        for edge in self.edges:
            tgt = clean_name(edge.target_name)
            sources = [clean_name(s) for s in edge.source_names]
            sources = [s for s in sources if s and s != tgt]

            if not sources or not tgt:
                continue

            if edge.is_interaction and len(sources) > 1:
                interaction_edges.append((sources, tgt, edge.direction, edge.interaction))
                # Add edges to graph for layout
                for src in sources:
                    G.add_edge(src, tgt)
            else:
                for src in sources:
                    simple_edges.append((src, tgt, edge.direction))
                    G.add_edge(src, tgt)

        if G.number_of_nodes() == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('off')
            return fig, ax

        # Compute layout
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                # Try graphviz for hierarchical layout
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR -Gnodesep=0.8')
            except Exception:
                try:
                    if nx.is_directed_acyclic_graph(G):
                        gens = list(nx.topological_generations(G))
                        pos = {}
                        for gi, gen in enumerate(gens):
                            for ni, node in enumerate(sorted(gen)):
                                pos[node] = (gi * 2.5, ni - len(gen) / 2)
                    else:
                        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
                except Exception:
                    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')

        # Scaling for node size based on layout bounds
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            x_range = max(xs) - min(xs) if len(xs) > 1 else 1
            y_range = max(ys) - min(ys) if len(ys) > 1 else 1
            scale = min(x_range, y_range) / 10 if min(x_range, y_range) > 0 else 0.3
            node_radius = max(0.15, min(0.4, scale))
        else:
            node_radius = 0.3

        # Calculate shrink values based on node shape
        if node_shape == "rectangle":
            shrink_val = max(node_width, node_height) * 35  # Shrink for rectangles
        else:
            shrink_val = node_radius * 50  # Shrink for circles

        # Draw simple edges (solid lines, color by direction)
        for src, tgt, direction in simple_edges:
            if src not in pos or tgt not in pos:
                continue
            x1, y1 = pos[src]
            x2, y2 = pos[tgt]
            edge_color = edge_colors.get(direction, '#333333')
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=1.5,
                               shrinkA=shrink_val, shrinkB=shrink_val,
                               connectionstyle='arc3,rad=0.1'))

        # Draw interaction edges with junction points
        # Color by direction (risk=red, protective=blue, causal=black)
        # Line style by interaction: solid=simple, dotted=AND, dashed=OR
        for sources, tgt, direction, interaction_type in interaction_edges:
            if tgt not in pos:
                continue
            valid_sources = [s for s in sources if s in pos]

            # Choose color based on direction (risk/protective/causal)
            edge_color = edge_colors.get(direction, '#333333')
            # Choose line style based on interaction type: dotted=AND, dashed=OR
            if interaction_type == "or":
                linestyle = '--'   # Dashed for OR
            elif interaction_type == "and":
                linestyle = ':'    # Dotted for AND
            else:
                linestyle = '-'    # Solid (fallback)

            if len(valid_sources) < 2:
                # Fall back to simple edges
                for src in valid_sources:
                    x1, y1 = pos[src]
                    x2, y2 = pos[tgt]
                    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=1.5,
                                       linestyle=linestyle,
                                       shrinkA=shrink_val, shrinkB=shrink_val,
                                       connectionstyle='arc3,rad=0.1'))
                continue

            # Calculate junction point (weighted average closer to target)
            src_positions = [pos[s] for s in valid_sources]
            tgt_x, tgt_y = pos[tgt]
            avg_x = sum(p[0] for p in src_positions) / len(src_positions)
            avg_y = sum(p[1] for p in src_positions) / len(src_positions)
            # Junction at 70% toward target
            junc_x = avg_x + 0.7 * (tgt_x - avg_x)
            junc_y = avg_y + 0.7 * (tgt_y - avg_y)

            # Draw lines from each source to junction (no arrowhead)
            for src in valid_sources:
                x1, y1 = pos[src]
                ax.annotate('', xy=(junc_x, junc_y), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-', color=edge_color, lw=1.5,
                                   linestyle=linestyle,
                                   shrinkA=shrink_val, shrinkB=0,
                                   connectionstyle='arc3,rad=0.05'))

            # Draw junction point (small filled circle)
            junc_circle = plt.Circle((junc_x, junc_y), node_radius * 0.3,
                                     color=edge_color, ec='white', lw=1, zorder=15)
            ax.add_patch(junc_circle)

            # Draw arrow from junction to target
            ax.annotate('', xy=(tgt_x, tgt_y), xytext=(junc_x, junc_y),
                arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=2,
                               linestyle=linestyle,
                               shrinkA=0, shrinkB=shrink_val))

        # Draw nodes and labels
        for node in G.nodes():
            if node not in pos:
                continue
            x, y = pos[node]
            ntype = node_types.get(node, 'unknown')
            color = type_colors.get(ntype, '#AAA')

            # Wrap long labels
            label = node
            max_chars = 18 if label_inside else 15
            if len(label) > max_chars:
                words = label.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > max_chars:
                        lines.append(' '.join(current_line))
                        current_line = []
                if current_line:
                    lines.append(' '.join(current_line))
                label = '\n'.join(lines)

            if node_shape == "rectangle":
                # Calculate node dimensions based on label length
                n_lines = label.count('\n') + 1
                rect_h = node_height * max(1, n_lines * 0.7)
                rect_w = node_width

                # Draw rounded rectangle
                rect = mpatches.FancyBboxPatch(
                    (x - rect_w / 2, y - rect_h / 2), rect_w, rect_h,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color, edgecolor='white', linewidth=2, zorder=20
                )
                ax.add_patch(rect)

                if label_inside:
                    # Label inside rectangle
                    ax.text(x, y, label, fontsize=font_size, ha='center', va='center',
                           fontweight='medium', color='white', zorder=25)
                else:
                    # Label beside rectangle
                    ax.text(x + rect_w / 2 + 0.1, y, label,
                           fontsize=font_size, ha='left', va='center',
                           fontweight='medium', color='#333')
            else:
                # Circle shape (original behavior)
                circle = plt.Circle((x, y), node_radius, color=color, ec='white', lw=2, zorder=20)
                ax.add_patch(circle)

                if label_inside:
                    ax.text(x, y, label, fontsize=font_size - 1, ha='center', va='center',
                           fontweight='medium', color='white', zorder=25)
                else:
                    ax.text(x + node_radius + 0.1, y, label,
                           fontsize=font_size, ha='left', va='center',
                           fontweight='medium', color='#333')

        # Legend
        if show_legend:
            legend_elements = []
            present_types = set(node_types.values())

            for ntype in ['condition', 'medication', 'lifestyle', 'finding', 'symptom', 'procedure', 'outcome']:
                if ntype in present_types:
                    color = type_colors.get(ntype, '#AAA')
                    legend_elements.append(
                        mpatches.Patch(facecolor=color, edgecolor='white', label=ntype.capitalize())
                    )

            # Edge direction legend (colors)
            all_directions = set(e[2] for e in simple_edges) | set(e[2] for e in interaction_edges)
            legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
            for direction in ['risk', 'protective', 'causal']:
                if direction in all_directions:
                    color = edge_colors[direction]
                    label_text = {'risk': 'Increases risk', 'protective': 'Protective', 'causal': 'Causes'}[direction]
                    legend_elements.append(
                        plt.Line2D([0], [0], color=color, lw=2, linestyle='-', label=label_text)
                    )

            # Line style legend (interaction types)
            has_and = any(e[3] == "and" for e in interaction_edges)
            has_or = any(e[3] == "or" for e in interaction_edges)
            if simple_edges or has_and or has_or:
                legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='white', label=''))
            if simple_edges:
                legend_elements.append(
                    plt.Line2D([0], [0], color='#666', lw=2, linestyle='-', label='Simple (A→B)')
                )
            if has_and:
                legend_elements.append(
                    plt.Line2D([0], [0], color='#666', lw=2, linestyle=':', label='AND (A+B→C)')
                )
            if has_or:
                legend_elements.append(
                    plt.Line2D([0], [0], color='#666', lw=2, linestyle='--', label='OR (A|B→C)')
                )

            if legend_elements:
                legend = ax.legend(
                    handles=legend_elements, loc='upper left',
                    bbox_to_anchor=(1.02, 1), frameon=True,
                    fontsize=9, title='Legend', title_fontsize=10,
                )
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('#CCC')

        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15, color='#333')

        # Clean up
        ax.set_aspect('equal')
        ax.axis('off')

        # Adjust limits with padding
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            margin = 1.5
            ax.set_xlim(min(xs) - margin, max(xs) + margin + 3)  # Extra space for labels
            ax.set_ylim(min(ys) - margin, max(ys) + margin)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved to {save_path}")

        return fig, ax

    def _plot_interactive(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        notebook: bool = True,
        height: str = "600px",
        width: str = "100%",
    ):
        """
        Create an interactive visualization of the causal graph using pyvis.

        Args:
            title: Plot title.
            save_path: If provided, save HTML to this path.
            notebook: If True, render inline in Jupyter notebook.
            height: Height of the plot.
            width: Width of the plot.

        Returns:
            pyvis Network object.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("pyvis required for interactive plots: pip install pyvis")

        # Color palette - nodes by type
        type_colors = {
            "condition": "#E15759",   # Red
            "medication": "#4E79A7",  # Blue
            "procedure": "#B07AA1",   # Purple
            "lifestyle": "#F28E2B",   # Orange
            "symptom": "#76B7B2",     # Teal
            "finding": "#59A14F",     # Green
            "outcome": "#555555",     # Dark gray
            "unknown": "#AAAAAA",     # Light gray
        }

        # Edge colors by direction
        edge_colors = {
            'risk': '#C44E52',       # Red - increases risk
            'protective': '#4C72B0', # Blue - decreases risk
            'causal': '#333333',     # Black - causes
        }

        def clean_name(s: str) -> str:
            """Strip [type] suffix and clean for display."""
            s = re.sub(r'\[[\w]+\]', '', s).strip()
            s = re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()
            return s.replace('_', ' ')

        # Create network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=notebook,
            bgcolor="#ffffff",
            font_color="#333333",
        )

        # Configure physics for better layout
        net.set_options("""
        {
            "nodes": {
                "font": {"size": 14, "face": "arial"},
                "borderWidth": 2,
                "borderWidthSelected": 3
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "smooth": {"type": "curvedCW", "roundness": 0.2},
                "font": {"size": 10, "align": "middle"}
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 150,
                    "springConstant": 0.01,
                    "nodeDistance": 180
                },
                "solver": "hierarchicalRepulsion"
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 200,
                    "nodeSpacing": 100
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": {"enabled": true}
            }
        }
        """)

        if len(self.edges) == 0:
            net.add_node("empty", label="No causal relationships found", color="#888888")
            if save_path:
                net.save_graph(save_path)
            if notebook:
                return net.show(save_path or "causal_graph.html")
            return net

        # Collect all nodes
        node_types = {}
        for node in self.get_nodes():
            name = clean_name(node.name)
            if name:
                node_types[name] = node.node_type

        # Add nodes
        for name, ntype in node_types.items():
            color = type_colors.get(ntype, "#AAAAAA")
            net.add_node(
                name,
                label=name,
                color=color,
                title=f"{name}\nType: {ntype.capitalize()}",
                shape="box",
                font={"color": "white"},
            )

        # Add edges
        edge_id = 0
        for edge in self.edges:
            tgt = clean_name(edge.target_name)
            sources = [clean_name(s) for s in edge.source_names]
            sources = [s for s in sources if s and s != tgt and s in node_types]

            if not sources or not tgt or tgt not in node_types:
                continue

            edge_color = edge_colors.get(edge.direction, '#333333')

            # Determine line style (dashes) based on interaction type
            if edge.is_interaction and edge.interaction == "and":
                dashes = [5, 5]  # Dotted for AND
                interaction_label = " (AND)"
            elif edge.is_interaction and edge.interaction == "or":
                dashes = [10, 5]  # Dashed for OR
                interaction_label = " (OR)"
            else:
                dashes = False
                interaction_label = ""

            # Build tooltip
            direction_label = {
                'risk': 'increases risk of',
                'protective': 'protects against',
                'causal': 'causes'
            }.get(edge.direction, 'affects')

            for src in sources:
                tooltip = f"{src} {direction_label} {tgt}{interaction_label}"
                net.add_edge(
                    src,
                    tgt,
                    color=edge_color,
                    title=tooltip,
                    dashes=dashes,
                    width=2,
                )
                edge_id += 1

        # Add title if provided
        if title:
            net.heading = title

        # Save or show
        if save_path:
            net.save_graph(save_path)
            print(f"Saved interactive graph to {save_path}")

        if notebook:
            return net.show(save_path or "causal_graph.html")

        return net


def _parse_interaction_sources(source_str: str) -> tuple[list[str], Optional[str]]:
    """
    Parse source string for interaction operators (+ for AND, | for OR).

    Args:
        source_str: Source part of a causal expression

    Returns:
        (list of sources, interaction type or None)

    Examples:
        "DrugA + DrugB" -> (["DrugA", "DrugB"], "and")
        "BRCA1 | BRCA2" -> (["BRCA1", "BRCA2"], "or")
        "Obesity" -> (["Obesity"], None)
    """
    # Check for AND interaction (but not inside brackets or edge operators)
    # Use word boundary to avoid matching ++ in edge types
    if ' + ' in source_str and not any(et in source_str for et in CAUSAL_EDGE_TYPES.keys()):
        parts = [p.strip() for p in source_str.split(' + ') if p.strip()]
        if len(parts) > 1:
            return parts, "and"

    # Check for OR interaction
    if ' | ' in source_str:
        parts = [p.strip() for p in source_str.split(' | ') if p.strip()]
        if len(parts) > 1:
            return parts, "or"

    return [source_str], None


def parse_causal_graph(text: str) -> CausalGraph:
    """
    Parse causal relationships from text into a CausalGraph.

    Recognizes the notation:
    - ++>  strongly increases probability
    - +>   increases probability
    - ?+>  possibly increases
    - -->  strongly decreases probability
    - ->   decreases probability
    - ?->  possibly decreases
    - =>   direct causation

    Also supports interaction notation:
    - A + B => C  (A AND B together cause C)
    - A | B => C  (A OR B causes C, either sufficient)

    Args:
        text: Text containing causal relationships (e.g., from SOAP note assessment)

    Returns:
        CausalGraph with parsed edges

    Example:
        >>> text = '''
        ... Obesity[lifestyle] ++> Diabetes[condition]
        ... DrugA[medication] + DrugB[medication] => Liver_failure[condition]
        ... BRCA1[finding] | BRCA2[finding] ++> Breast_cancer[condition]
        ... '''
        >>> graph = parse_causal_graph(text)
        >>> print(len(graph.edges))
        3
        >>> print(graph.edges[1].interaction)
        and
    """
    edges = []

    # Edge types sorted by length (longest first) to match correctly
    edge_types_pattern = "|".join(
        re.escape(et) for et in sorted(CAUSAL_EDGE_TYPES.keys(), key=len, reverse=True)
    )

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('```'):
            continue

        # Strip leading # or - (markdown list/header markers) but keep content
        if line.startswith('#'):
            line = line.lstrip('#').strip()
        if line.startswith('-'):
            line = line.lstrip('-').strip()
        if line.startswith('*'):
            line = line.lstrip('*').strip()

        if not line:
            continue

        # Check if this line contains any edge types
        if not any(et in line for et in CAUSAL_EDGE_TYPES.keys()):
            continue

        # Split on edge operators, keeping them as separators
        parts = re.split(rf'({edge_types_pattern})', line)
        parts = [p.strip() for p in parts if p.strip()]

        # Process pairs: (source, edge_type, target, edge_type, target2, ...)
        i = 0
        while i < len(parts) - 2:
            source_str = parts[i]
            if parts[i + 1] in CAUSAL_EDGE_TYPES:
                edge_type = parts[i + 1]
                target = parts[i + 2]

                # Clean up source and target
                source_str = source_str.strip(' -,')
                target = target.strip(' -,')

                # Parse interaction operators in source
                sources, interaction = _parse_interaction_sources(source_str)

                if sources and target:
                    edge = CausalEdge(
                        sources=sources,
                        target=target,
                        edge_type=edge_type,
                        interaction=interaction,
                    )
                    # Only add if not a duplicate (using __eq__ for comparison)
                    if edge not in edges:
                        edges.append(edge)
                i += 2  # Move to target, which becomes next source
            else:
                i += 1

    return CausalGraph(edges=edges, raw_text=text)


# =============================================================================
# BioMCP Variant Annotation (Optional)
# =============================================================================


async def _annotate_variant_biomcp(rsid: str) -> Optional[dict]:
    """
    Annotate a single variant using BioMCP.

    Args:
        rsid: rsID of the variant (e.g., "rs121913529")

    Returns:
        Dict with variant annotations or None if not found/error
    """
    if not _biomcp_available:
        return None

    try:
        import asyncio
        # Try to get variant info - this is an async call
        # Note: This requires the biomcp-python package
        from biomcp.variants.get import variant_getter
        result = await variant_getter(variant_id=rsid)
        if result:
            # Extract phenotypes/disease associations
            phenotypes = []
            if hasattr(result, "phenotypes"):
                phenotypes = getattr(result, "phenotypes", [])
            elif hasattr(result, "conditions"):
                phenotypes = getattr(result, "conditions", [])

            # Extract drug associations (pharmacogenomics)
            drug_associations = []
            if hasattr(result, "drugs"):
                drug_associations = getattr(result, "drugs", [])
            elif hasattr(result, "drug_associations"):
                drug_associations = getattr(result, "drug_associations", [])

            # Extract clinical actionability
            actionability = None
            if hasattr(result, "actionability"):
                actionability = getattr(result, "actionability", None)

            # Get review status for confidence
            review_status = None
            if hasattr(result, "review_status"):
                review_status = getattr(result, "review_status", None)

            return {
                "rsid": rsid,
                "clinical_significance": getattr(result, "clinical_significance", None),
                "conditions": phenotypes if phenotypes else getattr(result, "conditions", []),
                "phenotypes": phenotypes,
                "gene": getattr(result, "gene", {}).get("symbol") if hasattr(result, "gene") else None,
                "protein_change": getattr(result, "protein_change", None),
                "frequencies": getattr(result, "frequencies", {}),
                "predictions": getattr(result, "predictions", {}),
                "drug_associations": drug_associations,
                "actionability": actionability,
                "review_status": review_status,
                # Try to get effect direction from phenotype associations
                "effect_type": getattr(result, "effect_type", None),  # protective, risk, etc.
            }
    except Exception:
        pass
    return None


def annotate_variants_biomcp(rsids: list[str], max_variants: int = 20) -> dict[str, dict]:
    """
    Annotate multiple variants using BioMCP (synchronous wrapper).

    This function queries BioMCP for detailed variant annotations including
    clinical significance, associated conditions, population frequencies,
    and functional predictions.

    Args:
        rsids: List of rsIDs to annotate
        max_variants: Maximum number of variants to query (to avoid rate limits)

    Returns:
        Dict mapping rsID to annotation dict

    Example:
        >>> annotations = annotate_variants_biomcp(["rs121913529", "rs6025"])
        >>> print(annotations["rs121913529"]["clinical_significance"])
        "Pathogenic"
    """
    if not _biomcp_available:
        return {}

    import asyncio

    async def _annotate_batch():
        results = {}
        for rsid in rsids[:max_variants]:
            annotation = await _annotate_variant_biomcp(rsid)
            if annotation:
                results[rsid] = annotation
        return results

    try:
        # Run async function
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _annotate_batch())
                return future.result()
        else:
            return asyncio.run(_annotate_batch())
    except Exception:
        return {}


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
    ) -> tuple[str, dict[str, Any]]:
        """
        Format patient data as structured text for LLM input.

        Args:
            patient: Patient object from MultimodalDataset
            max_tokens: Approximate max tokens (characters / 4)
            use_biomcp: If True, use BioMCP to enrich genetic variant annotations.
                        Requires: pip install biomcp-python

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

        # Genomics (if available)
        if hasattr(patient, 'genomics') and patient.genomics is not None:
            try:
                genomics_text, genomics_biomcp = FHIRFormatter._format_genomics(
                    patient.genomics,
                    use_biomcp=use_biomcp,
                )
                sections.append(genomics_text)
                biomcp_annotations = genomics_biomcp
            except Exception:
                pass  # Skip if genomics formatting fails

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

        Returns:
            Tuple of (formatted_text, biomcp_annotations_dict)
            - formatted_text: Formatted string with genetic findings
            - biomcp_annotations_dict: Raw BioMCP annotations (empty if not used)
        """
        biomcp_raw = {}  # Store raw bioMCP annotations
        lines = ["## GENETIC / GENOMIC FINDINGS"]

        if genomics_df is None or len(genomics_df) == 0:
            lines.append("No genetic data available.")
            return "\n".join(lines), biomcp_raw

        # Filter to variants the patient has (VARIANT == true)
        try:
            patient_variants = genomics_df.filter(genomics_df["VARIANT"] == True)
        except Exception:
            # Try string comparison if boolean doesn't work
            try:
                patient_variants = genomics_df.filter(genomics_df["VARIANT"] == "true")
            except Exception:
                patient_variants = genomics_df

        if len(patient_variants) == 0:
            lines.append("No clinically significant genetic variants detected.")
            return "\n".join(lines), biomcp_raw

        # Categorize by clinical significance
        significance_order = [
            "Pathogenic",
            "Likely Pathogenic",
            "Risk Factor",
            "Uncertain",
        ]

        # Get unique variants (dedupe by rsID)
        seen_variants = set()
        categorized = {sig: [] for sig in significance_order}

        for row in patient_variants.iter_rows(named=True):
            rsid = row.get("INDEX_PREFIX", row.get("INDEX", "Unknown"))
            if rsid in seen_variants:
                continue
            seen_variants.add(rsid)

            gene = row.get("GENE", "Unknown")
            significance = row.get("CLINICAL_SIGNIFICANCE", "Unknown")
            chrom = row.get("CHROMOSOME", "?")
            allele = row.get("ALLELE", "")
            ancestral = row.get("ANCESTRAL_ALLELE", "")

            # Determine if it's a variant vs reference
            is_variant = allele != ancestral if ancestral else True

            if not is_variant:
                continue

            # Categorize
            for sig in significance_order:
                if sig.lower() in str(significance).lower():
                    categorized[sig].append({
                        "rsid": rsid,
                        "gene": gene,
                        "chrom": chrom,
                        "significance": significance,
                        "allele": allele,
                    })
                    break

        # Format output - prioritize clinically significant variants
        for sig in ["Pathogenic", "Likely Pathogenic", "Risk Factor"]:
            variants = categorized.get(sig, [])
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

        # Clinical notes for specific well-known genes
        clinically_relevant_genes = {
            "BRCA1": "breast/ovarian cancer risk",
            "BRCA2": "breast/ovarian cancer risk",
            "PCSK9": "cholesterol metabolism, cardiovascular risk",
            "F5": "Factor V Leiden, thrombosis risk",
            "APOE": "Alzheimer's disease, cardiovascular risk",
            "MTHFR": "folate metabolism",
            "CYP2C19": "drug metabolism (clopidogrel, PPIs)",
            "CYP2D6": "drug metabolism (codeine, tamoxifen)",
            "TNNT2": "cardiomyopathy risk",
            "PSEN2": "early-onset Alzheimer's disease",
            "LDLR": "familial hypercholesterolemia",
            "MYH7": "cardiomyopathy risk",
        }

        noted_genes = set()
        for sig in ["Pathogenic", "Likely Pathogenic", "Risk Factor"]:
            for v in categorized.get(sig, []):
                gene = v['gene']
                if gene in clinically_relevant_genes and gene not in noted_genes:
                    noted_genes.add(gene)

        if noted_genes:
            lines.append("\n### Clinical Relevance Notes")
            for gene in sorted(noted_genes):
                lines.append(f"- {gene}: Associated with {clinically_relevant_genes[gene]}")

        # BioMCP enrichment (optional)
        if use_biomcp and _biomcp_available:
            # Get rsIDs of ALL variants for enrichment (not just pathogenic)
            all_rsids = []
            for category_variants in categorized.values():
                for v in category_variants:
                    if v['rsid'].startswith("rs"):
                        all_rsids.append(v['rsid'])

            if all_rsids:
                # Prioritize: pathogenic first, then risk factors, then others
                priority_order = ["Pathogenic", "Likely Pathogenic", "Risk Factor", "Protective"]
                prioritized_rsids = []
                seen = set()
                for sig in priority_order:
                    for v in categorized.get(sig, []):
                        if v['rsid'].startswith("rs") and v['rsid'] not in seen:
                            prioritized_rsids.append(v['rsid'])
                            seen.add(v['rsid'])
                # Add remaining rsIDs
                for rsid in all_rsids:
                    if rsid not in seen:
                        prioritized_rsids.append(rsid)
                        seen.add(rsid)

                biomcp_annotations = annotate_variants_biomcp(prioritized_rsids, max_variants=20)
                if biomcp_annotations:
                    # Store raw annotations before summarizing
                    biomcp_raw = biomcp_annotations

                    lines.append("\n### Clinical Variant Annotations (BioMCP)")
                    lines.append("*Curated disease associations and clinical interpretations:*\n")

                    for rsid, ann in biomcp_annotations.items():
                        gene = ann.get("gene", "")
                        gene_str = f" ({gene})" if gene else ""
                        ann_lines = [f"**{rsid}{gene_str}**"]

                        # Clinical significance with review status
                        clin_sig = ann.get("clinical_significance")
                        if clin_sig:
                            review = ann.get("review_status", "")
                            review_str = f" [{review}]" if review else ""
                            ann_lines.append(f"  - Clinical significance: {clin_sig}{review_str}")

                        # Disease/phenotype associations - THIS IS KEY
                        conditions = ann.get("conditions") or ann.get("phenotypes") or []
                        if conditions:
                            if len(conditions) <= 3:
                                conditions_str = ", ".join(str(c) for c in conditions)
                            else:
                                conditions_str = ", ".join(str(c) for c in conditions[:3]) + f" (+{len(conditions)-3} more)"
                            ann_lines.append(f"  - Disease associations: {conditions_str}")

                        # Effect type (protective vs risk) if available
                        effect = ann.get("effect_type")
                        if effect:
                            ann_lines.append(f"  - Effect: {effect}")

                        # Drug associations (pharmacogenomics)
                        drugs = ann.get("drug_associations") or []
                        if drugs:
                            drugs_str = ", ".join(str(d) for d in drugs[:3])
                            ann_lines.append(f"  - Drug interactions: {drugs_str}")

                        # Actionability
                        if ann.get("actionability"):
                            ann_lines.append(f"  - Actionability: {ann['actionability']}")

                        # Functional predictions
                        predictions = ann.get("predictions", {})
                        pred_parts = []
                        if predictions.get("cadd"):
                            pred_parts.append(f"CADD={predictions['cadd']}")
                        if predictions.get("polyphen"):
                            pred_parts.append(f"PolyPhen={predictions['polyphen']}")
                        if predictions.get("sift"):
                            pred_parts.append(f"SIFT={predictions['sift']}")
                        if pred_parts:
                            ann_lines.append(f"  - Predictions: {', '.join(pred_parts)}")

                        # Protein change
                        if ann.get("protein_change"):
                            ann_lines.append(f"  - Protein change: {ann['protein_change']}")

                        # Population frequency
                        freq = ann.get("frequencies", {}).get("gnomad")
                        if freq:
                            ann_lines.append(f"  - Population freq (gnomAD): {freq:.4f}")

                        lines.extend(ann_lines)
                        lines.append("")  # Blank line between variants

        elif use_biomcp and not _biomcp_available:
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

    SOAP_USER_PROMPT_NO_IMAGES = """Generate a SOAP note from this patient record:

{patient_data}

Format:

# Patient Story
1-2 paragraph narrative of health journey, focusing on key events and how they connect to current status.

# Subjective
Symptoms, complaints, relevant history.

# Objective
Vitals, labs (highlight abnormals), exam findings.
If genetic/genomic data is present, summarize key variants and their clinical significance.

# Assessment
Active diagnoses, disease progression, risk factors, clinical reasoning.
GENETIC INTERPRETATION: If genetic variants are present with BioMCP annotations:
- Discuss each variant's disease associations and whether they increase or decrease risk
- Explain how variants relate to the patient's current conditions
- Note any protective variants that may be beneficial
- Mention pharmacogenomic implications for drug selection/dosing if relevant

## Causal Graph
List DIRECT causal relationships, one per line:
Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Diabetic_Nephropathy)
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic
- ARROW = ++> (strong risk), +> (risk), ?+> (uncertain), --> (strong protection), -> (protection), => (causes)

Example:
Obesity[lifestyle] ++> Diabetes[condition]
Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Stroke[condition] => Cognitive_Impairment[condition]
Metformin[medication] --> Blood_glucose[finding]

# Plan
Treatment, monitoring, preventive care, follow-up.

# Future Considerations
Which causal chains may progress? Interventions to interrupt harmful chains? Protective factors to reinforce?

# Summary
2-3 sentences on key findings and causal relationships."""

    SOAP_USER_PROMPT_WITH_IMAGES = """Generate a SOAP note from this patient record. {num_images} image(s) attached - analyze them.

{patient_data}

Format:

# Patient Story
1-2 paragraph narrative of health journey.

# Subjective
Symptoms, complaints, relevant history.

# Objective
Vitals, labs (highlight abnormals), exam findings.
Imaging Findings: Describe attached images - modality, anatomy, abnormalities, quality.
If genetic/genomic data is present, summarize key variants and their clinical significance.

# Assessment
Diagnoses, disease progression, risk factors. Integrate imaging findings.
GENETIC INTERPRETATION: If genetic variants are present with BioMCP/clinical annotations:
- For EACH annotated variant, explain its disease associations and clinical impact
- State whether variants INCREASE or DECREASE risk for specific conditions
- Connect variants to the patient's actual diagnoses (e.g., "rs699 associated with decreased CAD risk may be protective")
- Note any pharmacogenomic implications for current medications

## Causal Graph
List DIRECT causal relationships, one per line:
Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Lung_cancer)
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic
- ARROW = ++> (strong risk), +> (risk), ?+> (uncertain), --> (strong protection), -> (protection), => (causes)

Example:
Smoking[lifestyle] ++> Lung_nodule[finding]
Lung_nodule[finding] ?+> Lung_cancer[condition]
Lung_cancer[condition] => Biopsy[procedure]

# Plan
Treatment, monitoring, preventive care, follow-up.

# Future Considerations
Causal chains likely to progress? Interventions? Protective factors?

# Summary
2-3 sentences on key findings."""

    CHUNK_SUMMARY_PROMPT = """Summarize this time period from a patient's history (2-3 paragraphs):

{chunk_data}

Cover: diagnoses, treatments (and their effects), health trajectory.
Note DIRECT causal relationships using STRICT FORMAT (one per line, no explanations):
Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic

Examples:
Hypertension[condition] ++> Stroke[condition]
Lisinopril[medication] --> Blood_pressure[finding]
Appendicitis[condition] => Appendectomy[procedure]"""

    AGGREGATE_PROMPT_NO_IMAGES = """Synthesize SOAP note from time period summaries.

Patient: {patient_name}

{summaries}
{genetics_section}
Sections: Patient Story, Subjective, Objective, Assessment (with Causal Graph), Plan, Future Considerations, Summary.

## Causal Graph
List DIRECT causal relationships, one per line:
Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Diabetic_Nephropathy)
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic
- ARROW = ++> (strong risk), +> (risk), --> (strong protection), -> (protection), => (causes)
- Include genetic risk factors in causal relationships (e.g., BRCA1_mutation[genetic] ++> Breast_cancer[condition])

Example:
Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Hypertension[condition] ++> Chronic_Kidney_Disease[condition]
Metformin[medication] --> Blood_glucose[finding]
APOE4_variant[genetic] ++> Alzheimers_disease[condition]"""

    AGGREGATE_PROMPT_WITH_IMAGES = """Synthesize SOAP note from time period summaries. {num_images} image(s) attached - analyze them.

Patient: {patient_name}

{summaries}
{genetics_section}
Sections: Patient Story, Subjective, Objective (with Imaging Findings), Assessment (with Causal Graph), Plan, Future Considerations, Summary.

## Causal Graph
List DIRECT causal relationships, one per line:
Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Lung_cancer)
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic
- ARROW = ++> (strong risk), +> (risk), --> (strong protection), -> (protection), => (causes)
- Include genetic risk factors in causal relationships (e.g., BRCA1_mutation[genetic] ++> Breast_cancer[condition])

Example:
Lung_nodule[finding] ?+> Lung_cancer[condition]
Lung_cancer[condition] => CT_guided_biopsy[procedure]
Bronchodilator[medication] --> Wheezing[symptom]"""

    # Prompts for separate_causal_graph mode (SOAP without causal graph)
    SOAP_NO_GRAPH_PROMPT = """Generate a SOAP note from this patient record:

{patient_data}

Format:

# Patient Story
1-2 paragraph narrative of health journey, focusing on key events and how they connect to current status.

# Subjective
Symptoms, complaints, relevant history.

# Objective
Vitals, labs (highlight abnormals), exam findings.
If genetic/genomic data is present, summarize key variants and their clinical significance.

# Assessment
Active diagnoses, disease progression, risk factors, clinical reasoning.
GENETIC INTERPRETATION: If genetic variants are present with BioMCP/clinical annotations:
- For EACH annotated variant, explain its disease associations and clinical impact
- State whether variants INCREASE or DECREASE risk for specific conditions
- Connect variants to the patient's actual diagnoses
- Note any pharmacogenomic implications for current medications
(Causal graph will be generated separately)

# Plan
Treatment, monitoring, preventive care, follow-up.
Consider genetic factors when recommending screening or preventive measures.

# Future Considerations
Which conditions may progress? Interventions needed? Protective factors to reinforce?
Include genetic predispositions in risk assessment.

# Summary
2-3 sentences on key findings."""

    SOAP_NO_GRAPH_WITH_IMAGES_PROMPT = """Generate a SOAP note from this patient record. {num_images} image(s) attached - analyze them.

{patient_data}

Format:

# Patient Story
1-2 paragraph narrative of health journey.

# Subjective
Symptoms, complaints, relevant history.

# Objective
Vitals, labs (highlight abnormals), exam findings.
Imaging Findings: Describe attached images - modality, anatomy, abnormalities, quality.
If genetic/genomic data is present, summarize key variants and their clinical significance.

# Assessment
Diagnoses, disease progression, risk factors. Integrate imaging findings.
GENETIC INTERPRETATION: If genetic variants are present with BioMCP/clinical annotations:
- For EACH annotated variant, explain its disease associations and clinical impact
- State whether variants INCREASE or DECREASE risk for specific conditions
- Connect variants to the patient's actual diagnoses (e.g., "rs699 associated with decreased CAD risk")
- Note any pharmacogenomic implications for current medications
(Causal graph will be generated separately)

# Plan
Treatment, monitoring, preventive care, follow-up.
Consider genetic factors when recommending screening or preventive measures.

# Future Considerations
Which conditions may progress? Interventions needed?
Include genetic predispositions in risk assessment.

# Summary
2-3 sentences on key findings."""

    CAUSAL_GRAPH_PROMPT = """Based on this SOAP note, generate a causal graph showing medical relationships.

SOAP Note:
{soap_note}

Generate DIRECT causal relationships, one per line.
Format: Cause[type] ARROW Effect[type]
- Use underscores for multi-word terms (e.g., Diabetic_Nephropathy)
- [type] = condition, medication, procedure, lifestyle, symptom, finding, outcome, or genetic
- ARROW = ++> (strong risk), +> (risk), ?+> (uncertain), --> (strong protection), -> (protection), => (causes)

Example:
Diabetes[condition] ++> Diabetic_Nephropathy[condition]
Stroke[condition] => Cognitive_Impairment[condition]
Hypertension[condition] ++> Cardiovascular_Disease[condition]
Metformin[medication] --> Blood_glucose[finding]

Output the relationships now:"""

    # Two-stage grounded causal graph prompts
    ENTITY_EXTRACTION_PROMPT = """Extract all medical entities from this clinical note.

SOAP Note:
{soap_note}

List each unique medical entity (conditions, medications, procedures, symptoms, findings, lifestyle factors).
Output one entity per line, using standard medical terminology.
Be specific and use proper clinical terms (e.g., "Type 2 diabetes mellitus" not "diabetes").

Format: entity_name | type
Types: condition, medication, procedure, symptom, finding, lifestyle, genetic

Example output:
Type 2 diabetes mellitus | condition
Metformin | medication
Hypertension | condition
Chronic kidney disease | condition
Elevated creatinine | finding
Smoking | lifestyle

Extract entities now:"""

    GROUNDED_RELATIONSHIP_PROMPT = """Identify causal relationships between these medical concepts.

Available concepts:
{concept_list}

Based on medical knowledge, identify direct causal relationships between these concepts.
Use the LABEL (the short identifier before the equals sign) for each concept.

Format: LABEL ARROW LABEL
- ARROW types: ++> (strong risk), +> (risk), --> (strong protection), -> (protection), => (causes)

Examples:
C1 ++> C2
C3 => C4
C5 -> C6

List all valid medical relationships (one per line, using only LABELs from the list above):"""

    # Aggregate prompts without causal graph (for separate_causal_graph mode)
    AGGREGATE_NO_GRAPH_PROMPT = """Synthesize SOAP note from time period summaries.

Patient: {patient_name}

{summaries}
{genetics_section}
Sections: Patient Story, Subjective, Objective, Assessment, Plan, Future Considerations, Summary.
(Causal graph will be generated separately)

Focus on synthesizing the key clinical findings across all time periods. Integrate relevant genetic findings into the assessment and plan."""

    AGGREGATE_NO_GRAPH_WITH_IMAGES_PROMPT = """Synthesize SOAP note from time period summaries. {num_images} image(s) attached - analyze them.

Patient: {patient_name}

{summaries}
{genetics_section}
Sections: Patient Story, Subjective, Objective (with Imaging Findings), Assessment, Plan, Future Considerations, Summary.
(Causal graph will be generated separately)

Focus on synthesizing the key clinical findings and integrating imaging results. Integrate relevant genetic findings into the assessment and plan."""

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
        verbose: bool = True,
        approximate_tokens: bool = False,
        multi_gpu: bool = False,
        gpu_memory_fraction: Optional[float] = None,
        batch_size: Optional[int] = None,
        include_imaging: bool = True,
        max_images: int = 3,
        use_biomcp: bool = False,
        separate_causal_graph: bool = False,
        include_admissions: bool = False,
        ground_snomed: bool = False,
        snomed_embedding_model: Optional[str] = None,
        grounded_graph_mode: bool = True,
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
            separate_causal_graph: If True, generate the causal graph in a separate
                        prompt after the main SOAP note. This avoids hitting token
                        generation limits. The causal graph is only generated for
                        the final SOAP note, not for chunk summaries. (default: False)
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
            grounded_graph_mode: If True (default when ground_snomed=True), use a
                        two-stage approach: first extract and ground entities to SNOMED,
                        then identify relationships between grounded concepts. This ensures
                        100% grounding by construction. If False, generates free-form
                        causal graph then attempts retrospective SNOMED mapping.
        """
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
        self.separate_causal_graph = separate_causal_graph
        self.include_admissions = include_admissions
        self.ground_snomed = ground_snomed
        self.snomed_embedding_model = snomed_embedding_model
        self.grounded_graph_mode = grounded_graph_mode

        # Validate SNOMED grounding dependencies early (fail fast)
        if self.ground_snomed:
            self._validate_snomed_dependencies()

        self._model = None
        self._processor = None
        self._pipe = None
        self._tokenizer = None
        self._optimal_batch_size = None  # Cached after first calculation
        self._snomed_linker = None  # Lazy-loaded SNOMEDLinker for grounding

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
            # Load processor with fast image processor
            processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)

            pipeline_kwargs = {
                "task": "image-text-to-text",
                "model": self.model_id,
                "dtype": dtype,
                "use_fast": True,
                "processor": processor,
            }

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
        """Remove thinking tokens and other artifacts from model output."""
        import re

        # Remove thinking blocks (e.g., <unused94>thought ... </unused94>)
        text = re.sub(r'<unused\d+>thought.*?(?:</unused\d+>|$)', '', text, flags=re.DOTALL)
        # Remove any remaining <unusedXX> tokens
        text = re.sub(r'<unused\d+>', '', text)
        # Remove <thinking> blocks if present
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

        return text.strip()

    def _generate_text(
        self,
        prompt: str,
        images: Optional[list] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the model."""
        self._load_model()

        # Build message content
        content = []

        if images:
            for img in images:
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Build generate kwargs
        generate_kwargs = {
            "repetition_penalty": 1.2,  # Discourage repetition
        }
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        try:
            output = self._pipe(text=messages, **generate_kwargs)
            response = output[0]["generated_text"][-1]["content"]
            return self._clean_response(response)
        except Exception as e:
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
            "do_sample": False,  # Greedy decoding for consistent output
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

        Returns:
            SOAPNote object with structured output
        """
        # Use instance defaults if not overridden
        include_imaging = include_imaging if include_imaging is not None else self.include_imaging
        max_images = max_images if max_images is not None else self.max_images
        use_biomcp = use_biomcp if use_biomcp is not None else self.use_biomcp
        ground_snomed = ground_snomed if ground_snomed is not None else self.ground_snomed
        snomed_embedding_model = snomed_embedding_model if snomed_embedding_model is not None else self.snomed_embedding_model

        patient_name = patient.name or patient.patient_id[:8]

        # Get patient data (with optional BioMCP timing)
        biomcp_seconds = 0.0
        if use_biomcp:
            biomcp_start = time.perf_counter()
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(patient, use_biomcp=True)
            biomcp_seconds = time.perf_counter() - biomcp_start
            if self.verbose:
                print(f"  BioMCP enrichment: {biomcp_seconds:.2f}s")
        else:
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(patient, use_biomcp=False)
        text_tokens = len(patient_text) // 4  # Rough estimate

        # Load images if requested
        images = []
        if include_imaging and patient.dicom_paths:
            images = self._load_patient_images(patient, max_images)

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

        # Report generation mode
        if self.verbose:
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

        if use_hierarchical:
            result = self._generate_hierarchical(patient, images, biomcp_annotations, chunks, biomcp_seconds)
        else:
            result = self._generate_direct(patient, patient_text, images, biomcp_annotations, biomcp_seconds)

        # Ground causal graph to SNOMED CT if enabled
        if ground_snomed:
            if self.grounded_graph_mode:
                # Two-stage approach: extract entities, ground, then build relationships
                result = self._generate_grounded_causal_graph(result, embedding_model=snomed_embedding_model)
            else:
                # Retrospective grounding of free-form graph
                result = self._ground_causal_graph(result, embedding_model=snomed_embedding_model)

        return result

    def _load_snomed_linker(self, embedding_model: Optional[str] = None) -> "SNOMEDLinker":
        """Load or return cached SNOMEDLinker.

        Args:
            embedding_model: Override the embedding model (if different from cached linker,
                           a new linker will be created)
        """
        from synthlab.snomed import SNOMEDLinker, EMBEDDING_MODELS, get_sample_snomed_concepts

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

        self._snomed_linker = SNOMEDLinker(
            model_id=model_id,
            verbose=self.verbose,
        )

        # Auto-build index with sample concepts if it doesn't exist
        if not self._snomed_linker.is_ready():
            if self.verbose:
                print(f"  Building SNOMED index (first time for {model_name})...")
            concepts = get_sample_snomed_concepts()
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
            linker = self._load_snomed_linker(embedding_model=embedding_model)

            # Stage 1: Extract entities from SOAP note
            if self.verbose:
                print("  Stage 1: Extracting medical entities...")

            soap_text = str(soap_note)
            entity_prompt = self.ENTITY_EXTRACTION_PROMPT.format(soap_note=soap_text)
            entity_response = self._generate_text(entity_prompt, max_new_tokens=2000)

            # Parse extracted entities
            entities = []
            entity_types = {}
            for line in entity_response.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Parse "entity_name | type" format
                if "|" in line:
                    parts = line.split("|")
                    entity_name = parts[0].strip()
                    entity_type = parts[1].strip().lower() if len(parts) > 1 else "condition"
                else:
                    entity_name = line.strip()
                    entity_type = "condition"

                if entity_name and len(entity_name) > 1:
                    entities.append(entity_name)
                    entity_types[entity_name] = entity_type

            if self.verbose:
                print(f"    Extracted {len(entities)} entities")

            if not entities:
                if self.verbose:
                    print("  No entities extracted, skipping causal graph")
                return soap_note

            # Stage 2: Ground entities to SNOMED
            if self.verbose:
                print("  Stage 2: Grounding to SNOMED CT...")

            # Link all entities
            results = linker.link_batch_with_cache(
                entities,
                k=1,
                threshold=0.5,
                track_gaps=True,
            )

            # Build grounded nodes (only keep high-confidence matches)
            grounded_concepts = {}  # concept_id -> GroundedNode
            entity_to_concept = {}  # entity_name -> concept_id

            # Track mapping statistics
            matched_count = 0
            unmatched_entities = []
            duplicate_mappings = 0

            for entity_name, matches in results:
                if matches and matches[0].score >= 0.5:
                    match = matches[0]
                    matched_count += 1
                    # Avoid duplicates (same concept from different mentions)
                    if match.concept_id not in grounded_concepts:
                        grounded_concepts[match.concept_id] = GroundedNode(
                            mention=entity_name,
                            concept_id=match.concept_id,
                            term=match.term,
                            node_type=entity_types.get(entity_name, "condition"),
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
            if self.verbose:
                print("  Stage 3: Identifying causal relationships...")

            # Build concept list with short labels (easier for LLM to use)
            concept_lines = []
            label_to_cid = {}  # Map short label -> concept_id
            cid_to_label = {}  # Map concept_id -> short label
            for i, (cid, node) in enumerate(grounded_concepts.items(), 1):
                label = f"C{i}"
                label_to_cid[label] = cid
                cid_to_label[cid] = label
                concept_lines.append(f"{label} = {node.term} [{node.node_type}]")
            concept_list = "\n".join(concept_lines)

            relationship_prompt = self.GROUNDED_RELATIONSHIP_PROMPT.format(
                concept_list=concept_list
            )
            relationship_response = self._generate_text(relationship_prompt, max_new_tokens=2000)

            # Parse relationships
            grounded_edges = []
            edge_type_map = {
                "++>": "risk",
                "+>": "risk",
                "?+>": "risk",
                "-->": "protective",
                "->": "protective",
                "=>": "causal",
            }

            # Debug: show first few lines of response
            if self.verbose:
                response_lines = relationship_response.strip().split("\n")[:5]
                print(f"    LLM response (first 5 lines): {response_lines}")

            parsed_count = 0
            invalid_refs = 0
            for line in relationship_response.strip().split("\n"):
                line = line.strip().upper()  # Normalize to uppercase for C1, C2 matching
                if not line or line.startswith("#"):
                    continue

                # Parse "LABEL ARROW LABEL" format (e.g., "C1 ++> C2")
                for arrow, edge_type in edge_type_map.items():
                    if arrow in line:
                        parts = line.split(arrow)
                        if len(parts) == 2:
                            source_label = parts[0].strip()
                            target_label = parts[1].strip()
                            parsed_count += 1

                            # Convert labels to concept IDs
                            source_cid = label_to_cid.get(source_label)
                            target_cid = label_to_cid.get(target_label)

                            # Validate both concepts exist
                            if source_cid and target_cid:
                                grounded_edges.append(GroundedEdge(
                                    source=grounded_concepts[source_cid],
                                    target=grounded_concepts[target_cid],
                                    relation=arrow,
                                ))
                            else:
                                invalid_refs += 1
                                if self.verbose and invalid_refs <= 3:
                                    print(f"    Invalid ref: '{source_label}' -> '{target_label}' (not in label list)")
                        break

            if self.verbose:
                print(f"    Parsed {parsed_count} relationships, {len(grounded_edges)} valid, {invalid_refs} invalid refs")

            # Build grounded causal graph
            grounded_graph = GroundedCausalGraph(
                nodes=list(grounded_concepts.values()),
                edges=grounded_edges,
            )

            soap_note.grounded_causal_graph = grounded_graph

            # Generate text representation of grounded graph for display
            graph_lines = ["```grounded_graph"]
            for edge in grounded_edges:
                src = edge.source
                tgt = edge.target
                # Format: SNOMED_Term[type] ARROW SNOMED_Term[type] (SCTID:xxx -> SCTID:yyy)
                graph_lines.append(
                    f"{src.term.replace(' ', '_')}[{src.node_type}] {edge.relation} "
                    f"{tgt.term.replace(' ', '_')}[{tgt.node_type}]"
                )
            graph_lines.append("```")
            graph_lines.append("")
            graph_lines.append(f"*Grounded to SNOMED CT: {len(grounded_graph.nodes)} concepts, {len(grounded_graph.edges)} relationships*")

            grounded_graph_text = "\n".join(graph_lines)

            # Update the SOAP note's raw text to include grounded graph
            if hasattr(soap_note, 'raw_response') and soap_note.raw_response:
                soap_note.raw_response = soap_note.raw_response.replace(
                    "(Grounded causal graph will be generated using SNOMED CT concepts)",
                    grounded_graph_text
                )

            # Also update the causal_graph field
            soap_note.causal_graph = grounded_graph_text

            if self.verbose:
                print(f"  Grounded graph: {len(grounded_graph.nodes)} nodes, {len(grounded_graph.edges)} edges")

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

        for i, patient in enumerate(patients):
            patient_text, biomcp_annotations = FHIRFormatter.format_patient_for_llm(patient, use_biomcp=use_biomcp)
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

            for batch_start in range(0, len(direct_patients), batch_size):
                batch = direct_patients[batch_start:batch_start + batch_size]

                # Prepare batch inputs
                batch_prompts = []
                batch_images = []

                for idx, patient, patient_text, images, biomcp_ann, n_chunks in batch:
                    if images:
                        prompt = self.SOAP_USER_PROMPT_WITH_IMAGES.format(
                            patient_data=patient_text,
                            num_images=len(images)
                        )
                    else:
                        prompt = self.SOAP_USER_PROMPT_NO_IMAGES.format(patient_data=patient_text)

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

    def _generate_direct(
        self,
        patient: Any,
        patient_text: str,
        images: list,
        biomcp_annotations: Optional[dict[str, Any]] = None,
        biomcp_seconds: float = 0.0,
    ) -> SOAPNote:
        """Generate SOAP note directly (for shorter histories)."""
        if biomcp_annotations is None:
            biomcp_annotations = {}
        # Token counting
        input_tokens = self._count_tokens(patient_text)

        # Determine steps: 2 for normal, 3 for separate causal graph
        total_steps = 3 if self.separate_causal_graph else 2

        # Progress bar for direct generation
        if _tqdm_available and self.verbose:
            pbar = _tqdm(total=total_steps, desc="Generating SOAP note")
            pbar.set_postfix_str("Analyzing patient data" + (f" + {len(images)} images" if images else ""))
        else:
            pbar = None

        if self.verbose and not _tqdm_available:
            print(f"\n  Token Statistics:")
            print(f"    Input (patient data): {input_tokens:,} tokens")

        # Select prompt based on whether images are present and graph generation mode
        # Skip inline causal graph if we'll generate grounded graph, or if separate_causal_graph is enabled
        skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)

        if skip_inline_graph:
            # Use prompts without causal graph (grounded graph will be generated later)
            if images:
                prompt = self.SOAP_NO_GRAPH_WITH_IMAGES_PROMPT.format(
                    patient_data=patient_text,
                    num_images=len(images)
                )
            else:
                prompt = self.SOAP_NO_GRAPH_PROMPT.format(patient_data=patient_text)
        else:
            # Use normal prompts with causal graph included
            if images:
                prompt = self.SOAP_USER_PROMPT_WITH_IMAGES.format(
                    patient_data=patient_text,
                    num_images=len(images)
                )
            else:
                prompt = self.SOAP_USER_PROMPT_NO_IMAGES.format(patient_data=patient_text)

        # Add admission exclusion instruction if needed (for inline causal graph mode)
        if not skip_inline_graph:
            prompt += self._get_admission_exclusion_instruction()

        soap_start = time.perf_counter()
        response = self._generate_text(
            prompt,
            images=images if images else None,
            max_new_tokens=self.max_new_tokens_final,
        )
        soap_seconds = time.perf_counter() - soap_start

        # Count output tokens
        output_tokens = self._count_tokens(response)
        if self.verbose and not _tqdm_available:
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
            graph_start = time.perf_counter()
            causal_graph_response = self._generate_text(
                graph_prompt,
                max_new_tokens=2000,
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
            # Add placeholder - will be replaced with grounded graph
            response = response.rstrip() + "\n\n# Causal Graph\n(Grounded causal graph will be generated using SNOMED CT concepts)"
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
            pbar.close()

        # Print token stats for tqdm mode too
        if self.verbose and _tqdm_available:
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
            patient_text, fallback_biomcp = FHIRFormatter.format_patient_for_llm(patient, use_biomcp=self.use_biomcp)
            # Merge any new biomcp annotations
            if fallback_biomcp:
                biomcp_annotations.update(fallback_biomcp)
            return self._generate_direct(patient, patient_text, images, biomcp_annotations)

        # Token tracking
        chunk_stats = []  # List of (period, input_tokens, output_tokens, summary_seconds)
        total_input_tokens = 0
        total_summary_tokens = 0

        # Create progress bar for all steps
        # Steps: n chunks + 1 aggregation + (1 causal graph if separate) + 1 parsing
        total_steps = len(chunks) + 2 + (1 if self.separate_causal_graph else 0)

        if _tqdm_available and self.verbose:
            pbar = _tqdm(total=total_steps, desc="Generating SOAP note")
        else:
            pbar = None

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

            prompt = self.CHUNK_SUMMARY_PROMPT.format(chunk_data=chunk_text)
            chunk_start_time = time.perf_counter()
            summary = self._generate_text(prompt, max_new_tokens=self.max_new_tokens_chunk)
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

        # Format genetics section for inclusion in hierarchical prompts
        # (In direct mode, genetics is already in patient_text; in hierarchical mode, chunks don't include it)
        genetics_section = ""
        if hasattr(patient, 'genomics') and patient.genomics is not None:
            try:
                genetics_text, _ = FHIRFormatter._format_genomics(
                    patient.genomics,
                    use_biomcp=self.use_biomcp,
                )
                if genetics_text and genetics_text.strip():
                    genetics_section = f"\n{genetics_text}\n"
            except Exception:
                pass  # Skip if genomics formatting fails

        # Select prompt based on whether images are present and graph generation mode
        # Skip inline causal graph if we'll generate grounded graph, or if separate_causal_graph is enabled
        skip_inline_graph = self.separate_causal_graph or (self.ground_snomed and self.grounded_graph_mode)

        if skip_inline_graph:
            # Use prompts without causal graph (grounded graph will be generated later)
            if images:
                aggregate_prompt = self.AGGREGATE_NO_GRAPH_WITH_IMAGES_PROMPT.format(
                    patient_name=patient_name,
                    summaries=summaries_text,
                    genetics_section=genetics_section,
                    num_images=len(images),
                )
            else:
                aggregate_prompt = self.AGGREGATE_NO_GRAPH_PROMPT.format(
                    patient_name=patient_name,
                    summaries=summaries_text,
                    genetics_section=genetics_section,
                )
        else:
            # Use normal prompts with causal graph
            if images:
                aggregate_prompt = self.AGGREGATE_PROMPT_WITH_IMAGES.format(
                    patient_name=patient_name,
                    summaries=summaries_text,
                    genetics_section=genetics_section,
                    num_images=len(images),
                )
            else:
                aggregate_prompt = self.AGGREGATE_PROMPT_NO_IMAGES.format(
                    patient_name=patient_name,
                    summaries=summaries_text,
                    genetics_section=genetics_section,
                )

        # Add admission exclusion instruction if needed (for inline causal graph mode)
        if not skip_inline_graph:
            aggregate_prompt += self._get_admission_exclusion_instruction()

        aggregate_start = time.perf_counter()
        response = self._generate_text(
            aggregate_prompt,
            images=images if images else None,
            max_new_tokens=self.max_new_tokens_final,
        )
        aggregate_seconds = time.perf_counter() - aggregate_start

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
            graph_start = time.perf_counter()
            causal_graph_response = self._generate_text(
                graph_prompt,
                max_new_tokens=2000,
            )
            causal_graph_seconds = time.perf_counter() - graph_start
            causal_graph_tokens = self._count_tokens(causal_graph_response)

            # Append causal graph as its own section at the end
            response = response.rstrip() + "\n\n# Causal Graph\n" + causal_graph_response.strip()
            final_output_tokens += causal_graph_tokens

            if pbar:
                pbar.update(1)
        elif self.separate_causal_graph and skip_freeform_graph:
            # Add placeholder - will be replaced with grounded graph
            response = response.rstrip() + "\n\n# Causal Graph\n(Grounded causal graph will be generated using SNOMED CT concepts)"
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
            pbar.close()

        # Print comprehensive token statistics
        if self.verbose:
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

        for line in response.split("\n"):
            stripped = line.strip()
            line_upper = stripped.upper()

            # Skip empty lines and horizontal rules
            if not stripped or stripped in ["---", "***", "___"]:
                if current_section:
                    current_content.append(line)
                continue

            # Remove markdown formatting for header detection
            clean_upper = line_upper.lstrip("#").lstrip("*").lstrip("-").strip()
            clean_upper = clean_upper.rstrip(":").rstrip("*").strip()

            # Check for section headers
            found_section = None
            for alias, section in alias_to_section.items():
                if clean_upper == alias or clean_upper.startswith(alias + " "):
                    found_section = section
                    break

            if found_section:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = found_section
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

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
    verbose: bool = True,
    approximate_tokens: bool = False,
    use_biomcp: bool = False,
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
        separate_causal_graph=separate_causal_graph,
        include_admissions=include_admissions,
        chunk_period_years=chunk_period_years,
    )
    return generator.generate(patient)
