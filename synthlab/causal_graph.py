#!/usr/bin/env python3
"""
Causal Graph extraction and analysis for clinical notes.

This module provides classes and functions for representing, parsing,
and analyzing causal relationships extracted from clinical text.

Classes:
    CausalNode: A node in a causal graph representing a clinical entity
    CausalEdge: A directed edge representing a causal relationship
    CausalGraph: A directed graph of causal relationships

Functions:
    parse_causal_graph: Parse causal relationships from text into a CausalGraph
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synthlab.snomed import SNOMEDLinker, GroundedCausalGraph


# =============================================================================
# Causal Graph Types and Parsing
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
    "?=>": {"name": "possibly_causes", "direction": "causal", "strength": "uncertain", "weight": 0.5},
    "<=>": {"name": "bidirectional", "direction": "causal", "strength": "direct", "weight": 1.0},
    "~>": {"name": "associated", "direction": "causal", "strength": "uncertain", "weight": 0.2},
    # Allow text-based indicators too (normalized during parsing)
    "increases": {"name": "increases", "direction": "risk", "strength": "moderate", "weight": 0.5},
    "decreases": {"name": "decreases", "direction": "protective", "strength": "moderate", "weight": -0.5},
    "causes": {"name": "causes", "direction": "causal", "strength": "direct", "weight": 1.0},
}


def _extract_node_type(node_str: str) -> tuple[str, str]:
    """
    Extract node name and type from bracket notation.

    Examples:
        "Diabetes[condition]" -> ("Diabetes", "condition")
        "Metformin[medication]" -> ("Metformin", "medication")
        "Obesity" -> ("Obesity", "unknown")

    Args:
        node_str: Node string, potentially with [type] suffix

    Returns:
        (name, type) tuple
    """
    match = re.match(r'(.+?)\[(\w+)\]', node_str.strip())
    if match:
        return match.group(1).strip(), match.group(2).lower()
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
        op = " && " if self.interaction == "and" else " || " if self.interaction == "or" else ", "
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
        font_color: str = "#000000",
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
            font_color: Font color for node labels.
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
                font_color=font_color,
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
        for sources, tgt, direction, interaction_type in interaction_edges:
            if tgt not in pos:
                continue
            valid_sources = [s for s in sources if s in pos]
            if len(valid_sources) < 2:
                # Fall back to simple edges
                for src in valid_sources:
                    if src in pos:
                        x1, y1 = pos[src]
                        x2, y2 = pos[tgt]
                        edge_color = edge_colors.get(direction, '#333333')
                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=1.5,
                                           shrinkA=shrink_val, shrinkB=shrink_val))
                continue

            # Calculate junction point (weighted midpoint between sources and target)
            src_positions = [pos[s] for s in valid_sources]
            src_center = (
                sum(p[0] for p in src_positions) / len(src_positions),
                sum(p[1] for p in src_positions) / len(src_positions)
            )
            tgt_pos = pos[tgt]

            # Junction is 70% of the way from sources to target
            junction = (
                src_center[0] * 0.3 + tgt_pos[0] * 0.7,
                src_center[1] * 0.3 + tgt_pos[1] * 0.7
            )

            edge_color = edge_colors.get(direction, '#333333')
            linestyle = ':' if interaction_type == "and" else '--'

            # Draw lines from each source to junction
            for src in valid_sources:
                x1, y1 = pos[src]
                ax.plot([x1, junction[0]], [y1, junction[1]],
                       color=edge_color, lw=1.5, linestyle=linestyle,
                       solid_capstyle='round')

            # Draw line from junction to target with arrow
            ax.annotate('', xy=tgt_pos, xytext=junction,
                arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=1.5,
                               shrinkA=0, shrinkB=shrink_val))

            # Small marker at junction point
            marker = 'o' if interaction_type == "and" else 's'
            ax.plot(junction[0], junction[1], marker=marker, color=edge_color,
                   markersize=6, markeredgecolor='white', markeredgewidth=1)

        # Draw nodes
        for node_name in G.nodes():
            if node_name not in pos:
                continue
            x, y = pos[node_name]
            ntype = node_types.get(node_name, "unknown")
            color = type_colors.get(ntype, "#AAAAAA")

            if node_shape == "rectangle":
                rect = mpatches.FancyBboxPatch(
                    (x - node_width/2, y - node_height/2),
                    node_width, node_height,
                    boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                    facecolor=color, edgecolor='white', linewidth=2
                )
                ax.add_patch(rect)
            else:
                circle = plt.Circle((x, y), node_radius, color=color, ec='white', lw=2)
                ax.add_patch(circle)

            # Add label
            if label_inside:
                # Wrap long names
                display_name = node_name
                if len(display_name) > 15:
                    words = display_name.split()
                    if len(words) > 1:
                        mid = len(words) // 2
                        display_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                ax.text(x, y, display_name, ha='center', va='center',
                       fontsize=font_size, fontweight='bold', color=font_color,
                       wrap=True)
            else:
                ax.text(x + node_width/2 + 0.1, y, node_name, ha='left', va='center',
                       fontsize=font_size, color=font_color)

        # Add legend
        if show_legend:
            legend_elements = []

            # Node types that appear in the graph
            for ntype, color in type_colors.items():
                if any(node_types.get(n) == ntype for n in G.nodes()):
                    legend_elements.append(
                        mpatches.Patch(facecolor=color, edgecolor='white', label=ntype.capitalize())
                    )

            # Edge directions that appear
            has_risk = any(d == 'risk' for _, _, d in simple_edges)
            has_protective = any(d == 'protective' for _, _, d in simple_edges)
            has_causal = any(d == 'causal' for _, _, d in simple_edges)

            if has_risk:
                legend_elements.append(
                    plt.Line2D([0], [0], color=edge_colors['risk'], lw=2, label='Increases risk')
                )
            if has_protective:
                legend_elements.append(
                    plt.Line2D([0], [0], color=edge_colors['protective'], lw=2, label='Protective')
                )
            if has_causal:
                legend_elements.append(
                    plt.Line2D([0], [0], color=edge_colors['causal'], lw=2, label='Causes')
                )

            # Interaction types
            has_and = any(i == "and" for _, _, _, i in interaction_edges)
            has_or = any(i == "or" for _, _, _, i in interaction_edges)

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
        font_color: str = "#000000",
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
            font_color=font_color,
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
                font={"color": font_color},
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
    Parse source string for interaction operators (&& for AND, || for OR).

    Args:
        source_str: Source part of a causal expression

    Returns:
        (list of sources, interaction type or None)

    Examples:
        "DrugA && DrugB" -> (["DrugA", "DrugB"], "and")
        "BRCA1 || BRCA2" -> (["BRCA1", "BRCA2"], "or")
        "Obesity" -> (["Obesity"], None)
    """
    # Check for AND interaction (&&)
    if ' && ' in source_str:
        parts = [p.strip() for p in source_str.split(' && ') if p.strip()]
        if len(parts) > 1:
            return parts, "and"

    # Check for OR interaction (||)
    if ' || ' in source_str:
        parts = [p.strip() for p in source_str.split(' || ') if p.strip()]
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
    - A && B => C  (A AND B together cause C)
    - A || B => C  (A OR B causes C, either sufficient)

    Args:
        text: Text containing causal relationships (e.g., from SOAP note assessment)

    Returns:
        CausalGraph with parsed edges

    Example:
        >>> text = '''
        ... Obesity[lifestyle] ++> Diabetes[condition]
        ... DrugA[medication] && DrugB[medication] => Liver_failure[condition]
        ... BRCA1[finding] || BRCA2[finding] ++> Breast_cancer[condition]
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
# Exports
# =============================================================================

__all__ = [
    "CAUSAL_EDGE_TYPES",
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "parse_causal_graph",
]
