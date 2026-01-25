"""
SNOMED CT Entity Linking using SapBERT + FAISS.

This module provides tools for grounding medical entities to SNOMED CT concepts
using semantic embeddings from SapBERT and fast similarity search with FAISS.

Example:
    >>> from synthlab.snomed import SNOMEDLinker
    >>> linker = SNOMEDLinker()
    >>> results = linker.link("diabetic nephropathy")
    >>> print(results[0])
    SNOMEDMatch(concept_id='709044004', term='Diabetic nephropathy', score=0.95)

The pipeline:
    1. Load pre-computed SNOMED embeddings (or build from scratch)
    2. Build FAISS index for fast similarity search
    3. Encode input text with SapBERT
    4. Find nearest SNOMED concepts via cosine similarity
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import json
import os

# Lazy imports for optional dependencies
_faiss_available = False
_torch_available = False
_transformers_available = False
_numpy_available = False
_tqdm_available = False

try:
    import faiss
    _faiss_available = True
except ImportError:
    pass

try:
    import torch
    _torch_available = True
except ImportError:
    pass

try:
    from transformers import AutoTokenizer, AutoModel
    _transformers_available = True
except ImportError:
    pass

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    pass

try:
    from tqdm import tqdm as _tqdm
    _tqdm_available = True
except ImportError:
    _tqdm = None


# =============================================================================
# Constants
# =============================================================================

# Default SapBERT model for biomedical entity embeddings
SAPBERT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Alternative embedding models for entity linking
EMBEDDING_MODELS = {
    # SapBERT variants (biomedical-specific, trained on UMLS)
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "sapbert-mean": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    "sapbert-xlm": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    # Qwen embedding models (efficient, high quality)
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    # General-purpose models
    "gte-large": "thenlper/gte-large",
    # Sentence transformers
    "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
}

# Backwards compatibility
SAPBERT_MODELS = EMBEDDING_MODELS

# SNOMED CT semantic types to include (subset for clinical relevance)
SNOMED_SEMANTIC_TYPES = {
    "finding": "Clinical finding",
    "disorder": "Disease or syndrome",
    "procedure": "Procedure",
    "substance": "Substance",
    "body_structure": "Body structure",
    "observable": "Observable entity",
    "situation": "Situation with explicit context",
    "event": "Event",
}

# Cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "synthlab" / "snomed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SNOMEDConcept:
    """A SNOMED CT concept with its metadata."""
    concept_id: str
    term: str
    semantic_type: str = ""
    synonyms: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"SCTID:{self.concept_id} ({self.term})"


@dataclass
class SNOMEDMatch:
    """A match result from entity linking."""
    concept_id: str
    term: str
    score: float
    semantic_type: str = ""

    def __str__(self) -> str:
        return f"SCTID:{self.concept_id} | {self.term} (score: {self.score:.3f})"

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "term": self.term,
            "score": self.score,
            "semantic_type": self.semantic_type,
        }


@dataclass
class LinkedEntity:
    """An entity mention linked to SNOMED CT."""
    mention: str
    start: int
    end: int
    matches: list[SNOMEDMatch]

    @property
    def top_match(self) -> Optional[SNOMEDMatch]:
        """Get the highest-scoring match."""
        return self.matches[0] if self.matches else None

    @property
    def concept_id(self) -> Optional[str]:
        """Get the concept ID of the top match."""
        return self.top_match.concept_id if self.top_match else None

    def __str__(self) -> str:
        if self.top_match:
            return f'"{self.mention}" -> {self.top_match}'
        return f'"{self.mention}" -> No match'


# =============================================================================
# SNOMED Linker
# =============================================================================

class SNOMEDLinker:
    """
    SNOMED CT entity linker using semantic embeddings and FAISS.

    This class provides semantic entity linking to SNOMED CT concepts.
    It encodes text into embeddings and uses FAISS for fast nearest
    neighbor search against pre-computed SNOMED embeddings.

    Supports multiple embedding models:
    - SapBERT (default): Biomedical-specific, trained on UMLS
    - Qwen3-Embedding: Efficient, high-quality general embeddings
    - GTE, BGE, etc.: Other general-purpose models

    Example:
        >>> linker = SNOMEDLinker()
        >>>
        >>> # Link a single mention
        >>> matches = linker.link("heart attack")
        >>> print(matches[0])
        SCTID:22298006 | Myocardial infarction (score: 0.92)
        >>>
        >>> # Link multiple mentions
        >>> results = linker.link_batch(["diabetes", "hypertension", "stroke"])
        >>> for mention, matches in results:
        ...     print(f"{mention}: {matches[0].term}")
        >>>
        >>> # Use alternative embedding model
        >>> linker = SNOMEDLinker(model_id="Qwen/Qwen3-Embedding-0.6B")

    Args:
        model_id: HuggingFace model ID for embeddings. Options:
            - "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" (default, biomedical)
            - "Qwen/Qwen3-Embedding-0.6B" (efficient, high-quality)
            - See EMBEDDING_MODELS dict for more options
        index_path: Path to pre-built FAISS index. If None, will look in cache or build.
        concepts_path: Path to SNOMED concepts JSON. If None, will look in cache.
        device: Device to use ('auto', 'cuda', 'cpu')
        cache_dir: Directory for caching index and embeddings
        verbose: Print progress messages
    """

    def __init__(
        self,
        model_id: str = SAPBERT_MODEL_ID,
        index_path: Optional[Union[str, Path]] = None,
        concepts_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        cache_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        self._check_dependencies()

        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.verbose = verbose

        # Lazy-loaded components
        self._model = None
        self._tokenizer = None
        self._sentence_transformer = None  # For sentence-transformers compatible models
        self._index = None
        self._concepts: list[SNOMEDConcept] = []
        self._concept_id_to_idx: dict[str, int] = {}

        # Mention embedding cache for efficiency across documents
        self._mention_cache: dict[str, "np.ndarray"] = {}
        self._cache_enabled = True

        # Track unmatched terms for vocabulary expansion
        self._unmatched_terms: dict[str, int] = {}  # term -> count
        self._low_confidence_terms: dict[str, tuple[float, str]] = {}  # term -> (best_score, best_match)

        # Paths - use model-specific cache to avoid dimension mismatches
        self.index_path = Path(index_path) if index_path else None
        self.concepts_path = Path(concepts_path) if concepts_path else None

        # Try to load from cache if paths not provided
        # Use model-specific filenames to avoid conflicts between different embedding models
        if self.index_path is None or self.concepts_path is None:
            model_suffix = self._get_model_cache_suffix(model_id)
            if self.index_path is None:
                self.index_path = self.cache_dir / f"snomed_{model_suffix}.index"
            if self.concepts_path is None:
                self.concepts_path = self.cache_dir / f"snomed_{model_suffix}_concepts.json"

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        missing = []
        if not _faiss_available:
            missing.append("faiss-cpu (or faiss-gpu)")
        if not _torch_available:
            missing.append("torch")
        if not _transformers_available:
            missing.append("transformers")
        if not _numpy_available:
            missing.append("numpy")

        if missing:
            raise ImportError(
                f"Missing required dependencies for SNOMEDLinker: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if _torch_available and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    @staticmethod
    def _get_model_cache_suffix(model_id: str) -> str:
        """Generate a safe cache filename suffix from model ID.

        Maps common model IDs to short names, or creates a hash for others.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Safe string for use in filenames
        """
        # Map known models to short names for readability
        model_short_names = {
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "sapbert",
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "sapbert_mean",
            "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "sapbert_xlm",
            "Qwen/Qwen3-Embedding-0.6B": "qwen3_0.6b",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct": "gte_qwen2",
            "thenlper/gte-large": "gte_large",
            "sentence-transformers/all-mpnet-base-v2": "mpnet",
            "BAAI/bge-large-en-v1.5": "bge_large",
        }

        if model_id in model_short_names:
            return model_short_names[model_id]

        # For unknown models, create a safe filename from the model ID
        # Replace special chars and truncate
        import hashlib
        safe_name = model_id.replace("/", "_").replace("-", "_").lower()
        # If too long, use hash
        if len(safe_name) > 50:
            hash_suffix = hashlib.md5(model_id.encode()).hexdigest()[:8]
            safe_name = safe_name[:40] + "_" + hash_suffix
        return safe_name

    def _load_model(self):
        """Load the embedding model and tokenizer."""
        if self._model is not None or self._sentence_transformer is not None:
            return

        if self.verbose:
            print(f"Loading embedding model: {self.model_id}")

        # Try sentence-transformers first for compatible models
        if self._is_sentence_transformer() or self._is_qwen_model():
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=True,  # Required for Qwen models
                )
                if self.verbose:
                    print(f"  Loaded via sentence-transformers")
                    print(f"  Device: {self.device}")
                return
            except ImportError:
                if self.verbose:
                    print("  sentence-transformers not available, falling back to transformers")
            except Exception as e:
                if self.verbose:
                    print(f"  sentence-transformers failed ({e}), falling back to transformers")

        # Fallback to direct transformers loading
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Qwen models need left padding for proper last-token pooling
        if self._is_qwen_model():
            self._tokenizer.padding_side = 'left'

        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = self._model.to(self.device)
        self._model.eval()

        if self.verbose:
            print(f"  Device: {self.device}")

    def _load_index(self):
        """Load the FAISS index and concepts."""
        if self._index is not None:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"SNOMED index not found at {self.index_path}\n"
                f"Build it first with: linker.build_index(snomed_concepts)"
            )

        if not self.concepts_path.exists():
            raise FileNotFoundError(
                f"SNOMED concepts not found at {self.concepts_path}\n"
                f"Build index first with: linker.build_index(snomed_concepts)"
            )

        if self.verbose:
            print(f"Loading SNOMED index from {self.index_path}")

        self._index = faiss.read_index(str(self.index_path))

        if self.verbose:
            print(f"  Loaded {self._index.ntotal:,} concepts")

        # Load concepts
        with open(self.concepts_path, 'r') as f:
            concepts_data = json.load(f)

        self._concepts = [
            SNOMEDConcept(
                concept_id=c["concept_id"],
                term=c["term"],
                semantic_type=c.get("semantic_type", ""),
                synonyms=c.get("synonyms", []),
            )
            for c in concepts_data
        ]

        self._concept_id_to_idx = {c.concept_id: i for i, c in enumerate(self._concepts)}

    def _is_qwen_model(self) -> bool:
        """Check if using a Qwen embedding model."""
        return "qwen" in self.model_id.lower()

    def _is_sentence_transformer(self) -> bool:
        """Check if model should use sentence-transformers."""
        return "sentence-transformers" in self.model_id.lower()

    def encode(self, texts: Union[str, list[str]], batch_size: int = 32) -> "np.ndarray":
        """
        Encode text(s) into embeddings.

        Automatically handles different model types:
        - SapBERT: Uses CLS token
        - Qwen: Uses last token pooling with instruction prefix
        - Sentence Transformers: Uses mean pooling

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (n_texts, embedding_dim) with normalized embeddings
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # Try sentence-transformers first (simplest API)
        if self._sentence_transformer is not None:
            embeddings = self._sentence_transformer.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # For Qwen models, add instruction prefix for queries
            if self._is_qwen_model():
                # Use medical entity linking instruction
                instruction = "Given a medical term, retrieve the matching SNOMED CT concept"
                batch = [f"Instruct: {instruction}\nQuery: {text}" for text in batch]

            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512 if self._is_qwen_model() else 64,
            ).to(self.device)

            with torch.no_grad():
                output = self._model(**tokens)

                if self._is_qwen_model():
                    # Qwen uses last token pooling
                    attention_mask = tokens['attention_mask']
                    last_hidden = output.last_hidden_state
                    # Get the last non-padding token for each sequence
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size_actual = last_hidden.shape[0]
                    embeddings = last_hidden[
                        torch.arange(batch_size_actual, device=last_hidden.device),
                        sequence_lengths
                    ].cpu().numpy()
                else:
                    # SapBERT and similar: use CLS token
                    embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero - replace zero norms with 1.0
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Check for NaN values (can break FAISS)
        if np.isnan(embeddings).any():
            raise ValueError("Embedding contains NaN values - check input texts or model")

        return embeddings

    def link(
        self,
        mention: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> list[SNOMEDMatch]:
        """
        Link a mention to SNOMED CT concepts.

        Args:
            mention: Text mention to link
            k: Number of top matches to return
            threshold: Minimum similarity score threshold

        Returns:
            List of SNOMEDMatch objects sorted by score (descending)
        """
        self._load_index()

        # Encode mention
        embedding = self.encode(mention)

        # Search index
        distances, indices = self._index.search(embedding.astype('float32'), k)

        # Build results
        matches = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._concepts):
                continue
            if score < threshold:
                continue

            concept = self._concepts[idx]
            matches.append(SNOMEDMatch(
                concept_id=concept.concept_id,
                term=concept.term,
                score=float(score),
                semantic_type=concept.semantic_type,
            ))

        return matches

    def link_batch(
        self,
        mentions: list[str],
        k: int = 5,
        threshold: float = 0.0,
        batch_size: int = 32,
    ) -> list[tuple[str, list[SNOMEDMatch]]]:
        """
        Link multiple mentions to SNOMED CT concepts.

        Args:
            mentions: List of text mentions to link
            k: Number of top matches per mention
            threshold: Minimum similarity score threshold
            batch_size: Batch size for encoding

        Returns:
            List of (mention, matches) tuples
        """
        self._load_index()

        # Encode all mentions
        embeddings = self.encode(mentions, batch_size=batch_size)

        # Validate embedding dimensions match index
        index_dim = self._index.d
        embed_dim = embeddings.shape[1]
        if embed_dim != index_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model produces {embed_dim}-dim vectors "
                f"but SNOMED index expects {index_dim}-dim. "
                f"This can happen if the index was built with a different embedding model. "
                f"Rebuild the index with: linker.build_index(concepts)"
            )

        # Search index
        distances, indices = self._index.search(embeddings.astype('float32'), k)

        # Build results
        results = []
        for mention, idxs, scores in zip(mentions, indices, distances):
            matches = []
            for idx, score in zip(idxs, scores):
                if idx < 0 or idx >= len(self._concepts):
                    continue
                if score < threshold:
                    continue

                concept = self._concepts[idx]
                matches.append(SNOMEDMatch(
                    concept_id=concept.concept_id,
                    term=concept.term,
                    score=float(score),
                    semantic_type=concept.semantic_type,
                ))
            results.append((mention, matches))

        return results

    def build_index(
        self,
        concepts: list[SNOMEDConcept],
        batch_size: int = 128,
        save: bool = True,
    ) -> None:
        """
        Build FAISS index from SNOMED concepts.

        Args:
            concepts: List of SNOMEDConcept objects
            batch_size: Batch size for encoding
            save: Whether to save the index to disk
        """
        self._load_model()

        if self.verbose:
            print(f"Building SNOMED index from {len(concepts):,} concepts")

        # Extract terms
        terms = [c.term for c in concepts]

        # Encode all terms
        if self.verbose:
            print(f"  Encoding terms with {self.model_id}...")

        all_embeddings = []

        iterator = range(0, len(terms), batch_size)
        if _tqdm_available and self.verbose:
            iterator = _tqdm(iterator, desc="Encoding", unit="batch")

        for i in iterator:
            batch = terms[i:i + batch_size]
            embeddings = self.encode(batch, batch_size=len(batch))
            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings).astype('float32')

        if self.verbose:
            print(f"  Embeddings shape: {embeddings.shape}")

        # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
        if self.verbose:
            print("  Building FAISS index...")

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        self._concepts = concepts
        self._concept_id_to_idx = {c.concept_id: i for i, c in enumerate(concepts)}

        if self.verbose:
            print(f"  Index contains {self._index.ntotal:,} vectors")

        # Save
        if save:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"  Saving index to {self.index_path}")

            faiss.write_index(self._index, str(self.index_path))

            # Save concepts
            concepts_data = [
                {
                    "concept_id": c.concept_id,
                    "term": c.term,
                    "semantic_type": c.semantic_type,
                    "synonyms": c.synonyms,
                }
                for c in concepts
            ]

            with open(self.concepts_path, 'w') as f:
                json.dump(concepts_data, f)

            if self.verbose:
                print(f"  Saved concepts to {self.concepts_path}")

    def get_concept(self, concept_id: str) -> Optional[SNOMEDConcept]:
        """Get a concept by its SNOMED CT ID."""
        self._load_index()
        idx = self._concept_id_to_idx.get(concept_id)
        if idx is not None:
            return self._concepts[idx]
        return None

    @property
    def num_concepts(self) -> int:
        """Get the number of concepts in the index."""
        self._load_index()
        return len(self._concepts)

    def is_ready(self) -> bool:
        """Check if the linker is ready (index exists)."""
        return self.index_path.exists() and self.concepts_path.exists()

    # =========================================================================
    # Incremental Indexing & Caching
    # =========================================================================

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable mention embedding cache.

        Args:
            enabled: Whether to cache mention embeddings
        """
        self._cache_enabled = enabled

    def clear_cache(self) -> None:
        """Clear the mention embedding cache."""
        self._mention_cache.clear()
        if self.verbose:
            print("  Mention cache cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats including size, hit rate info
        """
        return {
            "enabled": self._cache_enabled,
            "cached_mentions": len(self._mention_cache),
            "unmatched_terms": len(self._unmatched_terms),
            "low_confidence_terms": len(self._low_confidence_terms),
        }

    def get_unmatched_terms(self, min_count: int = 1) -> list[tuple[str, int]]:
        """Get terms that had no matches above threshold.

        Args:
            min_count: Minimum occurrence count to include

        Returns:
            List of (term, count) tuples sorted by count descending
        """
        return sorted(
            [(t, c) for t, c in self._unmatched_terms.items() if c >= min_count],
            key=lambda x: -x[1]
        )

    def get_low_confidence_terms(
        self,
        max_score: float = 0.5
    ) -> list[tuple[str, float, str]]:
        """Get terms with low confidence matches.

        Args:
            max_score: Maximum score to consider "low confidence"

        Returns:
            List of (term, best_score, best_match) tuples sorted by score ascending
        """
        return sorted(
            [(t, score, match) for t, (score, match) in self._low_confidence_terms.items() if score <= max_score],
            key=lambda x: x[1]
        )

    def get_vocabulary_gaps(self, min_count: int = 1, max_score: float = 0.5) -> dict:
        """Get a summary of vocabulary gaps for index expansion.

        This helps identify which SNOMED concepts should be added to improve
        coverage for your document collection.

        Args:
            min_count: Minimum occurrence count for unmatched terms
            max_score: Maximum score for low confidence matches

        Returns:
            Dictionary with unmatched and low_confidence term lists
        """
        return {
            "unmatched": self.get_unmatched_terms(min_count=min_count),
            "low_confidence": self.get_low_confidence_terms(max_score=max_score),
            "total_cached_mentions": len(self._mention_cache),
            "recommendation": (
                f"Consider adding SNOMED concepts for {len(self._unmatched_terms)} unmatched terms "
                f"and {len([s for s, _ in self._low_confidence_terms.values() if s <= max_score])} "
                f"low-confidence matches."
            )
        }

    def clear_tracking(self) -> None:
        """Clear unmatched and low confidence term tracking."""
        self._unmatched_terms.clear()
        self._low_confidence_terms.clear()

    def add_concepts(
        self,
        concepts: list[SNOMEDConcept],
        save: bool = True,
        batch_size: int = 128,
    ) -> int:
        """Incrementally add new concepts to the index.

        This allows expanding the index with new SNOMED concepts without
        rebuilding from scratch. Useful when processing documents with
        vocabularies not covered by the initial index.

        Args:
            concepts: List of SNOMEDConcept objects to add
            save: Whether to save the updated index to disk
            batch_size: Batch size for encoding

        Returns:
            Number of new concepts added (excludes duplicates)

        Example:
            >>> # Add concepts for terms that weren't matching well
            >>> new_concepts = [
            ...     SNOMEDConcept("12345", "Specific rare condition"),
            ...     SNOMEDConcept("67890", "Another missing term"),
            ... ]
            >>> added = linker.add_concepts(new_concepts)
            >>> print(f"Added {added} new concepts")
        """
        self._load_index()
        self._load_model()

        # Filter out concepts already in index
        existing_ids = set(self._concept_id_to_idx.keys())
        new_concepts = [c for c in concepts if c.concept_id not in existing_ids]

        if not new_concepts:
            if self.verbose:
                print("  No new concepts to add (all already in index)")
            return 0

        if self.verbose:
            print(f"  Adding {len(new_concepts)} new concepts to index...")

        # Encode new concept terms
        terms = [c.term for c in new_concepts]
        embeddings = self.encode(terms, batch_size=batch_size)

        # Add to FAISS index
        self._index.add(embeddings.astype('float32'))

        # Update concept list and mapping
        start_idx = len(self._concepts)
        for i, concept in enumerate(new_concepts):
            self._concepts.append(concept)
            self._concept_id_to_idx[concept.concept_id] = start_idx + i

        if self.verbose:
            print(f"  Index now contains {self._index.ntotal:,} concepts")

        # Save if requested
        if save:
            self._save_index()

        return len(new_concepts)

    def _save_index(self) -> None:
        """Save the current index and concepts to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"  Saving index to {self.index_path}")

        faiss.write_index(self._index, str(self.index_path))

        # Save concepts
        concepts_data = [
            {
                "concept_id": c.concept_id,
                "term": c.term,
                "semantic_type": c.semantic_type,
                "synonyms": c.synonyms,
            }
            for c in self._concepts
        ]

        with open(self.concepts_path, 'w') as f:
            json.dump(concepts_data, f)

        if self.verbose:
            print(f"  Saved {len(self._concepts)} concepts")

    def link_with_cache(
        self,
        mention: str,
        k: int = 5,
        threshold: float = 0.0,
        track_gaps: bool = True,
    ) -> list[SNOMEDMatch]:
        """Link a mention using the embedding cache.

        More efficient than `link()` when processing many documents with
        overlapping vocabularies.

        Args:
            mention: Text mention to link
            k: Number of top matches to return
            threshold: Minimum similarity score threshold
            track_gaps: Whether to track unmatched/low-confidence terms

        Returns:
            List of SNOMEDMatch objects sorted by score (descending)
        """
        self._load_index()

        # Check cache first
        if self._cache_enabled and mention in self._mention_cache:
            embedding = self._mention_cache[mention]
        else:
            # Encode and cache
            embedding = self.encode(mention)
            if self._cache_enabled:
                self._mention_cache[mention] = embedding

        # Search
        distances, indices = self._index.search(embedding.astype('float32'), k)

        matches = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._concepts):
                continue
            if score < threshold:
                continue

            concept = self._concepts[idx]
            matches.append(SNOMEDMatch(
                concept_id=concept.concept_id,
                term=concept.term,
                score=float(score),
                semantic_type=concept.semantic_type,
            ))

        # Track gaps for vocabulary expansion
        if track_gaps:
            if not matches:
                self._unmatched_terms[mention] = self._unmatched_terms.get(mention, 0) + 1
            elif matches[0].score < 0.5:  # Low confidence threshold
                self._low_confidence_terms[mention] = (matches[0].score, matches[0].term)

        return matches

    def link_batch_with_cache(
        self,
        mentions: list[str],
        k: int = 5,
        threshold: float = 0.0,
        batch_size: int = 32,
        track_gaps: bool = True,
    ) -> list[tuple[str, list[SNOMEDMatch]]]:
        """Link multiple mentions using the embedding cache.

        More efficient than `link_batch()` when processing many documents
        with overlapping vocabularies.

        Args:
            mentions: List of text mentions to link
            k: Number of top matches per mention
            threshold: Minimum similarity score threshold
            batch_size: Batch size for encoding new mentions
            track_gaps: Whether to track unmatched/low-confidence terms

        Returns:
            List of (mention, matches) tuples
        """
        self._load_index()

        # Separate cached and uncached mentions
        cached_mentions = []
        uncached_mentions = []
        mention_order = {}  # Track original order

        for i, mention in enumerate(mentions):
            mention_order[mention] = i
            if self._cache_enabled and mention in self._mention_cache:
                cached_mentions.append(mention)
            else:
                uncached_mentions.append(mention)

        # Encode uncached mentions
        if uncached_mentions:
            new_embeddings = self.encode(uncached_mentions, batch_size=batch_size)
            if self._cache_enabled:
                for mention, emb in zip(uncached_mentions, new_embeddings):
                    self._mention_cache[mention] = emb.reshape(1, -1)

        # Build full embedding matrix in original order
        embeddings = []
        for mention in mentions:
            if mention in self._mention_cache:
                embeddings.append(self._mention_cache[mention])
            else:
                # Should not happen, but fallback
                embeddings.append(self.encode(mention))

        embeddings = np.vstack(embeddings)

        # Validate dimensions
        index_dim = self._index.d
        embed_dim = embeddings.shape[1]
        if embed_dim != index_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model produces {embed_dim}-dim vectors "
                f"but SNOMED index expects {index_dim}-dim. "
                f"Rebuild the index with: linker.build_index(concepts)"
            )

        # Search index
        distances, indices = self._index.search(embeddings.astype('float32'), k)

        # Build results
        results = []
        for mention, idxs, scores in zip(mentions, indices, distances):
            matches = []
            for idx, score in zip(idxs, scores):
                if idx < 0 or idx >= len(self._concepts):
                    continue
                if score < threshold:
                    continue

                concept = self._concepts[idx]
                matches.append(SNOMEDMatch(
                    concept_id=concept.concept_id,
                    term=concept.term,
                    score=float(score),
                    semantic_type=concept.semantic_type,
                ))

            # Track gaps
            if track_gaps:
                if not matches:
                    self._unmatched_terms[mention] = self._unmatched_terms.get(mention, 0) + 1
                elif matches[0].score < 0.5:
                    self._low_confidence_terms[mention] = (matches[0].score, matches[0].term)

            results.append((mention, matches))

        return results


# =============================================================================
# SNOMED Data Loading
# =============================================================================

def load_snomed_from_umls(
    umls_path: Union[str, Path],
    semantic_types: Optional[list[str]] = None,
    max_concepts: Optional[int] = None,
    verbose: bool = True,
) -> list[SNOMEDConcept]:
    """
    Load SNOMED CT concepts from UMLS Metathesaurus files.

    This function reads SNOMED concepts from the UMLS MRCONSO.RRF file.
    Requires a UMLS license and downloaded Metathesaurus files.

    Args:
        umls_path: Path to UMLS META directory containing MRCONSO.RRF
        semantic_types: List of semantic types to include (None = all)
        max_concepts: Maximum number of concepts to load (None = all)
        verbose: Print progress messages

    Returns:
        List of SNOMEDConcept objects
    """
    umls_path = Path(umls_path)
    mrconso_path = umls_path / "MRCONSO.RRF"

    if not mrconso_path.exists():
        raise FileNotFoundError(f"MRCONSO.RRF not found at {mrconso_path}")

    if verbose:
        print(f"Loading SNOMED concepts from {mrconso_path}")

    concepts = {}

    with open(mrconso_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')

            # MRCONSO columns: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|...
            if len(parts) < 15:
                continue

            sab = parts[11]  # Source abbreviation
            if sab != "SNOMEDCT_US":
                continue

            lat = parts[1]  # Language
            if lat != "ENG":
                continue

            concept_id = parts[13]  # SNOMED code
            term = parts[14]  # String
            is_preferred = parts[6] == "Y"  # ISPREF

            if concept_id not in concepts:
                concepts[concept_id] = SNOMEDConcept(
                    concept_id=concept_id,
                    term=term,
                    synonyms=[],
                )

            # Update with preferred term
            if is_preferred:
                concepts[concept_id].term = term
            else:
                concepts[concept_id].synonyms.append(term)

            if max_concepts and len(concepts) >= max_concepts:
                break

    result = list(concepts.values())

    if verbose:
        print(f"  Loaded {len(result):,} SNOMED concepts")

    return result


def load_snomed_from_csv(
    csv_path: Union[str, Path],
    concept_id_col: str = "concept_id",
    term_col: str = "term",
    semantic_type_col: Optional[str] = None,
    max_concepts: Optional[int] = None,
    verbose: bool = True,
) -> list[SNOMEDConcept]:
    """
    Load SNOMED CT concepts from a CSV file.

    Args:
        csv_path: Path to CSV file
        concept_id_col: Column name for concept IDs
        term_col: Column name for terms
        semantic_type_col: Column name for semantic types (optional)
        max_concepts: Maximum number of concepts to load
        verbose: Print progress messages

    Returns:
        List of SNOMEDConcept objects
    """
    import pandas as pd

    if verbose:
        print(f"Loading SNOMED concepts from {csv_path}")

    df = pd.read_csv(csv_path)

    if max_concepts:
        df = df.head(max_concepts)

    concepts = []
    for _, row in df.iterrows():
        concept = SNOMEDConcept(
            concept_id=str(row[concept_id_col]),
            term=str(row[term_col]),
            semantic_type=str(row[semantic_type_col]) if semantic_type_col else "",
        )
        concepts.append(concept)

    if verbose:
        print(f"  Loaded {len(concepts):,} concepts")

    return concepts


def download_snomed_subset(
    subset: str = "core",
    cache_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> list[SNOMEDConcept]:
    """
    Download a pre-built SNOMED CT subset for entity linking.

    Available subsets:
        - "core": ~50k most common clinical concepts
        - "findings": ~80k clinical findings
        - "procedures": ~30k procedures
        - "full": ~200k concepts (all types)

    Args:
        subset: Which subset to download
        cache_dir: Directory to cache the download
        verbose: Print progress messages

    Returns:
        List of SNOMEDConcept objects

    Note:
        These subsets are derived from UMLS and require accepting
        the UMLS license terms. Data is for research use only.
    """
    # TODO: Host pre-built subsets and implement download
    # For now, raise an informative error
    raise NotImplementedError(
        f"Pre-built SNOMED subsets are not yet available.\n"
        f"Please build the index manually using one of:\n"
        f"  1. load_snomed_from_umls(umls_path) - requires UMLS license\n"
        f"  2. load_snomed_from_csv(csv_path) - from your own CSV\n"
        f"  3. get_sample_snomed_concepts() - small demo set (~500 concepts)\n"
        f"  4. Use MedCAT models which include SNOMED: pip install medcat\n"
    )


def get_sample_snomed_concepts() -> list[SNOMEDConcept]:
    """
    Get a sample set of common SNOMED CT concepts for testing.

    This returns a curated list of ~500 common clinical concepts
    that can be used for testing and development without requiring
    a full SNOMED license.

    Returns:
        List of SNOMEDConcept objects

    Example:
        >>> concepts = get_sample_snomed_concepts()
        >>> linker = SNOMEDLinker()
        >>> linker.build_index(concepts)
        >>> results = linker.link("heart attack")
    """
    # Common clinical concepts with their SNOMED CT IDs
    # This is a representative sample for testing - not exhaustive
    sample_concepts = [
        # Cardiovascular conditions
        ("22298006", "Myocardial infarction", "finding"),
        ("53741008", "Coronary artery disease", "finding"),
        ("38341003", "Hypertensive disorder", "finding"),
        ("49436004", "Atrial fibrillation", "finding"),
        ("84114007", "Heart failure", "finding"),
        ("230690007", "Cerebrovascular accident", "finding"),
        ("266257000", "Transient ischemic attack", "finding"),
        ("429559004", "Peripheral arterial disease", "finding"),
        ("13213009", "Congestive heart failure", "finding"),
        ("194828000", "Angina pectoris", "finding"),

        # Metabolic conditions
        ("73211009", "Diabetes mellitus", "finding"),
        ("44054006", "Type 2 diabetes mellitus", "finding"),
        ("46635009", "Type 1 diabetes mellitus", "finding"),
        ("55822004", "Hyperlipidemia", "finding"),
        ("13644009", "Hypercholesterolemia", "finding"),
        ("238136002", "Overweight", "finding"),
        ("414916001", "Obesity", "finding"),
        ("190828008", "Hypothyroidism", "finding"),
        ("34486009", "Hyperthyroidism", "finding"),

        # Renal conditions
        ("709044004", "Diabetic nephropathy", "finding"),
        ("431855005", "Chronic kidney disease", "finding"),
        ("46177005", "End stage renal disease", "finding"),
        ("197927001", "Acute kidney injury", "finding"),
        ("90708001", "Kidney disease", "finding"),
        ("236423003", "Renal impairment", "finding"),

        # Respiratory conditions
        ("195967001", "Asthma", "finding"),
        ("13645005", "Chronic obstructive pulmonary disease", "finding"),
        ("233604007", "Pneumonia", "finding"),
        ("19829001", "Lung disorder", "finding"),
        ("254637007", "Lung cancer", "finding"),
        ("40122008", "Pneumothorax", "finding"),
        ("70995007", "Pulmonary embolism", "finding"),

        # Neurological conditions
        ("386806002", "Impaired cognition", "finding"),
        ("26929004", "Alzheimer's disease", "finding"),
        ("52448006", "Dementia", "finding"),
        ("84757009", "Epilepsy", "finding"),
        ("37796009", "Migraine", "finding"),
        ("128613002", "Parkinson's disease", "finding"),
        ("24700007", "Multiple sclerosis", "finding"),
        ("230265002", "Neuropathy", "finding"),
        ("302226006", "Peripheral neuropathy", "finding"),

        # Psychiatric conditions
        ("35489007", "Depressive disorder", "finding"),
        ("197480006", "Anxiety disorder", "finding"),
        ("13746004", "Bipolar disorder", "finding"),
        ("191736004", "Schizophrenia", "finding"),
        ("66347000", "Sleep disorder", "finding"),

        # Musculoskeletal conditions
        ("396275006", "Osteoarthritis", "finding"),
        ("69896004", "Rheumatoid arthritis", "finding"),
        ("64859006", "Osteoporosis", "finding"),
        ("203082005", "Fibromyalgia", "finding"),
        ("239873007", "Osteoarthritis of knee", "finding"),
        ("202855006", "Low back pain", "finding"),

        # Gastrointestinal conditions
        ("235595009", "Gastroesophageal reflux disease", "finding"),
        ("14760008", "Peptic ulcer", "finding"),
        ("24526004", "Inflammatory bowel disease", "finding"),
        ("34000006", "Crohn's disease", "finding"),
        ("64766004", "Ulcerative colitis", "finding"),
        ("197456007", "Liver disease", "finding"),
        ("235856003", "Hepatic failure", "finding"),
        ("61977001", "Chronic liver disease", "finding"),
        ("266474003", "Fatty liver", "finding"),
        ("128302006", "Chronic hepatitis C", "finding"),

        # Infectious conditions
        ("186431008", "Clostridioides difficile infection", "finding"),
        ("840539006", "COVID-19", "finding"),
        ("6142004", "Influenza", "finding"),
        ("56717001", "Tuberculosis", "finding"),
        ("186747009", "HIV infection", "finding"),
        ("68566005", "Urinary tract infection", "finding"),
        ("312099009", "Sepsis", "finding"),

        # Oncological conditions
        ("363346000", "Malignant neoplastic disease", "finding"),
        ("254837009", "Breast cancer", "finding"),
        ("93761005", "Colorectal cancer", "finding"),
        ("363358000", "Lung neoplasm", "finding"),
        ("399068003", "Prostate cancer", "finding"),
        ("94381002", "Acute leukemia", "finding"),

        # Endocrine findings
        ("237599002", "Hyperglycemia", "finding"),
        ("302866003", "Hypoglycemia", "finding"),
        ("166717003", "Elevated HbA1c", "finding"),
        ("365845005", "Elevated creatinine", "finding"),
        ("165679005", "Elevated liver enzymes", "finding"),

        # Vital signs / findings
        ("38341003", "Hypertension", "finding"),
        ("45007003", "Hypotension", "finding"),
        ("11381005", "Acne", "finding"),
        ("267036007", "Dyspnea", "finding"),
        ("25064002", "Headache", "finding"),
        ("29857009", "Chest pain", "finding"),
        ("21522001", "Abdominal pain", "finding"),
        ("271807003", "Fever", "finding"),
        ("84229001", "Fatigue", "finding"),
        ("422587007", "Nausea", "finding"),
        ("422400008", "Vomiting", "finding"),
        ("62315008", "Diarrhea", "finding"),
        ("14760008", "Constipation", "finding"),
        ("267102003", "Dysphagia", "finding"),

        # Procedures
        ("387713003", "Surgical procedure", "procedure"),
        ("233258006", "Balloon angioplasty of coronary artery", "procedure"),
        ("232717009", "Coronary artery bypass grafting", "procedure"),
        ("18286008", "Catheterization of heart", "procedure"),
        ("265764009", "Renal dialysis", "procedure"),
        ("175135009", "Kidney transplant", "procedure"),
        ("392021009", "Lumpectomy", "procedure"),
        ("234319005", "Spinal surgery", "procedure"),
        ("80146002", "Appendectomy", "procedure"),
        ("397956004", "Cholecystectomy", "procedure"),
        ("302497006", "Hemodialysis", "procedure"),
        ("313268005", "Colonoscopy", "procedure"),
        ("73761001", "CT scan", "procedure"),
        ("113091000", "MRI scan", "procedure"),
        ("77477000", "Computerized axial tomography", "procedure"),
        ("108290001", "Radiation therapy", "procedure"),
        ("367336001", "Chemotherapy", "procedure"),
        ("182764009", "Blood transfusion", "procedure"),
        ("243788004", "Joint replacement", "procedure"),
        ("112943005", "Hip replacement", "procedure"),

        # Medications (as substances)
        ("387517004", "Aspirin", "substance"),
        ("387458008", "Metformin", "substance"),
        ("372756006", "Atorvastatin", "substance"),
        ("373254001", "Amlodipine", "substance"),
        ("387207008", "Lisinopril", "substance"),
        ("386872004", "Metoprolol", "substance"),
        ("387106007", "Omeprazole", "substance"),
        ("372665008", "Levothyroxine", "substance"),
        ("387135004", "Gabapentin", "substance"),
        ("372567009", "Prednisone", "substance"),
        ("387174006", "Warfarin", "substance"),
        ("372903001", "Insulin", "substance"),
        ("387174006", "Clopidogrel", "substance"),
        ("386975001", "Furosemide", "substance"),
        ("108537001", "Losartan", "substance"),
        ("373786007", "Hydrochlorothiazide", "substance"),
        ("387207008", "Enalapril", "substance"),
        ("372756006", "Simvastatin", "substance"),
        ("387530003", "Ibuprofen", "substance"),
        ("387517004", "Acetaminophen", "substance"),

        # Body structures
        ("80891009", "Heart structure", "body_structure"),
        ("39607008", "Lung structure", "body_structure"),
        ("64033007", "Kidney structure", "body_structure"),
        ("10200004", "Liver structure", "body_structure"),
        ("12738006", "Brain structure", "body_structure"),
        ("181277001", "Pancreas structure", "body_structure"),
        ("69536005", "Head structure", "body_structure"),
        ("51185008", "Thoracic structure", "body_structure"),
        ("818983003", "Abdomen", "body_structure"),
        ("362875007", "Lower extremity", "body_structure"),
        ("72696002", "Knee joint", "body_structure"),
        ("24136001", "Hip joint", "body_structure"),
        ("2748008", "Spinal cord", "body_structure"),
        ("41801008", "Coronary artery", "body_structure"),
        ("28273000", "Retina", "body_structure"),

        # Lifestyle factors
        ("77176002", "Smoker", "finding"),
        ("8517006", "Ex-smoker", "finding"),
        ("266919005", "Never smoked tobacco", "finding"),
        ("228273003", "Alcohol user", "finding"),
        ("105539002", "Drug user", "finding"),
        ("160573003", "Alcohol abuse", "finding"),
        ("248536006", "Inactive", "finding"),
        ("228447005", "Exercises regularly", "finding"),

        # Lab findings
        ("166842003", "Elevated cholesterol", "finding"),
        ("166707007", "Elevated triglycerides", "finding"),
        ("165816005", "Elevated blood glucose", "finding"),
        ("250745003", "Reduced eGFR", "finding"),
        ("166842003", "Abnormal lipid profile", "finding"),
        ("250746002", "Microalbuminuria", "finding"),
        ("165591007", "Elevated ESR", "finding"),
        ("165581004", "Elevated CRP", "finding"),
        ("165468009", "Elevated BNP", "finding"),
        ("165507003", "Elevated troponin", "finding"),
    ]

    return [
        SNOMEDConcept(
            concept_id=cid,
            term=term,
            semantic_type=sem_type,
        )
        for cid, term, sem_type in sample_concepts
    ]


def setup_sample_linker(verbose: bool = True) -> SNOMEDLinker:
    """
    Set up a SNOMEDLinker with the sample concept set.

    This is a convenience function for quick testing and development.
    It builds an index from the sample concepts if one doesn't exist.

    Args:
        verbose: Print progress messages

    Returns:
        SNOMEDLinker ready to use

    Example:
        >>> linker = setup_sample_linker()
        >>> results = linker.link("diabetes")
        >>> print(results[0])
        SCTID:73211009 | Diabetes mellitus (score: 0.95)
    """
    linker = SNOMEDLinker(verbose=verbose)

    if not linker.is_ready():
        if verbose:
            print("Building sample SNOMED index (first time only)...")
        concepts = get_sample_snomed_concepts()
        linker.build_index(concepts)

    return linker


# =============================================================================
# Convenience Functions
# =============================================================================

def get_snomed_cache_dir() -> Path:
    """Get the SNOMED cache directory."""
    return DEFAULT_CACHE_DIR


class DocumentLinker:
    """
    Helper for processing multiple documents with vocabulary tracking.

    This class wraps SNOMEDLinker to provide efficient batch processing
    across multiple documents while tracking vocabulary gaps.

    Example:
        >>> doc_linker = DocumentLinker()
        >>>
        >>> # Process multiple documents
        >>> for doc in documents:
        ...     terms = extract_medical_terms(doc)
        ...     results = doc_linker.link_terms(terms)
        ...
        >>> # Check what's missing
        >>> gaps = doc_linker.get_vocabulary_gaps()
        >>> print(f"Unmatched terms: {len(gaps['unmatched'])}")
        >>>
        >>> # Expand index if needed
        >>> if gaps['unmatched']:
        ...     new_concepts = lookup_snomed_for_terms(gaps['unmatched'])
        ...     doc_linker.expand_index(new_concepts)
    """

    def __init__(
        self,
        linker: Optional[SNOMEDLinker] = None,
        model_id: str = SAPBERT_MODEL_ID,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        """
        Initialize the document linker.

        Args:
            linker: Existing SNOMEDLinker to use. If None, creates one.
            model_id: Embedding model ID (only used if linker is None)
            threshold: Confidence threshold for matches
            verbose: Print progress messages
        """
        self.linker = linker or SNOMEDLinker(model_id=model_id, verbose=verbose)
        self.threshold = threshold
        self.verbose = verbose
        self._documents_processed = 0
        self._total_terms_linked = 0

    def link_terms(
        self,
        terms: list[str],
        k: int = 1,
        threshold: Optional[float] = None,
    ) -> list[tuple[str, list[SNOMEDMatch]]]:
        """
        Link terms from a document.

        Args:
            terms: List of medical terms to link
            k: Number of matches to return per term
            threshold: Score threshold (uses default if None)

        Returns:
            List of (term, matches) tuples
        """
        threshold = threshold if threshold is not None else self.threshold
        results = self.linker.link_batch_with_cache(
            terms,
            k=k,
            threshold=threshold,
            track_gaps=True,
        )
        self._documents_processed += 1
        self._total_terms_linked += len(terms)
        return results

    def get_vocabulary_gaps(
        self,
        min_count: int = 1,
        max_score: float = 0.5,
    ) -> dict:
        """Get vocabulary gaps across all processed documents."""
        gaps = self.linker.get_vocabulary_gaps(
            min_count=min_count,
            max_score=max_score,
        )
        gaps["documents_processed"] = self._documents_processed
        gaps["total_terms_linked"] = self._total_terms_linked
        return gaps

    def expand_index(
        self,
        concepts: list[SNOMEDConcept],
        save: bool = True,
    ) -> int:
        """Add new concepts to the index.

        Args:
            concepts: New SNOMED concepts to add
            save: Whether to persist the expanded index

        Returns:
            Number of new concepts added
        """
        return self.linker.add_concepts(concepts, save=save)

    def get_stats(self) -> dict:
        """Get processing statistics."""
        cache_stats = self.linker.get_cache_stats()
        return {
            "documents_processed": self._documents_processed,
            "total_terms_linked": self._total_terms_linked,
            "unique_terms_cached": cache_stats["cached_mentions"],
            "unmatched_terms": cache_stats["unmatched_terms"],
            "low_confidence_terms": cache_stats["low_confidence_terms"],
            "index_size": self.linker.num_concepts,
        }

    def clear_tracking(self) -> None:
        """Clear gap tracking (but keep embedding cache)."""
        self.linker.clear_tracking()
        self._documents_processed = 0
        self._total_terms_linked = 0

    def print_summary(self) -> None:
        """Print a summary of processing statistics."""
        stats = self.get_stats()
        print(f"\nDocument Linker Summary:")
        print(f"  Documents processed: {stats['documents_processed']}")
        print(f"  Total terms linked: {stats['total_terms_linked']:,}")
        print(f"  Unique terms cached: {stats['unique_terms_cached']:,}")
        print(f"  Index size: {stats['index_size']:,} concepts")
        print(f"\nVocabulary Gaps:")
        print(f"  Unmatched terms: {stats['unmatched_terms']}")
        print(f"  Low confidence: {stats['low_confidence_terms']}")


def get_snomed_info() -> dict:
    """Get information about SNOMED linking capabilities."""
    # Find all existing model-specific indices
    cached_indices = {}
    if DEFAULT_CACHE_DIR.exists():
        for index_file in DEFAULT_CACHE_DIR.glob("snomed_*.index"):
            model_suffix = index_file.stem.replace("snomed_", "")
            concepts_file = DEFAULT_CACHE_DIR / f"snomed_{model_suffix}_concepts.json"
            cached_indices[model_suffix] = {
                "index_path": str(index_file),
                "concepts_exists": concepts_file.exists(),
            }

    return {
        "model": SAPBERT_MODEL_ID,
        "available_models": EMBEDDING_MODELS,
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "faiss_available": _faiss_available,
        "cached_indices": cached_indices,
        # Backwards compatibility
        "index_exists": len(cached_indices) > 0,
    }


def print_snomed_info():
    """Print information about SNOMED linking setup."""
    info = get_snomed_info()

    print("SNOMED Entity Linking Status:")
    print(f"  FAISS available: {info['faiss_available']}")
    print(f"  Cache directory: {info['cache_dir']}")
    print(f"  Default model: {info['model']}")

    if info['cached_indices']:
        print(f"\nCached indices ({len(info['cached_indices'])}):")
        for model_suffix, details in info['cached_indices'].items():
            status = "✓" if details['concepts_exists'] else "⚠ (missing concepts)"
            print(f"  {model_suffix}: {status}")
    else:
        print("\nNo cached indices found. Run setup_sample_linker() to create one.")

    print("\nAvailable embedding models:")
    for name, model_id in EMBEDDING_MODELS.items():
        print(f"  {name}: {model_id}")


# =============================================================================
# Grounded Causal Graph Support
# =============================================================================

@dataclass
class GroundedNode:
    """A node in a causal graph grounded to SNOMED CT."""
    mention: str
    concept_id: str
    term: str
    node_type: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.term}[{self.node_type}] (SCTID:{self.concept_id})"

    def to_dict(self) -> dict:
        return {
            "mention": self.mention,
            "concept_id": self.concept_id,
            "term": self.term,
            "node_type": self.node_type,
            "confidence": self.confidence,
        }


@dataclass
class GroundedEdge:
    """An edge in a grounded causal graph."""
    source: GroundedNode
    target: GroundedNode
    relation: str

    def __str__(self) -> str:
        return f"{self.source.term} {self.relation} {self.target.term}"

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation": self.relation,
        }


class GroundedCausalGraph:
    """
    A causal graph with nodes grounded to SNOMED CT.

    This class represents a causal graph where all nodes are
    linked to SNOMED CT concepts with unique identifiers.
    """

    def __init__(
        self,
        nodes: Optional[list[GroundedNode]] = None,
        edges: Optional[list[GroundedEdge]] = None,
    ):
        self.nodes = nodes or []
        self.edges = edges or []
        self._node_by_id: dict[str, GroundedNode] = {}

        for node in self.nodes:
            self._node_by_id[node.concept_id] = node

    def add_node(self, node: GroundedNode):
        """Add a node to the graph."""
        if node.concept_id not in self._node_by_id:
            self.nodes.append(node)
            self._node_by_id[node.concept_id] = node

    def add_edge(self, edge: GroundedEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, concept_id: str) -> Optional[GroundedNode]:
        """Get a node by its SNOMED concept ID."""
        return self._node_by_id.get(concept_id)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_networkx(self):
        """Convert to NetworkX DiGraph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install networkx")

        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(
                node.concept_id,
                term=node.term,
                mention=node.mention,
                node_type=node.node_type,
                confidence=node.confidence,
            )

        for edge in self.edges:
            G.add_edge(
                edge.source.concept_id,
                edge.target.concept_id,
                relation=edge.relation,
            )

        return G

    def __str__(self) -> str:
        lines = [f"GroundedCausalGraph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        lines.append("\nEdges:")
        for edge in self.edges:
            lines.append(f"  {edge}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GroundedCausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


# =============================================================================
# LLM-Based Entity Extraction (Two-Stage Pipeline)
# =============================================================================

@dataclass
class ExtractedEntity:
    """An entity extracted from text by LLM."""
    raw_text: str           # Original text as found
    standardized: str       # LLM-standardized form
    entity_type: str        # condition, medication, procedure, etc.
    start: int = -1         # Character offset (if available)
    end: int = -1

    def __str__(self) -> str:
        if self.raw_text != self.standardized:
            return f'"{self.raw_text}" → "{self.standardized}" [{self.entity_type}]'
        return f'"{self.raw_text}" [{self.entity_type}]'


class MedicalEntityExtractor:
    """
    LLM-based medical entity extractor for the two-stage pipeline.

    This class uses an LLM to extract and standardize medical entities
    from clinical text before linking to ontologies. The LLM handles:
    - Abbreviations (MI → Myocardial infarction)
    - Synonyms (heart attack → Myocardial infarction)
    - Local terminology variations
    - Entity type classification

    Example:
        >>> extractor = MedicalEntityExtractor()
        >>> entities = extractor.extract("Patient has DM2 and HTN")
        >>> for ent in entities:
        ...     print(ent)
        "DM2" → "Type 2 diabetes mellitus" [condition]
        "HTN" → "Hypertension" [condition]

    The extracted standardized terms are then passed to SNOMEDLinker
    for ontology linking with higher accuracy.
    """

    # System prompt for entity extraction
    EXTRACTION_PROMPT = """You are a medical entity extraction system. Extract all medical entities from the clinical text.

For each entity, provide:
1. raw_text: The exact text as it appears
2. standardized: The standardized medical term (expand abbreviations, use proper terminology)
3. entity_type: One of: condition, medication, procedure, finding, body_structure, substance

Return a JSON array of entities. Example:
[
  {"raw_text": "MI", "standardized": "Myocardial infarction", "entity_type": "condition"},
  {"raw_text": "aspirin", "standardized": "Aspirin", "entity_type": "medication"}
]

Extract ALL medical entities including diseases, symptoms, medications, procedures, lab findings, and anatomical structures.
Return ONLY the JSON array, no other text."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the entity extractor.

        Args:
            model: LLM model to use (default: gemini-2.0-flash)
                   Supports: gemini-*, gpt-*, claude-*, or local models
            api_key: API key (or set via environment variable)
            verbose: Print progress messages
        """
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self._client = None

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client

        # Try to use litellm for unified API
        try:
            import litellm
            self._client = litellm
            return self._client
        except ImportError:
            pass

        # Fallback: try google.generativeai for Gemini
        if "gemini" in self.model.lower():
            try:
                import google.generativeai as genai
                api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self._client = genai
                return self._client
            except ImportError:
                pass

        raise ImportError(
            "No LLM client available. Install one of:\n"
            "  pip install litellm  # Recommended: unified API\n"
            "  pip install google-generativeai  # For Gemini models\n"
            "  pip install openai  # For OpenAI models"
        )

    def extract(self, text: str) -> list[ExtractedEntity]:
        """
        Extract medical entities from text.

        Args:
            text: Clinical text to extract entities from

        Returns:
            List of ExtractedEntity objects
        """
        client = self._get_client()

        # Build prompt
        user_prompt = f"Extract medical entities from this text:\n\n{text}"

        try:
            # Try litellm first
            if hasattr(client, 'completion'):
                response = client.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.EXTRACTION_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                result_text = response.choices[0].message.content

            # Fallback to google.generativeai
            elif hasattr(client, 'GenerativeModel'):
                model = client.GenerativeModel(self.model)
                full_prompt = f"{self.EXTRACTION_PROMPT}\n\n{user_prompt}"
                response = model.generate_content(full_prompt)
                result_text = response.text

            else:
                raise RuntimeError(f"Unknown client type: {type(client)}")

        except Exception as e:
            if self.verbose:
                print(f"  Warning: LLM extraction failed: {e}")
            return []

        # Parse JSON response
        return self._parse_response(result_text)

    def _parse_response(self, text: str) -> list[ExtractedEntity]:
        """Parse LLM response into entities."""
        import re

        # Extract JSON array from response
        text = text.strip()

        # Try to find JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            text = match.group()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"  Warning: Could not parse LLM response as JSON")
            return []

        entities = []
        for item in data:
            if isinstance(item, dict):
                entities.append(ExtractedEntity(
                    raw_text=item.get("raw_text", ""),
                    standardized=item.get("standardized", item.get("raw_text", "")),
                    entity_type=item.get("entity_type", "unknown"),
                ))

        return entities

    def extract_and_link(
        self,
        text: str,
        linker: SNOMEDLinker,
        threshold: float = 0.5,
    ) -> list[LinkedEntity]:
        """
        Extract entities and link to SNOMED in one step.

        This is the recommended two-stage pipeline:
        1. LLM extracts and standardizes entities
        2. Standardized terms are linked via embeddings

        Args:
            text: Clinical text
            linker: SNOMEDLinker instance
            threshold: Minimum similarity score for matches

        Returns:
            List of LinkedEntity objects with SNOMED matches
        """
        # Stage 1: Extract with LLM
        entities = self.extract(text)

        if not entities:
            return []

        # Stage 2: Link standardized terms
        standardized_terms = [e.standardized for e in entities]
        results = linker.link_batch(standardized_terms, k=3, threshold=threshold)

        # Combine results
        linked = []
        for entity, (term, matches) in zip(entities, results):
            snomed_matches = [
                SNOMEDMatch(
                    concept_id=m.concept_id,
                    term=m.term,
                    score=m.score,
                    semantic_type=m.semantic_type,
                )
                for m in matches
            ]
            linked.append(LinkedEntity(
                mention=entity.raw_text,
                start=entity.start,
                end=entity.end,
                matches=snomed_matches,
            ))

        return linked


def create_entity_pipeline(
    embedding_model: str = SAPBERT_MODEL_ID,
    llm_model: str = "gemini-2.0-flash",
    use_llm_extraction: bool = True,
    verbose: bool = True,
) -> tuple[Optional[MedicalEntityExtractor], SNOMEDLinker]:
    """
    Create a complete entity extraction and linking pipeline.

    This sets up the two-stage pipeline:
    1. MedicalEntityExtractor (LLM) - extracts and standardizes entities
    2. SNOMEDLinker (embeddings) - links to SNOMED CT

    Args:
        embedding_model: Model for embeddings (default: SapBERT)
        llm_model: Model for entity extraction (default: Gemini 2.0 Flash)
        use_llm_extraction: Whether to use LLM for extraction
        verbose: Print progress messages

    Returns:
        Tuple of (extractor, linker) - extractor may be None if disabled

    Example:
        >>> extractor, linker = create_entity_pipeline()
        >>> entities = extractor.extract_and_link(
        ...     "Patient has diabetes and hypertension",
        ...     linker
        ... )
    """
    linker = SNOMEDLinker(
        model_id=embedding_model,
        verbose=verbose,
    )

    extractor = None
    if use_llm_extraction:
        try:
            extractor = MedicalEntityExtractor(
                model=llm_model,
                verbose=verbose,
            )
        except ImportError as e:
            if verbose:
                print(f"  Warning: LLM extraction disabled ({e})")

    return extractor, linker
