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

# Common medical acronyms and their expansions
# Used to expand acronyms before SNOMED matching
MEDICAL_ACRONYMS = {
    # Vital signs and measurements
    "HR": "heart rate",
    "BP": "blood pressure",
    "SBP": "systolic blood pressure",
    "DBP": "diastolic blood pressure",
    "RR": "respiratory rate",
    "SpO2": "oxygen saturation",
    "O2 sat": "oxygen saturation",
    "BMI": "body mass index",
    "Temp": "temperature",
    "Wt": "weight",
    "Ht": "height",

    # Lab values
    "BUN": "blood urea nitrogen",
    "Cr": "creatinine",
    "eGFR": "estimated glomerular filtration rate",
    "GFR": "glomerular filtration rate",
    "HbA1c": "hemoglobin A1c",
    "A1c": "hemoglobin A1c",
    "LDL": "low density lipoprotein cholesterol",
    "HDL": "high density lipoprotein cholesterol",
    "TG": "triglycerides",
    "TC": "total cholesterol",
    "ALT": "alanine aminotransferase",
    "AST": "aspartate aminotransferase",
    "ALP": "alkaline phosphatase",
    "GGT": "gamma-glutamyl transferase",
    "WBC": "white blood cell count",
    "RBC": "red blood cell count",
    "Hgb": "hemoglobin",
    "Hb": "hemoglobin",
    "Hct": "hematocrit",
    "PLT": "platelet count",
    "INR": "international normalized ratio",
    "PT": "prothrombin time",
    "PTT": "partial thromboplastin time",
    "aPTT": "activated partial thromboplastin time",
    "BNP": "brain natriuretic peptide",
    "NT-proBNP": "N-terminal pro-brain natriuretic peptide",
    "CRP": "C-reactive protein",
    "ESR": "erythrocyte sedimentation rate",
    "TSH": "thyroid stimulating hormone",
    "T3": "triiodothyronine",
    "T4": "thyroxine",
    "PSA": "prostate specific antigen",
    "CEA": "carcinoembryonic antigen",
    "AFP": "alpha fetoprotein",
    "LFTs": "liver function tests",
    "RFTs": "renal function tests",
    "CBC": "complete blood count",
    "BMP": "basic metabolic panel",
    "CMP": "comprehensive metabolic panel",
    "ABG": "arterial blood gas",
    "VBG": "venous blood gas",
    "UA": "urinalysis",
    "U/A": "urinalysis",

    # Conditions and diagnoses
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "DM2": "type 2 diabetes mellitus",
    "DM1": "type 1 diabetes mellitus",
    "CAD": "coronary artery disease",
    "CHF": "congestive heart failure",
    "HF": "heart failure",
    "MI": "myocardial infarction",
    "STEMI": "ST elevation myocardial infarction",
    "NSTEMI": "non-ST elevation myocardial infarction",
    "AFib": "atrial fibrillation",
    "AF": "atrial fibrillation",
    "A-fib": "atrial fibrillation",
    "VFib": "ventricular fibrillation",
    "VT": "ventricular tachycardia",
    "SVT": "supraventricular tachycardia",
    "PVD": "peripheral vascular disease",
    "PAD": "peripheral arterial disease",
    "DVT": "deep vein thrombosis",
    "PE": "pulmonary embolism",
    "CVA": "cerebrovascular accident",
    "TIA": "transient ischemic attack",
    "COPD": "chronic obstructive pulmonary disease",
    "SOB": "shortness of breath",
    "DOE": "dyspnea on exertion",
    "CKD": "chronic kidney disease",
    "AKI": "acute kidney injury",
    "ESRD": "end stage renal disease",
    "UTI": "urinary tract infection",
    "URI": "upper respiratory infection",
    "GERD": "gastroesophageal reflux disease",
    "PUD": "peptic ulcer disease",
    "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome",
    "NAFLD": "non-alcoholic fatty liver disease",
    "OSA": "obstructive sleep apnea",
    "OA": "osteoarthritis",
    "RA": "rheumatoid arthritis",
    "SLE": "systemic lupus erythematosus",
    "MS": "multiple sclerosis",
    "PD": "Parkinson's disease",
    "AD": "Alzheimer's disease",
    "ALS": "amyotrophic lateral sclerosis",
    "BPH": "benign prostatic hyperplasia",
    "PCOS": "polycystic ovary syndrome",
    "PID": "pelvic inflammatory disease",
    "STI": "sexually transmitted infection",
    "STD": "sexually transmitted disease",
    "HIV": "human immunodeficiency virus infection",
    "AIDS": "acquired immunodeficiency syndrome",
    "HCV": "hepatitis C virus infection",
    "HBV": "hepatitis B virus infection",
    "TB": "tuberculosis",
    "MRSA": "methicillin-resistant Staphylococcus aureus infection",
    "C. diff": "Clostridioides difficile infection",
    "CDI": "Clostridioides difficile infection",

    # Procedures and tests
    "EKG": "electrocardiogram",
    "ECG": "electrocardiogram",
    "Echo": "echocardiogram",
    "TTE": "transthoracic echocardiogram",
    "TEE": "transesophageal echocardiogram",
    "CT": "computed tomography",
    "MRI": "magnetic resonance imaging",
    "CXR": "chest X-ray",
    "XR": "X-ray",
    "US": "ultrasound",
    "PET": "positron emission tomography",
    "EEG": "electroencephalogram",
    "EMG": "electromyography",
    "EGD": "esophagogastroduodenoscopy",
    "ERCP": "endoscopic retrograde cholangiopancreatography",
    "CABG": "coronary artery bypass grafting",
    "PCI": "percutaneous coronary intervention",
    "PTCA": "percutaneous transluminal coronary angioplasty",
    "ICD": "implantable cardioverter-defibrillator",
    "PPM": "permanent pacemaker",
    "AICD": "automatic implantable cardioverter-defibrillator",
    "TKR": "total knee replacement",
    "TKA": "total knee arthroplasty",
    "THR": "total hip replacement",
    "THA": "total hip arthroplasty",
    "Lap chole": "laparoscopic cholecystectomy",
    "Appy": "appendectomy",
    "HD": "hemodialysis",
    "PD": "peritoneal dialysis",
    "TURP": "transurethral resection of prostate",

    # Medications
    "ASA": "aspirin",
    "NSAID": "nonsteroidal anti-inflammatory drug",
    "NSAIDs": "nonsteroidal anti-inflammatory drugs",
    "PPI": "proton pump inhibitor",
    "ACEi": "ACE inhibitor",
    "ACEI": "ACE inhibitor",
    "ARB": "angiotensin receptor blocker",
    "BB": "beta blocker",
    "CCB": "calcium channel blocker",
    "SSRI": "selective serotonin reuptake inhibitor",
    "SNRI": "serotonin-norepinephrine reuptake inhibitor",
    "TCA": "tricyclic antidepressant",
    "MAOI": "monoamine oxidase inhibitor",
    "abx": "antibiotics",
    "Abx": "antibiotics",
    "OCP": "oral contraceptive pill",
    "HRT": "hormone replacement therapy",
    "TPN": "total parenteral nutrition",

    # Clinical terms
    "Hx": "history",
    "PMH": "past medical history",
    "PSH": "past surgical history",
    "FH": "family history",
    "SH": "social history",
    "ROS": "review of systems",
    "HPI": "history of present illness",
    "CC": "chief complaint",
    "Dx": "diagnosis",
    "DDx": "differential diagnosis",
    "Tx": "treatment",
    "Rx": "prescription",
    "Sx": "symptoms",
    "Px": "prognosis",
    "Bx": "biopsy",
    "Fx": "fracture",
    "Cx": "culture",
    "WNL": "within normal limits",
    "NAD": "no acute distress",
    "A&O": "alert and oriented",
    "AAOx3": "alert and oriented times three",
    "PERRLA": "pupils equal round reactive to light and accommodation",
    "RRR": "regular rate and rhythm",
    "CTA": "clear to auscultation",
    "CTAB": "clear to auscultation bilaterally",
    "NT/ND": "non-tender non-distended",
    "NKA": "no known allergies",
    "NKDA": "no known drug allergies",
    "PRN": "as needed",
    "QD": "once daily",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "HS": "at bedtime",
    "AC": "before meals",
    "PC": "after meals",
    "PO": "by mouth",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SQ": "subcutaneous",
    "SubQ": "subcutaneous",
    "PR": "per rectum",
    "SL": "sublingual",
    "R/O": "rule out",
    "S/P": "status post",
    "w/": "with",
    "w/o": "without",
    "c/o": "complaining of",
    "h/o": "history of",
    "f/u": "follow up",
    "y/o": "years old",
    "yo": "years old",
}


def expand_medical_acronyms(text: str) -> str:
    """
    Expand common medical acronyms in text.

    Args:
        text: Input text potentially containing acronyms

    Returns:
        Text with acronyms expanded (original + expansion in parentheses)

    Example:
        >>> expand_medical_acronyms("HTN")
        "HTN (hypertension)"
        >>> expand_medical_acronyms("elevated BUN")
        "elevated BUN (blood urea nitrogen)"
    """
    if not text:
        return text

    # Check if the entire text is an acronym
    text_upper = text.strip().upper()
    if text_upper in MEDICAL_ACRONYMS:
        return MEDICAL_ACRONYMS[text_upper]

    # Check case-sensitive matches (e.g., "HbA1c", "eGFR")
    text_stripped = text.strip()
    if text_stripped in MEDICAL_ACRONYMS:
        return MEDICAL_ACRONYMS[text_stripped]

    # If text contains multiple words, try to expand each
    import re
    words = re.split(r'(\s+)', text)
    expanded_words = []

    for word in words:
        word_upper = word.upper()
        if word_upper in MEDICAL_ACRONYMS:
            expanded_words.append(MEDICAL_ACRONYMS[word_upper])
        elif word in MEDICAL_ACRONYMS:  # Case-sensitive check
            expanded_words.append(MEDICAL_ACRONYMS[word])
        else:
            expanded_words.append(word)

    return "".join(expanded_words)


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
        use_flash_attention: Use Flash Attention 2 if available (faster on GPU)
        verbose: Print progress messages
    """

    def __init__(
        self,
        model_id: str = SAPBERT_MODEL_ID,
        index_path: Optional[Union[str, Path]] = None,
        concepts_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        cache_dir: Optional[Union[str, Path]] = None,
        use_flash_attention: bool = True,
        verbose: bool = True,
    ):
        self._check_dependencies()

        # Resolve model alias if provided (e.g., "qwen3-0.6b" -> "Qwen/Qwen3-Embedding-0.6B")
        self.model_id = EMBEDDING_MODELS.get(model_id, model_id)
        self.device = self._resolve_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.use_flash_attention = use_flash_attention
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

    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention 2 is available and model supports it."""
        if self.device != "cuda":
            return False

        # BERT-based models don't support Flash Attention 2
        # Only newer architectures (Qwen, LLaMA, Mistral, etc.) support it
        bert_models = [
            "sapbert", "pubmedbert", "biobert", "bert", "roberta",
            "distilbert", "albert", "electra", "deberta"
        ]
        model_lower = self.model_id.lower()
        if any(bert in model_lower for bert in bert_models):
            return False

        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Load the embedding model and tokenizer."""
        if self._model is not None or self._sentence_transformer is not None:
            return

        if self.verbose:
            print(f"Loading embedding model: {self.model_id}")

        # Check Flash Attention availability
        flash_available = self._check_flash_attention() if self.use_flash_attention else False

        # Try sentence-transformers first for compatible models
        if self._is_sentence_transformer() or self._is_qwen_model():
            try:
                from sentence_transformers import SentenceTransformer

                # sentence-transformers can use Flash Attention if model supports it
                model_kwargs = {"trust_remote_code": True}
                if flash_available:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    model_kwargs["torch_dtype"] = torch.float16

                self._sentence_transformer = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs,
                )
                if self.verbose:
                    print(f"  Loaded via sentence-transformers")
                    if flash_available:
                        print(f"  Flash Attention 2: enabled")
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

        # Build model kwargs with optional Flash Attention
        model_kwargs = {"trust_remote_code": True}
        if flash_available:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            if self.verbose:
                print(f"  Flash Attention 2: enabled")

        try:
            self._model = AutoModel.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as e:
            # Fall back without Flash Attention if it fails
            if flash_available and self.verbose:
                print(f"  Flash Attention failed ({e}), using standard attention")
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        self._model = self._model.to(self.device)
        self._model.eval()

        if self.verbose:
            print(f"  Device: {self.device}")

    def _load_index(self):
        """Load the FAISS index and concepts, rebuilding if dimension mismatch."""
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

        self._index = faiss.read_index(str(self.index_path))

        if self.verbose:
            print(f"Loading SNOMED index: {self.index_path.name} ({self._index.ntotal:,} concepts)")

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

        # Check dimension compatibility with the embedding model
        self._load_model()
        sample_emb = self.encode(["test"])
        model_dim = sample_emb.shape[1]
        index_dim = self._index.d

        if model_dim != index_dim:
            if self.verbose:
                print(f"  Dimension mismatch: model={model_dim}, index={index_dim}")
                print(f"  Rebuilding index with {self.model_id}...")

            # Rebuild the index with the current model
            self.build_index(self._concepts, save=True)

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
        expand_acronyms: bool = True,
        auto_expand: bool = False,
        min_score_for_auto_expand: float = 0.5,
    ) -> list[SNOMEDMatch]:
        """
        Link a mention to SNOMED CT concepts.

        Args:
            mention: Text mention to link
            k: Number of top matches to return
            threshold: Minimum similarity score threshold
            expand_acronyms: Whether to expand medical acronyms before matching
            auto_expand: Whether to try external lookup for low-confidence matches
            min_score_for_auto_expand: Score threshold below which to try auto-expansion

        Returns:
            List of SNOMEDMatch objects sorted by score (descending)
        """
        self._load_index()

        # Expand acronyms if enabled (e.g., "HR" -> "heart rate")
        search_mention = mention
        if expand_acronyms:
            expanded = expand_medical_acronyms(mention)
            if expanded != mention:
                search_mention = expanded

        # Encode mention
        embedding = self.encode(search_mention)

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

        # Auto-expand: if best match is low confidence, try external lookup
        if auto_expand and (not matches or matches[0].score < min_score_for_auto_expand):
            new_concept = self._lookup_external(mention)
            if new_concept:
                # Add to index and re-search
                self.add_concept(new_concept)
                # Recursive call without auto_expand to get updated results
                return self.link(mention, k=k, threshold=threshold,
                               expand_acronyms=expand_acronyms, auto_expand=False)

        return matches

    def add_concept(self, concept: SNOMEDConcept) -> bool:
        """
        Add a new concept to the index dynamically.

        This allows the index to grow adaptively as new concepts are discovered.

        Args:
            concept: SNOMEDConcept to add

        Returns:
            True if added successfully, False if already exists
        """
        # Check if concept already exists
        existing_ids = {c.concept_id for c in self._concepts}
        if concept.concept_id in existing_ids:
            return False

        # Add to concepts list
        self._concepts.append(concept)

        # Compute embedding and add to index
        embedding = self.encode(concept.term)
        self._index.add(embedding.astype('float32'))

        # Save updated index
        self._save_index()

        return True

    def _lookup_external(self, mention: str) -> Optional[SNOMEDConcept]:
        """
        Look up a concept via the SNOMED International browser API.

        No API key required - uses the public Snowstorm API.

        Args:
            mention: Text to look up

        Returns:
            SNOMEDConcept if found, None otherwise
        """
        # First try acronym expansion
        expanded = expand_medical_acronyms(mention)
        search_term = expanded if expanded != mention else mention

        return lookup_snomed_concept(search_term, verbose=self.verbose)

    def link_batch(
        self,
        mentions: list[str],
        k: int = 5,
        threshold: float = 0.0,
        batch_size: int = 32,
        expand_acronyms: bool = True,
    ) -> list[tuple[str, list[SNOMEDMatch]]]:
        """
        Link multiple mentions to SNOMED CT concepts.

        Args:
            mentions: List of text mentions to link
            k: Number of top matches per mention
            threshold: Minimum similarity score threshold
            batch_size: Batch size for encoding
            expand_acronyms: Whether to expand medical acronyms before matching

        Returns:
            List of (mention, matches) tuples
        """
        self._load_index()

        # Expand acronyms if enabled
        if expand_acronyms:
            search_mentions = [expand_medical_acronyms(m) for m in mentions]
        else:
            search_mentions = mentions

        # Encode all mentions
        embeddings = self.encode(search_mentions, batch_size=batch_size)

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
        batch_size: int = 512,
        save: bool = True,
        use_fp16: bool = True,
    ) -> None:
        """
        Build FAISS index from SNOMED concepts.

        Args:
            concepts: List of SNOMEDConcept objects
            batch_size: Batch size for encoding (larger = faster, more memory)
            save: Whether to save the index to disk
            use_fp16: Use half precision for faster encoding (GPU only)
        """
        import time
        start_time = time.time()

        self._load_model()

        n_concepts = len(concepts)
        n_batches = (n_concepts + batch_size - 1) // batch_size

        # Auto-tune batch size based on available GPU memory
        if self.device == "cuda" and _torch_available:
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Heuristic: ~1GB per 2k concepts for SapBERT, less for larger models
                suggested_batch = min(int(gpu_mem * 500), 1024)
                if batch_size < suggested_batch and n_concepts > 10000:
                    batch_size = suggested_batch
                    n_batches = (n_concepts + batch_size - 1) // batch_size
            except Exception:
                pass

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Building SNOMED FAISS Index")
            print(f"{'='*60}")
            print(f"  Concepts: {n_concepts:,}")
            print(f"  Model: {self.model_id}")
            print(f"  Device: {self.device}")
            print(f"  Batch size: {batch_size}")
            if use_fp16 and self.device == "cuda":
                print(f"  Precision: FP16 (faster)")
            print(f"{'='*60}\n")

        # Extract terms
        terms = [c.term for c in concepts]

        # Enable FP16 for faster encoding on GPU
        use_amp = use_fp16 and self.device == "cuda" and _torch_available
        if use_amp and self._model is not None:
            self._model = self._model.half()

        # Pre-allocate embedding array for efficiency
        # First encode a sample to get embedding dimension
        sample_emb = self.encode(terms[:1], batch_size=1)
        emb_dim = sample_emb.shape[1]
        embeddings = np.zeros((n_concepts, emb_dim), dtype=np.float32)

        if _tqdm_available and self.verbose:
            pbar = _tqdm(
                total=n_concepts,
                desc=f"Encoding {n_concepts:,} SNOMED concepts",
                unit="concepts",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                mininterval=0.5,
            )
        else:
            pbar = None

        # Process in batches, writing directly to pre-allocated array
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i + batch_size]
            batch_embeddings = self.encode(batch, batch_size=len(batch))

            # Write directly to pre-allocated array
            end_idx = min(i + batch_size, n_concepts)
            embeddings[i:end_idx] = batch_embeddings

            if pbar:
                pbar.update(len(batch))

            # Clear GPU cache periodically to prevent OOM
            if self.device == "cuda" and i > 0 and i % (batch_size * 10) == 0:
                if _torch_available:
                    torch.cuda.empty_cache()

        if pbar:
            pbar.close()

        # Restore model to FP32 if we converted
        if use_amp and self._model is not None:
            self._model = self._model.float()

        encode_time = time.time() - start_time

        # Build FAISS index
        if self.verbose:
            print(f"\nBuilding FAISS index ({emb_dim}-dim vectors)...")

        faiss_start = time.time()
        self._index = faiss.IndexFlatIP(emb_dim)
        self._index.add(embeddings)
        faiss_time = time.time() - faiss_start

        self._concepts = concepts
        self._concept_id_to_idx = {c.concept_id: i for i, c in enumerate(concepts)}

        # Save
        if save:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"Saving index to {self.index_path}...")

            save_start = time.time()
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
            save_time = time.time() - save_start

        elapsed = time.time() - start_time
        index_size_mb = self.index_path.stat().st_size / (1024 * 1024) if save and self.index_path.exists() else 0
        rate = n_concepts / encode_time if encode_time > 0 else 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SNOMED Index Built Successfully!")
            print(f"{'='*60}")
            print(f"  Vectors: {self._index.ntotal:,}")
            print(f"  Dimensions: {emb_dim}")
            if save:
                print(f"  Index size: {index_size_mb:.1f} MB")
                print(f"  Saved to: {self.index_path}")
            print(f"  Encoding: {encode_time:.1f}s ({rate:,.0f} concepts/sec)")
            print(f"  FAISS build: {faiss_time:.1f}s")
            if save:
                print(f"  Save: {save_time:.1f}s")
            print(f"  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"{'='*60}\n")

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
        expand_acronyms: bool = True,
        auto_expand: bool = False,
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
            expand_acronyms: Whether to expand medical acronyms before matching
            auto_expand: Whether to look up missing concepts from SNOMED browser

        Returns:
            List of (mention, matches) tuples
        """
        self._load_index()

        # Expand acronyms if enabled (e.g., "HR" -> "heart rate")
        original_to_expanded = {}
        if expand_acronyms:
            expanded_mentions = []
            for mention in mentions:
                expanded = expand_medical_acronyms(mention)
                original_to_expanded[mention] = expanded
                expanded_mentions.append(expanded)
            search_mentions = expanded_mentions
        else:
            search_mentions = mentions
            original_to_expanded = {m: m for m in mentions}

        # Separate cached and uncached mentions
        cached_mentions = []
        uncached_mentions = []
        mention_order = {}  # Track original order

        for i, mention in enumerate(search_mentions):
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
        for search_mention in search_mentions:
            if search_mention in self._mention_cache:
                embeddings.append(self._mention_cache[search_mention])
            else:
                # Should not happen, but fallback
                embeddings.append(self.encode(search_mention))

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

        # Build results - use original mentions as keys
        results = []
        needs_auto_expand = []  # Collect low-confidence matches for auto-expand

        for original_mention, search_mention, idxs, scores in zip(mentions, search_mentions, indices, distances):
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

            # Track gaps and collect for auto-expand
            if track_gaps:
                if not matches:
                    self._unmatched_terms[original_mention] = self._unmatched_terms.get(original_mention, 0) + 1
                    if auto_expand:
                        needs_auto_expand.append((original_mention, search_mention))
                elif matches[0].score < 0.5:
                    self._low_confidence_terms[original_mention] = (matches[0].score, matches[0].term)
                    if auto_expand:
                        needs_auto_expand.append((original_mention, search_mention))

            results.append((original_mention, matches))

        # Auto-expand: look up missing concepts from SNOMED browser
        if auto_expand and needs_auto_expand:
            expanded_concepts = []
            for original, search_term in needs_auto_expand:
                concept = self._lookup_external(search_term)
                if concept:
                    expanded_concepts.append(concept)
                    if self.verbose:
                        print(f"    Auto-expanded: '{original}' -> {concept.term} ({concept.concept_id})")

            # Add new concepts to index and re-link those mentions
            if expanded_concepts:
                for concept in expanded_concepts:
                    self.add_concept(concept)

                # Re-link the expanded mentions
                expanded_originals = [m[0] for m in needs_auto_expand]
                relinked = self.link_batch_with_cache(
                    expanded_originals, k=k, threshold=threshold,
                    batch_size=batch_size, track_gaps=False,
                    expand_acronyms=expand_acronyms, auto_expand=False,  # Don't recurse
                )

                # Update results with relinked matches
                relinked_dict = dict(relinked)
                results = [
                    (m, relinked_dict.get(m, matches) if m in relinked_dict else matches)
                    for m, matches in results
                ]

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


def load_snomed_from_omop(
    omop_path: Union[str, Path],
    vocabulary_id: str = "SNOMED",
    standard_only: bool = True,
    active_only: bool = True,
    domains: Optional[list[str]] = None,
    max_concepts: Optional[int] = None,
    verbose: bool = True,
) -> list[SNOMEDConcept]:
    """
    Load SNOMED CT concepts from an OMOP CDM CONCEPT.csv file.

    The OMOP Common Data Model stores concepts in a standardized format.
    This function reads the CONCEPT.csv file and filters for SNOMED concepts.

    OMOP CONCEPT.csv columns:
        - concept_id: Unique identifier
        - concept_name: Readable name
        - domain_id: Domain (Condition, Drug, Procedure, etc.)
        - vocabulary_id: Source vocabulary (SNOMED, ICD10, RxNorm, etc.)
        - concept_class_id: Class within vocabulary
        - standard_concept: 'S' for standard, 'C' for classification, empty for non-standard
        - concept_code: Code in source vocabulary (SNOMED CT ID)
        - valid_start_date: Start of validity period
        - valid_end_date: End of validity period
        - invalid_reason: 'D' for deleted, 'U' for upgraded, empty for active

    Args:
        omop_path: Path to CONCEPT.csv file (tab or comma delimited)
        vocabulary_id: Vocabulary to filter for (default: "SNOMED")
        standard_only: Only include standard concepts (standard_concept='S')
        active_only: Only include active concepts (invalid_reason is empty)
        domains: List of domains to include (None = all, e.g., ["Condition", "Procedure"])
        max_concepts: Maximum number of concepts to load (None = all)
        verbose: Print progress messages

    Returns:
        List of SNOMEDConcept objects

    Example:
        >>> concepts = load_snomed_from_omop("/path/to/CONCEPT.csv")
        >>> print(f"Loaded {len(concepts):,} SNOMED concepts")
        Loaded 350,000 SNOMED concepts

        >>> # Load only conditions
        >>> conditions = load_snomed_from_omop(
        ...     "/path/to/CONCEPT.csv",
        ...     domains=["Condition"]
        ... )
    """
    import pandas as pd

    omop_path = Path(omop_path)

    if not omop_path.exists():
        raise FileNotFoundError(f"CONCEPT.csv not found at {omop_path}")

    if verbose:
        print(f"Loading SNOMED concepts from OMOP file: {omop_path}")

    # Try to detect delimiter (tab vs comma)
    with open(omop_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','

    # Load the file
    df = pd.read_csv(omop_path, sep=delimiter, low_memory=False)

    if verbose:
        print(f"  Total concepts in file: {len(df):,}")

    # Filter for vocabulary
    if vocabulary_id:
        df = df[df['vocabulary_id'] == vocabulary_id]
        if verbose:
            print(f"  After vocabulary filter ({vocabulary_id}): {len(df):,}")

    # Filter for standard concepts
    if standard_only:
        df = df[df['standard_concept'] == 'S']
        if verbose:
            print(f"  After standard_concept='S' filter: {len(df):,}")

    # Filter for active concepts (invalid_reason should be empty/NaN)
    if active_only:
        df = df[df['invalid_reason'].isna() | (df['invalid_reason'] == '')]
        if verbose:
            print(f"  After active filter (no invalid_reason): {len(df):,}")

    # Filter for specific domains
    if domains:
        df = df[df['domain_id'].isin(domains)]
        if verbose:
            print(f"  After domain filter ({domains}): {len(df):,}")

    # Limit number of concepts
    if max_concepts:
        df = df.head(max_concepts)
        if verbose:
            print(f"  After max_concepts limit: {len(df):,}")

    # Build concept list
    concepts = []
    for _, row in df.iterrows():
        # Use concept_code (SNOMED CT ID) as the concept_id, not the OMOP concept_id
        concept = SNOMEDConcept(
            concept_id=str(row['concept_code']),  # SNOMED CT ID
            term=str(row['concept_name']),
            semantic_type=str(row.get('concept_class_id', '')),
        )
        concepts.append(concept)

    if verbose:
        print(f"  Loaded {len(concepts):,} SNOMED concepts")

        # Show domain distribution
        if len(df) > 0:
            domain_counts = df['domain_id'].value_counts()
            print("  Domain distribution:")
            for domain, count in domain_counts.head(10).items():
                print(f"    {domain}: {count:,}")

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
    # TODO: Host pre-built subsets - for now use SNOMED browser API
    if verbose:
        print(f"Downloading SNOMED concepts via Snowstorm browser API...")
        print(f"Subset: {subset}")

    concepts = fetch_snomed_from_browser(subset=subset, verbose=verbose)

    if not concepts:
        raise RuntimeError(
            f"Could not download SNOMED subset '{subset}'. Try:\n"
            f"  1. load_snomed_from_umls(umls_path) - requires UMLS license\n"
            f"  2. load_snomed_from_csv(csv_path) - from your own CSV\n"
            f"  3. get_sample_snomed_concepts() - small demo set\n"
        )

    # Cache the downloaded concepts
    if cache_dir:
        cache_path = Path(cache_dir) / f"snomed_{subset}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump([
                {"concept_id": c.concept_id, "term": c.term, "semantic_type": c.semantic_type}
                for c in concepts
            ], f)
        if verbose:
            print(f"Cached {len(concepts)} concepts to {cache_path}")

    return concepts


def fetch_snomed_from_browser(
    subset: str = "core",
    verbose: bool = True,
) -> list[SNOMEDConcept]:
    """
    Fetch SNOMED CT concepts from the SNOMED International browser API.

    Uses the public Snowstorm API at browser.ihtsdotools.org.
    No authentication required.

    Args:
        subset: Which subset to fetch:
            - "core": Common clinical findings and disorders
            - "findings": Clinical findings hierarchy
            - "procedures": Procedure hierarchy
        verbose: Print progress

    Returns:
        List of SNOMEDConcept objects
    """
    import urllib.request
    import urllib.parse

    base_url = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
    branch = "MAIN/2024-03-01"  # Use stable release

    # Define ECL queries for different subsets
    ecl_queries = {
        "core": "< 404684003 OR < 71388002 OR < 123037004",  # Clinical finding OR Procedure OR Body structure
        "findings": "< 404684003",  # Descendant of Clinical finding
        "procedures": "< 71388002",  # Descendant of Procedure
        "substances": "< 105590001",  # Descendant of Substance
    }

    ecl = ecl_queries.get(subset, ecl_queries["core"])

    concepts = []
    offset = 0
    limit = 1000  # API limit
    total = None

    if verbose:
        print(f"Fetching SNOMED concepts (this may take a few minutes)...")

    while True:
        try:
            params = urllib.parse.urlencode({
                "ecl": ecl,
                "offset": offset,
                "limit": limit,
                "active": "true",
            })
            url = f"{base_url}/browser/{branch}/concepts?{params}"

            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("Accept-Language", "en")

            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode())

            if total is None:
                total = data.get("total", 0)
                if verbose:
                    print(f"  Total concepts available: {total:,}")

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                concept_id = item.get("conceptId", "")
                fsn = item.get("fsn", {}).get("term", "")
                pt = item.get("pt", {}).get("term", fsn)

                # Determine semantic type from FSN
                sem_type = "finding"
                if "(procedure)" in fsn.lower():
                    sem_type = "procedure"
                elif "(body structure)" in fsn.lower():
                    sem_type = "body_structure"
                elif "(substance)" in fsn.lower():
                    sem_type = "substance"
                elif "(disorder)" in fsn.lower():
                    sem_type = "disorder"
                elif "(finding)" in fsn.lower():
                    sem_type = "finding"

                concepts.append(SNOMEDConcept(
                    concept_id=concept_id,
                    term=pt,  # Use preferred term
                    semantic_type=sem_type,
                ))

            offset += limit
            if verbose and offset % 5000 == 0:
                print(f"  Downloaded {len(concepts):,} / {total:,} concepts...")

            # Limit to reasonable number for demo
            if len(concepts) >= 50000:
                if verbose:
                    print(f"  Limiting to 50,000 concepts for performance")
                break

        except Exception as e:
            if verbose:
                print(f"  API error at offset {offset}: {e}")
            break

    if verbose:
        print(f"  Downloaded {len(concepts):,} SNOMED concepts")

    return concepts


def lookup_snomed_concept(
    term: str,
    verbose: bool = False,
) -> Optional[SNOMEDConcept]:
    """
    Look up a single concept from the SNOMED browser API.

    This is useful for on-demand expansion when a term isn't in the index.

    Args:
        term: Term to look up
        verbose: Print debug info

    Returns:
        SNOMEDConcept if found, None otherwise
    """
    import urllib.request
    import urllib.parse

    try:
        base_url = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
        branch = "MAIN/2024-03-01"

        params = urllib.parse.urlencode({
            "term": term,
            "active": "true",
            "limit": 1,
        })
        url = f"{base_url}/browser/{branch}/descriptions?{params}"

        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("Accept-Language", "en")

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        items = data.get("items", [])
        if items:
            item = items[0]
            concept = item.get("concept", {})
            concept_id = concept.get("conceptId", "")
            fsn = concept.get("fsn", {}).get("term", "")
            pt = concept.get("pt", {}).get("term", item.get("term", term))

            # Determine semantic type
            sem_type = "finding"
            if "(procedure)" in fsn.lower():
                sem_type = "procedure"
            elif "(body structure)" in fsn.lower():
                sem_type = "body_structure"
            elif "(substance)" in fsn.lower():
                sem_type = "substance"

            if verbose:
                print(f"  Found: {pt} ({concept_id})")

            return SNOMEDConcept(
                concept_id=concept_id,
                term=pt,
                semantic_type=sem_type,
            )

    except Exception as e:
        if verbose:
            print(f"  Lookup failed for '{term}': {e}")

    return None


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
        ("267036007", "Dyspnea", "finding"),
        ("11833005", "Shortness of breath", "finding"),
        ("49727002", "Cough", "finding"),
        ("28743005", "Productive cough", "finding"),
        ("275498002", "Acute respiratory distress syndrome", "finding"),
        ("78275009", "Obstructive sleep apnea", "finding"),
        ("73430006", "Sleep apnea", "finding"),
        ("50043002", "Upper respiratory infection", "finding"),
        ("36971009", "Sinusitis", "finding"),
        ("40055000", "Chronic sinusitis", "finding"),
        ("15805002", "Acute sinusitis", "finding"),
        ("82272006", "Common cold", "finding"),
        ("195662009", "Acute bronchitis", "finding"),
        ("63480004", "Chronic bronchitis", "finding"),
        ("233678006", "Allergic rhinitis", "finding"),
        ("61582004", "Allergic rhinitis", "finding"),

        # ENT conditions
        ("194377003", "Otitis media", "finding"),
        ("65363002", "Otitis media", "finding"),
        ("3110003", "Acute otitis media", "finding"),
        ("21186006", "Chronic otitis media", "finding"),
        ("39498005", "Tonsillitis", "finding"),
        ("405737000", "Pharyngitis", "finding"),
        ("43878008", "Streptococcal pharyngitis", "finding"),
        ("126485001", "Laryngitis", "finding"),
        ("19471005", "Epistaxis", "finding"),
        ("162298006", "Nasal congestion", "finding"),
        ("64531003", "Nasal discharge", "finding"),
        ("60862001", "Tinnitus", "finding"),
        ("15188001", "Hearing loss", "finding"),
        ("44054006", "Sensorineural hearing loss", "finding"),
        ("95820000", "Conductive hearing loss", "finding"),
        ("422587007", "Vertigo", "finding"),

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
        ("166740000", "Elevated serum creatinine", "finding"),
        ("166923009", "Blood urea nitrogen elevated", "finding"),
        ("365764009", "Finding of heart rate", "finding"),
        ("364075005", "Heart rate", "observable"),
        ("271649006", "Systolic blood pressure", "observable"),
        ("271650006", "Diastolic blood pressure", "observable"),
        ("75367002", "Blood pressure", "observable"),
        ("86290005", "Respiratory rate", "observable"),
        ("103228002", "Hemoglobin saturation with oxygen", "observable"),
        ("363812007", "Head circumference", "observable"),
        ("27113001", "Body weight", "observable"),
        ("50373000", "Body height", "observable"),
        ("60621009", "Body mass index", "observable"),
        ("386725007", "Body temperature", "observable"),
        ("42419004", "Hemoglobin A1c measurement", "procedure"),
        ("271062006", "Fasting blood glucose", "observable"),
        ("33747003", "Glomerular filtration rate", "observable"),

        # Common clinical observations
        ("102594003", "Electrocardiogram finding", "finding"),
        ("164847006", "Electrocardiogram normal", "finding"),
        ("164861001", "ECG: atrial fibrillation", "finding"),
        ("164884008", "ECG: ST elevation", "finding"),
        ("164889003", "ECG: T wave inversion", "finding"),
        ("713197008", "Reduced left ventricular ejection fraction", "finding"),
        ("8867004", "Left ventricular hypertrophy", "finding"),
        ("399340007", "Stress test abnormal", "finding"),
        ("252569009", "Chest X-ray finding", "finding"),
        ("373945007", "Pericardial effusion", "finding"),
        ("38528000", "Pleural effusion", "finding"),
        ("840544004", "Pulmonary infiltrate", "finding"),
        ("129165009", "Edema", "finding"),
        ("267038008", "Peripheral edema", "finding"),
        ("271863002", "Ankle edema", "finding"),
        ("248499001", "Edema of lower extremity", "finding"),
        ("248490000", "Ascites", "finding"),
        ("271943001", "Jaundice", "finding"),
        ("267034002", "Pallor", "finding"),
        ("271807003", "Fever", "finding"),
        ("386661006", "Chills", "finding"),
        ("25064002", "Headache", "finding"),
        ("21522001", "Abdominal pain", "finding"),
        ("29857009", "Chest pain", "finding"),
        ("161891005", "Back pain", "finding"),
        ("57676002", "Joint pain", "finding"),
        ("68962001", "Muscle pain", "finding"),
        ("271757001", "Fatigue", "finding"),
        ("84229001", "Fatigue", "finding"),
        ("367391008", "Malaise", "finding"),
        ("422400008", "Vomiting", "finding"),
        ("422587007", "Nausea", "finding"),
        ("62315008", "Diarrhea", "finding"),
        ("14760008", "Constipation", "finding"),
        ("267102003", "Dysphagia", "finding"),
        ("300359004", "Finding of appetite", "finding"),
        ("79890006", "Loss of appetite", "finding"),
        ("161832001", "Weight loss", "finding"),
        ("8943002", "Weight gain", "finding"),
        ("22253000", "Pain", "finding"),
        ("373930000", "Cognitive deficit", "finding"),
        ("386807006", "Memory impairment", "finding"),
        ("40917007", "Confusion", "finding"),
        ("3006004", "Altered mental status", "finding"),
        ("419723007", "Mentally alert", "finding"),
        ("271587009", "Depressed mood", "finding"),
        ("48694002", "Anxiety", "finding"),
        ("26079004", "Panic attack", "finding"),
        ("28442001", "Suicidal ideation", "finding"),
        ("247592009", "Poor sleep", "finding"),
        ("193462001", "Insomnia", "finding"),

        # Common clinical terms
        ("248153007", "Family history of disease", "finding"),
        ("160303001", "Family history of diabetes mellitus", "finding"),
        ("134439009", "Family history of heart disease", "finding"),
        ("312824007", "Family history of cancer", "finding"),
        ("160357008", "Family history of hypertension", "finding"),
        ("429280009", "Medication adherence", "finding"),
        ("129832003", "Non-compliance with medication regimen", "finding"),
        ("276026009", "Mechanical fall", "finding"),
        ("161898004", "Falls", "finding"),
        ("20602000", "Indwelling urinary catheter", "finding"),
        ("373573001", "Nosocomial infection", "finding"),
        ("233678006", "Seasonal allergic rhinitis", "finding"),
        ("91936005", "Allergy", "finding"),
        ("416098002", "Drug allergy", "finding"),
        ("91935009", "Food allergy", "finding"),
        ("300916003", "Latex allergy", "finding"),
        ("91930004", "Penicillin allergy", "finding"),
        ("294505008", "Sulfonamide allergy", "finding"),
        ("417532006", "Adverse drug reaction", "finding"),
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


def build_snomed_index(
    omop_path: Optional[Union[str, Path]] = None,
    concepts: Optional[list[SNOMEDConcept]] = None,
    model_id: str = SAPBERT_MODEL_ID,
    index_name: str = "snomed_full",
    domains: Optional[list[str]] = None,
    batch_size: int = 512,
    use_fp16: bool = True,
    use_flash_attention: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Build and cache a SNOMED index from OMOP data (one-time operation).

    This is an OFFLINE operation that should be run once to pre-compute
    embeddings for all SNOMED concepts. The resulting index is cached
    and can be loaded instantly for runtime queries.

    Args:
        omop_path: Path to OMOP CONCEPT.csv file
        concepts: Pre-loaded list of SNOMEDConcept (alternative to omop_path)
        model_id: Embedding model to use
        index_name: Base name for the cached index (default: "snomed_full").
            The concept count is automatically appended, e.g., "snomed_full_349k".
        domains: Filter to specific domains (e.g., ["Condition", "Procedure"])
        batch_size: Batch size for embedding (larger = faster, more memory)
        use_fp16: Use half precision for faster GPU encoding
        use_flash_attention: Use Flash Attention 2 if available (faster on GPU)
        verbose: Print progress messages

    Returns:
        Path to the cached index file

    Example:
        # One-time setup (run once, takes 10-30 minutes for full SNOMED)
        >>> index_path = build_snomed_index("/path/to/CONCEPT.csv")
        Building SNOMED index from 349,211 concepts...
        To load this index, use:
          linker = sl.load_snomed_linker('snomed_full_349211')

        # Runtime (instant loading)
        >>> linker = load_snomed_linker('snomed_full_349211')
        >>> matches = linker.link("diabetes")
    """
    # Resolve model alias if provided
    model_id = EMBEDDING_MODELS.get(model_id, model_id)

    if omop_path is None and concepts is None:
        raise ValueError("Must provide either omop_path or concepts")

    if concepts is None:
        if verbose:
            print(f"Loading SNOMED concepts from {omop_path}...")
        concepts = load_snomed_from_omop(
            omop_path,
            domains=domains,
            verbose=verbose,
        )

    # Include exact concept count in index name for clarity
    n_concepts = len(concepts)
    full_index_name = f"{index_name}_{n_concepts}"

    # Create linker with custom index path
    cache_dir = DEFAULT_CACHE_DIR
    model_suffix = SNOMEDLinker._get_model_cache_suffix(model_id)
    index_path = cache_dir / f"{full_index_name}_{model_suffix}.index"
    concepts_path = cache_dir / f"{full_index_name}_{model_suffix}_concepts.json"

    linker = SNOMEDLinker(
        model_id=model_id,
        index_path=index_path,
        concepts_path=concepts_path,
        use_flash_attention=use_flash_attention,
        verbose=verbose,
    )

    linker.build_index(concepts, batch_size=batch_size, use_fp16=use_fp16)

    if verbose:
        print(f"\nTo load this index, use:")
        print(f"  linker = sl.load_snomed_linker('{full_index_name}')")

    return index_path


def load_snomed_linker(
    index_name: str = "snomed_full",
    model_id: str = SAPBERT_MODEL_ID,
    verbose: bool = True,
    auto_download: bool = True,
    force: bool = False,
) -> SNOMEDLinker:
    """
    Load a pre-built SNOMED index for fast runtime queries.

    This loads a cached index that was previously built with build_snomed_index().
    Loading is fast (seconds) compared to building (minutes).

    If no index exists and auto_download is True (default), the SNOMED vocabulary
    will be automatically downloaded from GitHub releases and the index built.

    Args:
        index_name: Name of the cached index (default: "snomed_full")
        model_id: Embedding model (must match what was used to build)
        verbose: Print progress messages
        auto_download: If True, automatically download vocabulary and build index
            if not found (default: True)
        force: If True, delete existing index and rebuild from scratch (default: False)

    Returns:
        SNOMEDLinker ready for queries

    Raises:
        FileNotFoundError: If the index hasn't been built and auto_download is False

    Example:
        >>> linker = load_snomed_linker()
        >>> matches = linker.link("chronic sinusitis")
        >>> print(matches[0])
        SCTID:40055000 | Chronic sinusitis (score: 0.98)

        # Force rebuild with a different model
        >>> linker = load_snomed_linker(model_id="sapbert", force=True)
    """
    # Resolve model alias if provided
    model_id = EMBEDDING_MODELS.get(model_id, model_id)

    cache_dir = DEFAULT_CACHE_DIR
    model_suffix = SNOMEDLinker._get_model_cache_suffix(model_id)

    # If force=True, delete any existing indices matching this name/model
    if force:
        cache_dir.mkdir(parents=True, exist_ok=True)
        deleted_count = 0
        for existing_index in cache_dir.glob(f"{index_name}_*_{model_suffix}.index"):
            existing_concepts = cache_dir / f"{existing_index.stem}_concepts.json"
            if verbose:
                print(f"Deleting existing index: {existing_index.name}")
            existing_index.unlink(missing_ok=True)
            existing_concepts.unlink(missing_ok=True)
            deleted_count += 1
        # Also check exact name pattern (backward compat)
        exact_index = cache_dir / f"{index_name}_{model_suffix}.index"
        exact_concepts = cache_dir / f"{index_name}_{model_suffix}_concepts.json"
        if exact_index.exists():
            if verbose:
                print(f"Deleting existing index: {exact_index.name}")
            exact_index.unlink(missing_ok=True)
            exact_concepts.unlink(missing_ok=True)
            deleted_count += 1
        if verbose and deleted_count > 0:
            print(f"Deleted {deleted_count} existing index file(s). Rebuilding...")

    # First check if any matching index exists (with concept count suffix)
    index_path = None
    concepts_path = None

    # Look for existing indices matching the pattern
    cache_dir.mkdir(parents=True, exist_ok=True)
    for existing_index in cache_dir.glob(f"{index_name}_*_{model_suffix}.index"):
        index_path = existing_index
        concepts_path = cache_dir / f"{existing_index.stem}_concepts.json"
        break

    # Fallback to exact name (for backwards compatibility)
    if index_path is None:
        index_path = cache_dir / f"{index_name}_{model_suffix}.index"
        concepts_path = cache_dir / f"{index_name}_{model_suffix}_concepts.json"

    if not index_path.exists():
        if not auto_download:
            if force:
                raise FileNotFoundError(
                    f"SNOMED index '{index_name}' was deleted (force=True) but cannot rebuild "
                    f"because auto_download=False.\n\n"
                    f"Either set auto_download=True to rebuild automatically, or build manually:\n\n"
                    f"    import synthlab as sl\n"
                    f"    sl.build_snomed_index('/path/to/CONCEPT.csv')\n"
                )
            raise FileNotFoundError(
                f"SNOMED index '{index_name}' not found at {index_path}\n\n"
                f"You need to build the index first (one-time operation):\n\n"
                f"    import synthlab as sl\n"
                f"    sl.build_snomed_index('/path/to/CONCEPT.csv')\n\n"
                f"Or enable auto_download=True to download and build automatically.\n"
            )

        # Auto-download and build
        if verbose:
            print("SNOMED index not found. Downloading vocabulary and building index...")
            print("(This is a one-time operation)")

        from synthlab.download_snomed import get_concept_csv_path
        concept_csv = get_concept_csv_path()

        if verbose:
            print(f"\nBuilding SNOMED index from {concept_csv}...")

        # Build the index - this will create files with concept count in name
        built_path = build_snomed_index(
            omop_path=concept_csv,
            model_id=model_id,
            index_name=index_name,
            verbose=verbose,
        )

        # Update paths to match what was built
        index_path = built_path
        concepts_path = cache_dir / f"{built_path.stem}_concepts.json"

    linker = SNOMEDLinker(
        model_id=model_id,
        index_path=index_path,
        concepts_path=concepts_path,
        verbose=verbose,
    )

    # Force load to verify
    linker._load_index()

    return linker


def list_snomed_indices(verbose: bool = True) -> dict[str, dict]:
    """
    List all cached SNOMED indices.

    Returns:
        Dictionary mapping index names to their info
    """
    indices = {}
    cache_dir = DEFAULT_CACHE_DIR

    if not cache_dir.exists():
        return indices

    for index_file in cache_dir.glob("*.index"):
        name = index_file.stem
        concepts_file = cache_dir / f"{name}_concepts.json"

        # Get concept count
        n_concepts = 0
        if concepts_file.exists():
            try:
                with open(concepts_file, 'r') as f:
                    data = json.load(f)
                    n_concepts = len(data)
            except Exception:
                pass

        indices[name] = {
            "index_path": str(index_file),
            "concepts_path": str(concepts_file),
            "n_concepts": n_concepts,
            "size_mb": index_file.stat().st_size / (1024 * 1024),
        }

    if verbose:
        print("Cached SNOMED indices:")
        if not indices:
            print("  (none)")
        for name, info in indices.items():
            print(f"  {name}: {info['n_concepts']:,} concepts ({info['size_mb']:.1f} MB)")

    return indices


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
    """
    An edge in a grounded causal graph, supporting interactions.

    Attributes:
        sources: List of source nodes (supports single or multiple for interactions)
        target: The effect/consequent node
        relation: The relationship type (e.g., "++>", "+>", "->", "=>")
        interaction: Type of interaction between sources: "and", "or", or None

    Examples:
        Simple edge: GroundedEdge(sources=[node1], target=node2, relation="++>")
        AND interaction: GroundedEdge(sources=[node1, node2], target=node3, relation="=>", interaction="and")
        OR interaction: GroundedEdge(sources=[node1, node2], target=node3, relation="++>", interaction="or")
    """
    sources: list[GroundedNode]
    target: GroundedNode
    relation: str
    interaction: Optional[str] = None  # "and", "or", or None

    def __init__(
        self,
        target: GroundedNode,
        relation: str,
        source: Optional[GroundedNode] = None,
        sources: Optional[list[GroundedNode]] = None,
        interaction: Optional[str] = None,
    ):
        """Initialize with backward-compatible single source or new multi-source format."""
        if sources is not None:
            self.sources = sources
        elif source is not None:
            self.sources = [source]
        else:
            self.sources = []
        self.target = target
        self.relation = relation
        self.interaction = interaction

    @property
    def source(self) -> Optional[GroundedNode]:
        """Get first source for backward compatibility."""
        return self.sources[0] if self.sources else None

    @property
    def is_interaction(self) -> bool:
        """True if this edge represents an interaction between multiple sources."""
        return len(self.sources) > 1 and self.interaction is not None

    def __str__(self) -> str:
        if self.is_interaction:
            op = " && " if self.interaction == "and" else " || "
            source_str = op.join(s.term for s in self.sources)
            return f"{source_str} {self.relation} {self.target.term}"
        return f"{self.source.term} {self.relation} {self.target.term}" if self.source else f"? {self.relation} {self.target.term}"

    def to_dict(self) -> dict:
        result = {
            "sources": [s.to_dict() for s in self.sources],
            "target": self.target.to_dict(),
            "relation": self.relation,
        }
        if self.interaction:
            result["interaction"] = self.interaction
        # Include legacy "source" key for backward compatibility
        if self.sources:
            result["source"] = self.sources[0].to_dict()
        return result


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

    def to_networkx(self, expand_interactions: bool = True):
        """
        Convert to NetworkX DiGraph with interaction support.

        Args:
            expand_interactions: If True (default), create intermediate nodes for
                A+B->C interactions. If False, create separate edges from each source.

        Returns:
            NetworkX DiGraph with node and edge attributes.
        """
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
                is_interaction=False,
            )

        for edge in self.edges:
            if edge.is_interaction and expand_interactions:
                # Create intermediate interaction node
                op = "AND" if edge.interaction == "and" else "OR"
                source_ids = [s.concept_id for s in edge.sources]
                interaction_node_id = f"({' {op} '.join(source_ids)})"

                # Add interaction node
                G.add_node(
                    interaction_node_id,
                    term=f"{op} of {len(edge.sources)} factors",
                    node_type="interaction",
                    is_interaction=True,
                    interaction_type=edge.interaction,
                    members=source_ids,
                )

                # Connect sources to interaction node
                for src in edge.sources:
                    G.add_edge(
                        src.concept_id,
                        interaction_node_id,
                        relation="member_of",
                    )

                # Connect interaction node to target
                G.add_edge(
                    interaction_node_id,
                    edge.target.concept_id,
                    relation=edge.relation,
                    interaction=edge.interaction,
                )
            else:
                # Simple edge or non-expanded interaction
                for src in edge.sources:
                    G.add_edge(
                        src.concept_id,
                        edge.target.concept_id,
                        relation=edge.relation,
                        interaction=edge.interaction,
                    )

        return G

    def interactions(self, interaction_type: Optional[str] = None) -> list[GroundedEdge]:
        """
        Get edges representing interactions (multiple sources).

        Args:
            interaction_type: Filter by "and" or "or", or None for all interactions

        Returns:
            List of GroundedEdge objects with multiple sources
        """
        if interaction_type:
            return [e for e in self.edges if e.interaction == interaction_type]
        return [e for e in self.edges if e.is_interaction]

    def __str__(self) -> str:
        interaction_count = len(self.interactions())
        lines = [f"GroundedCausalGraph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        if interaction_count > 0:
            lines[0] += f" ({interaction_count} interactions)"
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
        # Normalize model name for litellm (google/gemini-* -> gemini/gemini-*)
        if model.startswith("google/gemini"):
            model = model.replace("google/", "gemini/")
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self._client = None

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client

        # Load .env file if present (for API keys)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not installed, skip

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
                print(f"  ERROR: LLM extraction failed: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Entity extraction failed: {e}") from e

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
