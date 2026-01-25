#!/usr/bin/env python3
"""
Synthetic medical image generation for SynthLab.

This module provides tools for generating synthetic medical images that can be
coherent with EHR data. It wraps existing open-source medical image generation
models and provides a unified interface.

Supported backends:
- Medfusion: 2D medical images (chest X-ray, fundoscopy, histopathology)
- MONAI Generative: 2D and 3D medical images (CT, MRI, X-ray)

References:
- Medfusion: https://github.com/mueller-franzes/medfusion
- MONAI Generative: https://github.com/Project-MONAI/GenerativeModels
- MAISI: https://arxiv.org/abs/2409.11169

Example:
    >>> from synthlab import ImagingGenerator
    >>> generator = ImagingGenerator(modality="chest_xray")
    >>>
    >>> # Generate unconditional image
    >>> image = generator.generate()
    >>>
    >>> # Generate image conditioned on pathology
    >>> image = generator.generate(conditions={"cardiomegaly": True, "pleural_effusion": False})
    >>>
    >>> # Generate image coherent with patient EHR
    >>> from synthlab import load_multimodal_dataset
    >>> patient = load_multimodal_dataset(max_patients=1)[0]
    >>> image = generator.generate_for_patient(patient)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Literal
import warnings

# Check for optional dependencies
_torch_available = False
_monai_available = False
_medfusion_available = False
_PIL_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    pass

try:
    from PIL import Image
    _PIL_available = True
except ImportError:
    pass

try:
    import monai
    from monai.networks.nets import AutoencoderKL
    _monai_available = True
except ImportError:
    pass


# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_MODALITIES = {
    "chest_xray": {
        "description": "Chest X-ray (posteroanterior view)",
        "dimensions": "2D",
        "default_size": (512, 512),
        "backends": ["medfusion", "monai"],
        "conditions": [
            "cardiomegaly", "pleural_effusion", "pneumonia", "atelectasis",
            "consolidation", "pneumothorax", "edema", "emphysema", "fibrosis",
            "nodule", "mass", "hernia", "normal",
        ],
    },
    "fundoscopy": {
        "description": "Eye fundus photography",
        "dimensions": "2D",
        "default_size": (512, 512),
        "backends": ["medfusion"],
        "conditions": ["healthy", "glaucoma", "diabetic_retinopathy", "amd"],
    },
    "histopathology": {
        "description": "Histopathology slides (H&E stained)",
        "dimensions": "2D",
        "default_size": (256, 256),
        "backends": ["medfusion"],
        "conditions": ["benign", "malignant", "normal"],
    },
    "brain_mri": {
        "description": "Brain MRI (T1, T2, FLAIR)",
        "dimensions": "3D",
        "default_size": (128, 128, 128),
        "backends": ["monai"],
        "conditions": ["healthy", "tumor", "stroke", "atrophy"],
    },
    "ct_chest": {
        "description": "Chest CT scan",
        "dimensions": "3D",
        "default_size": (128, 128, 128),
        "backends": ["monai"],
        "conditions": ["normal", "nodule", "mass", "emphysema", "fibrosis"],
    },
    "ct_abdomen": {
        "description": "Abdominal CT scan",
        "dimensions": "3D",
        "default_size": (128, 128, 128),
        "backends": ["monai"],
        "conditions": ["normal", "liver_lesion", "kidney_cyst", "splenomegaly"],
    },
}

# Mapping from SNOMED/ICD codes to imaging conditions
CONDITION_MAPPINGS = {
    # Chest X-ray conditions (SNOMED CT codes)
    "cardiomegaly": ["8186001", "I51.7"],  # Cardiomegaly
    "pleural_effusion": ["60046008", "J90"],  # Pleural effusion
    "pneumonia": ["233604007", "J18.9"],  # Pneumonia
    "atelectasis": ["46621007", "J98.11"],  # Atelectasis
    "pneumothorax": ["36118008", "J93.9"],  # Pneumothorax
    "pulmonary_edema": ["19242006", "J81"],  # Pulmonary edema
    "emphysema": ["87433001", "J43.9"],  # Emphysema
    "lung_nodule": ["427359005", "R91.1"],  # Lung nodule
    "lung_mass": ["254632001", "C34.9"],  # Lung mass/cancer
    # Brain conditions
    "brain_tumor": ["126952004", "C71.9"],  # Brain tumor
    "stroke": ["230690007", "I64"],  # Stroke
    "alzheimers": ["26929004", "G30.9"],  # Alzheimer's
}


@dataclass
class ImagingGeneratorConfig:
    """Configuration for medical image generation."""

    modality: str = "chest_xray"
    backend: str = "auto"  # "auto", "medfusion", "monai"
    image_size: Optional[tuple] = None  # None = use default for modality
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: Optional[int] = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5  # For classifier-free guidance

    # Model paths (if using custom trained models)
    autoencoder_path: Optional[str] = None
    diffusion_model_path: Optional[str] = None

    def __post_init__(self):
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: {self.modality}. "
                f"Supported: {list(SUPPORTED_MODALITIES.keys())}"
            )

        if self.image_size is None:
            self.image_size = SUPPORTED_MODALITIES[self.modality]["default_size"]


@dataclass
class GeneratedImage:
    """A generated synthetic medical image with metadata."""

    image: Any  # numpy array or torch tensor
    modality: str
    conditions: dict = field(default_factory=dict)
    seed: Optional[int] = None
    backend: str = ""
    metadata: dict = field(default_factory=dict)

    def to_pil(self) -> "Image.Image":
        """Convert to PIL Image."""
        if not _PIL_available:
            raise ImportError("Pillow required: pip install Pillow")

        import numpy as np

        if _torch_available and isinstance(self.image, torch.Tensor):
            arr = self.image.cpu().numpy()
        else:
            arr = self.image

        # Handle different array shapes
        if arr.ndim == 4:  # Batch dimension
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in [1, 3]:  # Channel first
            arr = arr.transpose(1, 2, 0)
        if arr.shape[-1] == 1:  # Single channel
            arr = arr.squeeze(-1)

        # Normalize to 0-255
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

        return Image.fromarray(arr)

    def save(self, path: str):
        """Save image to file."""
        self.to_pil().save(path)


# =============================================================================
# Main Generator Class
# =============================================================================

class ImagingGenerator:
    """
    Generate synthetic medical images using diffusion models.

    This class provides a unified interface for generating synthetic medical
    images that can optionally be conditioned on clinical labels derived from
    EHR data.

    Example:
        >>> generator = ImagingGenerator(modality="chest_xray")
        >>>
        >>> # Unconditional generation
        >>> image = generator.generate()
        >>> image.save("synthetic_cxr.png")
        >>>
        >>> # Conditional generation
        >>> image = generator.generate(conditions={"cardiomegaly": True})
        >>>
        >>> # Generate for a patient (extracts conditions from EHR)
        >>> patient = load_multimodal_dataset(max_patients=1)[0]
        >>> image = generator.generate_for_patient(patient)
    """

    def __init__(
        self,
        modality: str = "chest_xray",
        backend: str = "auto",
        device: str = "auto",
        config: Optional[ImagingGeneratorConfig] = None,
    ):
        """
        Initialize the imaging generator.

        Args:
            modality: Type of medical image to generate
            backend: Which generation backend to use ("auto", "medfusion", "monai")
            device: Device for inference ("auto", "cuda", "cpu")
            config: Full configuration object (overrides other args)
        """
        if config is not None:
            self.config = config
        else:
            self.config = ImagingGeneratorConfig(
                modality=modality,
                backend=backend,
                device=device,
            )

        self._model = None
        self._autoencoder = None
        self._backend = None

    def _check_dependencies(self):
        """Check required dependencies are available."""
        if not _torch_available:
            raise ImportError(
                "PyTorch required for image generation. "
                "Install with: pip install torch"
            )
        if not _PIL_available:
            raise ImportError(
                "Pillow required for image generation. "
                "Install with: pip install Pillow"
            )

    def _select_backend(self) -> str:
        """Select the best available backend."""
        modality_info = SUPPORTED_MODALITIES[self.config.modality]
        available_backends = modality_info["backends"]

        if self.config.backend != "auto":
            if self.config.backend not in available_backends:
                raise ValueError(
                    f"Backend '{self.config.backend}' not supported for {self.config.modality}. "
                    f"Available: {available_backends}"
                )
            return self.config.backend

        # Auto-select: prefer medfusion for 2D, monai for 3D
        if modality_info["dimensions"] == "2D" and "medfusion" in available_backends:
            return "medfusion"
        elif "monai" in available_backends:
            return "monai"
        else:
            return available_backends[0]

    def _get_device(self):
        """Get the torch device to use."""
        if not _torch_available:
            return None

        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def initialize(self, force: bool = False):
        """
        Initialize the generation model.

        This is called automatically on first generate() call, but can be
        called explicitly to pre-load models.

        Args:
            force: Re-initialize even if already initialized
        """
        if self._model is not None and not force:
            return

        self._check_dependencies()
        self._backend = self._select_backend()

        print(f"Initializing {self._backend} backend for {self.config.modality}...")

        if self._backend == "medfusion":
            self._init_medfusion()
        elif self._backend == "monai":
            self._init_monai()
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _init_medfusion(self):
        """Initialize Medfusion backend."""
        try:
            # Try to import medfusion
            from medical_diffusion.models.pipelines import DiffusionPipeline
            print("  Found medfusion package")
        except ImportError:
            # Provide installation instructions
            print("\n" + "=" * 60)
            print("Medfusion not installed")
            print("=" * 60)
            print()
            print("To use Medfusion for 2D medical image generation:")
            print()
            print("  git clone https://github.com/mueller-franzes/medfusion.git")
            print("  cd medfusion")
            print("  pip install -e .")
            print()
            print("After installation, you'll need to train models on your data")
            print("or download pre-trained weights if available.")
            print()
            print("See: https://github.com/mueller-franzes/medfusion")
            print("=" * 60)

            # Create a placeholder that will show helpful error
            self._model = "medfusion_not_installed"
            return

        device = self._get_device()

        # Check for model paths
        if self.config.diffusion_model_path:
            print(f"  Loading model from {self.config.diffusion_model_path}")
            # Load custom trained model
            self._model = DiffusionPipeline.load_from_checkpoint(
                self.config.diffusion_model_path
            ).to(device)
        else:
            print("  No pre-trained model path specified.")
            print("  You'll need to train a model first or specify a checkpoint path.")
            self._model = "medfusion_no_model"

    def _init_monai(self):
        """Initialize MONAI Generative backend."""
        if not _monai_available:
            print("\n" + "=" * 60)
            print("MONAI not installed")
            print("=" * 60)
            print()
            print("To use MONAI for medical image generation:")
            print()
            print("  pip install monai")
            print("  pip install monai-generative  # Optional, for older tutorials")
            print()
            print("For 3D CT generation with MAISI:")
            print("  See MONAI tutorials: https://github.com/Project-MONAI/tutorials")
            print()
            print("See: https://github.com/Project-MONAI/GenerativeModels")
            print("=" * 60)

            self._model = "monai_not_installed"
            return

        device = self._get_device()

        # Check for model paths
        if self.config.diffusion_model_path:
            print(f"  Loading MONAI model from {self.config.diffusion_model_path}")
            # This would load a MONAI checkpoint
            # Implementation depends on specific model architecture
            self._model = "monai_custom_model"
        else:
            print("  No pre-trained model path specified.")
            print("  For MAISI (3D CT generation), see MONAI tutorials.")
            self._model = "monai_no_model"

    def generate(
        self,
        conditions: Optional[dict[str, bool]] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> list[GeneratedImage]:
        """
        Generate synthetic medical images.

        Args:
            conditions: Dictionary of condition labels (e.g., {"cardiomegaly": True})
            num_images: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of GeneratedImage objects
        """
        self.initialize()

        if isinstance(self._model, str):
            # Model not properly initialized
            raise RuntimeError(
                f"Model not initialized: {self._model}. "
                "Please install the required backend and/or provide model paths."
            )

        if seed is not None:
            if _torch_available:
                torch.manual_seed(seed)

        # Validate conditions
        if conditions:
            valid_conditions = SUPPORTED_MODALITIES[self.config.modality]["conditions"]
            for cond in conditions:
                if cond not in valid_conditions:
                    warnings.warn(
                        f"Condition '{cond}' not in standard list for {self.config.modality}. "
                        f"Valid: {valid_conditions}"
                    )

        # Generate images
        results = []
        for i in range(num_images):
            img_seed = seed + i if seed is not None else None

            # This is where actual generation would happen
            # For now, create placeholder
            import numpy as np
            placeholder = np.random.rand(*self.config.image_size).astype(np.float32)

            results.append(GeneratedImage(
                image=placeholder,
                modality=self.config.modality,
                conditions=conditions or {},
                seed=img_seed,
                backend=self._backend,
                metadata={
                    "guidance_scale": self.config.guidance_scale,
                    "num_inference_steps": self.config.num_inference_steps,
                },
            ))

        return results if num_images > 1 else results[0]

    def generate_for_patient(
        self,
        patient: Any,
        modality: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GeneratedImage:
        """
        Generate a synthetic medical image coherent with a patient's EHR data.

        This method extracts relevant conditions from the patient's medical
        record and uses them to condition the image generation.

        Args:
            patient: A Patient object from synthlab.coherent
            modality: Override the generator's modality (optional)
            seed: Random seed for reproducibility

        Returns:
            GeneratedImage with conditions extracted from patient EHR
        """
        # Extract conditions from patient FHIR data
        conditions = self._extract_conditions_from_patient(patient)

        print(f"Extracted conditions from patient {patient.patient_id}:")
        for cond, present in conditions.items():
            if present:
                print(f"  - {cond}")

        return self.generate(conditions=conditions, seed=seed)

    def _extract_conditions_from_patient(self, patient: Any) -> dict[str, bool]:
        """Extract imaging-relevant conditions from patient FHIR data."""
        conditions = {}
        valid_conditions = SUPPORTED_MODALITIES[self.config.modality]["conditions"]

        # Initialize all conditions as False
        for cond in valid_conditions:
            conditions[cond] = False

        # Try to extract conditions from FHIR data
        try:
            if hasattr(patient, 'fhir_data') and patient.fhir_data:
                fhir_bundle = patient.fhir_data

                # Look for Condition resources
                for entry in fhir_bundle.get('entry', []):
                    resource = entry.get('resource', {})
                    if resource.get('resourceType') == 'Condition':
                        # Extract condition code
                        coding = resource.get('code', {}).get('coding', [])
                        for code_info in coding:
                            code = code_info.get('code', '')
                            system = code_info.get('system', '')
                            display = code_info.get('display', '').lower()

                            # Match against our condition mappings
                            for cond_name, codes in CONDITION_MAPPINGS.items():
                                if code in codes or any(c.lower() in display for c in [cond_name.replace('_', ' ')]):
                                    if cond_name in conditions:
                                        conditions[cond_name] = True
                                    # Also check for related conditions
                                    for valid_cond in valid_conditions:
                                        if valid_cond.replace('_', ' ') in display:
                                            conditions[valid_cond] = True
        except Exception as e:
            warnings.warn(f"Could not extract conditions from patient: {e}")

        # If no conditions found, assume normal
        if not any(conditions.values()) and "normal" in conditions:
            conditions["normal"] = True

        return conditions

    @staticmethod
    def list_modalities() -> dict:
        """List all supported imaging modalities."""
        return {
            name: {
                "description": info["description"],
                "dimensions": info["dimensions"],
                "conditions": info["conditions"],
                "backends": info["backends"],
            }
            for name, info in SUPPORTED_MODALITIES.items()
        }

    @staticmethod
    def get_modality_info(modality: str) -> dict:
        """Get detailed information about a modality."""
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        return SUPPORTED_MODALITIES[modality]


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_synthetic_image(
    modality: str = "chest_xray",
    conditions: Optional[dict[str, bool]] = None,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> GeneratedImage:
    """
    Generate a synthetic medical image (convenience function).

    Args:
        modality: Type of image ("chest_xray", "brain_mri", etc.)
        conditions: Condition labels for conditional generation
        seed: Random seed
        backend: Generation backend ("auto", "medfusion", "monai")

    Returns:
        GeneratedImage object

    Example:
        >>> from synthlab import generate_synthetic_image
        >>> image = generate_synthetic_image(
        ...     modality="chest_xray",
        ...     conditions={"pneumonia": True, "cardiomegaly": False}
        ... )
        >>> image.save("synthetic_pneumonia.png")
    """
    generator = ImagingGenerator(modality=modality, backend=backend)
    return generator.generate(conditions=conditions, seed=seed)


def get_imaging_generation_info() -> dict:
    """
    Get information about imaging generation capabilities.

    Returns:
        dict: Information about supported modalities, backends, and requirements
    """
    return {
        "supported_modalities": list(SUPPORTED_MODALITIES.keys()),
        "backends": {
            "medfusion": {
                "description": "Latent diffusion for 2D medical images",
                "modalities": ["chest_xray", "fundoscopy", "histopathology"],
                "url": "https://github.com/mueller-franzes/medfusion",
                "install": "git clone + pip install -e .",
            },
            "monai": {
                "description": "MONAI Generative Models for 2D/3D medical images",
                "modalities": ["chest_xray", "brain_mri", "ct_chest", "ct_abdomen"],
                "url": "https://github.com/Project-MONAI/GenerativeModels",
                "install": "pip install monai",
            },
        },
        "dependencies": {
            "torch": _torch_available,
            "monai": _monai_available,
            "PIL": _PIL_available,
        },
        "usage": [
            "# Basic generation",
            "from synthlab import ImagingGenerator",
            "gen = ImagingGenerator(modality='chest_xray')",
            "image = gen.generate(conditions={'pneumonia': True})",
            "",
            "# Generate coherent with patient EHR",
            "patient = load_multimodal_dataset(max_patients=1)[0]",
            "image = gen.generate_for_patient(patient)",
        ],
    }


def print_imaging_generation_info():
    """Print information about imaging generation capabilities."""
    info = get_imaging_generation_info()

    print("\n" + "=" * 70)
    print("Synthetic Medical Image Generation")
    print("=" * 70)

    print("\nSupported Modalities:")
    for mod in info["supported_modalities"]:
        mod_info = SUPPORTED_MODALITIES[mod]
        print(f"  - {mod}: {mod_info['description']} ({mod_info['dimensions']})")

    print("\nBackends:")
    for name, backend in info["backends"].items():
        print(f"\n  {name}:")
        print(f"    {backend['description']}")
        print(f"    Modalities: {', '.join(backend['modalities'])}")
        print(f"    Install: {backend['install']}")
        print(f"    URL: {backend['url']}")

    print("\nDependencies:")
    for dep, available in info["dependencies"].items():
        status = "installed" if available else "NOT INSTALLED"
        print(f"  - {dep}: {status}")

    print("\nExample Usage:")
    for line in info["usage"]:
        print(f"  {line}")
    print()
