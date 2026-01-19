"""
Synthea wrapper for generating synthetic healthcare data.

Synthea is a synthetic patient population simulator that generates realistic
synthetic patient records. This module provides a Python wrapper around the
Synthea Java application.

References:
- https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running
"""

import pooch
import os
import subprocess
import shlex
import json
from typing import Optional, Literal, Any
from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_subprocess(cmd, shell=True, env=None, verbose=True, text=True):
    """Run a command using subprocess and return result dictionary."""
    if verbose:
        print(f"\n$ {cmd}")
    try:
        completed = subprocess.run(
            cmd, shell=shell, env=env, capture_output=True, text=text
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as e:
        logging.error(f"Error running command: {cmd}\nException: {str(e)}")
        return {"returncode": 1, "stdout": "", "stderr": str(e)}


# Default JAR URL
SYNTHEA_JAR_URL = (
    "https://github.com/synthetichealth/synthea/releases/download/"
    "master-branch-latest/synthea-with-dependencies.jar"
)


@dataclass
class SyntheaConfig:
    """Configuration for Synthea simulation runs."""
    
    # Population settings
    population_size: int = 1
    seed: Optional[int] = None
    clinician_seed: Optional[int] = None
    reference_date: Optional[str] = None  # Format: YYYYMMDD
    
    # Demographics
    gender: Optional[Literal["M", "F"]] = None
    min_age: int = 0
    max_age: int = 140
    
    # Location
    state: Optional[str] = None
    city: Optional[str] = None
    
    # Paths
    config_file: Optional[str] = None
    modules_dir: Optional[str] = None
    output_dir: str = "output"
    
    # Exporters (as command-line flags)
    exporter_flags: dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.min_age < 0 or self.max_age > 140:
            raise ValueError("Age range must be 0-140")
        if self.min_age > self.max_age:
            raise ValueError("min_age must be <= max_age")
        if self.gender not in [None, "M", "F"]:
            raise ValueError("gender must be None, 'M', or 'F'")
        if self.reference_date and len(self.reference_date) != 8:
            raise ValueError("reference_date must be in YYYYMMDD format")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheaConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    def __eq__(self, other: object) -> bool:
        """Compare two configs for equality (including seed)."""
        if not isinstance(other, SyntheaConfig):
            return False
        return self.to_dict() == other.to_dict()


class SyntheaRunner:
    """
    Wrapper class for running Synthea synthetic patient data generator.
    
    This class handles downloading the Synthea JAR file and provides
    methods to run Synthea simulations with various configurations.
    
    Example:
        >>> runner = SyntheaRunner()
        >>> config = SyntheaConfig(
        ...     population_size=100,
        ...     state="Massachusetts",
        ...     seed=12345
        ... )
        >>> result = runner.run(config)
        >>> print(f"Generated {result['returncode']} patients")
    """
    
    def __init__(
        self,
        jar_path: Optional[str] = None,
        jar_url: str = SYNTHEA_JAR_URL,
        cache_dir: Optional[str] = None,
        java_executable: str = "java",
    ) -> None:
        """
        Initialize Synthea runner.
        
        Parameters
        ----------
        jar_path : str, optional
            Path to existing Synthea JAR file. If None, will download.
        jar_url : str
            URL to download Synthea JAR from if not found locally.
        cache_dir : str, optional
            Directory to cache downloaded JAR. Defaults to pooch's OS cache.
        java_executable : str
            Path to Java executable. Defaults to "java" (must be in PATH).
        """
        self.jar_url: str = jar_url
        # Use pooch's default OS cache directory if not specified
        if cache_dir is None:
            # pooch.os_cache() returns the default cache directory for the OS
            cache_dir = str(pooch.os_cache("synthea"))
        self.cache_dir: str = cache_dir
        self.java_executable: str = java_executable
        
        # Check Java availability
        self._check_java()
        
        # Get or download JAR
        if jar_path and os.path.exists(jar_path):
            self.jar_path: str = jar_path
        else:
            self.jar_path = self._download_jar()
    
    def _check_java(self) -> None:
        """Check if Java is available and version >= 11."""
        result = run_subprocess(
            f"{self.java_executable} -version",
            verbose=False
        )
        if result["returncode"] != 0:
            raise RuntimeError(
                f"Java not found. Please install Java 11 or newer. "
                f"Attempted: {self.java_executable}"
            )  # noqa: E501
        
        # Parse version (output goes to stderr)
        version_output = result["stderr"]
        # Try to extract version number
        import re
        version_match = re.search(r'version "(\d+)', str(version_output))
        if version_match:
            version = int(version_match.group(1))
            if version < 11:
                raise RuntimeError(
                    f"Java 11+ required, found version {version}"
                )
    
    def _download_jar(self) -> str:
        """Download Synthea JAR file if not present."""
        jar_filename = "synthea-with-dependencies.jar"
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if JAR already exists in cache
        jar_path = os.path.join(self.cache_dir, jar_filename)
        if os.path.exists(jar_path):
            print(f"Using existing Synthea JAR: {jar_path}")
            return jar_path
        
        print(f"Downloading Synthea JAR from {self.jar_url}...")
        print(f"Cache directory: {self.cache_dir}")
        try:
            jar_path = pooch.retrieve(
                url=self.jar_url,
                known_hash=None,
                progressbar=True,
                fname=jar_filename,
                path=self.cache_dir,
            )
            print(f"Downloaded Synthea JAR to: {jar_path}")
            return jar_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Synthea JAR: {e}\n"
                f"URL: {self.jar_url}"
            )  # noqa: E501
    
    def _build_command(self, config: SyntheaConfig) -> str:
        """
        Build Synthea command-line string from configuration.
        
        Parameters
        ----------
        config : SyntheaConfig
            Configuration object.
        
        Returns
        -------
        str
            Complete command string to run Synthea.
        """
        # Build command parts list - all parts will be properly quoted by shlex.join
        cmd_parts = [self.java_executable, "-jar", self.jar_path]
        
        # Seed
        if config.seed is not None:
            cmd_parts.extend(["-s", str(config.seed)])
        
        # Reference date
        if config.reference_date:
            cmd_parts.extend(["-r", config.reference_date])
        
        # Clinician seed
        if config.clinician_seed is not None:
            cmd_parts.extend(["-cs", str(config.clinician_seed)])
        
        # Population size
        if config.population_size != 1:
            cmd_parts.extend(["-p", str(config.population_size)])
        
        # Gender
        if config.gender:
            cmd_parts.extend(["-g", config.gender])
        
        # Age range
        if config.min_age != 0 or config.max_age != 140:
            cmd_parts.extend(["-a", f"{config.min_age}-{config.max_age}"])
        
        # Config file
        if config.config_file:
            cmd_parts.extend(["-c", config.config_file])
        
        # Modules directory
        if config.modules_dir:
            cmd_parts.extend(["-d", config.modules_dir])
        
        # Exporter flags (e.g., --exporter.fhir.use_us_core_ig true)
        for flag, value in config.exporter_flags.items():
            if isinstance(value, bool):
                cmd_parts.append(f"--{flag}={str(value).lower()}")
            else:
                cmd_parts.append(f"--{flag}={value}")
        
        # Location (state and optionally city)
        if config.state:
            cmd_parts.append(config.state)
            if config.city:
                cmd_parts.append(config.city)
        
        # Properly quote arguments that contain spaces
        # Use shlex.join for proper shell escaping (Python 3.8+)
        # Fall back to manual quoting for older Python versions
        try:
            return shlex.join(cmd_parts)
        except AttributeError:
            # Python < 3.8: manually quote arguments
            quoted_parts = []
            for part in cmd_parts:
                if " " in part or any(char in part for char in ['"', "'", "$", "`", "\\"]):
                    # Quote arguments with spaces or special characters
                    quoted_parts.append(shlex.quote(part))
                else:
                    quoted_parts.append(part)
            return " ".join(quoted_parts)
    
    def _check_existing_output(
        self, output_dir: str, config: SyntheaConfig
    ) -> tuple[bool, Optional[SyntheaConfig]]:
        """
        Check if output directory exists and has a saved config.
        
        Returns
        -------
        tuple[bool, Optional[SyntheaConfig]]
            (output_exists, saved_config)
        """
        output_path = Path(output_dir)
        config_file = output_path / ".synthea_config.json"
        
        if not output_path.exists():
            return False, None
        
        # Check if directory has any CSV files (indicates Synthea output)
        csv_files = list(output_path.glob("*.csv"))
        csv_subdir = output_path / "csv"
        if csv_subdir.exists():
            csv_files.extend(list(csv_subdir.glob("*.csv")))
        
        if not csv_files:
            return False, None
        
        # Try to load saved config
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    saved_config_dict = json.load(f)
                saved_config = SyntheaConfig.from_dict(saved_config_dict)
                return True, saved_config
            except (json.JSONDecodeError, KeyError, TypeError):
                # Config file exists but is invalid
                return True, None
        
        # Output exists but no config file
        return True, None
    
    def _save_config(self, output_dir: str, config: SyntheaConfig) -> None:
        """Save config to output directory for future comparison."""
        config_file = Path(output_dir) / ".synthea_config.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def run(
        self,
        config: Optional[SyntheaConfig] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        force: int = 0,
    ) -> dict[str, Any]:
        """
        Run Synthea simulation with given configuration.
        
        Parameters
        ----------
        config : SyntheaConfig, optional
            Configuration for the simulation. If None, uses defaults.
        verbose : bool
            Whether to print command and output.
        output_dir : str, optional
            Override output directory. If None, uses config.output_dir.
            Note: Synthea outputs to current directory by default.
            This sets the working directory for the command.
        force : int
            Force regeneration behavior:
            - 0: Use existing output if it exists (skip generation)
            - 1: Replace unless config (including seed) matches exactly
            - 2: Always regenerate (replace everything)
        
        Returns
        -------
        dict[str, Any]
            Result dictionary with keys:
            - returncode: int (0 = success, or -1 if skipped)
            - stdout: str
            - stderr: str
            - command: str (the command that was run, or "skipped" if force=0)
            - output_dir: str (where files were generated)
            - skipped: bool (True if generation was skipped due to force=0)
        """
        if config is None:
            config = SyntheaConfig()
        
        config.validate()
        
        # Set output directory
        if output_dir is None:
            output_dir = config.output_dir
        
        # Check existing output based on force level
        output_exists, saved_config = self._check_existing_output(output_dir, config)
        
        if output_exists:
            if force == 0:
                # Use existing output
                if verbose:
                    print(f"Output directory exists: {output_dir}")
                    print("Skipping generation (force=0). Use force=1 or force=2 to regenerate.")
                return {
                    "returncode": -1,  # Special code for skipped
                    "stdout": "",
                    "stderr": "",
                    "command": "skipped",
                    "output_dir": os.path.abspath(output_dir),
                    "skipped": True,
                }
            elif force == 1:
                # Replace unless config matches exactly (including seed)
                if saved_config is not None and config == saved_config:
                    if verbose:
                        print(f"Output directory exists with identical config: {output_dir}")
                        print("Skipping generation (force=1, config matches). Use force=2 to force regeneration.")
                    return {
                        "returncode": -1,  # Special code for skipped
                        "stdout": "",
                        "stderr": "",
                        "command": "skipped",
                        "output_dir": os.path.abspath(output_dir),
                        "skipped": True,
                    }
                else:
                    if verbose:
                        if saved_config is None:
                            print(f"Output directory exists but no saved config found: {output_dir}")
                        else:
                            print(f"Output directory exists with different config: {output_dir}")
                        print("Regenerating (force=1, config differs)...")
            # force == 2: Always regenerate (no check needed)
            elif force == 2:
                if verbose:
                    print(f"Force regenerating (force=2): {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = self._build_command(config)
        
        # Run in output directory
        original_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            result = run_subprocess(cmd, verbose=verbose)
        finally:
            os.chdir(original_cwd)
        
        # Save config for future comparison
        if result["returncode"] == 0:
            self._save_config(output_dir, config)
        
        # Add metadata to result
        result["command"] = cmd
        result["output_dir"] = os.path.abspath(output_dir)
        result["skipped"] = False
        
        return result
    
    def run_quick(
        self,
        population_size: int = 10,
        state: Optional[str] = "Massachusetts",
        seed: Optional[int] = None,
        output_dir: str = "output",
        force: int = 0,
    ) -> dict[str, Any]:
        """
        Convenience method for quick test runs.
        
        Parameters
        ----------
        population_size : int
            Number of patients to generate.
        state : str, optional
            US state to generate patients in.
        seed : int, optional
            Random seed for reproducibility.
        output_dir : str
            Directory to save output files.
        force : int
            Force regeneration behavior (0=use existing, 1=replace if config differs, 2=always replace).
        
        Returns
        -------
        dict
            Result dictionary from run().
        """
        config = SyntheaConfig(
            population_size=population_size,
            state=state,
            seed=seed,
            output_dir=output_dir,
        )
        return self.run(config, force=force)
    
    def run_custom_location(
        self,
        state: str,
        city: Optional[str] = None,
        population_size: int = 100,
        seed: Optional[int] = None,
        output_dir: str = "output",
        force: int = 0,
    ) -> dict[str, Any]:
        """
        Generate population for a specific location.
        
        Parameters
        ----------
        state : str
            US state name.
        city : str, optional
            City name within the state.
        population_size : int
            Number of patients to generate.
        seed : int, optional
            Random seed for reproducibility.
        output_dir : str
            Directory to save output files.
        force : int
            Force regeneration behavior (0=use existing, 1=replace if config differs, 2=always replace).
        
        Returns
        -------
        dict[str, Any]
            Result dictionary from run().
        """
        config = SyntheaConfig(
            population_size=population_size,
            state=state,
            city=city,
            seed=seed,
            output_dir=output_dir,
        )
        return self.run(config, force=force)
    
    def run_age_specific(
        self,
        min_age: int,
        max_age: int,
        population_size: int = 100,
        state: Optional[str] = None,
        seed: Optional[int] = None,
        output_dir: str = "output",
        force: int = 0,
    ) -> dict[str, Any]:
        """
        Generate population within specific age range.
        
        Parameters
        ----------
        min_age : int
            Minimum age (0-140).
        max_age : int
            Maximum age (0-140).
        population_size : int
            Number of patients to generate.
        state : str, optional
            US state name.
        seed : int, optional
            Random seed for reproducibility.
        output_dir : str
            Directory to save output files.
        force : int
            Force regeneration behavior (0=use existing, 1=replace if config differs, 2=always replace).
        
        Returns
        -------
        dict[str, Any]
            Result dictionary from run().
        """
        config = SyntheaConfig(
            population_size=population_size,
            min_age=min_age,
            max_age=max_age,
            state=state,
            seed=seed,
            output_dir=output_dir,
        )
        return self.run(config, force=force)
    
    def show_help(self) -> str:
        """
        Display Synthea help message.
        
        Returns
        -------
        str
            Help text from Synthea.
        """
        cmd = f"{self.java_executable} -jar {self.jar_path} -h"
        result = run_subprocess(cmd, verbose=False)
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        return stdout + stderr
    
    def convert_to_omop(
        self,
        synthea_output_dir: str,
        omop_output_dir: Optional[str] = None,
        cdm_version: str = "5.4",
        output_format: Literal["parquet", "csv"] = "parquet",
        verbose: bool = True,
    ) -> dict[str, str]:
        """
        Convert Synthea CSV output to OMOP CDM format.
        
        Convenience method that calls convert_synthea_to_omop() with the
        Synthea output directory from a previous run.
        
        Parameters
        ----------
        synthea_output_dir : str
            Directory containing Synthea CSV output files (typically the
            output_dir from a previous run() call).
        omop_output_dir : str, optional
            Directory where OMOP CDM tables will be saved.
            If None, uses {synthea_output_dir}/omop
        cdm_version : str
            OMOP CDM version (e.g., "5.3", "5.4"). Default: "5.4"
        output_format : str
            Output format: "parquet" or "csv". Default: "parquet"
        verbose : bool
            Whether to print progress messages.
        
        Returns
        -------
        dict[str, str]
            Dictionary mapping OMOP table names to their output file paths.
        """
        if omop_output_dir is None:
            omop_output_dir = os.path.join(synthea_output_dir, "omop")
        
        return convert_synthea_to_omop(
            synthea_csv_dir=synthea_output_dir,
            output_dir=omop_output_dir,
            cdm_version=cdm_version,
            output_format=output_format,
            verbose=verbose,
        )


def convert_synthea_to_omop(
    synthea_csv_dir: str,
    output_dir: str,
    cdm_version: str = "5.4",
    synthea_version: str = "3.0.0",
    output_format: Literal["parquet", "csv"] = "parquet",
    concept_mapping: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Convert Synthea CSV output to OMOP CDM format.
    
    This function reads Synthea-generated CSV files and converts them to OMOP CDM
    tables. Inspired by the OHDSI ETL-Synthea R package.
    
    Parameters
    ----------
    synthea_csv_dir : str
        Directory containing Synthea CSV output files.
    output_dir : str
        Directory where OMOP CDM tables will be saved.
    cdm_version : str
        OMOP CDM version (e.g., "5.3", "5.4"). Default: "5.4"
    synthea_version : str
        Synthea version used (e.g., "2.7.0", "3.0.0"). Default: "3.0.0"
    output_format : str
        Output format: "parquet" or "csv". Default: "parquet"
    concept_mapping : dict, optional
        Optional mapping dictionary for concept codes. If None, uses SNOMED codes
        directly (may need vocabulary lookup for proper concept_ids).
    verbose : bool
        Whether to print progress messages.
    
    Returns
    -------
    dict[str, str]
        Dictionary mapping OMOP table names to their output file paths.
    
    References
    ----------
    - OHDSI ETL-Synthea: https://github.com/OHDSI/ETL-Synthea
    - OMOP CDM: https://ohdsi.github.io/CommonDataModel/
    
    Notes
    -----
    - Requires Synthea CSV export (set exporter.csv.export=true in synthea.properties)
    - Concept mappings may need vocabulary tables for full OMOP compliance
    - Some OMOP tables (e.g., concept, vocabulary) are not generated and must
      be loaded separately from OMOP vocabulary releases
    """
    try:
        import pandas as pd
        import polars as pl
    except ImportError as e:
        raise ImportError(
            "pandas and polars are required for OMOP conversion. "
            "Install with: pip install pandas polars"
        ) from e
    
    synthea_dir = Path(synthea_csv_dir)
    omop_dir = Path(output_dir)
    omop_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Converting Synthea CSV to OMOP CDM v{cdm_version}")
        print(f"Input directory: {synthea_dir}")
        print(f"Output directory: {omop_dir}")
    
    # Track generated files
    output_files = {}
    
    # Helper to log progress
    def log_msg(msg: str):
        if verbose:
            print(f"  {msg}")
    
    # ============================================================
    # 1. PERSON TABLE
    # ============================================================
    log_msg("Converting person table...")
    patients_file = synthea_dir / "patients.csv"
    
    # Create mapping from Synthea IDs (may be UUIDs or integers) to sequential person_ids
    person_id_map = {}
    
    if patients_file.exists():
        patients_df = pd.read_csv(patients_file)
        
        # Synthea IDs may be UUIDs (strings) or integers - handle both
        # Create sequential integer person_ids for OMOP compliance
        patient_ids = patients_df["Id"].astype(str)
        person_ids = range(1, len(patients_df) + 1)
        person_id_map = dict(zip(patient_ids, person_ids))
        
        # Map Synthea patients to OMOP person
        person_df = pd.DataFrame({
            "person_id": list(person_ids),
            "gender_concept_id": patients_df["GENDER"].map({
                "M": 8507,  # Male
                "F": 8532,  # Female
                "OTHER": 8551,  # Other
            }).fillna(value=0),  # Unknown = 0
            "year_of_birth": pd.to_datetime(patients_df["BIRTHDATE"]).dt.year,
            "month_of_birth": pd.to_datetime(patients_df["BIRTHDATE"]).dt.month,
            "day_of_birth": pd.to_datetime(patients_df["BIRTHDATE"]).dt.day,
            "birth_datetime": pd.to_datetime(patients_df["BIRTHDATE"]),
            "race_concept_id": 0,  # Default to 0 (No matching concept)
            "ethnicity_concept_id": 0,  # Default to 0
            "location_id": None,
            "provider_id": None,
            "care_site_id": None,
            "person_source_value": patient_ids.astype(str),
            "gender_source_value": patients_df["GENDER"],
            "gender_source_concept_id": 0,
            "race_source_value": None,
            "race_source_concept_id": 0,
            "ethnicity_source_value": None,
            "ethnicity_source_concept_id": 0,
        })
        
        # Save person table
        output_path = omop_dir / f"person.{output_format}"
        if output_format == "parquet":
            pl.from_pandas(person_df).write_parquet(output_path)
        else:
            person_df.to_csv(output_path, index=False)
        output_files["person"] = str(output_path)
        log_msg(f"  ✓ Created person table: {len(person_df)} rows")
    
    # ============================================================
    # 2. VISIT_OCCURRENCE TABLE
    # ============================================================
    log_msg("Converting visit_occurrence table...")
    encounters_file = synthea_dir / "encounters.csv"
    if encounters_file.exists():
        encounters_df = pd.read_csv(encounters_file)
        
        # Map encounter types to visit concept IDs (simplified mapping)
        visit_type_map = {
            "ambulatory": 9202,  # Outpatient Visit
            "emergency": 9203,   # Emergency Room Visit
            "inpatient": 9201,    # Inpatient Visit
            "wellness": 9202,    # Outpatient Visit
            "urgentcare": 9203,  # Emergency Room Visit
        }
        
        # Map patient IDs to person_ids (handle empty person_id_map)
        if person_id_map:
            encounter_person_ids = encounters_df["PATIENT"].astype(str).map(
                person_id_map
            ).fillna(value=0).astype(int)
        else:
            # If no person_id_map, create sequential IDs
            encounter_person_ids = range(1, len(encounters_df) + 1)
        
        visit_occurrence_df = pd.DataFrame({
            "visit_occurrence_id": range(1, len(encounters_df) + 1),
            "person_id": encounter_person_ids,
            "visit_concept_id": encounters_df["ENCOUNTERCLASS"].map(
                visit_type_map
            ).fillna(value=9202),  # Default to outpatient
            "visit_start_date": pd.to_datetime(encounters_df["START"]).dt.date,
            "visit_start_datetime": pd.to_datetime(encounters_df["START"]),
            "visit_end_date": pd.to_datetime(encounters_df["STOP"]).dt.date,
            "visit_end_datetime": pd.to_datetime(encounters_df["STOP"]),
            "visit_type_concept_id": 32827,  # 'EHR encounter record'
            "provider_id": None,
            "care_site_id": None,
            "visit_source_value": encounters_df["Id"].astype(str),
            "visit_source_concept_id": 0,
            "admitting_source_concept_id": 0,
            "admitting_source_value": None,
            "discharge_to_concept_id": 0,
            "discharge_to_source_value": None,
            "preceding_visit_occurrence_id": None,
        })
        
        output_path = omop_dir / f"visit_occurrence.{output_format}"
        if output_format == "parquet":
            pl.from_pandas(visit_occurrence_df).write_parquet(output_path)
        else:
            visit_occurrence_df.to_csv(output_path, index=False)
        output_files["visit_occurrence"] = str(output_path)
        log_msg(f"  ✓ Created visit_occurrence table: {len(visit_occurrence_df)} rows")
        
        # Create visit_id mapping for later joins
        visit_id_map = dict(zip(
            encounters_df["Id"].astype(str),
            visit_occurrence_df["visit_occurrence_id"]
        ))
    else:
        visit_id_map = {}
    
    # ============================================================
    # 3. CONDITION_OCCURRENCE TABLE
    # ============================================================
    log_msg("Converting condition_occurrence table...")
    conditions_file = synthea_dir / "conditions.csv"
    if conditions_file.exists():
        conditions_df = pd.read_csv(conditions_file)
        
        # Map patient IDs to person_ids (handle empty person_id_map)
        if person_id_map:
            condition_person_ids = conditions_df["PATIENT"].astype(str).map(
                person_id_map
            ).fillna(value=0).astype(int)
        else:
            condition_person_ids = range(1, len(conditions_df) + 1)
        
        # Map to visit_occurrence_id if available
        if visit_id_map:
            condition_visit_ids = conditions_df["ENCOUNTER"].astype(str).map(
                visit_id_map
            )
            # Convert NaN to None for nullable integer type
            condition_visit_ids = condition_visit_ids.mask(condition_visit_ids.isna(), None)
            condition_visit_ids = condition_visit_ids.astype("Int64")
        else:
            condition_visit_ids = pd.Series([None] * len(conditions_df), dtype="Int64")
        
        condition_occurrence_df = pd.DataFrame({
            "condition_occurrence_id": range(1, len(conditions_df) + 1),
            "person_id": condition_person_ids,
            "condition_concept_id": 0,  # Will need vocabulary lookup
            "condition_start_date": pd.to_datetime(conditions_df["START"]).dt.date,
            "condition_start_datetime": pd.to_datetime(conditions_df["START"]),
            "condition_end_date": pd.to_datetime(conditions_df["STOP"]).dt.date,
            "condition_end_datetime": pd.to_datetime(conditions_df["STOP"]),
            "condition_type_concept_id": 32879,  # 'EHR problem list'
            "condition_status_concept_id": 0,
            "stop_reason": None,
            "provider_id": None,
            "visit_occurrence_id": condition_visit_ids,
            "visit_detail_id": None,
            "condition_source_value": conditions_df["CODE"].astype(str),
            "condition_source_concept_id": 0,
            "condition_source_concept_name": conditions_df.get("DESCRIPTION", None),
            "condition_status_source_value": None,
        })
        
        output_path = omop_dir / f"condition_occurrence.{output_format}"
        if output_format == "parquet":
            pl.from_pandas(condition_occurrence_df).write_parquet(output_path)
        else:
            condition_occurrence_df.to_csv(output_path, index=False)
        output_files["condition_occurrence"] = str(output_path)
        log_msg(f"  ✓ Created condition_occurrence table: {len(condition_occurrence_df)} rows")
    
    # ============================================================
    # 4. DRUG_EXPOSURE TABLE
    # ============================================================
    log_msg("Converting drug_exposure table...")
    medications_file = synthea_dir / "medications.csv"
    if medications_file.exists():
        medications_df = pd.read_csv(medications_file)
        
        # Map patient IDs to person_ids (handle empty person_id_map)
        if person_id_map:
            medication_person_ids = medications_df["PATIENT"].astype(str).map(
                person_id_map
            ).fillna(value=0).astype(int)
        else:
            medication_person_ids = range(1, len(medications_df) + 1)
        
        if visit_id_map:
            medication_visit_ids = medications_df["ENCOUNTER"].astype(str).map(
                visit_id_map
            )
            medication_visit_ids = medication_visit_ids.mask(medication_visit_ids.isna(), None)
            medication_visit_ids = medication_visit_ids.astype("Int64")
        else:
            medication_visit_ids = pd.Series([None] * len(medications_df), dtype="Int64")
        
        drug_exposure_df = pd.DataFrame({
            "drug_exposure_id": range(1, len(medications_df) + 1),
            "person_id": medication_person_ids,
            "drug_concept_id": 0,  # Will need vocabulary lookup
            "drug_exposure_start_date": pd.to_datetime(medications_df["START"]).dt.date,
            "drug_exposure_start_datetime": pd.to_datetime(medications_df["START"]),
            "drug_exposure_end_date": pd.to_datetime(medications_df["STOP"]).dt.date,
            "drug_exposure_end_datetime": pd.to_datetime(medications_df["STOP"]),
            "verbatim_end_date": None,
            "drug_type_concept_id": 38000177,  # 'Prescription written'
            "stop_reason": None,
            "refills": None,
            "quantity": None,
            "days_supply": (
                pd.to_datetime(medications_df["STOP"]) - 
                pd.to_datetime(medications_df["START"])
            ).dt.days,
            "sig": medications_df.get("DESCRIPTION", None),
            "route_concept_id": 0,
            "lot_number": None,
            "provider_id": None,
            "visit_occurrence_id": medication_visit_ids,
            "visit_detail_id": None,
            "drug_source_value": medications_df["CODE"].astype(str),
            "drug_source_concept_id": 0,
            "drug_source_concept_name": medications_df.get("DESCRIPTION", None),
            "route_source_value": None,
            "dose_unit_source_value": None,
            "quantity_source_value": None,
        })
        
        output_path = omop_dir / f"drug_exposure.{output_format}"
        if output_format == "parquet":
            pl.from_pandas(drug_exposure_df).write_parquet(output_path)
        else:
            drug_exposure_df.to_csv(output_path, index=False)
        output_files["drug_exposure"] = str(output_path)
        log_msg(f"  ✓ Created drug_exposure table: {len(drug_exposure_df)} rows")
    
    # ============================================================
    # 5. PROCEDURE_OCCURRENCE TABLE
    # ============================================================
    log_msg("Converting procedure_occurrence table...")
    procedures_file = synthea_dir / "procedures.csv"
    if procedures_file.exists():
        procedures_df = pd.read_csv(procedures_file)
        
        # Check for date column (Synthea may use DATE or START)
        date_col = None
        for col in ["DATE", "START"]:
            if col in procedures_df.columns:
                date_col = col
                break
        
        if date_col is None:
            log_msg(f"  ⚠️  Skipping procedure_occurrence: No DATE or START column found")
            log_msg(f"     Available columns: {list(procedures_df.columns)}")
        else:
            # Map patient IDs to person_ids (handle empty person_id_map)
            if person_id_map:
                procedure_person_ids = procedures_df["PATIENT"].astype(str).map(
                    person_id_map
                ).fillna(value=0).astype(int)
            else:
                procedure_person_ids = range(1, len(procedures_df) + 1)
            
            if visit_id_map:
                procedure_visit_ids = procedures_df["ENCOUNTER"].astype(str).map(
                    visit_id_map
                )
                procedure_visit_ids = procedure_visit_ids.mask(procedure_visit_ids.isna(), None)
                procedure_visit_ids = procedure_visit_ids.astype("Int64")
            else:
                procedure_visit_ids = pd.Series([None] * len(procedures_df), dtype="Int64")
            
            procedure_occurrence_df = pd.DataFrame({
                "procedure_occurrence_id": range(1, len(procedures_df) + 1),
                "person_id": procedure_person_ids,
                "procedure_concept_id": 0,  # Will need vocabulary lookup
                "procedure_date": pd.to_datetime(procedures_df[date_col]).dt.date,
                "procedure_datetime": pd.to_datetime(procedures_df[date_col]),
                "procedure_type_concept_id": 32817,  # 'EHR order list'
                "modifier_concept_id": 0,
                "quantity": None,
                "provider_id": None,
                "visit_occurrence_id": procedure_visit_ids,
                "visit_detail_id": None,
                "procedure_source_value": procedures_df["CODE"].astype(str),
                "procedure_source_concept_id": 0,
                "procedure_source_concept_name": procedures_df.get("DESCRIPTION", None),
                "modifier_source_value": None,
            })
            
            output_path = omop_dir / f"procedure_occurrence.{output_format}"
            if output_format == "parquet":
                pl.from_pandas(procedure_occurrence_df).write_parquet(output_path)
            else:
                procedure_occurrence_df.to_csv(output_path, index=False)
            output_files["procedure_occurrence"] = str(output_path)
            log_msg(f"  ✓ Created procedure_occurrence table: {len(procedure_occurrence_df)} rows")
    
    # ============================================================
    # 6. MEASUREMENT TABLE
    # ============================================================
    log_msg("Converting measurement table...")
    observations_file = synthea_dir / "observations.csv"
    if observations_file.exists():
        observations_df = pd.read_csv(observations_file)
        
        # Check for date column (Synthea may use DATE or START)
        date_col = None
        for col in ["DATE", "START"]:
            if col in observations_df.columns:
                date_col = col
                break
        
        if date_col is None:
            log_msg(f"  ⚠️  Skipping measurement: No DATE or START column found in observations.csv")
            log_msg(f"     Available columns: {list(observations_df.columns)}")
        else:
            # Filter for numeric observations (measurements)
            numeric_obs = observations_df[
                pd.to_numeric(observations_df.get("VALUE", ""), errors="coerce").notna()
            ].copy()
            
            if len(numeric_obs) > 0:
                # Map patient IDs to person_ids (handle empty person_id_map)
                if person_id_map:
                    measurement_person_ids = numeric_obs["PATIENT"].astype(str).map(
                        person_id_map
                    ).fillna(value=0).astype(int)
                else:
                    measurement_person_ids = range(1, len(numeric_obs) + 1)
                
                if visit_id_map:
                    measurement_visit_ids = numeric_obs["ENCOUNTER"].astype(str).map(
                        visit_id_map
                    )
                    measurement_visit_ids = measurement_visit_ids.mask(measurement_visit_ids.isna(), None)
                    measurement_visit_ids = measurement_visit_ids.astype("Int64")
                else:
                    measurement_visit_ids = pd.Series([None] * len(numeric_obs), dtype="Int64")
                
                measurement_df = pd.DataFrame({
                    "measurement_id": range(1, len(numeric_obs) + 1),
                    "person_id": measurement_person_ids,
                    "measurement_concept_id": 0,  # Will need vocabulary lookup
                    "measurement_date": pd.to_datetime(numeric_obs[date_col]).dt.date,
                    "measurement_datetime": pd.to_datetime(numeric_obs[date_col]),
                    "measurement_type_concept_id": 32856,  # 'Lab'
                    "operator_concept_id": 0,
                    "value_as_number": pd.to_numeric(numeric_obs.get("VALUE", None), errors="coerce"),
                    "value_as_concept_id": 0,
                    "unit_concept_id": 0,
                    "range_low": None,
                    "range_high": None,
                    "provider_id": None,
                    "visit_occurrence_id": measurement_visit_ids,
                    "visit_detail_id": None,
                    "measurement_source_value": numeric_obs["CODE"].astype(str),
                    "measurement_source_concept_id": 0,
                    "measurement_source_concept_name": numeric_obs.get("DESCRIPTION", None),
                    "unit_source_value": numeric_obs.get("UNITS", None),
                    "value_source_value": numeric_obs.get("VALUE", None),
                })
                
                output_path = omop_dir / f"measurement.{output_format}"
                if output_format == "parquet":
                    pl.from_pandas(measurement_df).write_parquet(output_path)
                else:
                    measurement_df.to_csv(output_path, index=False)
                output_files["measurement"] = str(output_path)
                log_msg(f"  ✓ Created measurement table: {len(measurement_df)} rows")
    
    # ============================================================
    # 7. OBSERVATION TABLE
    # ============================================================
    log_msg("Converting observation table...")
    observations_file = synthea_dir / "observations.csv"
    if observations_file.exists():
        observations_df = pd.read_csv(observations_file)
        
        # Check for date column (Synthea may use DATE or START)
        date_col = None
        for col in ["DATE", "START"]:
            if col in observations_df.columns:
                date_col = col
                break
        
        if date_col is None:
            log_msg(f"  ⚠️  Skipping observation: No DATE or START column found in observations.csv")
            log_msg(f"     Available columns: {list(observations_df.columns)}")
        else:
            # Non-numeric observations go to observation table
            non_numeric_obs = observations_df[
                pd.to_numeric(observations_df.get("VALUE", ""), errors="coerce").isna()
            ].copy()
            
            if len(non_numeric_obs) > 0:
                # Map patient IDs to person_ids (handle empty person_id_map)
                if person_id_map:
                    observation_person_ids = non_numeric_obs["PATIENT"].astype(str).map(
                        person_id_map
                    ).fillna(value=0).astype(int)
                else:
                    observation_person_ids = range(1, len(non_numeric_obs) + 1)
                
                if visit_id_map:
                    observation_visit_ids = non_numeric_obs["ENCOUNTER"].astype(str).map(
                        visit_id_map
                    )
                    observation_visit_ids = observation_visit_ids.mask(observation_visit_ids.isna(), None)
                    observation_visit_ids = observation_visit_ids.astype("Int64")
                else:
                    observation_visit_ids = pd.Series([None] * len(non_numeric_obs), dtype="Int64")
                
                observation_df = pd.DataFrame({
                    "observation_id": range(1, len(non_numeric_obs) + 1),
                    "person_id": observation_person_ids,
                    "observation_concept_id": 0,  # Will need vocabulary lookup
                    "observation_date": pd.to_datetime(non_numeric_obs[date_col]).dt.date,
                    "observation_datetime": pd.to_datetime(non_numeric_obs[date_col]),
                    "observation_type_concept_id": 32856,  # 'Lab'
                    "value_as_number": None,
                    "value_as_string": non_numeric_obs.get("VALUE", None).astype(str),
                    "value_as_concept_id": 0,
                    "qualifier_concept_id": 0,
                    "unit_concept_id": 0,
                    "provider_id": None,
                    "visit_occurrence_id": observation_visit_ids,
                    "visit_detail_id": None,
                    "observation_source_value": non_numeric_obs["CODE"].astype(str),
                    "observation_source_concept_id": 0,
                    "observation_source_concept_name": non_numeric_obs.get("DESCRIPTION", None),
                    "unit_source_value": non_numeric_obs.get("UNITS", None),
                    "qualifier_source_value": None,
                })
                
                output_path = omop_dir / f"observation.{output_format}"
                if output_format == "parquet":
                    pl.from_pandas(observation_df).write_parquet(output_path)
                else:
                    observation_df.to_csv(output_path, index=False)
                output_files["observation"] = str(output_path)
                log_msg(f"  ✓ Created observation table: {len(observation_df)} rows")
    
    # ============================================================
    # 8. DEATH TABLE
    # ============================================================
    log_msg("Converting death table...")
    patients_file = synthea_dir / "patients.csv"
    if patients_file.exists():
        patients_df = pd.read_csv(patients_file)
        # Check for deaths (patients with DEATHDATE)
        patients_with_death = patients_df[patients_df["DEATHDATE"].notna()].copy()
        
        if len(patients_with_death) > 0:
            # Map patient IDs to person_ids (handle empty person_id_map)
            if person_id_map:
                death_person_ids = patients_with_death["Id"].astype(str).map(
                    person_id_map
                ).fillna(value=0).astype(int)
            else:
                death_person_ids = range(1, len(patients_with_death) + 1)
            
            death_df = pd.DataFrame({
                "person_id": death_person_ids,
                "death_date": pd.to_datetime(patients_with_death["DEATHDATE"]).dt.date,
                "death_datetime": pd.to_datetime(patients_with_death["DEATHDATE"]),
                "death_type_concept_id": 32815,  # 'EHR record'
                "cause_concept_id": 0,
                "cause_source_value": None,
                "cause_source_concept_id": 0,
            })
            
            output_path = omop_dir / f"death.{output_format}"
            if output_format == "parquet":
                pl.from_pandas(death_df).write_parquet(output_path)
            else:
                death_df.to_csv(output_path, index=False)
            output_files["death"] = str(output_path)
            log_msg(f"  ✓ Created death table: {len(death_df)} rows")
    
    # ============================================================
    # 9. OBSERVATION_PERIOD TABLE
    # ============================================================
    log_msg("Converting observation_period table...")
    patients_file = synthea_dir / "patients.csv"
    encounters_file = synthea_dir / "encounters.csv"
    if patients_file.exists() and encounters_file.exists():
        patients_df = pd.read_csv(patients_file)
        encounters_df = pd.read_csv(encounters_file)
        # Create observation periods from first to last encounter (or death)
        person_periods = []
        # Get person_df if it was created earlier
        if "person" in output_files:
            person_df = pd.read_csv(output_files["person"]) if output_format == "csv" else pl.read_parquet(output_files["person"]).to_pandas()
        else:
            # Create minimal person_df from patients using person_id_map
            person_df = pd.DataFrame({"person_id": list(person_id_map.values())})
        
        for person_id in person_df["person_id"]:
            # Find the original patient ID for this person_id
            original_patient_id = [k for k, v in person_id_map.items() if v == person_id][0]
            
            # Look up encounters using the original patient ID (may be UUID)
            person_encounters = encounters_df[encounters_df["PATIENT"].astype(str) == original_patient_id]
            if len(person_encounters) > 0:
                start_date = pd.to_datetime(person_encounters["START"]).min()
                
                # End date is death date if available, otherwise last encounter
                person_data = patients_df[patients_df["Id"].astype(str) == original_patient_id].iloc[0]
                if pd.notna(person_data.get("DEATHDATE")):
                    end_date = pd.to_datetime(person_data["DEATHDATE"])
                else:
                    end_date = pd.to_datetime(person_encounters["STOP"]).max()
                
                person_periods.append({
                    "observation_period_id": len(person_periods) + 1,
                    "person_id": person_id,
                    "observation_period_start_date": start_date.date(),
                    "observation_period_end_date": end_date.date(),
                    "period_type_concept_id": 44814724,  # 'Period covering healthcare encounters'
                })
        
        if person_periods:
            observation_period_df = pd.DataFrame(person_periods)
            
            output_path = omop_dir / f"observation_period.{output_format}"
            if output_format == "parquet":
                pl.from_pandas(observation_period_df).write_parquet(output_path)
            else:
                observation_period_df.to_csv(output_path, index=False)
            output_files["observation_period"] = str(output_path)
            log_msg(f"  ✓ Created observation_period table: {len(observation_period_df)} rows")
    
    # ============================================================
    # 10. CDM_SOURCE TABLE (Metadata)
    # ============================================================
    log_msg("Creating cdm_source table...")
    cdm_source_df = pd.DataFrame([{
        "cdm_source_name": "Synthea Synthetic Data",
        "cdm_source_abbreviation": "SYNTHEA",
        "cdm_holder": "Synthea",
        "source_description": f"Synthetic patient data generated by Synthea {synthea_version}",
        "source_documentation_reference": "https://github.com/synthetichealth/synthea",
        "cdm_etl_reference": "https://github.com/OHDSI/ETL-Synthea",
        "source_release_date": datetime.now().date(),
        "cdm_release_date": datetime.now().date(),
        "cdm_version": cdm_version,
        "cdm_version_concept_id": 0,  # Would need vocabulary lookup
        "vocabulary_version": "Unknown",
    }])
    
    output_path = omop_dir / f"cdm_source.{output_format}"
    if output_format == "parquet":
        pl.from_pandas(cdm_source_df).write_parquet(output_path)
    else:
        cdm_source_df.to_csv(output_path, index=False)
    output_files["cdm_source"] = str(output_path)
    log_msg(f"  ✓ Created cdm_source table")
    
    if verbose:
        print(f"\n✓ Conversion complete!")
        print(f"  Generated {len(output_files)} OMOP tables")
        print(f"  Output directory: {omop_dir}")
        print("\n⚠️  Note: concept_id fields are set to 0 and require vocabulary lookup")
        print("  for full OMOP compliance. Use source_value fields for concept mapping.")
        print("\n💡 To get concept names:")
        print("  1. Load OMOP vocabulary/concept table")
        print("  2. Join using *_source_value fields with concept.concept_code")
        print("  3. Or use add_concept_names() helper function")
    
    return output_files


def add_concept_names(
    omop_table: pd.DataFrame,
    concept_table: pd.DataFrame,
    source_value_col: str,
    concept_code_col: str = "concept_code",
    concept_name_col: str = "concept_name",
    vocabulary_id_col: str = "vocabulary_id",
    vocabulary_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add concept names to an OMOP table by joining with concept table.
    
    This function joins an OMOP table with a concept table using the source_value
    fields to look up concept names. Useful for adding human-readable names to
    Synthea-generated OMOP tables.
    
    Parameters
    ----------
    omop_table : pd.DataFrame
        OMOP table (e.g., condition_occurrence, drug_exposure) with source_value columns.
    concept_table : pd.DataFrame
        OMOP concept table with columns: concept_code, concept_name, vocabulary_id, etc.
    source_value_col : str
        Name of the source_value column in omop_table (e.g., "condition_source_value").
    concept_code_col : str
        Name of the concept_code column in concept_table. Default: "concept_code".
    concept_name_col : str
        Name of the concept_name column in concept_table. Default: "concept_name".
    vocabulary_id_col : str
        Name of the vocabulary_id column in concept_table. Default: "vocabulary_id".
    vocabulary_id : str, optional
        Filter concept_table to specific vocabulary (e.g., "SNOMED"). If None, uses all.
    
    Returns
    -------
    pd.DataFrame
        Original omop_table with added concept_name column.
    
    Example
    -------
    >>> import pandas as pd
    >>> # Load your OMOP tables
    >>> conditions = pd.read_parquet("omop/condition_occurrence.parquet")
    >>> concepts = pd.read_parquet("omop_vocab/concept.parquet")  # Or load from OMOP vocabulary
    >>> 
    >>> # Add concept names
    >>> conditions_with_names = add_concept_names(
    ...     omop_table=conditions,
    ...     concept_table=concepts,
    ...     source_value_col="condition_source_value",
    ...     vocabulary_id="SNOMED"
    ... )
    >>> 
    >>> # Now you can see condition names
    >>> print(conditions_with_names[["condition_source_value", "concept_name"]].head())
    """
    # Filter concept table by vocabulary if specified
    if vocabulary_id is not None:
        concept_subset = concept_table[
            concept_table[vocabulary_id_col] == vocabulary_id
        ].copy()
    else:
        concept_subset = concept_table.copy()
    
    # Create lookup dictionary: concept_code -> concept_name
    # Handle potential duplicates by taking first match
    concept_lookup = concept_subset.groupby(concept_code_col)[concept_name_col].first().to_dict()
    
    # Map source values to concept names
    omop_table = omop_table.copy()
    omop_table["concept_name"] = omop_table[source_value_col].astype(str).map(concept_lookup)
    
    return omop_table


# Convenience function for backward compatibility
def download_synthea_jar(
    jar_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download Synthea JAR file.
    
    Parameters
    ----------
    jar_path : str, optional
        Desired path for JAR file. Not currently used.
    cache_dir : str, optional
        Directory to cache JAR. Defaults to pooch's OS cache.
    
    Returns
    -------
    str
        Path to downloaded JAR file.
    """
    _ = jar_path  # Unused parameter for API compatibility
    runner = SyntheaRunner(cache_dir=cache_dir)
    return runner.jar_path


# Example usage
if __name__ == "__main__":
    # Quick example
    runner = SyntheaRunner()
    
    # Generate 10 patients in Massachusetts
    result = runner.run_quick(
        population_size=10,
        state="Massachusetts",
        seed=42,
        output_dir="synthea_output"
    )
    
    print(f"\nSimulation completed with return code: {result['returncode']}")
    print(f"Output directory: {result['output_dir']}")
