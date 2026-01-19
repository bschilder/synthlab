#!/usr/bin/env python3
"""
Basic usage example for SynthLab.

This example demonstrates how to:
1. Generate synthetic patient data using Synthea
2. Convert the output to OMOP CDM format
"""

from synthlab import SyntheaRunner, SyntheaConfig, convert_synthea_to_omop


def main():
    print("=" * 60)
    print("SynthLab Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create a Synthea runner
    print("\n1. Initializing Synthea runner...")
    runner = SyntheaRunner()
    print(f"   Synthea JAR: {runner.jar_path}")
    
    # Step 2: Configure a simulation
    print("\n2. Configuring simulation...")
    config = SyntheaConfig(
        population_size=50,
        state="Massachusetts",
        seed=42,
        output_dir="output/example_run"
    )
    print(f"   Population size: {config.population_size}")
    print(f"   State: {config.state}")
    print(f"   Seed: {config.seed}")
    
    # Step 3: Run the simulation
    print("\n3. Running Synthea simulation...")
    result = runner.run(config, verbose=True)
    
    if result['returncode'] == 0:
        print(f"\n✓ Simulation completed successfully!")
        print(f"   Output directory: {result['output_dir']}")
        
        # Step 4: Convert to OMOP CDM
        print("\n4. Converting to OMOP CDM format...")
        omop_files = convert_synthea_to_omop(
            synthea_csv_dir=result['output_dir'],
            output_dir=f"{result['output_dir']}/omop",
            cdm_version="5.4",
            output_format="parquet",
            verbose=True
        )
        
        print(f"\n✓ Conversion complete!")
        print(f"   Generated {len(omop_files)} OMOP tables")
        print(f"   OMOP directory: {result['output_dir']}/omop")
    else:
        print(f"\n✗ Simulation failed with return code: {result['returncode']}")
        if result['stderr']:
            print(f"   Error: {result['stderr']}")


if __name__ == "__main__":
    main()
