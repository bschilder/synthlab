# Conda Environment for SynthLab

This directory contains the conda environment configuration for SynthLab.

## Creating the Environment

To create the conda environment:

```bash
cd /path/to/synthlab
conda env create -f envs/environment.yml
```

## Activating the Environment

```bash
conda activate medagent
```

## Installing SynthLab in Development Mode

After activating the environment:

```bash
cd /path/to/synthlab
pip install -e .
```

## Updating the Environment

If you modify `environment.yml`, update the environment with:

```bash
conda env update -f envs/environment.yml --prune
```

## Removing the Environment

```bash
conda env remove -n medagent
```
