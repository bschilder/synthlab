# Conda Environment for SynthLab

This directory contains the conda environment configuration for SynthLab.

## Creating the Environment

To create the conda environment:

```bash
cd /home/schilder/projects/synthlab
conda env create -f conda/synthlab.yml
```

## Activating the Environment

```bash
conda activate synthlab
```

## Installing SynthLab in Development Mode

After activating the environment:

```bash
cd /home/schilder/projects/synthlab
pip install -e .
```

## Updating the Environment

If you modify `synthlab.yml`, update the environment with:

```bash
conda env update -f conda/synthlab.yml --prune
```

## Removing the Environment

```bash
conda env remove -n synthlab
```
