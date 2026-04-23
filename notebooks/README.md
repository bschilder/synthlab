# SynthLab notebooks

Demos and walkthroughs for the [SynthLab](https://github.com/bschilder/synthlab)
synthetic-healthcare-data toolkit.

## Index

| Notebook | Module | Description |
| -------- | ------ | ----------- |
| [`olink_demo.ipynb`](olink_demo.ipynb) | [`synthlab.olink`](../synthlab/olink.py) | Olink NPX proteomics simulator — generate, analyse, visualise case vs control with LOD missingness + plate batch effects. |
| [`Synthea.ipynb`](Synthea.ipynb) | [`synthlab.synthea`](../synthlab/synthea.py) | Run Synthea and convert CSV to OMOP. |
| [`Coherent_MultimodalDataset.ipynb`](Coherent_MultimodalDataset.ipynb) | [`synthlab.coherent`](../synthlab/coherent.py) | Load and explore the Synthea Coherent multimodal dataset. |
| [`Generate_MultimodalDataset.ipynb`](Generate_MultimodalDataset.ipynb) | [`synthlab.coherent`](../synthlab/coherent.py) | Generate multimodal synthetic cohorts. |
| [`MedGemma_SOAP_Notes.ipynb`](MedGemma_SOAP_Notes.ipynb) | [`synthlab.soap`](../synthlab/soap.py) | MedGemma-based SOAP note generation with causal graph analysis. |
| [`SNOMED_Entity_Linking.ipynb`](SNOMED_Entity_Linking.ipynb) | [`synthlab.snomed`](../synthlab/snomed.py) | SNOMED entity linking with SapBERT and FAISS. |
| [`UKBiobank_Synthetic.ipynb`](UKBiobank_Synthetic.ipynb) | [`synthlab.download_ukbiobank_synthetic`](../synthlab/download_ukbiobank_synthetic.py) | Download and explore the UK Biobank Synthetic Dataset. |

## Running the Olink demo

The Olink NPX demo notebook needs the `viz` optional dependency group
(`matplotlib`, `seaborn`, `scikit-learn`, `umap-learn`). Install alongside the
core SynthLab package:

```bash
pip install 'synthlab[viz]'
```

Then open [`olink_demo.ipynb`](olink_demo.ipynb) in Jupyter — it runs
end-to-end in under a minute on CPU with no external data dependencies.

## Reproducing `olink_demo.ipynb`

The notebook is assembled from
[`_build_olink_demo.py`](_build_olink_demo.py) so the cell layout stays
reviewable under version control. After editing the builder:

```bash
python notebooks/_build_olink_demo.py                             # rebuild cells
jupyter nbconvert --to notebook --execute --inplace \\
    notebooks/olink_demo.ipynb                                    # embed outputs
```
