# Existing content of README.md

---

## 📦 Pretrained Model Checkpoints & Zenodo Archive

> **Notice:**  
> This GitHub repository contains all code, data processing scripts, and Jupyter notebooks for the study.  
> **Model checkpoint files for Random Forest (RF), CNN, and RGCN are NOT included here due to GitHub file size limitations.**  
> To access all pretrained model checkpoints and supplementary files, visit our Zenodo archive:  
> [https://doi.org/10.5281/zenodo.17104898](https://doi.org/10.5281/zenodo.17104898)

### How to Use Pretrained Models

- Download the model checkpoint files from Zenodo.
- Place them in the correct `model_checkpoints/` folders under `Random_Forest/`, `CNN/`, or `RGCN/` as needed.
- This enables exact reproducibility and pretrained model loading in the provided notebooks.

### What is on Zenodo?

- All files available in this GitHub repo
- **PLUS:**  
  - All trained model checkpoints for Random Forest, CNN, and RGCN (see each method’s `model_checkpoints/` directory)
  - Any additional large supplementary files

---

## 📑 Pip Requirements

For pip users, a unified `requirements.txt` is provided for convenience:

```bash
pip install -r requirements.txt
```

*Note: For RDKit, `conda install -c rdkit rdkit` is generally preferred, but `rdkit-pypi` is included for pip compatibility.*

---

## 📬 Contact

- For questions, bug reports, or collaboration, please open a GitHub issue or contact the authors directly.

---