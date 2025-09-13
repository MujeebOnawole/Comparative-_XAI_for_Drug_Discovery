# Explainable AI for Bioactivity Prediction: Multi-Architecture Comparative Study

## 📋 Overview

This repository contains the complete codebase for a comparative study of explainable AI approaches in bioactivity prediction against *Staphylococcus aureus*. The project implements and compares three distinct machine learning architectures: **Random Forest**, **Convolutional Neural Networks (CNN)**, and **Relational Graph Convolutional Networks (RGCN)**, each with specialized explainability methods.

### 🎯 Purpose & Applications

- **Research Reproducibility**: Full pipelines for training, validation, and testing of bioactivity prediction models
- **Model Comparison**: Side-by-side evaluation of three complementary ML approaches
- **Explainability Analysis**: Implementation of LIME, Grad-CAM, and substructure masking
- **Interactive Testing**: Ready-to-use Jupyter notebooks for model testing and visualization
- **Educational Resource**: Well-documented code for learning XAI in drug discovery

## 📊 Dataset

**Input**: `S_aureus.csv`  
- This file contains curated bioactivity data for 43,777 compounds against *Staphylococcus aureus*.
- **Note:** `S_aureus.csv` is **not included in this repository** due to file size.  
- **Access:** The dataset is available via Zenodo: [https://doi.org/10.5281/zenodo.17104898](https://doi.org/10.5281/zenodo.17104898)
- This CSV is used as the input to train all three model architectures.

### Key Dataset Features:
- **Molecular Properties**: MW, LogP, PSA, Lipinski descriptors
- **Activity Data**: MIC values, standardized activity classification
- **Chemical Diversity**: Broad chemical coverage
- **Quality Control**: Standardized SMILES, validated structures, consistent activity thresholds

## 🏗️ Architecture Overview

| Model           | Molecular Representation | Explainability Method  | Key Strengths                        |
|-----------------|-------------------------|-----------------------|--------------------------------------|
| Random Forest   | Fragment Descriptors    | LIME                  | Interpretable features, fast training|
| CNN             | SMILES Sequences        | Grad-CAM              | Sequence patterns, character-level attention |
| RGCN            | Molecular Graphs        | Substructure Masking  | Relational information, chemical intuition |

## 📁 Repository Structure

```
├── Random_Forest/                  # Random Forest implementation
│   ├── RF_LIME_Visualization.ipynb
│   ├── descriptor.py
│   ├── RF_CV.py
│   ├── ML_SA.py
│   ├── RF_test_evaluation.py
│   ├── Functional_Group_mapping.txt
│   └── SA_FG_fragments.csv
│
├── CNN/                           
│   ├── CNN_Grad_CAM_visual.ipynb
│   ├── main.py
│   ├── model.py
│   ├── data_preprocessing.py
│   ├── hyperparameter_opt.py
│   ├── cross_validation.py
│   ├── config.py
│   └── best_models.json
│
├── RGCN/                         
│   ├── RGCN_SME_Visualization.ipynb
│   ├── model.py
│   ├── build_data.py
│   ├── RGCN_CV.py
│   ├── hyper_RGCN.py
│   ├── rgcn_test_evaluation.py
│   ├── config.py
│   └── ensemble_classification_attribution.csv
│
└── README.md                      # This file
```

> **Note:** All code files are provided directly in the repository (not zipped) for easy browsing.  
> **Pre-trained models, large output files, and the input dataset** are available on Zenodo: [https://doi.org/10.5281/zenodo.17104898](https://doi.org/10.5281/zenodo.17104898)

## 🚀 Quick Start: Interactive Notebooks

Each model folder includes a ready-to-use Jupyter notebook for immediate testing and visualization.

### 1️⃣ Random Forest + LIME
```bash
cd Random_Forest
jupyter notebook RF_LIME_Visualization.ipynb
```
- LIME-based explanations for molecular fragments
- Interactive feature importance visualization

### 2️⃣ CNN + Grad-CAM  
```bash
cd CNN
jupyter notebook CNN_Grad_CAM_visual.ipynb
```
- Grad-CAM heatmaps over SMILES
- Attention visualization for sequence patterns

### 3️⃣ RGCN + Substructure Masking
```bash
cd RGCN
jupyter notebook RGCN_SME_Visualization.ipynb
```
- Murcko scaffold attribution analysis
- Interactive substructure highlighting

## 🔧 Environment Setup

<... keep as in your original ...>

## 💻 Model Training (Optional)

All notebooks can be retrained from scratch using the scripts provided in each folder.

## 📋 Data Format Requirements

### Input Data Structure
```csv
COMPOUND_ID,PROCESSED_SMILES,TARGET,group
CHEMBL123,CC(=O)OC1=CC=CC=C1C(=O)O,1,training
CHEMBL456,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0,validation
```

## 🔄 Reproducibility

- Deterministic training (fixed random seeds)
- Consistent data splits
- Pinned dependency versions

## 📊 Output Files & Results

- **Pre-trained model checkpoints, large output files, and the input dataset are available on Zenodo:**  
  [https://doi.org/10.5281/zenodo.17104898](https://doi.org/10.5281/zenodo.17104898)

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{xai-bioactivity-2025,
  title={Bridging the Explainability Gap in AI-Driven Drug Discovery: A Systematic Comparison of Interpretable Methods for Antimicrobial Prediction},
  author={Abdulmujeeb T. Onawole, Mark A. T. Blaskovich, Johannes Zuegg},
  year={2025},
  note={Working paper},
  publisher={Zenodo},
  doi={10.5281/zenodo.17104898}
}
```


