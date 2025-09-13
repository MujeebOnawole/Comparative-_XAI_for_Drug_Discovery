# Explainable AI for Bioactivity Prediction: Multi-Architecture Comparative Study

## 📋 Overview

This repository contains the complete codebase and pre-trained models for a comprehensive comparative study of explainable AI approaches in bioactivity prediction against *Staphylococcus aureus*. The research implements and compares three distinct machine learning architectures: **Random Forest**, **Convolutional Neural Networks (CNN)**, and **Relational Graph Convolutional Networks (RGCN)**, each with specialized explainability methods.

### 🎯 Purpose & Applications

- **Research Reproducibility**: Complete pipelines for training, validation, and testing of bioactivity prediction models
- **Model Comparison**: Side-by-side evaluation of three complementary ML approaches with different molecular representations
- **Explainability Analysis**: Implementation of LIME, Grad-CAM, and substructure masking for model interpretation
- **Interactive Testing**: Ready-to-use Jupyter notebooks for immediate model testing and visualization
- **Educational Resource**: Well-documented implementations for learning XAI in drug discovery

## 📊 Dataset

**File**: `S_aureus.csv` (43,777 compounds)
- **Source**: Curated bioactivity data against *Staphylococcus aureus*
- **Features**: Chemical structures (SMILES), bioactivity labels, molecular descriptors
- **Splits**: Pre-stratified training/validation/test sets for reproducible evaluation
- **Target**: Binary classification (Active/Inactive antimicrobial compounds)

### Key Dataset Features:
- **Molecular Properties**: MW, LogP, PSA, Lipinski descriptors
- **Activity Data**: MIC values, standardized activity classification
- **Chemical Diversity**: Broad coverage of chemical space including quinolones, β-lactams, and other antibiotic classes
- **Quality Control**: Standardized SMILES, validated structures, consistent activity thresholds

## 🏗️ Architecture Overview

| Model | Molecular Representation | Explainability Method | Key Strengths |
|-------|-------------------------|----------------------|---------------|
| **Random Forest** | Fragment Descriptors | LIME | Interpretable features, fast training |
| **CNN** | SMILES Sequences | Grad-CAM | Sequence patterns, character-level attention |
| **RGCN** | Molecular Graphs | Substructure Masking | Relational information, chemical intuition |

## 📁 Repository Structure

```
├── S_aureus.csv                    # Main dataset (43,777 compounds)
├── Random_Forest/                  # Random Forest implementation
│   ├── RF_LIME_Visualization.ipynb    # 🎯 Interactive LIME explanations
│   ├── descriptor.py                  # RDKit fragment descriptor generation  
│   ├── RF_CV.py                      # Cross-validation pipeline
│   ├── ML_SA.py                      # Multi-algorithm comparison
│   ├── RF_test_evaluation.py         # Final evaluation & ensemble
│   ├── model_checkpoints/            # Trained RF models & scalers
│   ├── Functional_Group_mapping.txt  # Fragment interpretation guide
│   └── SA_FG_fragments.csv          # Generated molecular descriptors
│
├── CNN/                           # CNN implementation  
│   ├── CNN_Grad_CAM_visual.ipynb     # 🎯 Interactive Grad-CAM visualization
│   ├── main.py                       # Full training pipeline
│   ├── model.py                      # CNN architecture definition
│   ├── data_preprocessing.py         # SMILES tokenization & augmentation
│   ├── hyperparameter_opt.py         # Optuna optimization
│   ├── cross_validation.py           # K-fold cross-validation
│   ├── model_checkpoints/            # Trained CNN models (.ckpt)
│   ├── config.py                     # Model configuration
│   └── best_models.json              # Cross-validation results
│
├── RGCN/                          # RGCN implementation
│   ├── RGCN_SME_Visualization.ipynb  # 🎯 Interactive substructure masking
│   ├── model.py                      # RGCN architecture definition  
│   ├── build_data.py                 # Molecular graph construction
│   ├── RGCN_CV.py                    # Cross-validation pipeline
│   ├── hyper_RGCN.py                 # Hyperparameter optimization
│   ├── rgcn_test_evaluation.py       # Final evaluation
│   ├── model_checkpoints/            # Trained RGCN models (.ckpt)
│   ├── config.py                     # Model configuration
│   └── ensemble_classification_attribution.csv  # Attribution results
│
└── README.md                      # This file
```

## 🚀 Quick Start: Interactive Notebooks

Each architecture includes a ready-to-use Jupyter notebook for immediate testing and visualization:

### 1️⃣ Random Forest + LIME
```bash
# Navigate to Random Forest folder
cd Random_Forest
jupyter notebook RF_LIME_Visualization.ipynb
```
**Features**:
- LIME-based explanations for molecular fragments
- Interactive visualization of feature importance  
- Ensemble predictions from cross-validation models
- Export explanations to CSV for further analysis

### 2️⃣ CNN + Grad-CAM  
```bash
# Navigate to CNN folder
cd CNN  
jupyter notebook CNN_Grad_CAM_visual.ipynb
```
**Features**:
- Grad-CAM heatmaps over SMILES characters
- Attention visualization for sequence patterns
- Functional group impact analysis
- High-resolution molecular visualizations

### 3️⃣ RGCN + Substructure Masking
```bash
# Navigate to RGCN folder  
cd RGCN
jupyter notebook RGCN_SME_Visualization.ipynb
```
**Features**:
- Murcko scaffold attribution analysis
- Interactive substructure highlighting
- Graph-based molecular explanations  
- Chemical intuition through structural masking

## 🔧 Environment Setup

### Option 1: Individual Environments (Recommended)

**Random Forest Environment**:
```bash
conda create -n xai-rf python=3.10 -y
conda activate xai-rf
conda install -c rdkit rdkit -y
pip install scikit-learn joblib pandas numpy matplotlib seaborn lime
```

**CNN Environment**:
```bash
conda create -n xai-cnn python=3.10 -y  
conda activate xai-cnn
pip install torch torchvision torchaudio
pip install pytorch-lightning torchmetrics optuna rdkit-pypi pandas numpy scikit-learn matplotlib
```

**RGCN Environment**:
```bash
conda create -n xai-rgcn python=3.10 -y
conda activate xai-rgcn  
pip install torch torchvision torchaudio
pip install pytorch-lightning torchmetrics rdkit-pypi pandas numpy scikit-learn
pip install torch-geometric
```

### Option 2: Unified Environment
```bash
conda create -n xai-bioactivity python=3.10 -y
conda activate xai-bioactivity
conda install -c rdkit rdkit -y
pip install torch torchvision torchaudio torch-geometric
pip install pytorch-lightning torchmetrics optuna scikit-learn lime
pip install pandas numpy matplotlib seaborn joblib
```

## 💻 Model Training (Optional)

All notebooks include pre-trained models, but you can retrain from scratch:

### Random Forest Pipeline
```bash
cd Random_Forest
python descriptor.py          # Generate RDKit descriptors
python RF_CV.py              # 5×5 cross-validation  
python ML_SA.py              # Multi-algorithm comparison
python RF_test_evaluation.py # Final ensemble evaluation
```

### CNN Pipeline
```bash
cd CNN  
python main.py --trials 50 --seed 42
# Outputs: model_checkpoints/, hyperparameter_results.csv, cross_validation_results.csv
```

### RGCN Pipeline  
```bash
cd RGCN
python RGCN_CV.py           # Cross-validation
python hyper_RGCN.py        # Hyperparameter optimization (optional)
python rgcn_test_evaluation.py  # Final evaluation
```



## 🔍 Explainability Methods

### LIME (Random Forest)
- **Method**: Local surrogate model explanations
- **Granularity**: Individual molecular fragments
- **Output**: Feature importance scores, fragment contributions
- **Use Case**: Understanding which chemical fragments drive predictions

### Grad-CAM (CNN)
- **Method**: Gradient-weighted class activation mapping
- **Granularity**: SMILES character positions  
- **Output**: Attention heatmaps, sequence highlighting
- **Use Case**: Identifying important sequence patterns and motifs

### Substructure Masking (RGCN)
- **Method**: Structural perturbation analysis
- **Granularity**: Murcko scaffolds and functional groups
- **Output**: Attribution scores, structural highlighting  
- **Use Case**: Chemical intuition through structural relationships

## 📋 Data Format Requirements

### Input Data Structure
```csv
COMPOUND_ID,PROCESSED_SMILES,TARGET,group
CHEMBL123,CC(=O)OC1=CC=CC=C1C(=O)O,1,training
CHEMBL456,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0,validation
```

### Required Columns:
- `COMPOUND_ID`: Unique identifier
- `PROCESSED_SMILES`: Canonical SMILES representation
- `TARGET`: Binary activity label (0/1)  
- `group`: Data split (`training`/`valid`/`test`)

## 🔄 Reproducibility

- **Deterministic Training**: Fixed random seeds across all experiments
- **Cross-Validation**: Consistent fold splits for fair comparison
- **Version Control**: Pinned dependency versions in environments
- **Model Checkpoints**: Saved models enable exact result reproduction
- **Configuration Files**: All hyperparameters documented and version-controlled

## 📊 Output Files & Results

### Random Forest
- `model_checkpoints/*.ckpt`: Cross-validation model checkpoints

### CNN  
- `model_checkpoints/*.ckpt`: PyTorch Lightning checkpoints


### RGCN
- `model_checkpoints/*.ckpt`: Trained graph neural networks  
tion analysis
- Cross-validation logs and metrics

## 🎓 Educational Use

This repository serves as a comprehensive educational resource for:
- **XAI in Drug Discovery**: Practical implementation of explainable AI methods
- **Multi-Modal ML**: Comparison of different molecular representations
- **Bioactivity Prediction**: End-to-end pipeline from data to interpretation
- **Reproducible Research**: Best practices in computational chemistry

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{xai-bioactivity-2024,
  title={Bridging the Explainability Gap in AI-Driven Drug Discovery: A Systematic Comparison of Interpretable Methods for Antimicrobial Prediction},
  author={[Abdulmujeeb T.Onawole, Mark A.T. Blaskovich and Johannes Zuegg ]},
  year={2025},
  publisher={Zenodo},
  doi={[DOI will be added upon publication]}
}
```

### Key Dependencies
- **RDKit**: Cheminformatics toolkit  
- **PyTorch & PyTorch Lightning**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Scikit-learn**: Machine learning algorithms
- **LIME**: Model-agnostic explanations
- **Optuna**: Hyperparameter optimization

## 🤝 Contributing & Support

- **Issues**: Report bugs or request features via GitHub issues
- **Documentation**: Comprehensive docstrings and comments throughout code  
- **Notebooks**: Step-by-step tutorials with detailed explanations
- **Contact**: group-blaskovich@imb.uq.edu.au for questions and collaboration


