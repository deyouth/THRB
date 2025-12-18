# THRB Binary Classification Model Project

This is a machine learning project for predicting compound activity against Thyroid Hormone Receptor Beta (THRB). Based on the ChEMBL database, the project uses **6 algorithms** (including traditional machine learning and Graph Neural Networks GNN) to build binary classification prediction models.

## Project Overview

### Core Features
- **Objective**: Predict activity of unknown compounds against THRB (active/inactive)
- **Dataset**: ~5400 THRB-related compound data (filtered from 81000+ records)
- **Models**: 6 algorithms (5 traditional ML + 1 GNN)
- **Performance**: ROC AUC 0.75-0.85+
- **Activity Definition**: pchembl_value ≥ **6.0** defines active compounds

## 📁 Project Structure

```
THRB/
│
├── Core Scripts
│   ├── main.py                     # Main entry point (traditional ML pipeline)
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── feature_extraction.py       # Feature extraction
│   ├── model_training.py           # Model training (6 models)
│   ├── model_evaluation.py         # Model evaluation and visualization
│   ├── model_gnn.py               # GNN model implementation ⭐
│   ├── predict.py                  # Prediction module
│   └── test_gnn.py                # GNN environment testing ⭐
│
├── Data Files
│   ├── nr_activities.csv           # Raw dataset (81000+ records)
│   ├── data/
│   │   ├── thrb_processed.csv      # Processed data (~5400 records)
│   │   ├── data_statistics.txt     # Data statistics
│   │   ├── features_combined.npz   # Combined features (2058 dims)
│   │   └── features_morgan.npz     # Morgan fingerprints (2048 dims)
│
├── Model Files
│   ├── models/
│   │   ├── best_model.pkl          # Best model
│   │   ├── scaler.pkl              # Feature scaler
│   │   ├── model_random_forest.pkl
│   │   ├── model_xgboost.pkl
│   │   ├── model_gradient_boosting.pkl
│   │   ├── model_svm.pkl
│   │   ├── model_logistic_regression.pkl
│   │   ├── model_thrb_gnn.pkl     # GNN model ⭐
│   │   ├── smiles_test.npy        # Test set SMILES ⭐
│   │   ├── model_comparison.csv    # Performance comparison
│   │   └── evaluation_report.txt   # Detailed report
│
├── Results Files
│   └── results/
│       ├── roc_curves.png          # ROC curves (6 models)
│       ├── precision_recall_curves.png
│       ├── confusion_matrices.png  # Confusion matrices (6 models)
│       ├── model_comparison.png
│       ├── prediction_distributions.png
│        classification_reports.txt
│
└── Documentation and Configuration
    ├── README.md                   # This document
    └── requirements.txt            # Python dependencies
```


### Model Architecture

#### Traditional Machine Learning Models

**Random Forest**
```python
n_estimators=200
max_depth=20
min_samples_split=5
class_weight='balanced'  # Handle imbalance
```

**XGBoost**
```python
n_estimators=200
max_depth=6
learning_rate=0.1
subsample=0.8
scale_pos_weight=2.6  # Handle imbalance
```

**Class Imbalance Handling**:
- SMOTE oversampling
- Class weight adjustment
- From 3938:1489 → Balanced dataset

**GNN Hyperparameters**:
```python
hidden_dim=128        # Hidden dimension
num_epochs=100        # Number of epochs
batch_size=32         # Batch size
learning_rate=0.001   # Learning rate
dropout=0.3           # Dropout ratio
```

### Complete Modeling Pipeline

```bash
python main.py full
```

#### Step-by-Step Execution

```bash
# Step 1: Data preprocessing
python main.py preprocess
# Or: python data_preprocessing.py

# Step 2: Feature extraction
python main.py extract
# Or: python feature_extraction.py

# Step 3: Model training
python main.py train
# Or: python model_training.py

# Step 4: Model evaluation
python main.py evaluate
# Or: python model_evaluation.py
```

#### Command Line Prediction

### 1. Predict single compound using best model
```bash
python predict.py --smiles "CCOc1ccc(C2NC(=O)NC2=O)cc1"
```
### 2. Predict using GNN model
```bash
python predict.py --model gnn --smiles "Cc1ccc(O)cc1"
```
### 3. Predict using XGBoost model
```bash
python predict.py --model xgboost --smiles "c1ccccc1"
```
### 4. Compare predictions from all models
```bash
python predict.py --compare --smiles "Cc1ccc(O)cc1"
```
### 5. Batch prediction from file (using GNN)
```bash
python predict.py --model gnn --input example_compounds.csv --output gnn_predictions.csv
```
### 6. Specify SMILES column name
```bash
python predict.py --input mydata.csv --smiles-column compound_smiles --output results.csv
```
## 📖 References

1. **ChEMBL Database**  
   Gaulton A, et al. (2017) The ChEMBL database in 2017. Nucleic Acids Res.  
   https://www.ebi.ac.uk/chembl/

2. **RDKit: Open-Source Cheminformatics**  
   https://www.rdkit.org/

3. **Morgan Fingerprints (ECFP)**  
   Rogers D & Hahn M. (2010) Extended-Connectivity Fingerprints. J. Chem. Inf. Model.

4. **SMOTE**  
   Chawla NV, et al. (2002) SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

5. **XGBoost**  
   Chen T & Guestrin C. (2016) XGBoost: A Scalable Tree Boosting System. KDD.

6. **Graph Neural Networks**  
   Kipf TN & Welling M. (2017) Semi-Supervised Classification with Graph Convolutional Networks. ICLR.

7. **PyTorch Geometric**  
   Fey M & Lenssen JE. (2019) Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop.

8. **Drug Discovery with GNNs**  
   Stokes JM, et al. (2020) A Deep Learning Approach to Antibiotic Discovery. Cell.
