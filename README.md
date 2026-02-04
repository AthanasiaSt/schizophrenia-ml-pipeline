# schizophrenia-gene-expression-ml

## Machine Learning Pipeline for Schizophrenia Classification Using Brain Gene Expression

This repository contains an automated machine learning pipeline developed as part of a Master’s project for the classification of schizophrenia using post-mortem brain gene expression data.

The pipeline is built around the GEO dataset **GDS4523** and integrates preprocessing, unsupervised analysis for visualization, feature selection, and supervised classification with extensive cross-validation.

Read the readme.txt for more information

---

## Dataset

- **Source:** GEO (GDS4523)
- **Tissue:** Anterior prefrontal cortex (Brodmann Area 10, BA10)
- **Samples:** 51 post-mortem brain samples
- **Features:** ~54,675 gene expression measurements
- **Task:** Binary classification (schizophrenia vs control)

---

## Pipeline Overview

The workflow is fully automated and includes:

1. Data download and cleaning (optional)
2. Target extraction (disease, sex, age)
3. Preprocessing
   - Mean imputation
   - Standard scaling
4. Unsupervised analysis (visualization)
   - PCA
   - K-means clustering with silhouette optimization
5. Feature selection
   - Lasso regression
6. Supervised classification
   - Support Vector Machine (RBF)
   - Random Forest
7. Model evaluation and selection
   - Repeated stratified k-fold cross-validation
   - Metrics: Accuracy, F1 score, ROC-AUC

---

## Methods

### Preprocessing
- Mean imputation for missing values
- Z-score standardization

### Dimensionality Reduction & Clustering
- PCA (components selected to explain ~90% of total variance)
- K-means clustering
- Silhouette score for cluster number selection

### Feature Selection
- Lasso regression for sparse feature selection
- Automatic fallback if zero features are selected

### Classifiers
- Support Vector Machine (RBF kernel)
- Random Forest

### Hyperparameter Optimization
The following hyperparameters are explored:

```python
C = [1e-3, 1e-2, 1, 1e2, 1e3]
gamma = [1e-2, 1e-1, 1, 1e1, 1e2]
max_depth = [20, 40, 60, 80, None]
estimators = [100, 200, 300, 400, 500]
alpha = np.logspace(-4, -0.5, 8)
```

A total of 200 configurations are evaluated.

---

## How to Run

### Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

---

### Run the full pipeline

If you already have the dataset locally:

```python
SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download='no')
```

To automatically download the dataset from GEO:

```python
SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download='yes')
```

---

## Results (Default Run)

- Runtime: ~45 minutes
- Best performing model: Support Vector Machine (RBF)

Best SVM configuration:
- C = 100.0
- gamma = 0.01
- alpha = 0.000316

Performance:
- Accuracy: ~0.58
- F1 score: ~0.54
- ROC-AUC: ~0.57

---

## Notes

- This project was developed for academic purposes as part of a Master’s degree.
- The goal is to demonstrate a complete ML pipeline for transcriptomics data.
- Performance is limited by sample size and the high dimensionality of gene expression data.

---

## Disclaimer

This code is intended for research and educational purposes only and should not be used for clinical decision making.
