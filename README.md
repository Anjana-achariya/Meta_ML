Project Objective : 

The objective of this project is to build a Meta-Machine Learning system that can recommend the most suitable classical ML model for a given dataset before training. 
Instead of trial-and-error model training, the system analyzes dataset characteristics (meta-features such as size, feature distribution, and missing values) and predicts which model is likely to perform best, while also considering training time and resource constraints. 
This helps users make faster, more informed model selection decisions.

Architecture Overview : 

Dataset (CSV)
     ↓
Meta-Feature Extraction (streaming, no downsampling)
     ↓
Feasibility Rules (size & resource constraints)
     ↓
Performance Meta-Model (predicts best model)
     ↓
Training-Time Estimator (predicts cost)
     ↓
Resource-Aware Ranking
     ↓
Recommended ML Model(s)

Dataset Meta-Features Used : 

Number of rows
Number of features
Numeric vs categorical feature ratio
Missing value ratio
Mean skewness of numeric features
Mean entropy of feature distributions
Feature-to-row ratio
Log-scaled dataset size features

All features are extracted in a streaming-safe manner, allowing the system to handle large datasets.

Classical ML Models Compared

The system benchmarks and recommends from the following classical machine learning models:

Logistic Regression
Random Forest
XGBoost
Linear SVM
k-Nearest Neighbors (kNN)

These models are evaluated offline using cross-validation, and their historical performance is used to train the meta-model.

Output : 

For a user-provided dataset, the system returns:
Recommended ML model (or top-K models)
Estimated training time per model
Expected relative performance
Exclusion of infeasible models based on dataset size or resource limits

The output is designed to be practical, fast, and resource-aware, without training multiple models at runtime.