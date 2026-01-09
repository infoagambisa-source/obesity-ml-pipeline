# Obesity Level Prediction — Machine Learning Pipeline

## Overview
This project builds an end-to-end machine learning workflow to predict **obesity level categories** based on eating habits and physical condition data.  
The final solution is packaged as a **production-ready scikit-learn pipeline** for training and inference.

## Dataset
- **Source:** UCI Machine Learning Repository  
- **Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition  
- **Countries:** Mexico, Peru, Colombia  
- **Instances:** 2,111  
- **Features:** 16  
- **Target:** `NObeyesdad` (7 obesity level classes)
- **Missing Values:** None (per inspection)
- **Notes:** ~77% of records were synthetically generated using SMOTE

### Target Classes
- Insufficient Weight  
- Normal Weight  
- Overweight Level I  
- Overweight Level II  
- Obesity Type I  
- Obesity Type II  
- Obesity Type III  

## Problem Statement
Predict an individual’s obesity level category (`NObeyesdad`) using lifestyle, dietary, and physical activity features.

## Task Type
- **Multi-class classification**

## Evaluation Metrics
- **Primary:** Macro F1-score (treats all classes equally) 
- **Secondary:** Accuracy  

## Approach
1. **Data loading & inspection** (via `ucimlrepo`)
2. **EDA** (target distribution, numeric summary, correlations)
3. **Leakage-safe splitting** (train/validation/test with stratification)
4. **Preprocessing** using `ColumnTransformer`
   - Numeric: imputation + scaling
   - Categorical: imputation + one-hot encoding (`handle_unknown="ignore"`)
5. **Baseline model:** Logistic Regression
6. **Model comparison:** Logistic Regression vs Random Forest vs Gradient Boosting
7. **Hyperparameter tuning:** RandomizedSearchCV (optimized for Macro F1)
8. **Final evaluation:** held-out test set

## Results

### Validation (baseline)
- Logistic Regression:
  - **Macro F1:** 0.8479
  - **Accuracy:** 0.8555

### Validation (model comparison)
- Gradient Boosting:
  - **Macro F1:** 0.9437
  - **Accuracy:** 0.9455
- Random Forest:
  - **Macro F1:** 0.9365
  - **Accuracy:** 0.9384

### Tuning (cross-validation on training set)
Best Gradient Boosting parameters:
- `n_estimators=300`, `learning_rate=0.2`, `max_depth=3`, `subsample=0.8`

Best cross-validated score:
- **Macro F1:** 0.9646

### Test (final model)
- Tuned Gradient Boosting:
  - **Macro F1:** 0.9476
  - **Accuracy:** 0.9480

**Confusion matrix insight:** misclassifications occur primarily between **adjacent obesity categories**  
(e.g., boundary cases like Normal ↔ Insufficient or Overweight Level II ↔ Obesity Type I).  
No extreme category flips were observed (e.g., Insufficient predicted as Obesity Type III).


## Repository Structure
obesity-ml-pipeline/
├─ notebooks/ # EDA + experimentation
├─ src/ # Training & inference code
├─ models/ # Saved ML pipeline
├─ data/ # Raw / processed data
├─ requirements.txt
└─ README.md


## How to Run (to be finalized)
```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.predict --input data/raw/sample.csv --output predictions.csv
