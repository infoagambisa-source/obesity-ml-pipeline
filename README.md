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
- **Missing Values:** None
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
Predict an individual’s obesity level category using lifestyle, dietary, and physical activity features.

## Task Type
- **Multi-class classification**

## Evaluation Metrics
- **Primary:** Macro F1-score  
- **Secondary:** Accuracy  

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
python -m src.predict --input data/raw/sample.csv
