# Squat Form Classification using Interpretable Machine Learning

This project presents an interpretable machine learning framework for classifying squat form using biomechanical features derived from pose estimation.

## Overview
We extract sequence-level biomechanical descriptors such as knee flexion and trunk inclination to classify squat form into:
- Correct
- Excessive Lean
- Shallow

## Methods
- Feature Engineering: Angular biomechanical features
- Models:
  - Random Forest (primary)
  - Support Vector Machine
  - Logistic Regression
- Evaluation:
  - 5-fold Stratified Cross Validation
  - Paired t-test statistical analysis

## Results
- Random Forest achieved highest accuracy (~97%)
- Strong class separability observed
- Robust to moderate noise levels

## How to Run

```bash
pip install pandas numpy scikit-learn scipy
python train.py
