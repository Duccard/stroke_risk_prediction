# DS.v2.5.3.2.5

# **Stroke Risk Prediction Model: Johns Hopkins Hospital**

## Background

In this project I will explore dataset designed to identify stroke risk in patients. Stroke is a major global health concern, ranking as the second leading cause of death worldwide. It is also a leading cause of disability. Annually, millions experience strokes, and a significant number of these result in death. After data exploration and analysis, I will build, train and deploy a model which could predict whether patients is likely to have stroke or not, based on parameters like age, gender, lifestyle choices, bmi and so on. 

## Overview

This project develops a machine learning models to predict the risk of stroke using patient demographic and health data. It includes data preprocessing, exploratory analysis, feature engineering, model selection and tuning, evaluation with business-focused metrics, and deployment preparation.

## Problem

It’s hard to spot which patients are at high risk of stroke, making it tough to catch it early and prevent it.

## Goal

The objective is to support early identification of high-risk patients to inform healthcare interventions and reduce the burden of stroke.

## Files

- StrokeRiskPredictionModel.ipynb - Jupyter notebook containing the full stroke risk prediction workflow from data loading to evaluation and  model deployment.

- healthcare-dataset-stroke-data.csv - CSV file with patient data and stroke labels used for training and testing.

- best_catboost_pipeline.pkl - saved CatBoost model pipeline with preprocessing steps for easy reuse.

- best_lr_pipeline.pkl - saved Logistic Regression pipeline including all preprocessing.

- best_rf_pipeline.pkl - saved Random Forest pipeline with feature engineering and preprocessing.

- stroke_risk_pipeline_ensemble_model.joblib - Ensemble model pipeline combining multiple classifiers.

- pipelinestrokefunctions.py- pipeline functions containing all features

- pipelinestrokeselectedfeaturesfunctions.py - contains pipeline functions for selected, engineered feature subsets.

- graphscustomfunctions.py - custom plot functions

- customstrokefunctions.py - custom helper functions.

- README.md - project documentation and overview.

- requirements.txt - info about packages and their versions.

- DS_Store, __pycache__/ - system and Python cache files

## Hypotheses

1. Older age increases likelyhood of stroke.
- **H₀:** Age is not associated with stroke risk.
- **H₁:** Increasing age is associated with higher stroke risk

2. Males are more likely to have a stroke than females.
- **H₀:** Gender has no effect on stroke risk.
- **H₁:** Male patients have a higher probability of experiencing a stroke than female patients.

3. Obesity increases risk of a stroke.
- **H₀:** BMI is not associated with stroke risk.
- **H₁:** Higher BMI is associated with an increased likelihood of stroke.

4. Having a hypertension increases chances of having a stroke
- **H₀:** Hypertension is not associated with stroke risk.
- **H₁:** Patients with hypertension have higher odds of experiencing a stroke.

## Main Evaluation Metric:

- F2. Similar to F1 score, but it weights recall more heavily than precision.

- Recall (Class 1). Measures how many actual positives the model successfully finds.

- Macro F1 Score. Balances precision and recall across classes to ensure fair assesment of performance on both stroke and non-strokes case in an imbalanced dataset.

## Secondary Evaluation Metric:

- Precision (Class 1). It tells how many of positive predictions were actually correct. 

- ROC AUC. Measures how well a model can distinguish between classes.

- PR AUC. Measures how well a model balances precision and recall across thresholds.

- Confusion Matrix. Provides detailed breakdown of correct and incorrect predictions.

## Typical High Risk Stroke Groups

According to the this specific dataset the most typical people to suffer from strokes are:

1. Older patients

2. Patients with heart diseases*

3. Patients with hypertension*

4. Self-employed

5. People with higher average glucose levels*

6. Higher BMI patients*

7. Former smokers*

8. Ever married*


**Important note:** *As people get older, they're more likely to have conditions like heart disease and hypertension, to have smoked in the past, to have been married, and to carry more weight—though BMI can drop in the very old age.

## Best Model

Ensemble model was chosen as best model, was made from of three base/advanced models (LR, RF, CatBoost). On final evaluation it achieved:

- F2 Score: 0.37

- F1 Macro: 0.55

- F1: 0.23

- Recall(1 Class): 66

- Precision(1 Class): 0.14

- ROC AUC: 0.77

- PR AUC: 0.145


Evaluation metrics confirmed strong performance for the majority class and highlighted the challenge of identifying minority stroke cases—an expected difficulty given the data imbalance.

Overall, this ensemble model can serve as a helpful tool to flag high-risk cases, supporting early detection efforts. However, given its tendency for both false positives and missed positives, it should not be relied on as the sole decision-maker. Medical professionals should use its predictions alongside clinical judgment and other diagnostic information to ensure safe and accurate patient care.

## License

LICENSE
These projects are licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.