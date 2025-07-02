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

Macro F1 Score. Balances precision and recall across classes to ensure fair assesment of performance on both stroke and non-strokes case in an imbalanced dataset.