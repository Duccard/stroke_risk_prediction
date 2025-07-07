from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def train_evaluate_dummy_classifier(
    X_train, y_train, pipeline_builder, title_suffix="Training Data"
):
    """
    Trains and evaluates a DummyClassifier pipeline on the training data.
    Prints Precision, Recall, F1 (binary), and F1 Macro scores.
    """
    print(f"\n🔹 Training Dummy Classifier Pipeline on {title_suffix} only...")
    dummy_classifier = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline_pipeline = pipeline_builder(dummy_classifier)
    baseline_pipeline.fit(X_train, y_train)
    print("Training complete.")

    y_pred = baseline_pipeline.predict(X_train)
    y_prob = baseline_pipeline.predict_proba(X_train)[:, 1]

    print(f"\n--- Dummy Classifier Performance on {title_suffix} ---")
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1_binary = f1_score(y_train, y_pred, zero_division=0)
    f1_macro = f1_score(y_train, y_pred, average="macro", zero_division=0)

    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
