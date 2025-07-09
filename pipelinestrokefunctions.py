import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

age_bins = [0, 18, 39, 59, 64, 120]
age_labels = ["Under 18", "Young Adult", "Middle-Aged Adult", "Senior Adult", "Elderly"]

bmi_bins = [0, 18.5, 25, 30, 100]
bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]

glucose_bins = [0, np.log(100), np.log(126), np.inf]
glucose_labels = ["Normal", "Prediabetes", "Diabetes"]


class CombinedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        age_bins=age_bins,
        age_labels=age_labels,
        bmi_bins=bmi_bins,
        bmi_labels=bmi_labels,
        bmi_cap_quantile=0.99,
        glucose_bins=glucose_bins,
        glucose_labels=glucose_labels,
    ):
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.bmi_bins = bmi_bins
        self.bmi_labels = bmi_labels
        self.bmi_cap_quantile = bmi_cap_quantile
        self.glucose_bins = glucose_bins
        self.glucose_labels = glucose_labels
        self.medians_ = {}
        self.global_bmi_median_ = None
        self.cap_value_ = None

    def fit(self, X, y=None):
        X_temp = X.copy()
        X_temp.columns = X_temp.columns.str.lower()
        X_temp = X_temp[X_temp["gender"] != "Other"]

        X_temp["age_group"] = pd.cut(
            X_temp["age"],
            bins=self.age_bins,
            labels=self.age_labels,
            include_lowest=True,
        )

        self.medians_ = (
            X_temp.groupby("age_group", observed=False)["bmi"].median().to_dict()
        )
        self.global_bmi_median_ = X_temp["bmi"].median()
        self.cap_value_ = X_temp["bmi"].quantile(self.bmi_cap_quantile)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.columns = X_transformed.columns.str.lower()

        X_transformed["age_group"] = pd.cut(
            X_transformed["age"],
            bins=self.age_bins,
            labels=self.age_labels,
            include_lowest=True,
        )

        X_transformed["bmi"] = X_transformed.apply(
            lambda row: (
                row["bmi"]
                if pd.notna(row["bmi"])
                else self.medians_.get(row["age_group"], self.global_bmi_median_)
            ),
            axis=1,
        )

        X_transformed["bmi_capped"] = np.where(
            X_transformed["bmi"] > self.cap_value_,
            self.cap_value_,
            X_transformed["bmi"],
        )

        X_transformed["bmi_category"] = pd.cut(
            X_transformed["bmi_capped"], bins=self.bmi_bins, labels=self.bmi_labels
        )

        X_transformed["avg_glucose_level_log"] = np.log(
            X_transformed["avg_glucose_level"]
        )

        X_transformed["glucose_category_log"] = pd.cut(
            X_transformed["avg_glucose_level_log"],
            bins=self.glucose_bins,
            labels=self.glucose_labels,
        )

        return X_transformed


class SimpleRandomUndersampler(BaseEstimator, TransformerMixin):
    """
    Undersamples majority class to balance dataset. Only used during training.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, Xy):
        if isinstance(Xy, tuple):
            X, y = Xy
        else:
            raise ValueError("Expected (X, y) tuple as input to undersampler")

        df = X.copy()
        df["target"] = y

        majority = df[df["target"] == 0]
        minority = df[df["target"] == 1]

        majority_downsampled = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=self.random_state,
        )

        balanced = pd.concat([majority_downsampled, minority])
        X_balanced = balanced.drop("target", axis=1)
        y_balanced = balanced["target"]

        return (X_balanced, y_balanced)


numerical_final_features = ["hypertension", "heart_disease"]
categorical_final_features = [
    "gender",
    "ever_married",
    "work_type",
    "residence_type",
    "smoking_status",
    "age_group",
    "bmi_category",
    "glucose_category_log",
]

perfect_data_preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), numerical_final_features),
        (
            "cat_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_final_features,
        ),
    ],
    remainder="drop",
)


def pipeline_stroke(estimator, undersample=True, random_state=42):
    steps = [
        ("feature_engineering", CombinedFeatureTransformer()),
        ("final_preprocessing", perfect_data_preprocessor),
    ]

    if undersample:
        steps.append(("undersampler", RandomUnderSampler(random_state=random_state)))

    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


def evaluate_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    undersample=False,
    use_weights=False,
    title_suffix="",
):
    if use_weights:
        pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        estimator = XGBClassifier(
            random_state=42, eval_metric="logloss", scale_pos_weight=pos_weight
        )
    else:
        estimator = XGBClassifier(random_state=42, eval_metric="logloss")

    xgb_pipeline = pipeline_stroke(estimator=estimator, undersample=undersample)

    print(f"\nTraining Pipeline (Undersample={undersample}, Weights={use_weights})...")
    xgb_pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")

    y_train_pred = xgb_pipeline.predict(X_train)
    print("\n--- Classification Report on Training Data ---")
    print(classification_report(y_train, y_train_pred))

    y_test_pred = xgb_pipeline.predict(X_test)
    print("\n--- Classification Report on Test Data ---")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="magma_r")
    plt.title(f"Confusion Matrix (Test Data) - {title_suffix}", weight="bold")
    plt.show()
