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
from typing import Callable, Any, Optional, Union, Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import fbeta_score
from typing import Callable, Any, Optional, Union, Dict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable, Any, Optional, Union
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from typing import Callable, Any, Optional
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable, Any, Optional, Union
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from typing import Callable, Any, Optional
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def train_and_evaluate_dummy_classifier(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    strategy: str = "most_frequent",
) -> None:
    """
    Train and evaluate a DummyClassifier baseline on training data only.
    Prints evaluation metrics including precision, recall, F1, F2, and macro F1.

    Args:
        pipeline_function (Callable): Function returning a pipeline with the DummyClassifier.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        strategy (str): Strategy for DummyClassifier (e.g. 'most_frequent', 'stratified').
    """
    dummy_classifier = DummyClassifier(strategy=strategy, random_state=42)

    pipeline_model: Pipeline = pipeline_function(
        estimator=dummy_classifier,
        undersample=False,  # Usually you don't undersample for dummy
    )

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"STRATEGY: {strategy}")
    print("============================\n")

    print("Training Dummy Classifier Pipeline on Training Data only...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f2_binary = fbeta_score(y_train, y_pred_train, beta=2, zero_division=0)
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n--- DUMMY CLASSIFIER TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("-----------------------------------------\n")


def train_and_evaluate_logistic_regression(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    undersample: bool,
    class_weight: Optional[Union[str, Dict]] = "balanced",
) -> None:
    """
    Train and evaluate a logistic regression model using a given pipeline function.
    Prints evaluation metrics and configuration details.
    """
    logistic_regression_classifier = LogisticRegression(
        random_state=42, max_iter=1000, class_weight=class_weight
    )

    pipeline_model: Pipeline = pipeline_function(
        estimator=logistic_regression_classifier, undersample=undersample
    )

    print("\n============================")
    print(f"PIPELINE: {pipeline_function.__name__}")
    print(f"CLASS WEIGHT: {class_weight}")
    print(f"UNDERSAMPLING: {undersample}")
    print("============================\n")

    print("Training Logistic Regression Pipeline...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)
    y_prob_train = pipeline_model.predict_proba(X_train)[:, 1]

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f2_binary = fbeta_score(y_train, y_pred_train, beta=2, zero_division=0)
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n--- LOGISTIC REGRESSION TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("--------------------------------------------\n")


def train_and_evaluate_random_forest(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    undersample: bool,
    class_weight: Optional[Union[str, Dict]] = "balanced",
    n_estimators: int = 100,
) -> None:
    """
    Train and evaluate a Random Forest model using a given pipeline function on training data.
    Prints evaluation metrics and configuration details.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators, class_weight=class_weight, random_state=42
    )

    pipeline_model: Pipeline = pipeline_function(
        estimator=rf_classifier, undersample=undersample
    )

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"CLASS WEIGHT: {class_weight}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    print("Training Random Forest Pipeline...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f2_binary = fbeta_score(y_train, y_pred_train, beta=2, zero_division=0)
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n--- RANDOM FOREST TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("---------------------------------------\n")


def train_and_evaluate_xgboost(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    undersample: bool,
    n_estimators: int = 100,
    weight: Optional[float] = None,
) -> None:
    """
    Train and evaluate an XGBoost model using a given pipeline function on training data.
    Prints evaluation metrics and configuration details.

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y_train.
    """
    if weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        weight = neg_count / pos_count

    xgb_classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        scale_pos_weight=weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    pipeline_model: Pipeline = pipeline_function(
        estimator=xgb_classifier, undersample=undersample
    )

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    print("Training XGBoost Pipeline...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
    f2_binary = fbeta_score(
        y_train, y_pred_train, average="binary", beta=2, zero_division=0
    )

    print("\n--- XGBOOST TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("---------------------------------\n")


def train_and_evaluate_lightgbm(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    undersample: bool,
    n_estimators: int = 100,
    weight: Optional[float] = None,
) -> None:
    """
    Train and evaluate a LightGBM model using a given pipeline function on training data.
    Prints evaluation metrics and configuration details.

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y_train.
    """
    if weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        weight = neg_count / pos_count

    lgbm_classifier = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        scale_pos_weight=weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline_model: Pipeline = pipeline_function(
        estimator=lgbm_classifier, undersample=undersample
    )

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    print("Training LightGBM Pipeline...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f2_binary = fbeta_score(
        y_train, y_pred_train, average="binary", beta=2, zero_division=0
    )
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n--- LIGHTGBM TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("---------------------------------\n")


def train_and_evaluate_catboost(
    pipeline_function: Callable[..., Pipeline],
    X_train: Any,
    y_train: Any,
    undersample: bool,
    iterations: int = 100,
    weight: Optional[float] = None,
) -> None:
    """
    Train and evaluate a CatBoost model using a given pipeline function on training data.
    Prints evaluation metrics and configuration details.

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y_train.
    """
    if weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        weight = neg_count / pos_count

    catboost_classifier = CatBoostClassifier(
        objective="Logloss",
        iterations=iterations,
        scale_pos_weight=weight,
        random_seed=42,
        verbose=0,
    )

    pipeline_model: Pipeline = pipeline_function(
        estimator=catboost_classifier, undersample=undersample
    )

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"ITERATIONS: {iterations}")
    print("============================\n")

    print("Training CatBoost Pipeline...")
    pipeline_model.fit(X_train, y_train)
    print("Training complete.")

    y_pred_train = pipeline_model.predict(X_train)

    precision = precision_score(y_train, y_pred_train, zero_division=0)
    recall = recall_score(y_train, y_pred_train, zero_division=0)
    f1_binary = f1_score(y_train, y_pred_train, average="binary", zero_division=0)
    f2_binary = fbeta_score(
        y_train, y_pred_train, average="binary", beta=2, zero_division=0
    )
    f1_macro = f1_score(y_train, y_pred_train, average="macro", zero_division=0)

    print("\n--- CATBOOST TRAINING METRICS ---")
    print(f"Precision (Class 1): {precision:.4f}")
    print(f"Recall (Class 1): {recall:.4f}")
    print(f"F1 Score (Class 1): {f1_binary:.4f}")
    print(f"F2 Score (Class 1): {f2_binary:.4f}")
    print(f"F1 Macro Score: {f1_macro:.4f}")
    print("---------------------------------\n")


from sklearn.linear_model import LogisticRegression
from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def cross_validate_logistic_regression(
    pipeline_function: Callable[..., Any],
    X: Any,
    y: Any,
    class_weight: Optional[str] = "balanced",
    undersample: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Perform stratified K-fold cross-validation for a Logistic Regression model
    and print average F1 Macro, F1 (Class 1), F2 (Class 1), Recall (Class 1), and Precision (Class 1).

    Args:
        pipeline_function (Callable): Function returning a pipeline given an estimator.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target labels.
        class_weight (str or None): Class weighting scheme.
        undersample (bool): Whether to include undersampling in the pipeline.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
    """
    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"CLASS WEIGHT: {class_weight}")
    print(f"UNDERSAMPLING: {undersample}")
    print("============================\n")

    lr_classifier = LogisticRegression(
        class_weight=class_weight, random_state=random_state, max_iter=1000
    )

    pipeline_model = pipeline_function(
        estimator=lr_classifier, undersample=undersample, random_state=random_state
    )

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    scoring = {
        "precision_class_1": make_scorer(precision_score, zero_division=0),
        "recall_class_1": make_scorer(recall_score, zero_division=0),
        "f1_class_1": make_scorer(f1_score, zero_division=0),
        "f2_class_1": f2_scorer,
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print(
        f"Performing {n_splits}-fold Stratified Cross-Validation for Logistic Regression..."
    )
    cv_results = cross_validate(
        pipeline_model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    print("Cross-validation complete.")

    print("\n--- Logistic Regression Cross-Validation Results ---")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.replace('_', ' ').title()}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}"
        )


from typing import Callable, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


def cross_validate_random_forest(
    pipeline_function: Callable[..., Any],
    X: Any,
    y: Any,
    n_estimators: int = 100,
    class_weight: Optional[str] = "balanced",
    undersample: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Perform stratified K-fold cross-validation for a Random Forest model
    and print average F1 Macro, F1 (Class 1), F2 (Class 1), Recall (Class 1), and Precision (Class 1).

    Args:
        pipeline_function (Callable): Function returning a pipeline given an estimator.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target labels.
        n_estimators (int): Number of trees in the forest.
        class_weight (str or None): Class weighting scheme.
        undersample (bool): Whether to include undersampling in the pipeline.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
    """
    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"CLASS WEIGHT: {class_weight}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
    )

    pipeline_model = pipeline_function(
        estimator=rf_classifier, undersample=undersample, random_state=random_state
    )

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    scoring = {
        "precision_class_1": make_scorer(precision_score, zero_division=0),
        "recall_class_1": make_scorer(recall_score, zero_division=0),
        "f1_class_1": make_scorer(f1_score, zero_division=0),
        "f2_class_1": f2_scorer,
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print(
        f"Performing {n_splits}-fold Stratified Cross-Validation for Random Forest..."
    )
    cv_results = cross_validate(
        pipeline_model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    print("Cross-validation complete.")

    print("\n--- Random Forest Cross-Validation Results ---")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.replace('_', ' ').title()}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}"
        )


from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


def cross_validate_xgboost(
    pipeline_function: Callable[..., Any],
    X: Any,
    y: Any,
    n_estimators: int = 200,
    weight: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 42,
    undersample: bool = False,
) -> None:
    """
    Perform stratified K-fold cross-validation for an XGBoost model
    and print average F1 Macro, F1 (Class 1), F2 (Class 1), Recall (Class 1), and Precision (Class 1).

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y.

    Args:
        pipeline_function (Callable): Function returning a pipeline given an estimator.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target labels.
        n_estimators (int): Number of boosting rounds.
        weight (float, optional): scale_pos_weight for XGBoost; if None, computed automatically.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        undersample (bool): Whether to enable undersampling in the pipeline.
    """
    if weight is None:
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        weight = neg_count / pos_count

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    xgb_classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        scale_pos_weight=weight,
        random_state=random_state,
        verbosity=0,
        eval_metric="logloss",
        disable_default_eval_metric=True,
    )

    pipeline_model = pipeline_function(
        estimator=xgb_classifier, undersample=undersample
    )

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    scoring = {
        "precision_class_1": make_scorer(precision_score, zero_division=0),
        "recall_class_1": make_scorer(recall_score, zero_division=0),
        "f1_class_1": make_scorer(f1_score, zero_division=0),
        "f2_class_1": f2_scorer,
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print(f"Performing {n_splits}-fold Stratified Cross-Validation for XGBoost...")
    cv_results = cross_validate(
        pipeline_model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )
    print("Cross-validation complete.")

    print("\n--- XGBoost Cross-Validation Results ---")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.replace('_', ' ').title()}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}"
        )


import warnings
import sys
import os
from contextlib import redirect_stderr
from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier


def cross_validate_lightgbm(
    pipeline_function: Callable[..., Any],
    X: Any,
    y: Any,
    n_estimators: int = 100,
    weight: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 42,
    undersample: bool = False,
) -> None:
    """
    Perform stratified K-fold cross-validation for a LightGBM model
    and print average F1 Macro, F1 (Class 1), F2 (Class 1), Recall (Class 1), and Precision (Class 1).
    Suppresses LightGBM logs and scikit-learn UserWarnings for cleaner output.

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y.

    Args:
        pipeline_function (Callable): Function returning a pipeline given an estimator.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target labels.
        n_estimators (int): Number of boosting rounds.
        weight (float, optional): scale_pos_weight for LightGBM; if None, computed automatically.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        undersample (bool): Whether to enable undersampling in the pipeline.
    """
    if weight is None:
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        weight = neg_count / pos_count

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"N_ESTIMATORS: {n_estimators}")
    print("============================\n")

    lgbm_classifier = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        scale_pos_weight=weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    pipeline_model = pipeline_function(
        estimator=lgbm_classifier, undersample=undersample
    )

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    scoring = {
        "precision_class_1": make_scorer(precision_score, zero_division=0),
        "recall_class_1": make_scorer(recall_score, zero_division=0),
        "f1_class_1": make_scorer(f1_score, zero_division=0),
        "f2_class_1": f2_scorer,
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print(f"Performing {n_splits}-fold Stratified Cross-Validation for LightGBM...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            cv_results = cross_validate(
                pipeline_model,
                X,
                y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )

    print("Cross-validation complete.")

    print("\n--- LightGBM Cross-Validation Results ---")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.replace('_', ' ').title()}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}"
        )


import warnings
import sys
import os
from contextlib import redirect_stderr
from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier


def cross_validate_catboost(
    pipeline_function: Callable[..., Any],
    X: Any,
    y: Any,
    iterations: int = 100,
    weight: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 42,
    undersample: bool = False,
) -> None:
    """
    Perform stratified K-fold cross-validation for a CatBoost model
    and print average F1 Macro, F1 (Class 1), F2 (Class 1), Recall (Class 1), and Precision (Class 1).
    Suppresses CatBoost logs and scikit-learn UserWarnings for cleaner output.

    If 'weight' is not provided, scale_pos_weight will be automatically calculated from y.

    Args:
        pipeline_function (Callable): Function returning a pipeline given an estimator.
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target labels.
        iterations (int): Number of boosting rounds.
        weight (float, optional): scale_pos_weight for CatBoost; if None, computed automatically.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        undersample (bool): Whether to enable undersampling in the pipeline.
    """
    if weight is None:
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        weight = neg_count / pos_count

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"SCALE_POS_WEIGHT: {weight:.4f}")
    print(f"UNDERSAMPLING: {undersample}")
    print(f"ITERATIONS: {iterations}")
    print("============================\n")

    catboost_classifier = CatBoostClassifier(
        iterations=iterations,
        loss_function="Logloss",
        scale_pos_weight=weight,
        random_seed=random_state,
        verbose=0,
    )

    pipeline_model = pipeline_function(
        estimator=catboost_classifier, undersample=undersample
    )

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    scoring = {
        "precision_class_1": make_scorer(precision_score, zero_division=0),
        "recall_class_1": make_scorer(recall_score, zero_division=0),
        "f1_class_1": make_scorer(f1_score, zero_division=0),
        "f2_class_1": f2_scorer,
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print(f"Performing {n_splits}-fold Stratified Cross-Validation for CatBoost...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            cv_results = cross_validate(
                pipeline_model,
                X,
                y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )

    print("Cross-validation complete.")

    print("\n--- CatBoost Cross-Validation Results ---")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(
            f"{metric.replace('_', ' ').title()}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}"
        )
