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
from typing import Callable, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import warnings
import os
from contextlib import redirect_stderr
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.exceptions import ConvergenceWarning

import warnings
import sys
import os
from contextlib import redirect_stderr
from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

import warnings
import sys
import os
from contextlib import redirect_stderr
from typing import Callable, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import warnings
import sys
import os
from contextlib import redirect_stderr
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from catboost import CatBoostClassifier
import optuna
import warnings
import os
from contextlib import redirect_stderr
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import chi2_contingency, fisher_exact, shapiro, mannwhitneyu
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    precision_score,
    fbeta_score,
)


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
        undersample=False,
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


def tune_logistic_regression_rs(
    X: Any,
    y: Any,
    pipeline_function: Any,
    param_distributions: Dict[str, Any],
    undersample: bool = False,
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Any:
    """
    Runs RandomizedSearchCV on a pipeline with LogisticRegression.
    Returns the best estimator and prints best params and cross-validation scores.
    """

    from sklearn.metrics import fbeta_score

    def fbeta_class1(y_true, y_pred, beta=2):
        return fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)

    lr_classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    pipeline = pipeline_function(
        estimator=lr_classifier, undersample=undersample, random_state=random_state
    )

    cv_strategy = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    scoring = {
        "f1_macro": "f1_macro",
        "f1_class_1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "f2_class_1": make_scorer(fbeta_class1),
        "recall_class_1": make_scorer(recall_score, pos_label=1, zero_division=0),
        "precision_class_1": make_scorer(precision_score, pos_label=1, zero_division=0),
    }

    print("\n============================")
    print(f"PIPELINE FUNCTION: {pipeline_function.__name__}")
    print(f"UNDERSAMPLING: {undersample}")
    print("============================\n")
    print(f"Performing RandomizedSearchCV with {n_iter} iterations...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                refit="f1_macro",
                cv=cv_strategy,
                verbose=0,
                n_jobs=-1,
                random_state=random_state,
                return_train_score=False,
            )
            random_search.fit(X, y)

    print("\nRandomizedSearchCV complete.")
    print("\nBest Parameters Found:")
    print(random_search.best_params_)

    best_index = random_search.best_index_
    cv_results = random_search.cv_results_

    print("\n--- Best Cross-Validation Scores ---")
    print(f"F1 Macro Score: {cv_results['mean_test_f1_macro'][best_index]:.4f}")
    print(f"F1 Class 1 Score: {cv_results['mean_test_f1_class_1'][best_index]:.4f}")
    print(f"F2 Class 1 Score: {cv_results['mean_test_f2_class_1'][best_index]:.4f}")
    print(f"Recall (Class 1): {cv_results['mean_test_recall_class_1'][best_index]:.4f}")
    print(
        f"Precision (Class 1): {cv_results['mean_test_precision_class_1'][best_index]:.4f}"
    )

    return random_search.best_estimator_


def tune_model_optuna_with_grid_rf(
    X,
    y,
    pipeline_function,
    model_class,
    param_distributions,
    n_trials=30,
    cv_folds=5,
    undersample=False,
    random_state=42,
):

    def fbeta_class1(y_true, y_pred, beta=2):
        from sklearn.metrics import fbeta_score

        return fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)

    def objective(trial):
        hyperparams = {}
        for param, values in param_distributions.items():
            if isinstance(values[0], float):
                # Continuous hyperparameters
                low, high = min(values), max(values)
                hyperparams[param] = trial.suggest_float(param, low, high)
            elif isinstance(values[0], int):
                # Integer hyperparameters
                low, high = min(values), max(values)
                hyperparams[param] = trial.suggest_int(param, low, high)
            else:
                # Categorical
                hyperparams[param] = trial.suggest_categorical(param, values)

        estimator = model_class(**hyperparams, random_state=random_state, n_jobs=-1)

        pipeline = pipeline_function(
            estimator=estimator, undersample=undersample, random_state=random_state
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        scoring = {
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_class_1": make_scorer(f1_score, pos_label=1, zero_division=0),
            "f2_class_1": make_scorer(fbeta_class1),
            "recall_class_1": make_scorer(recall_score, zero_division=0),
            "precision_class_1": make_scorer(precision_score, zero_division=0),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    return_train_score=False,
                )

        mean_f1 = scores["test_f1_macro"].mean()

        trial.set_user_attr("f1_class_1", scores["test_f1_class_1"].mean())
        trial.set_user_attr("f2_class_1", scores["test_f2_class_1"].mean())
        trial.set_user_attr("recall_class_1", scores["test_recall_class_1"].mean())
        trial.set_user_attr(
            "precision_class_1", scores["test_precision_class_1"].mean()
        )

        return 1.0 - mean_f1

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimization complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 Macro Score: {1.0 - study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nOther metrics on best trial:")
    print(f"F1 Class 1: {study.best_trial.user_attrs['f1_class_1']:.4f}")
    print(f"F2 Class 1: {study.best_trial.user_attrs['f2_class_1']:.4f}")
    print(f"Recall (Class 1): {study.best_trial.user_attrs['recall_class_1']:.4f}")
    print(
        f"Precision (Class 1): {study.best_trial.user_attrs['precision_class_1']:.4f}"
    )

    best_params = study.best_trial.params
    best_estimator = model_class(**best_params, random_state=random_state, n_jobs=-1)

    best_pipeline = pipeline_function(
        estimator=best_estimator, undersample=undersample, random_state=random_state
    )
    best_pipeline.fit(X, y)

    return best_pipeline


def tune_model_optuna_with_grid_cb(
    X,
    y,
    pipeline_function,
    param_distributions,
    n_trials=30,
    cv_folds=5,
    undersample=False,
    random_state=42,
):

    def fbeta_class1(y_true, y_pred, beta=2):
        from sklearn.metrics import fbeta_score

        return fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)

    def objective(trial):
        hyperparams = {}

        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type", param_distributions["bootstrap_type"]
        )
        hyperparams["bootstrap_type"] = bootstrap_type

        if bootstrap_type == "Bernoulli":
            subsample_range = param_distributions.get("subsample", [0.5, 0.9])
            hyperparams["subsample"] = trial.suggest_float(
                "subsample", min(subsample_range), max(subsample_range)
            )

        for param, values in param_distributions.items():
            if param in ["bootstrap_type", "subsample"]:
                continue
            if isinstance(values[0], float):
                low, high = min(values), max(values)
                hyperparams[param] = trial.suggest_float(param, low, high)
            elif isinstance(values[0], int):
                low, high = min(values), max(values)
                hyperparams[param] = trial.suggest_int(param, low, high)
            else:
                hyperparams[param] = trial.suggest_categorical(param, values)

        hyperparams["random_state"] = random_state
        hyperparams["verbose"] = 0

        estimator = CatBoostClassifier(**hyperparams)

        pipeline = pipeline_function(
            estimator=estimator, undersample=undersample, random_state=random_state
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        scoring = {
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_class_1": make_scorer(f1_score, pos_label=1, zero_division=0),
            "f2_class_1": make_scorer(fbeta_class1),
            "recall_class_1": make_scorer(recall_score, zero_division=0),
            "precision_class_1": make_scorer(precision_score, zero_division=0),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    return_train_score=False,
                )

        mean_f1 = scores["test_f1_macro"].mean()

        trial.set_user_attr("f1_class_1", scores["test_f1_class_1"].mean())
        trial.set_user_attr("f2_class_1", scores["test_f2_class_1"].mean())
        trial.set_user_attr("recall_class_1", scores["test_recall_class_1"].mean())
        trial.set_user_attr(
            "precision_class_1", scores["test_precision_class_1"].mean()
        )

        return 1.0 - mean_f1

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna optimization complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 Macro Score: {1.0 - study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nOther metrics on best trial:")
    print(f"F1 Class 1: {study.best_trial.user_attrs['f1_class_1']:.4f}")
    print(f"F2 Class 1: {study.best_trial.user_attrs['f2_class_1']:.4f}")
    print(f"Recall (Class 1): {study.best_trial.user_attrs['recall_class_1']:.4f}")
    print(
        f"Precision (Class 1): {study.best_trial.user_attrs['precision_class_1']:.4f}"
    )

    best_params = study.best_trial.params
    best_params["random_state"] = random_state
    best_params["verbose"] = 0
    best_estimator = CatBoostClassifier(**best_params)

    best_pipeline = pipeline_function(
        estimator=best_estimator, undersample=undersample, random_state=random_state
    )
    best_pipeline.fit(X, y)

    return best_pipeline


def bootstrap_diff_median_age(group1, group2, n_bootstrap=1000):
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(np.median(sample2) - np.median(sample1))
    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    return lower, upper


def bootstrap_diff_median_bmi(group1, group2, n_bootstrap=1000):
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(np.median(sample2) - np.median(sample1))
    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    return lower, upper


def compare_smoking_groups(df, group1, group2):
    sub_df = df[df["smoking_status"].isin([group1, group2])]
    table = pd.crosstab(sub_df["smoking_status"], sub_df["stroke"])

    print(f"\nContingency Table: {group1} vs. {group2}")
    print(table)

    chi2, p, dof, expected = chi2_contingency(table)
    print("Expected Counts:")
    print(expected)

    if (expected < 5).any():
        print("Low expected counts detected, using Fisher's Exact Test")
        oddsratio, p_fisher = fisher_exact(table)
        print(f"Fisher's Exact Test p-value: {p_fisher:.4f}")
        print(f"Odds Ratio: {oddsratio:.2f}")
    else:
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"p-value: {p:.4f}")


def fbeta_class1(y_true, y_pred, beta=2):
    return fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)


def evaluate_model_cv(estimator, X, y, cv=5, n_jobs=-1):
    """
    Runs cross-validation on the given estimator with predefined scoring.
    Prints mean cross-validation scores.
    """
    scoring = {
        "f1_macro": "f1_macro",
        "f1_class_1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "f2_class_1": make_scorer(fbeta_class1),
        "recall_class_1": make_scorer(recall_score, pos_label=1, zero_division=0),
        "precision_class_1": make_scorer(precision_score, pos_label=1, zero_division=0),
    }

    print("\n============================")
    print(f"Evaluating model: {estimator.__class__.__name__}")
    print(f"Cross-validation folds: {cv}")
    print("============================\n")

    cv_results = cross_validate(
        estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, return_train_score=False
    )

    for metric in scoring.keys():
        mean_score = cv_results["test_" + metric].mean()
        print(f"{metric}: {mean_score:.4f}")

    return cv_results
