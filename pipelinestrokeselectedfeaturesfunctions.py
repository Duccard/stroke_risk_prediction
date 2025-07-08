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
from imblearn.combine import SMOTETomek


class CombinedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        age_bins=[0, 18, 39, 59, 64, 120],
        age_labels=[
            "Under 18",
            "Young Adult",
            "Middle-Aged Adult",
            "Senior Adult",
            "Elderly",
        ],
        bmi_bins=[0, 18.5, 25, 30, 100],
        bmi_labels=["Underweight", "Normal", "Overweight", "Obese"],
        bmi_cap_quantile=0.99,
        glucose_bins=[0, np.log(100), np.log(126), np.inf],
        glucose_labels=["Normal", "Prediabetes", "Diabetes"],
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
        X_transformed = X_transformed[X_transformed["gender"] != "Other"]

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


categorical_final_features_selected = [
    "age_group",
    "smoking_status",
    "bmi_category",
    "glucose_category_log",
    "work_type",
]

explicit_categories_for_ohe_selected = {
    "age_group": [
        "Under 18",
        "Young Adult",
        "Middle-Aged Adult",
        "Senior Adult",
        "Elderly",
    ],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "unknown"],
    "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
    "glucose_category_log": ["Normal", "Prediabetes", "Diabetes"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
}

ohe_categories_list_selected = [
    explicit_categories_for_ohe_selected[col.lower()]
    for col in categorical_final_features_selected
]

perfect_data_preprocessor_selected = ColumnTransformer(
    transformers=[
        (
            "cat_encoder",
            OneHotEncoder(
                categories=ohe_categories_list_selected, handle_unknown="ignore"
            ),
            categorical_final_features_selected,
        )
    ],
    remainder="drop",
)


def pipeline_stroke_selected_features(estimator, undersample=True, random_state=42):
    steps = [
        ("feature_engineering", CombinedFeatureTransformer()),
        ("final_preprocessing", perfect_data_preprocessor_selected),
    ]

    if undersample:
        steps.append(("undersampler", RandomUnderSampler(random_state=random_state)))

    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


def pipeline_stroke_selected_features_ratio_2_to_1(
    estimator, undersample=True, random_state=42
):
    steps = [
        ("feature_engineering", CombinedFeatureTransformer()),
        ("final_preprocessing", perfect_data_preprocessor_selected),
    ]

    if undersample:
        steps.append(
            (
                "undersampler",
                RandomUnderSampler(sampling_strategy=0.5, random_state=random_state),
            )
        )

    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


def pipeline_stroke_selected_features_smote_tomek(
    estimator, undersample=True, random_state=42
):
    """
    Creates a pipeline with feature engineering, preprocessing, optional SMOTETomek balancing, and classifier.
    """
    steps = [
        ("feature_engineering", CombinedFeatureTransformer()),
        ("final_preprocessing", perfect_data_preprocessor_selected),
    ]

    if undersample:
        steps.append(("smote_tomek", SMOTETomek(random_state=random_state)))

    steps.append(("classifier", estimator))
    return ImbPipeline(steps)


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def evaluate_pipeline_selected_features(
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

    xgb_pipeline = pipeline_stroke_selected_features(
        estimator=estimator, undersample=undersample
    )

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


def evaluate_pipeline_selected_features_ratio_2_to_1(
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

    xgb_pipeline = pipeline_stroke_selected_features_ratio_2_to_1(
        estimator=estimator, undersample=undersample
    )

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


def evaluate_pipeline_selected_features_smotetomek(
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

    xgb_pipeline = pipeline_stroke_selected_features_smote_tomek(
        estimator=estimator, undersample=undersample
    )

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
    from catboost import CatBoostClassifier
    import optuna
    import warnings
    import os
    from contextlib import redirect_stderr
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
    from sklearn.exceptions import ConvergenceWarning

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
