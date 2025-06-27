# === Standard Library ===
import warnings
from matplotlib.patches import Patch
from typing import Any, Dict, List, Tuple

# === Data Manipulation & Scientific Computing ===
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# == Phi-K ==
import phik
from phik.report import plot_correlation_matrix
from phik import resources

# === Statistical Testing ===
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Scikit-learn Core ===
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

# === Scikit-learn Models ===
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold


# === XGBoost ===
from xgboost import XGBClassifier

# === Imbalanced-Learn ===
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def plot_numeric_boxplots(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Plots boxplots for a list of numeric columns in a DataFrame.

    Args:
        df (pd.DataFrame): The dataset containing the numeric columns.
        numeric_cols (List[str]): A list of column names to plot.

    Returns:
        None
    """
    sns.set(style="whitegrid", palette="mako")
    fig, axes = plt.subplots(
        nrows=1, ncols=len(numeric_cols), figsize=(6 * len(numeric_cols), 5)
    )
    colors = sns.color_palette("mako", n_colors=len(numeric_cols))

    for i, (col, color) in enumerate(zip(numeric_cols, colors)):
        sns.boxplot(data=df, y=col, ax=axes[i], color=color)
        axes[i].set_title(f"Boxplot of {col}", fontsize=14)

    plt.suptitle(
        "Distribution of Key Numeric Features", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    plt.show()


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    sns.set(style="whitegrid")

    n_cols = len(numeric_cols)
    fig_width = max(6 * n_cols, 8)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 5))

    if n_cols == 1:
        axes = [axes]

    palette = sns.color_palette("mako", n_cols)
    legend_handles = []

    for i, col in enumerate(numeric_cols):
        unique_vals = df[col].nunique()
        bins = min(unique_vals, 30)
        sns.histplot(
            data=df,
            x=col,
            bins=bins,
            kde=False,
            ax=axes[i],
            color=palette[i],
        )
        mean_val = df[col].mean()
        axes[i].axvline(mean_val, color="red", linestyle="--", linewidth=2)

        axes[i].set_title(f"Distribution of {col}", fontsize=16, fontweight="bold")
        legend_handles.append(Patch(color=palette[i], label=col))

    legend_handles.append(
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Mean")
    )

    fig.suptitle("Histograms of Key Numeric Features", fontsize=18, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(numeric_cols) + 1,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_categorical_counts(df: pd.DataFrame, categorical_cols: List[str]) -> None:
    """
    Plots count plots for a list of categorical columns in a DataFrame with count labels.

    Args:
        df (pd.DataFrame): The dataset containing the categorical columns.
        categorical_cols (List[str]): A list of column names to plot.

    Returns:
        None
    """
    label_map = {
        "Private Sector/Self Employed": "Private Sector\nor Self Employed",
        "Government Sector": "Government\nSector",
        "Yes": "Yes",
        "No": "No",
    }

    df_viz = df.copy()
    for col in categorical_cols:
        df_viz[col] = df_viz[col].replace(label_map)

    colors = sns.color_palette("mako", n_colors=len(categorical_cols))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.flatten()

    for i, (col, color) in enumerate(zip(categorical_cols, colors)):
        ax = axes[i]
        barplot = sns.countplot(data=df_viz, x=col, ax=ax, color=color)
        ax.set_title(f"Value Counts of {col}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="x", rotation=0)

        for container in barplot.containers:
            barplot.bar_label(container, padding=3, fontsize=10)

    if len(categorical_cols) < len(axes):
        fig.delaxes(axes[len(categorical_cols)])

    plt.suptitle(
        "Univariate Analysis of Categorical Features",
        fontsize=17,
        fontweight="bold",
        y=0.92,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def plot_phik_and_pearson_correlation(
    df, cols: List[str], title: str = "Phi-k and Pearson Correlation"
) -> None:
    phik_corr = df[cols].phik_matrix()
    numeric_cols = df[cols].select_dtypes(include=["int64", "float64"]).columns
    pearson_corr = df[numeric_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    mask = np.triu(np.ones_like(phik_corr, dtype=bool))
    sns.heatmap(
        phik_corr,
        mask=mask,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="mako_r",
        linewidths=0.5,
        square=True,
    )
    axes[0].set_title("Phi-k Correlation", fontsize=14, fontweight="bold")

    if not pearson_corr.empty:
        mask_pearson = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(
            pearson_corr,
            mask=mask_pearson,
            ax=axes[1],
            annot=True,
            fmt=".2f",
            cmap="mako_r",
            linewidths=0.5,
            square=True,
            vmin=-1,
            vmax=1,
        )
        axes[1].set_title("Pearson Correlation", fontsize=14, fontweight="bold")
    else:
        axes[1].axis("off")
        axes[1].set_title(
            "No numeric columns to compute Pearson correlation.", fontsize=12
        )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_insurance_by_categorical_features(
    df: pd.DataFrame, categorical_features: List[str]
) -> None:
    """
    Plots bar charts showing the proportion of customers buying travel insurance
    for each categorical feature, with consistent y-axis and selective axis labeling.

    Args:
        df (pd.DataFrame): DataFrame containing 'TravelInsurance' and categorical features.
        categorical_features (List[str]): List of categorical column names to visualize.

    Returns:
        None
    """

    colors = sns.color_palette("mako", n_colors=len(categorical_features))
    df_bar = df.copy()

    df_bar["Employment Type"] = df_bar["Employment Type"].replace(
        {
            "Private Sector/Self Employed": "Private Sector\nor Self Employed",
            "Government Sector": "Government\nSector",
        }
    )
    df_bar["GraduateOrNot"] = df_bar["GraduateOrNot"].replace(
        {"Yes": "Graduate", "No": "Not Graduate"}
    )
    df_bar["FrequentFlyer"] = df_bar["FrequentFlyer"].replace(
        {"Yes": "Frequent Flyer", "No": "Not Frequent"}
    )
    df_bar["EverTravelledAbroad"] = df_bar["EverTravelledAbroad"].replace(
        {"Yes": "Traveled Abroad", "No": "Never Traveled"}
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, color) in enumerate(zip(categorical_features, colors)):
        group_rates = df_bar.groupby(col)["TravelInsurance"].mean().reset_index()
        group_rates["TravelInsurance"] *= 100

        sns.barplot(
            data=group_rates,
            x=col,
            y="TravelInsurance",
            ax=axes[i],
            color=color,
            errorbar=None,
        )
        axes[i].set_title(f"Insurance Rate by {col}", fontweight="bold", fontsize=15)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=0)
        axes[i].set_ylim(0, 100)

        if i not in [0, 2]:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel("Proportion (%) Buying Insurance")

        for bar in axes[i].containers[0]:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.suptitle(
        "Proportion of Customers Buying Travel Insurance by Categorical Feature",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_violin_numerical_by_target(
    df: pd.DataFrame, target_col: str = "TravelInsurance"
) -> None:
    """
    Plots violin plots of numerical features split by the target class.

    Args:
        df (pd.DataFrame): The dataset containing the features and target.
        target_col (str, optional): The name of the binary target column. Defaults to "TravelInsurance".

    Returns:
        None
    """
    numeric_cols = ["Age", "AnnualIncome", "FamilyMembers"]
    colors = sns.color_palette("mako", n_colors=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vp = None

    for i, col in enumerate(numeric_cols):
        vp = sns.violinplot(
            data=df,
            x=target_col,
            y=col,
            hue=target_col,
            palette=colors,
            ax=axes[i],
            split=True,
        )
        axes[i].set_title(f"{col} by {target_col}", fontsize=15, fontweight="bold")
        axes[i].set_xlabel(f"{target_col} (0 = No, 1 = Yes)")
        axes[i].legend_.remove()

    handles, labels = vp.get_legend_handles_labels()
    fig.legend(
        handles,
        ["No", "Yes"],
        title="Travel Insurance",
        loc="upper center",
        bbox_to_anchor=(0.01, 0.98),
        ncol=2,
        frameon=False,
    )

    plt.suptitle(
        "Violin Plot Comparison of Numerical Features by Insurance Purchase",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.show()


def test_abroad_vs_insurance(df: pd.DataFrame) -> None:
    """
    Performs a Z-test to check if there's a significant difference in insurance uptake
    between individuals who have and have not traveled abroad.

    Args:
        df (pd.DataFrame): The dataset containing 'EverTravelledAbroad' and 'TravelInsurance'.

    Returns:
        None
    """
    df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({"Yes": 1, "No": 0})

    insured_abroad = df[
        (df["EverTravelledAbroad"] == 1) & (df["TravelInsurance"] == 1)
    ].shape[0]
    total_abroad = df[df["EverTravelledAbroad"] == 1].shape[0]

    insured_no_abroad = df[
        (df["EverTravelledAbroad"] == 0) & (df["TravelInsurance"] == 1)
    ].shape[0]
    total_no_abroad = df[df["EverTravelledAbroad"] == 0].shape[0]

    p1_hat = insured_abroad / total_abroad if total_abroad > 0 else 0
    p2_hat = insured_no_abroad / total_no_abroad if total_no_abroad > 0 else 0

    print("Traveled Abroad:")
    print("  Insured:", insured_abroad)
    print("  Not Insured:", total_abroad - insured_abroad)
    print("  n*p̂:", total_abroad * p1_hat)
    print("  n*(1-p̂):", total_abroad * (1 - p1_hat))

    print("\nNever Traveled Abroad:")
    print("  Insured:", insured_no_abroad)
    print("  Not Insured:", total_no_abroad - insured_no_abroad)
    print("  n*p̂:", total_no_abroad * p2_hat)
    print("  n*(1-p̂):", total_no_abroad * (1 - p2_hat))

    if all(
        [
            total_abroad * p1_hat >= 5,
            total_abroad * (1 - p1_hat) >= 5,
            total_no_abroad * p2_hat >= 5,
            total_no_abroad * (1 - p2_hat) >= 5,
        ]
    ):
        successes = [insured_abroad, insured_no_abroad]
        nobs = [total_abroad, total_no_abroad]
        z_stat, p_value = proportions_ztest(successes, nobs, alternative="larger")

        print("\nZ-test result:")
        print(f"  Z-statistic: {z_stat:.3f}")
        print(f"  P-value: {p_value:.10f}")
        if p_value < 0.05:
            print("  Result: Statistically significant difference.")
        else:
            print("  Result: Not statistically significant.")
    else:
        print("One or both groups do not meet the assumptions for the z-test.")


def test_income_difference(df: pd.DataFrame) -> None:
    """
    Tests if the average income differs between insured and uninsured customers
    using Welch's t-test when normality is assumed, or Mann–Whitney U test otherwise.

    Args:
        df (pd.DataFrame): Dataset with 'AnnualIncome' and 'TravelInsurance'.

    Returns:
        None
    """
    income_insured = df[df["TravelInsurance"] == 1]["AnnualIncome"]
    income_uninsured = df[df["TravelInsurance"] == 0]["AnnualIncome"]

    shapiro_insured = shapiro(income_insured.sample(min(500, len(income_insured))))
    shapiro_uninsured = shapiro(
        income_uninsured.sample(min(500, len(income_uninsured)))
    )

    print("Shapiro-Wilk Test (Insured):")
    print(
        f"  Statistic: {shapiro_insured.statistic:.3f}, P-value: {shapiro_insured.pvalue:.4f}"
    )
    print("Shapiro-Wilk Test (Uninsured):")
    print(
        f"  Statistic: {shapiro_uninsured.statistic:.3f}, P-value: {shapiro_uninsured.pvalue:.4f}"
    )

    normal = shapiro_insured.pvalue > 0.05 and shapiro_uninsured.pvalue > 0.05
    large_sample = len(income_insured) > 30 and len(income_uninsured) > 30

    if normal or large_sample:
        t_stat, p_value = ttest_ind(income_insured, income_uninsured, equal_var=False)
        one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)

        print("\nWelch's T-test:")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  One-tailed P-value: {one_tailed_p:.10f}")
        if one_tailed_p < 0.05:
            print(
                "  Result: Statistically significant. Insured individuals tend to have higher income."
            )
        else:
            print("  Result: Not statistically significant.")
    else:
        u_stat, p_value = mannwhitneyu(
            income_insured, income_uninsured, alternative="greater"
        )
        print("\nMann–Whitney U Test (non-parametric):")
        print(f"  U-statistic: {u_stat:.3f}")
        print(f"  One-tailed P-value: {p_value:.10f}")
        if p_value < 0.05:
            print(
                "  Result: Statistically significant. Insured individuals tend to have higher income."
            )
        else:
            print("  Result: Not statistically significant.")


def test_graduate_insurance_proportion(df: pd.DataFrame) -> None:
    """
    Tests if the proportion of insured customers differs between graduates and non-graduates.

    Args:
        df (pd.DataFrame): Dataset containing 'GraduateOrNot' and 'TravelInsurance'.

    Returns:
        None
    """
    df = df.copy()
    df["GraduateOrNot"] = df["GraduateOrNot"].map({"Yes": 1, "No": 0})

    insured_graduates = df[
        (df["GraduateOrNot"] == 1) & (df["TravelInsurance"] == 1)
    ].shape[0]
    total_graduates = df[df["GraduateOrNot"] == 1].shape[0]

    insured_nongraduates = df[
        (df["GraduateOrNot"] == 0) & (df["TravelInsurance"] == 1)
    ].shape[0]
    total_nongraduates = df[df["GraduateOrNot"] == 0].shape[0]

    p1_hat = insured_graduates / total_graduates if total_graduates > 0 else 0
    p2_hat = insured_nongraduates / total_nongraduates if total_nongraduates > 0 else 0

    print("Graduates:")
    print(f"  Insured: {insured_graduates}")
    print(f"  Not insured: {total_graduates - insured_graduates}")
    print(f"  n*p̂: {total_graduates * p1_hat:.2f}")
    print(f"  n*(1-p̂): {total_graduates * (1 - p1_hat):.2f}")

    print("\nNon-Graduates:")
    print(f"  Insured: {insured_nongraduates}")
    print(f"  Not insured: {total_nongraduates - insured_nongraduates}")
    print(f"  n*p̂: {total_nongraduates * p2_hat:.2f}")
    print(f"  n*(1-p̂): {total_nongraduates * (1 - p2_hat):.2f}")

    if all(
        [
            total_graduates * p1_hat >= 5,
            total_graduates * (1 - p1_hat) >= 5,
            total_nongraduates * p2_hat >= 5,
            total_nongraduates * (1 - p2_hat) >= 5,
        ]
    ):
        successes = [insured_graduates, insured_nongraduates]
        nobs = [total_graduates, total_nongraduates]

        z_stat, p_value = proportions_ztest(successes, nobs, alternative="larger")

        print("\nZ-test result:")
        print(f"  Z-statistic: {z_stat:.3f}")
        print(f"  P-value: {p_value:.10f}")
        if p_value < 0.05:
            print(
                "  Result: Statistically significant. Graduates are more likely to buy insurance."
            )
        else:
            print("  Result: Not statistically significant.")
    else:
        print("\nAssumptions not met: np or n(1-p) < 5. Consider an alternative test.")


def compute_ci_difference(df: pd.DataFrame, group_type: str = "abroad") -> None:
    """
    Computes and prints the 95% confidence interval for the difference in proportions
    of insured individuals between two groups (either 'abroad' vs. 'not abroad' or
    'graduate' vs. 'non-graduate').

    Args:
        df (pd.DataFrame): The dataset containing the relevant features.
        group_type (str): One of ["abroad", "graduate"] indicating which groups to compare.

    Raises:
        ValueError: If group_type is not one of the expected options.

    Returns:
        None
    """
    df = df.copy()

    if group_type == "abroad":
        df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
        group_col = "EverTravelledAbroad"
        label = "Travelled Abroad vs. Never Travelled"
    elif group_type == "graduate":
        df["GraduateOrNot"] = df["GraduateOrNot"].map({"No": 0, "Yes": 1})
        group_col = "GraduateOrNot"
        label = "Graduates vs. Non-Graduates"
    else:
        raise ValueError("group_type must be either 'abroad' or 'graduate'")

    group1 = df[df[group_col] == 1]
    group0 = df[df[group_col] == 0]

    insured_1 = group1["TravelInsurance"].sum()
    total_1 = group1.shape[0]
    insured_0 = group0["TravelInsurance"].sum()
    total_0 = group0.shape[0]

    if total_1 > 0 and total_0 > 0:
        ci_low, ci_upp = confint_proportions_2indep(
            count1=insured_1,
            nobs1=total_1,
            count2=insured_0,
            nobs2=total_0,
            method="wald",
        )
        print(
            f"95% CI for difference in proportions ({label}): ({ci_low:.3f}, {ci_upp:.3f})"
        )
    else:
        print("Cannot compute CI: one or both groups have zero observations.")


def compute_income_difference_ci(df: pd.DataFrame) -> None:
    """
    Computes and prints the 95% confidence interval for the difference in mean annual
    income between insured and uninsured individuals using Welch's t-interval formula.

    Args:
        df (pd.DataFrame): The dataset containing 'AnnualIncome' and 'TravelInsurance'.

    Returns:
        None
    """
    income_insured = df[df["TravelInsurance"] == 1]["AnnualIncome"]
    income_uninsured = df[df["TravelInsurance"] == 0]["AnnualIncome"]

    n1, n2 = len(income_insured), len(income_uninsured)
    mean1, mean2 = income_insured.mean(), income_uninsured.mean()
    var1, var2 = income_insured.var(ddof=1), income_uninsured.var(ddof=1)

    se = np.sqrt(var1 / n1 + var2 / n2)
    dof = (var1 / n1 + var2 / n2) ** 2 / (
        (var1**2 / (n1**2 * (n1 - 1))) + (var2**2 / (n2**2 * (n2 - 1)))
    )

    t_crit = stats.t.ppf(0.975, dof)
    mean_diff = mean1 - mean2
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    print(f"Mean Income Difference: {mean_diff:.2f}")
    print(f"95% CI for Difference in Means: ({ci_lower:.2f}, {ci_upper:.2f})")


def plot_confidence_intervals() -> None:
    colors = sns.color_palette("mako", 3)

    prop_labels = ["Traveled Abroad vs Not", "Graduate vs Non-Graduate"]
    prop_diffs = [0.528, 0.026]
    prop_errors = [0.574 - 0.528, 0.084 - 0.026]
    prop_cis = [(0.481, 0.574), (-0.033, 0.084)]

    income_label = ["Insured vs Uninsured"]
    income_diff_pct = [0.198]
    income_errors_pct = [0.219 - 0.198]
    income_cis_pct = [(0.177, 0.219)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(
        "Confidence Intervals for Group Differences (95%)",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    x_prop = np.arange(len(prop_labels))
    axes[0].bar(
        x_prop, prop_diffs, yerr=prop_errors, capsize=8, color=[colors[0], colors[1]]
    )
    axes[0].set_xticks(x_prop)
    axes[0].set_xticklabels(prop_labels)
    axes[0].set_ylabel("Difference in Proportions")
    axes[0].set_title("Proportional Differences", fontsize=16)
    axes[0].set_ylim(
        min(np.array(prop_diffs) - np.array(prop_errors)) - 0.05,
        max(np.array(prop_diffs) + np.array(prop_errors)) + 0.05,
    )

    for i, (ci_low, ci_upp) in enumerate(prop_cis):
        axes[0].text(
            i,
            prop_diffs[i] + prop_errors[i] + 0.02,
            f"{ci_low*100:.1f}% – {ci_upp*100:.1f}%",
            ha="center",
            fontsize=10,
        )

    x_income = np.arange(len(income_label))
    axes[1].bar(
        x_income, income_diff_pct, yerr=income_errors_pct, capsize=8, color=[colors[2]]
    )
    axes[1].set_xticks(x_income)
    axes[1].set_xticklabels(income_label)
    axes[1].set_ylabel("Difference in Mean Income (%)")
    axes[1].set_title("Income Difference", fontsize=16)
    axes[1].set_ylim(0, income_diff_pct[0] + income_errors_pct[0] + 0.05)

    axes[1].text(
        0,
        income_diff_pct[0] + income_errors_pct[0] + 0.01,
        f"{income_cis_pct[0][0]*100:.1f}% – {income_cis_pct[0][1]*100:.1f}%",
        ha="center",
        fontsize=10,
    )

    legend_handles = [
        Patch(color=colors[0], label="Travel Abroad Experience"),
        Patch(color=colors[1], label="Graduation"),
        Patch(color=colors[2], label="Income"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file and unnecessary index columns.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def create_preprocessing_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
    binary_features: list[str],
) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline for numeric, categorical, and binary features.

    Args:
        categorical_features (list[str]): Names of categorical features to one-hot encode.
        numerical_features (list[str]): Names of numerical features to standard scale.
        binary_features (list[str]): Names of binary features (e.g., Yes/No) to map.

    Returns:
        ColumnTransformer: A fitted transformer ready for data preprocessing.
    """

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "mapper",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x)
                    .replace({"Yes": 1, "No": 0})
                    .infer_objects(copy=False),
                    validate=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ]
    )

    return preprocessor


def split_and_preprocess_data(
    df: pd.DataFrame, preprocessor: ColumnTransformer
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Series,
    np.ndarray,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Splits the data into training, validation, and test sets; applies preprocessing;
    and returns resampled training and transformed validation/test sets.

    Args:
        df (pd.DataFrame): The full dataset with a 'TravelInsurance' target column.
        preprocessor (ColumnTransformer): Preprocessing pipeline created for features.

    Returns:
        tuple: (
            X_train_resampled, y_train_resampled,
            X_val_processed, y_val,
            X_test_processed, y_test,
            y_train_full, y_val, y_test
        )
    """
    X = df.drop("TravelInsurance", axis=1)
    y = df["TravelInsurance"]

    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y, test_size=0.36, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5555, stratify=y_temp, random_state=42
    )

    train_pipeline = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("smote", SMOTE(random_state=42)),
        ]
    )

    X_train_resampled, y_train_resampled = train_pipeline.fit_resample(
        X_train_full, y_train_full
    )

    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_resampled,
        y_train_resampled,
        X_val_processed,
        y_val,
        X_test_processed,
        y_test,
        y_train_full,
        y_val,
        y_test,
    )


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    ),
}


def plot_model_comparison(metrics: Dict[str, List]) -> None:
    """
    Plot a horizontal bar chart comparing performance metrics across multiple classification models.

    Args:
        metrics (Dict[str, List]): A dictionary where keys are metric names
            (e.g., "Accuracy", "F1 Score") and the values are lists of metric values
            for each model. Must include a "Model" key listing model names.

    Returns:
        None. Displays a seaborn bar plot visualizing the performance comparison.
    """
    df_metrics = pd.DataFrame(metrics)
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    ax = sns.barplot(
        data=df_melted, y="Model", x="Score", hue="Metric", palette="mako_r"
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=8)

    plt.title(
        "Model Comparison: Performance on Travel Insurance Prediction",
        fontsize=14,
        weight="bold",
    )
    plt.xlim(0.3, 1.00)
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0.75))
    plt.tight_layout()
    plt.show()
    plt.show()


def train_rf_with_grid_search(
    df: pd.DataFrame, target_col: str = "TravelInsurance"
) -> GridSearchCV:
    """
    Preprocesses the data, applies SMOTE to the training set, performs grid search
    with cross-validation on a Random Forest model, and evaluates it on a validation set.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target.
        target_col (str): Name of the target column.

    Returns:
        GridSearchCV: Trained GridSearchCV object with best parameters and CV results.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    numerical_features = ["Age", "AnnualIncome", "FamilyMembers"]
    categorical_features = ["Employment Type"]
    binary_features = ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "mapper",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x).replace({"Yes": 1, "No": 0}),
                    validate=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed, y_train
    )

    model_pipeline = Pipeline(
        [
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    grid_search_rf = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search_rf.fit(X_train_resampled, y_train_resampled)

    X_val_processed = preprocessor.transform(X_val)
    y_val_pred = grid_search_rf.predict(X_val_processed)

    print("Best hyperparameters:", grid_search_rf.best_params_)
    print("Best F1 score (CV):", grid_search_rf.best_score_)
    print(
        "Classification Report on Validation Set:\n",
        classification_report(y_val, y_val_pred),
    )

    return grid_search_rf


def train_and_evaluate_svm(
    X_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: pd.Series,
) -> GridSearchCV:
    """
    Performs grid search with cross-validation for an SVM model using resampled training data,
    then evaluates the best model on the test set.

    Parameters:
        X_train_resampled (np.ndarray): SMOTE-resampled training features.
        y_train_resampled (np.ndarray): SMOTE-resampled training labels.
        X_test_processed (np.ndarray): Preprocessed test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        GridSearchCV: Fitted GridSearchCV object containing the best SVM model and scores.
    """
    svm = SVC(probability=True)

    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1, 1],
        "kernel": ["rbf"],
    }

    grid_search_svm = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
        verbose=1,
    )

    grid_search_svm.fit(X_train_resampled, y_train_resampled)

    print("Best parameters:", grid_search_svm.best_params_)
    print("Best F1 score from CV:", grid_search_svm.best_score_)

    best_svm = grid_search_svm.best_estimator_
    y_pred_svm = best_svm.predict(X_test_processed)

    print("Classification Report:\n", classification_report(y_test, y_pred_svm))

    return grid_search_svm


def train_and_evaluate_xgboost(
    X_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
    X_test_processed: np.ndarray,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], float, Dict[str, Dict[str, float]]]:
    """
    Trains and evaluates an XGBoost classifier using grid search and F1 scoring.

    Parameters:
        X_train_resampled (np.ndarray): Resampled training features.
        y_train_resampled (np.ndarray): Resampled training labels.
        X_test_processed (np.ndarray): Preprocessed test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        Tuple containing:
            - best_params (Dict[str, Any]): Best hyperparameters from grid search.
            - best_score (float): Best F1 score from cross-validation.
            - report (Dict[str, Dict[str, float]]): Classification report as a dictionary.
    """
    xgb_param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }

    xgb = XGBClassifier(
        eval_metric="logloss", random_state=42, use_label_encoder=False, verbosity=0
    )

    xgb_grid = GridSearchCV(
        estimator=xgb,
        param_grid=xgb_param_grid,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    xgb_grid.fit(X_train_resampled, y_train_resampled)

    best_params = xgb_grid.best_params_
    best_score = xgb_grid.best_score_

    best_model = xgb_grid.best_estimator_
    y_pred = best_model.predict(X_test_processed)
    report = classification_report(y_test, y_pred, output_dict=True)

    return best_params, best_score, report


def plot_f1_comparison(
    model_names: List[str], before_f1: List[float], after_f1: List[float]
) -> None:
    """
    Plots F1 Score comparison before and after hyperparameter tuning for given models, horizontally.

    Parameters:
        model_names (List[str]): List of model names (must match the order in scores).
        before_f1 (List[float]): F1 scores before tuning for each model.
        after_f1 (List[float]): F1 scores after tuning for each model.

    Returns:
        None
    """
    df_plot = pd.DataFrame(
        {
            "Model": model_names * 2,
            "F1 Score": before_f1 + after_f1,
            "Stage": ["Before Tuning"] * len(model_names)
            + ["After Tuning"] * len(model_names),
        }
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_plot, y="Model", x="F1 Score", hue="Stage", palette="mako")

    plt.title(
        "F1 Score (Class 1) Before and After Hyperparameter Tuning", fontweight="bold"
    )
    plt.xlim(0.4, 0.65)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.legend(
        title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
    )

    plt.tight_layout()
    plt.show()


best_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10, random_state=42
    ),
    "SVM": SVC(C=1, gamma="scale", kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=150,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    ),
}


def evaluate_models_cv(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring_func=make_scorer(f1_score, pos_label=1),
) -> pd.DataFrame:
    """
    Evaluates models using cross-validation and F1 score.

    Parameters:
        models (Dict[str, object]): Dictionary of model name to estimator.
        X (pd.DataFrame): Preprocessed and resampled feature set.
        y (pd.Series): Corresponding target labels.
        cv (int): Number of folds for cross-validation. Default is 5.
        scoring_func: Scoring function for evaluation. Default is F1 score for class 1.

    Returns:
        pd.DataFrame: DataFrame with F1 scores per fold per model.
    """
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1,
        )
        cv_results[name] = scores

    results_df = pd.DataFrame(cv_results)
    print("5-Fold Cross-Validation F1 Scores:\n", results_df)
    print("\nMean F1 Scores:\n", results_df.mean().sort_values(ascending=False))
    print("\nStandard Deviation of F1 Scores:\n", results_df.std().sort_values())

    return results_df


def evaluate_voting_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, str]:
    """
    Fit and evaluate a soft voting ensemble classifier using Random Forest, Logistic Regression, and SVM.

    Args:
        X_train (np.ndarray): Preprocessed and resampled training features.
        y_train (np.ndarray): Resampled training labels.
        X_test (np.ndarray): Preprocessed test features.
        y_test (np.ndarray): Ground truth test labels.

    Returns:
        Tuple[float, str]: Accuracy and classification report on the test set.
    """
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf2 = LogisticRegression(max_iter=1000, random_state=42)
    clf3 = SVC(kernel="rbf", probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[("rf", clf1), ("lr", clf2), ("svc", clf3)],
        voting="soft",
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    return accuracy, report


def evaluate_weighted_ensemble_pipeline(
    df: pd.DataFrame, target_col: str = "TravelInsurance"
) -> Tuple[VotingClassifier, str]:
    """
    Builds and evaluates a soft-voting ensemble classifier using Random Forest, SVM, and XGBoost
    with preprocessing and SMOTE in a pipeline.

    Args:
        df (pd.DataFrame): Raw dataset including features and target.
        target_col (str): Name of the target column.

    Returns:
        Tuple[VotingClassifier, str]: Trained ensemble classifier and classification report on the test set.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numerical_features = ["Age", "AnnualIncome", "FamilyMembers"]
    categorical_features = ["Employment Type"]
    binary_features = ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
    binary_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "mapper",
                FunctionTransformer(
                    lambda x: pd.DataFrame(x).replace({"Yes": 1, "No": 0}),
                    validate=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10, random_state=42
    )
    svm = SVC(C=1, gamma="scale", kernel="rbf", probability=True, random_state=42)
    xgb = XGBClassifier(
        use_label_encoder=False,
        verbosity=0,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=150,
        eval_metric="logloss",
    )

    voting_clf = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)],
        voting="soft",
        weights=[1, 1, 2],
    )

    ensemble_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", voting_clf),
        ]
    )

    ensemble_pipeline.fit(X_train, y_train)
    y_pred = ensemble_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)

    print("Classification Report:\n", report)
    return voting_clf, report


def tune_threshold_on_validation_set(
    X_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
    X_val_processed: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[BaseEstimator, float, float]:
    """
    Trains a soft voting classifier and tunes the classification threshold
    based on F1-score performance on the validation set.

    Returns:
        Tuple containing:
        - trained voting classifier,
        - best classification threshold (float),
        - best F1 score on validation set (float)
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10, random_state=42
    )
    svm = SVC(C=1, gamma="scale", kernel="rbf", probability=True)
    xgb = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=150,
        eval_metric="logloss",
        random_state=42,
    )

    voting_clf = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)], voting="soft"
    )
    voting_clf.fit(X_train_resampled, y_train_resampled)

    y_val_proba = voting_clf.predict_proba(X_val_processed)[:, 1]

    thresholds = np.arange(0.3, 0.71, 0.05)
    results = []

    for thresh in thresholds:
        y_pred = (y_val_proba >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary"
        )
        accuracy = accuracy_score(y_val, y_pred)
        results.append((thresh, accuracy, precision, recall, f1))

    results = np.array(results)
    thresholds, accuracies, precisions, recalls, f1s = results.T

    colors = sns.color_palette("mako", n_colors=3)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, marker="o", label="F1-score", color=colors[0])
    plt.plot(
        thresholds, recalls, marker="s", label="Recall", linestyle="--", color=colors[1]
    )
    plt.plot(
        thresholds,
        precisions,
        marker="^",
        label="Precision",
        linestyle="--",
        color=colors[2],
    )
    plt.title(
        "Threshold Tuning (Validation Set): Precision, Recall, F1-score vs Threshold"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_index = np.argmax(f1s)
    return voting_clf, thresholds[best_index], f1s[best_index]


def evaluate_ensemble_with_threshold(
    file_path: str, threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Loads and preprocesses data, trains a soft-voting ensemble classifier,
    applies a probability threshold, and evaluates performance on test data.

    Args:
        file_path (str): Path to the input CSV dataset.
        threshold (float): Decision threshold to classify probabilities.

    Returns:
        Tuple[float, float]: Threshold and corresponding accuracy on the test set.
    """
    df = load_and_clean_data(file_path)

    X = df.drop("TravelInsurance", axis=1)
    y = df["TravelInsurance"]

    categorical_features = ["Employment Type"]
    numerical_features = ["Age", "AnnualIncome", "FamilyMembers"]
    binary_features = ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    preprocessor = create_preprocessing_pipeline(
        categorical_features, numerical_features, binary_features
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed, y_train
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10, random_state=42
    )
    svm = SVC(C=1, gamma="scale", kernel="rbf", probability=True, random_state=42)
    xgb = XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=150,
        eval_metric="logloss",
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)], voting="soft"
    )

    ensemble.fit(X_train_resampled, y_train_resampled)
    y_proba = ensemble.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(
        f"\nClassification Report (Threshold = {threshold}):\n",
        classification_report(y_test, y_pred),
    )

    return threshold, accuracy_score(y_test, y_pred)


def visualize_confusion_matrix_variants(cm: np.ndarray, labels: List[str]) -> None:
    """
    Visualizes a given confusion matrix in three formats:
    - Total-normalized (percentage of all predictions)
    - Row-normalized (recall-focused)
    - Column-normalized (precision-focused)

    Args:
        cm (np.ndarray): 2x2 confusion matrix as a NumPy array.
        labels (List[str]): List of class labels in order [negative, positive].

    Returns:
        None
    """
    custom_palette = sns.color_palette("mako")

    cm_total = cm / cm.sum() * 100
    cm_total_labels = np.array([[f"{val:.1f}%" for val in row] for row in cm_total])

    cm_row = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_row_labels = np.array([[f"{val:.1f}%" for val in row] for row in cm_row])

    cm_col = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :] * 100
    cm_col_labels = np.array([[f"{val:.1f}%" for val in row] for row in cm_col])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle("Confusion Matrix Visualizations", fontsize=17, weight="bold", y=0.93)

    sns.heatmap(
        cm_total,
        annot=cm_total_labels,
        fmt="",
        cmap=custom_palette,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axs[0],
        linewidths=1,
        linecolor="black",
    )
    axs[0].set_title("Total-Normalized (%)", fontsize=16)
    axs[0].set_xlabel("Predicted Label")
    axs[0].set_ylabel("True Label")

    sns.heatmap(
        cm_row,
        annot=cm_row_labels,
        fmt="",
        cmap=custom_palette,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axs[1],
        linewidths=1,
        linecolor="black",
    )
    axs[1].set_title("Recall-Focused (Row-Normalized %)", fontsize=16)
    axs[1].set_xlabel("Predicted Label")
    axs[1].set_ylabel("True Label")

    sns.heatmap(
        cm_col,
        annot=cm_col_labels,
        fmt="",
        cmap=custom_palette,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axs[2],
        linewidths=1,
        linecolor="black",
    )
    axs[2].set_title("Precision-Focused (Column-Normalized %)", fontsize=16)
    axs[2].set_xlabel("Predicted Label")
    axs[2].set_ylabel("True Label")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_cv_f1_scores(cv_df: pd.DataFrame) -> None:
    """
    Plots boxplots of F1 scores from cross-validation for each model, with a legend outside the plot.
    """
    sns.set(style="whitegrid")
    df_melted = cv_df.melt(var_name="Model", value_name="F1 Score")

    palette = sns.color_palette("mako", n_colors=df_melted["Model"].nunique())

    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=df_melted, x="Model", y="F1 Score", palette=palette)

    handles = [
        Patch(color=col, label=label) for col, label in zip(palette, cv_df.columns)
    ]
    plt.legend(
        handles=handles,
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.title(
        "5-Fold Cross-Validation F1 Score Distribution", fontsize=14, weight="bold"
    )
    plt.ylim(0.6, 0.85)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def cross_validate_voting_classifier(
    X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Tuple[float, np.ndarray]:
    """
    Perform stratified k-fold cross-validation on a soft voting ensemble classifier
    consisting of Random Forest, Logistic Regression, and SVM.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        cv (int): Number of cross-validation folds (default is 5).

    Returns:
        Tuple[float, np.ndarray]: Mean F1 score and array of F1 scores across folds.
    """
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf2 = LogisticRegression(max_iter=1000, random_state=42)
    clf3 = SVC(kernel="rbf", probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[("rf", clf1), ("lr", clf2), ("svc", clf3)], voting="soft"
    )

    f1_scorer = make_scorer(f1_score)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(voting_clf, X, y, cv=skf, scoring=f1_scorer)

    print(f"F1 scores per fold: {scores}")
    print(f"Mean F1 score: {scores.mean():.4f}")

    return scores.mean(), scores


def plot_vertical_model_comparison(metrics: Dict[str, List]) -> None:
    """
    Plot a vertical bar chart comparing performance metrics across multiple classification models.

    Args:
        metrics (Dict[str, List]): A dictionary where keys are metric names
            (e.g., "Accuracy", "F1 Score") and the values are lists of metric values
            for each model. Must include a "Model" key listing model names.

    Returns:
        None. Displays a seaborn bar plot visualizing the performance comparison.
    """
    df_metrics = pd.DataFrame(metrics)
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 6))
    sns.set(style="whitegrid")

    ax = sns.barplot(
        data=df_melted, x="Model", y="Score", hue="Metric", palette="mako_r"
    )

    for p in ax.patches:
        score = p.get_height()
        if score > 0.001:
            ax.annotate(
                f"{score:.2f}",
                (p.get_x() + p.get_width() / 2.0, score),
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
                fontweight="bold",
                xytext=(0, 4),
                textcoords="offset points",
            )

    plt.title(
        "Model Comparison: Performance on Travel Insurance Prediction",
        fontsize=17,
        weight="bold",
    )
    plt.ylim(0, 1.0)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Metric")
    plt.tight_layout()
    plt.show()
