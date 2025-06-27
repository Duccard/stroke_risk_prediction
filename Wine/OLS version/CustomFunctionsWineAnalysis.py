import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stats
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from phik import resources
from phik.report import plot_correlation_matrix
from phik import phik_matrix
import math
from scipy.stats import skew
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.stats import boxcox
from sklearn.preprocessing import PolynomialFeatures


def plot_feature_histograms(
    df, exclude_cols=None, color_palette="PuRd", title="Feature Distributions"
):
    """
    Plots histograms with KDE for all numerical features in a dataframe, excluding specified columns.

    Parameters:
        df (DataFrame): The input dataset.
        exclude_cols (list): List of column names to exclude from plotting (e.g., ['quality']).
        color_palette (str): Seaborn color palette to use for the plots.
        title (str): Supertitle for the entire plot grid.
    """
    sns.set(style="whitegrid")
    color = sns.color_palette(color_palette, 8)[5]

    if exclude_cols:
        features = df.columns.drop(exclude_cols)
    else:
        features = df.columns

    rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color=color)
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=22, y=1)
    plt.tight_layout()
    plt.show()


def plot_feature_violins_by_skew(
    df,
    exclude_cols=None,
    title="Feature Distributions via Violin Plots",
    cols_per_row=3,
):
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include="number").columns
    plot_cols = [col for col in numeric_cols if col not in exclude_cols]

    num_features = len(plot_cols)
    rows = math.ceil(num_features / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 5 * rows))
    axes = axes.flatten()

    base_palette = sns.color_palette("PuRd", 100)
    skewness_scores = [abs(skew(df[col])) for col in plot_cols]

    def get_color(score):
        if score >= 0.8:
            return base_palette[90]
        elif score >= 0.5:
            return base_palette[60]
        else:
            return base_palette[20]

    colors = [get_color(score) for score in skewness_scores]

    for i, col in enumerate(plot_cols):
        sns.violinplot(
            x=[""] * len(df[col]),
            y=df[col],
            hue=[""] * len(df[col]),
            palette=[colors[i]],
            legend=False,
            ax=axes[i],
        )
        col_title = col.replace("_", " ").title()
        axes[i].set_title(f"{col_title}", fontsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title.title(), fontsize=22, y=0.90)

    legend_elements = [
        Patch(facecolor=base_palette[20], label="Low skewness"),
        Patch(facecolor=base_palette[60], label="Moderate skewness"),
        Patch(facecolor=base_palette[90], label="High skewness"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False,
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.86])
    plt.show()


def plot_feature_vs_target_boxplots(
    df,
    target,
    exclude_cols=None,
    color_palette="PuRd",
    title="Feature vs. Target Boxplots",
    highlight_features=None,
):
    """
    Creates boxplots of each feature against a target (e.g., wine quality).
    Optionally highlights selected features (by bolding the title).

    Parameters:
        df (DataFrame): Dataset.
        target (str): Name of the categorical/ordinal target variable.
        exclude_cols (list): Columns to exclude from the feature list.
        color_palette (str): Seaborn color palette.
        title (str): Figure title.
        highlight_features (list): List of feature names to visually highlight.
    """

    sns.set(style="whitegrid")
    color = sns.color_palette(color_palette, 8)[5]

    features = df.select_dtypes(include="number").columns
    features = [
        f
        for f in features
        if f != target and (exclude_cols is None or f not in exclude_cols)
    ]

    rows = (len(features) + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.boxplot(
            x=target,
            y=feature,
            data=df,
            hue=target,
            palette=color_palette,
            ax=axes[i],
            legend=False,
        )
        title_str = f"{feature} vs. {target}"
        if highlight_features and feature in highlight_features:
            axes[i].set_title(title_str, weight="bold", fontsize=13, color="black")
        else:
            axes[i].set_title(title_str, fontsize=13)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=20, y=0.99)
    plt.tight_layout()
    plt.show()


def clean_and_cap_outliers(
    df, columns_to_cap, lower_quantile=0.01, upper_quantile=0.99
):
    """
    Drops duplicate rows and caps outliers for specified columns using given quantiles.

    Parameters:
        df (DataFrame): Input dataset.
        columns_to_cap (list): List of column names to apply outlier capping.
        lower_quantile (float): Lower quantile threshold (default 0.01).
        upper_quantile (float): Upper quantile threshold (default 0.99).

    Returns:
        DataFrame: Cleaned dataframe with duplicates removed and outliers capped.
    """
    df_cleaned = df.drop_duplicates().copy()

    def cap_outliers(series):
        lower = series.quantile(lower_quantile)
        upper = series.quantile(upper_quantile)
        return series.clip(lower, upper)

    for col in columns_to_cap:
        df_cleaned.loc[:, col] = cap_outliers(df_cleaned[col])

    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    return df_cleaned


def plot_combined_boxplots_comparison(
    df_raw,
    df_cleaned,
    exclude_cols=None,
    palette="PuRd",
    title="Boxplot Comparison: Raw vs Cleaned",
    xlim_max=300,
):
    """
    Plots two side-by-side boxplots for raw and cleaned datasets using the same x-axis scale.

    Parameters:
    - df_raw (DataFrame): Raw/original dataset before cleaning.
    - df_cleaned (DataFrame): Cleaned dataset after processing.
    - exclude_cols (list): List of column names to exclude (e.g., ['quality']).
    - palette (str): Color palette for Seaborn plots.
    - title (str): Main title for the plot.
    - xlim_max (float): Maximum x-axis limit for both plots to standardize comparison.
    """

    sns.set(style="whitegrid")

    features = df_raw.select_dtypes(include="number").columns
    if exclude_cols:
        features = [f for f in features if f not in exclude_cols]

    df_raw_melted = df_raw[features].melt(var_name="Feature", value_name="Value")
    df_cleaned_melted = df_cleaned[features].melt(
        var_name="Feature", value_name="Value"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    sns.boxplot(
        x="Value",
        y="Feature",
        hue="Feature",
        data=df_raw_melted,
        ax=axes[0],
        palette=palette,
    )
    axes[0].set_title("Before Cleaning")
    axes[0].set_xlabel("")
    axes[0].set_xlim(0, xlim_max)

    sns.boxplot(
        x="Value",
        y="Feature",
        hue="Feature",
        data=df_cleaned_melted,
        ax=axes[1],
        palette=palette,
    )
    axes[1].set_title("After Cleaning")
    axes[1].set_xlabel("")
    axes[1].set_xlim(0, xlim_max)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, title="Pearson Correlation Heatmap", cmap="PuRd"):
    """
    Plots a lower-triangle Pearson correlation heatmap for a given DataFrame,
    with a color scale fixed from -1 to 1.

    Parameters:
    - df: pandas DataFrame containing numeric features
    - title: string, title of the plot
    - cmap: string, seaborn/matplotlib color palette (default "PuRd")
    """
    corr_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        square=True,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def calculate_vif(df, target_col=None):
    """
    Calculates Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters:
    - df: pandas DataFrame with numeric columns
    - target_col: optional column to exclude (e.g., 'quality')

    Returns:
    - vif_df: DataFrame with features and their VIF scores
    """
    df = df.copy()
    if target_col:
        df = df.drop(columns=[target_col])

    X_with_const = add_constant(df)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
    ]

    return vif_data


def plot_feature_distributions(df, features, bins=30, palette_color=3):
    """
    Plots histograms with KDE for a list of features in a 2x2 grid layout.

    Parameters:
    - df: pandas DataFrame containing the data
    - features: list of column names to plot (up to 4 features)
    - bins: number of histogram bins (default: 30)
    - palette_color: index for PuRd color palette (default: 3)
    """
    if len(features) != 4:
        raise ValueError("Exactly 4 features must be provided for a 2x2 grid.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(
            df[col],
            kde=True,
            bins=bins,
            ax=axes[i],
            color=sns.color_palette("PuRd")[palette_color],
        )
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def transform_skewed_features(
    df, method="boxcox", exclude_cols=None, skew_threshold=0.5
):
    df_transformed = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df_transformed.select_dtypes(include="number").columns
    transform_cols = [
        col
        for col in numeric_cols
        if col not in exclude_cols and abs(skew(df_transformed[col])) > skew_threshold
    ]

    for col in transform_cols:
        if method == "log":
            df_transformed[col] = np.log1p(df_transformed[col])
        elif method == "boxcox":
            shift = (
                1 - df_transformed[col].min() if df_transformed[col].min() <= 0 else 0
            )
            df_transformed[col], _ = boxcox(df_transformed[col] + shift)
        elif method == "sqrt":
            df_transformed[col] = np.sqrt(df_transformed[col])
    return df_transformed


def plot_significant_coefficients(model, alpha=0.05, palette_name="PuRd"):
    """
    Plots statistically significant regression coefficients (p < alpha)
    with 95% confidence intervals and a color palette.
    """
    coef = model.params
    conf = model.conf_int()
    pvalues = model.pvalues

    sig_mask = (pvalues < alpha) & (pvalues.index != "const")
    sig_coef = coef[sig_mask]
    sig_conf = conf.loc[sig_mask]

    if sig_coef.empty:
        print("No significant coefficients found.")
        return

    coef_df = pd.DataFrame(
        {"coef": sig_coef, "lower": sig_conf[0], "upper": sig_conf[1]}
    ).sort_values(by="coef")

    colors = sns.color_palette(palette_name, len(coef_df))

    plt.figure(figsize=(10, 0.5 * len(coef_df) + 2))
    for i, (feature, row) in enumerate(coef_df.iterrows()):
        plt.barh(
            y=feature,
            width=row["coef"],
            color=colors[i],
            xerr=[[row["coef"] - row["lower"]], [row["upper"] - row["coef"]]],
            capsize=4,
            alpha=0.9,
        )

    plt.axvline(x=0, color="gray", linestyle="--")
    plt.title("Significant OLS Coefficients (p < 0.05)", fontsize=16)
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()


def plot_qq_residuals(
    y_true, y_pred, title="Q-Q Plot of Residuals", palette_name="PuRd", color_index=-4
):
    """
    Plots a Q-Q plot of residuals using a selected color from a seaborn palette.

    Parameters:
    - y_true: array-like of true target values
    - y_pred: array-like of predicted values
    - title: str, title for the Q-Q plot
    - palette_name: str, seaborn palette name
    - color_index: int, index for the color from the palette
    """

    residuals = y_true - y_pred
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    color = sns.color_palette(palette_name)[color_index]

    plt.figure(figsize=(6, 6))
    plt.scatter(osm, osr, alpha=0.7, color=color)
    plt.plot(osm, slope * osm + intercept, "r--", lw=2)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def prepare_polynomial_features(
    X_train_scaled_df, X_test_scaled_df, degree=2, interaction_only=True
):
    """
    Generates polynomial interaction-only features and adds constant term for OLS modeling.

    Parameters:
        X_train_scaled_df (pd.DataFrame): Scaled training features with original column names.
        X_test_scaled_df (pd.DataFrame): Scaled test features with original column names.
        degree (int): Degree of polynomial features.
        interaction_only (bool): Whether to include only interaction terms.

    Returns:
        X_train_poly_scaled_const (pd.DataFrame): Transformed training features with constant.
        X_test_poly_scaled_const (pd.DataFrame): Transformed test features with constant.
        poly (PolynomialFeatures): Fitted transformer for reference.
    """
    poly = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train_scaled_df)
    X_test_poly = poly.transform(X_test_scaled_df)

    feature_names = poly.get_feature_names_out(X_train_scaled_df.columns)

    X_train_poly_df = pd.DataFrame(
        X_train_poly, columns=feature_names, index=X_train_scaled_df.index
    )
    X_test_poly_df = pd.DataFrame(
        X_test_poly, columns=feature_names, index=X_test_scaled_df.index
    )

    X_train_poly_scaled_const = sm.add_constant(X_train_poly_df)
    X_test_poly_scaled_const = sm.add_constant(X_test_poly_df)

    return X_train_poly_scaled_const, X_test_poly_scaled_const, poly


def plot_significant_coefficients_poly(ols_model, alpha=0.05, palette_name="PuRd"):
    """
    Plots only statistically significant coefficients from an OLS model.

    Parameters:
        ols_model: statsmodels regression result
        alpha (float): significance threshold
        palette_name (str): seaborn color palette to use
    """
    coef_df = ols_model.summary2().tables[1].copy()
    coef_df = coef_df.rename(columns={"Coef.": "coef", "P>|t|": "pval"})
    coef_df = coef_df[coef_df.index != "const"]

    significant = coef_df[coef_df["pval"] < alpha].sort_values("coef")
    if significant.empty:
        print("No significant coefficients found at the specified alpha level.")
        return

    plt.figure(figsize=(8, len(significant) * 0.4 + 1))
    significant["feature"] = significant.index
    colors = sns.color_palette(palette_name, len(significant))
    sns.barplot(
        x="coef",
        y="feature",
        hue="feature",
        data=significant,
        palette=colors,
        legend=False,
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.title("Significant Coefficients (p < 0.05)", fontsize=16)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def compare_model_predictions(y_test, y_pred_baseline, y_pred_poly):
    """
    Creates a side-by-side scatter plot comparing actual vs predicted values
    for baseline and polynomial OLS models using custom purple and pink palette.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_baseline, color="#9b59b6", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.title("Baseline OLS: Actual vs Predicted")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_poly, color="#e91e63", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
    plt.title("Polynomial OLS: Actual vs Predicted")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")

    plt.tight_layout()
    plt.show()
