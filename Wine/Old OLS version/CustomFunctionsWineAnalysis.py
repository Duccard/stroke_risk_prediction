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

    plt.suptitle(title, fontsize=18, y=1)
    plt.tight_layout()
    plt.show()


def plot_feature_violins(
    df, exclude_cols=None, title="Feature Distributions via Violin Plots"
):
    """
    Plots violin plots for all numeric columns in the dataframe except excluded ones.

    Parameters:
    - df: pandas DataFrame
    - exclude_cols: list of column names to exclude from plotting
    - title: string, title of the whole plot
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include="number").columns
    plot_cols = [col for col in numeric_cols if col not in exclude_cols]

    df_melted = df[plot_cols].melt(var_name="Feature", value_name="Value")

    plt.figure(figsize=(14, 8))
    sns.violinplot(
        x="Feature",
        y="Value",
        data=df_melted,
        palette="PuRd",
        hue="Feature",
        legend=False,
    )
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
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
    import seaborn as sns
    import matplotlib.pyplot as plt

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

    plt.suptitle(title, fontsize=16, y=1.02)
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

    fig.suptitle(title, fontsize=16)
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


def plot_spearman_phik_heatmaps(
    df, drop_cols=None, label_col="quality", cutoff=7, cmap="PuRd"
):
    """
    Plots lower-triangle Spearman and Phi-K correlation heatmaps side by side,
    with appropriate color scales and alignment.

    Parameters:
    - df: pandas DataFrame with numeric columns
    - drop_cols: list of column names to exclude
    - label_col: column used to create a binary classification target
    - cutoff: threshold to define quality_label
    - cmap: heatmap color palette
    """

    df = df.copy()
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    df["quality_label"] = (df[label_col] >= cutoff).astype(int)

    columns = df.columns.tolist()

    spearman_corr = df.corr(method="spearman").loc[columns, columns]
    mask_spearman = np.triu(np.ones_like(spearman_corr, dtype=bool))

    phik_corr = df.phik_matrix(interval_cols=columns).loc[columns, columns]
    mask_phik = np.triu(np.ones_like(phik_corr, dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        spearman_corr,
        mask=mask_spearman,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        vmin=-1,
        vmax=1,
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Spearman Correlation")

    sns.heatmap(
        phik_corr,
        mask=mask_phik,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Phi-K Correlation")

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


def run_shapiro_on_model(
    df, target_col, drop_cols=None, test_size=0.2, random_state=42
):
    """
    Fits an OLS model after dropping specified columns and returns Shapiro-Wilk test results on residuals.

    Parameters:
    - df (DataFrame): Dataset with features and target
    - target_col (str): Name of the dependent variable
    - drop_cols (list): Columns to exclude from the model (optional)
    - test_size (float): Proportion of test data (default 0.2)
    - random_state (int): Random seed for reproducibility

    Returns:
    - (statistic, p-value) from Shapiro-Wilk test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)

    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train).fit()

    shapiro_stat, shapiro_p = shapiro(model.resid)
    return shapiro_stat, shapiro_p


def plot_cooks_distance(cooks_d, n_obs=None, palette="PuRd"):
    """
    Plots Cook's Distance values for all observations with threshold line.

    Parameters:
    - cooks_d (array-like): Array or Series of Cook's Distance values.
    - n_obs (int): Total number of observations (optional, auto-inferred if not provided).
    - palette (str): Seaborn color palette for the lines.
    """
    if n_obs is None:
        n_obs = len(cooks_d)

    color = sns.color_palette(palette)[3]

    plt.figure(figsize=(10, 5))
    plt.vlines(
        x=np.arange(len(cooks_d)), ymin=0, ymax=cooks_d, color=color, linewidth=1
    )
    plt.axhline(y=4 / n_obs, color="gray", linestyle="--", label="Threshold (4/n)")
    plt.title("Cook's Distance for All Observations")
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(model):
    """
    Plots a 2x2 grid of regression diagnostic plots for a given fitted model.

    Parameters:
    - model: A statsmodels regression result object (e.g., OLS)
    """
    residuals = model.resid
    fitted_vals = model.fittedvalues

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals vs Fitted
    sns.scatterplot(
        x=fitted_vals, y=residuals, ax=axes[0, 0], color=sns.color_palette("PuRd")[3]
    )
    axes[0, 0].axhline(0, linestyle="--", color="gray")
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")

    # 2. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color=sns.color_palette("PuRd")[2])
    axes[0, 1].set_title("Histogram of Residuals")
    axes[0, 1].set_xlabel("Residuals")

    # 3. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot of Residuals")

    # 4. Scale-Location Plot
    sns.scatterplot(
        x=fitted_vals,
        y=np.sqrt(np.abs(residuals)),
        ax=axes[1, 1],
        color=sns.color_palette("PuRd")[4],
    )
    axes[1, 1].set_title("Scale-Location Plot")
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("√|Residuals|")

    plt.tight_layout()
    plt.show()


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


def plot_roc_curve_decision_tree(
    df, feature_cols=None, label_col="quality", threshold=7, max_depth=5
):
    """
    Fits a Decision Tree Classifier and plots the ROC curve.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - feature_cols: list of feature column names. If None, all except 'quality' and the new label will be used.
    - label_col: name of the column with quality scores.
    - threshold: integer threshold to classify high vs low quality (default: 7).
    - max_depth: max depth for DecisionTreeClassifier.
    """
    df["quality_label"] = (df[label_col] >= threshold).astype(int)

    if feature_cols is None:
        feature_cols = df.drop(columns=[label_col, "quality_label"]).columns
    X = df[feature_cols]
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    roc_color = sns.color_palette("PuRd")[3]
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color=roc_color)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Decision Tree Classifier")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return clf, auc_score
