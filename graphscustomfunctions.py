# Standard Library
import warnings
from typing import List, Optional, Tuple, Sequence, Union

# Scientific / Data
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.inspection import permutation_importance

# Phik (for correlation)
from phik import phik_matrix


def plot_numerical_boxplots(
    df,
    num_cols: List[str],
    title: Optional[str] = "Boxplots of Numerical Features",
    palette: Optional[str] = "magma",
) -> None:
    """
    Plot side-by-side boxplots for numerical features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        num_cols (List[str]): List of numerical column names to plot.
        title (Optional[str]): The overall title for the figure.
        palette (Optional[str]): Seaborn color palette to use.
    """
    colors = sns.color_palette(palette, n_colors=len(num_cols))

    plt.figure(figsize=(5 * len(num_cols), 5))

    for i, (col, color) in enumerate(zip(num_cols, colors), 1):
        plt.subplot(1, len(num_cols), i)
        sns.boxplot(y=df[col], color=color)
        plt.title(f"Boxplot of {col}", fontsize=16, y=1.01)

    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.05, weight="bold")
    plt.show()


def plot_numerical_distributions_with_mean(
    df,
    num_cols: List[str],
    title: Optional[str] = "Distributions of Numerical Features with Mean Lines",
    palette: Optional[str] = "magma",
    bins: int = 30,
) -> None:
    """
    Plot histograms with KDE curves and mean lines for given numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        num_cols (List[str]): List of numerical columns to plot.
        title (Optional[str]): Overall title for the figure.
        palette (Optional[str]): Seaborn color palette for coloring the plots.
        bins (int): Number of histogram bins.
    """
    colors = sns.color_palette(palette, n_colors=len(num_cols))

    fig, axes = plt.subplots(1, len(num_cols), figsize=(6 * len(num_cols), 5))

    for ax, col, color in zip(axes, num_cols, colors):
        sns.histplot(df[col], bins=bins, kde=True, color=color, ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=16, y=1.02)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)

        mean_val = df[col].mean()
        ax.axvline(mean_val, color="red", linestyle="--")

    plt.suptitle(title, fontsize=20, weight="bold", y=1.02)

    mean_line_legend = mlines.Line2D([], [], color="red", linestyle="--", label="Mean")
    fig.legend(
        handles=[mean_line_legend], loc="upper right", fontsize=12, frameon=False
    )

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df,
    cat_cols: List[str],
    title: Optional[str] = "Distributions of Categorical Features with Percentages",
    palette_name: Optional[str] = "magma",
    n_cols: int = 3,
) -> None:
    """
    Plot countplots with percentages for categorical features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cat_cols (List[str]): List of categorical column names to plot.
        title (Optional[str]): Overall title for the figure.
        palette_name (Optional[str]): Name of seaborn color palette to use.
        n_cols (int): Number of columns in the subplot grid.
    """
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    total = len(df)

    for i, col in enumerate(cat_cols, 1):
        n_colors = df[col].nunique()
        palette = sns.color_palette(palette_name, n_colors=n_colors)

        plt.subplot(n_rows, n_cols, i)
        ax = sns.countplot(x=col, data=df, hue=col, palette=palette, legend=False)

        for p in ax.patches:
            height = p.get_height()
            percent = 100 * height / total
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                height + total * 0.005,
                f"{percent:.1f}%",
                ha="center",
                fontsize=10,
            )

        ax.set_title(f"Countplot of {col}", fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.suptitle(title, fontsize=20, weight="bold", y=1.02)
    plt.show()


def plot_binary_distributions(
    df,
    binary_cols: List[str],
    title: Optional[str] = "Binary Variables Distribution (Univariate Analysis)",
    palette: Optional[str] = "magma",
) -> None:
    """
    Plot countplots with percentages for binary features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        binary_cols (List[str]): List of binary column names to plot.
        title (Optional[str]): Overall title for the figure.
        palette (Optional[str]): Color palette for the plots.
    """
    plt.figure(figsize=(6 * len(binary_cols), 5))
    total = len(df)

    for i, col in enumerate(binary_cols, 1):
        plt.subplot(1, len(binary_cols), i)
        ax = sns.countplot(x=col, data=df, palette=palette)
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=14)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for p in ax.patches:
            height = p.get_height()
            percent = 100 * height / total
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                height + total * 0.005,
                f"{percent:.1f}%",
                ha="center",
                fontsize=11,
            )

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, weight="bold", y=1.05)
    plt.show()


def plot_violinplots_numerical_by_stroke(
    df,
    num_cols: List[str],
    target_col: str = "stroke",
    title: Optional[str] = "Violin Plots of Numerical Features by Stroke Status",
    palette: Optional[str] = "magma",
) -> None:
    """
    Plot violin plots for numerical features by stroke status.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        num_cols (List[str]): List of numerical columns to plot.
        target_col (str): The target categorical variable to split by (default "stroke").
        title (Optional[str]): Overall title for the figure.
        palette (Optional[str]): Seaborn color palette for the plots.
    """
    plt.figure(figsize=(6 * len(num_cols), 5))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(1, len(num_cols), i)
        sns.violinplot(
            x=target_col, y=col, data=df, palette=palette, hue=target_col, legend=False
        )
        plt.title(f"{col} vs {target_col.title()}", fontsize=12)
        plt.xlabel(target_col.title(), fontsize=11)
        plt.ylabel(col, fontsize=11)

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, weight="bold", y=1.02)
    plt.show()


def plot_countplots_categorical_by_stroke(
    df,
    cat_cols: List[str],
    target_col: str = "stroke",
    title: Optional[
        str
    ] = "Categorical Feature Distributions by Stroke Status with Percentages",
    palette_name: Optional[str] = "magma",
) -> None:
    """
    Plot countplots for categorical features split by stroke status, with percentages.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cat_cols (List[str]): List of categorical columns to plot.
        target_col (str): Target variable for hue (default 'stroke').
        title (Optional[str]): Overall title for the figure.
        palette_name (Optional[str]): Name of seaborn color palette.
    """
    n_rows = (len(cat_cols) + 2) // 3
    plt.figure(figsize=(6 * 3, 5 * n_rows))

    total = len(df)
    n_colors = df[target_col].nunique()
    palette = sns.color_palette(palette_name, n_colors=n_colors)

    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, 3, i)
        ax = sns.countplot(x=col, hue=target_col, data=df, palette=palette)
        ax.set_title(f"{col} by {target_col.title()}", fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        ax.legend(title=target_col.title(), fontsize=10, title_fontsize=11)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percent = 100 * height / total
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + total * 0.005,
                    f"{percent:.1f}%",
                    ha="center",
                    fontsize=11,
                )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.suptitle(title, fontsize=20, weight="bold", y=1.02)
    plt.show()


def plot_stroke_rate_by_category(
    df,
    cat_cols: List[str],
    target_col: str = "stroke",
    title: Optional[
        str
    ] = "Categorical Feature Stroke Rates (Proportional Within Each Group)",
    palette_name: Optional[str] = "magma",
    n_cols: int = 2,
) -> None:
    """
    Plot barplots of stroke rates (mean of target) for categorical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_cols (List[str]): List of categorical column names to plot.
        target_col (str): Target column for stroke (default "stroke").
        title (Optional[str]): Overall title for the figure.
        palette_name (Optional[str]): Name of seaborn color palette to use.
        n_cols (int): Number of columns in the grid layout.
    """
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        prop_df = (
            df.groupby(col)[target_col]
            .mean()
            .reset_index()
            .rename(columns={target_col: "Stroke Rate"})
            .sort_values("Stroke Rate", ascending=False)
        )

        n_colors = prop_df[col].nunique()
        palette = sns.color_palette(palette_name, n_colors=n_colors)

        sns.barplot(x=col, y="Stroke Rate", data=prop_df, palette=palette, ax=ax)

        ax.set_title(
            f"Stroke Rate by {col.replace('_', ' ').title()}", fontsize=16, y=1.03
        )
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=13)
        ax.set_ylabel("Stroke Rate", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=30)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 0.002,
                    f"{height * 100:.1f}%",
                    ha="center",
                    fontsize=11,
                )

    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.suptitle(title, fontsize=20, weight="bold", y=0.92)
    plt.show()


def plot_stroke_rate_by_category(
    df,
    cat_cols: List[str],
    target_col: str = "stroke",
    title: Optional[
        str
    ] = "Categorical Feature Stroke Rates (Proportional Within Each Group)",
    palette_name: Optional[str] = "magma",
    n_cols: int = 2,
) -> None:
    """
    Plot barplots of stroke rates (mean of target) for categorical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_cols (List[str]): List of categorical column names to plot.
        target_col (str): Target column for stroke (default "stroke").
        title (Optional[str]): Overall title for the figure.
        palette_name (Optional[str]): Name of seaborn color palette to use.
        n_cols (int): Number of columns in the grid layout.
    """
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        prop_df = (
            df.groupby(col)[target_col]
            .mean()
            .reset_index()
            .rename(columns={target_col: "Stroke Rate"})
            .sort_values("Stroke Rate", ascending=False)
        )

        n_colors = prop_df[col].nunique()
        palette = sns.color_palette(palette_name, n_colors=n_colors)

        sns.barplot(x=col, y="Stroke Rate", data=prop_df, palette=palette, ax=ax)

        ax.set_title(
            f"Stroke Rate by {col.replace('_', ' ').title()}", fontsize=15, y=1.015
        )
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Stroke Rate", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=30)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 0.002,
                    f"{height * 100:.1f}%",
                    ha="center",
                    fontsize=11,
                )

    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.suptitle(title, fontsize=20, weight="bold", y=0.92)
    plt.show()


def plot_stripplot_by_stroke(
    df,
    numerical_col: str,
    target_col: str = "stroke",
    hue_col: Optional[str] = None,
    title: Optional[str] = None,
    palette: Optional[str] = "magma",
    legend_loc: Optional[str] = "upper left",
    jitter: float = 0.25,
    alpha: float = 0.7,
) -> None:
    """
    Plot a stripplot of a numerical feature split by stroke status,
    with optional hue and customizable legend location.

    Args:
        df (pd.DataFrame): DataFrame with the data.
        numerical_col (str): Numerical column to plot on the y-axis.
        target_col (str): Target variable on the x-axis (default 'stroke').
        hue_col (Optional[str]): Column for hue split (e.g. 'age_group').
        title (Optional[str]): Title of the plot.
        palette (Optional[str]): Seaborn color palette.
        legend_loc (Optional[str]): Location of the legend.
        jitter (float): Jitter parameter for stripplot.
        alpha (float): Transparency for points.
    """
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        x=target_col,
        y=numerical_col,
        data=df,
        hue=hue_col,
        palette=palette,
        dodge=True if hue_col else False,
        jitter=jitter,
        alpha=alpha,
    )
    plt.title(
        title or f"{numerical_col.title()} Distribution by {target_col.title()}",
        fontsize=14,
    )
    plt.xlabel(target_col.title(), fontsize=12)
    plt.ylabel(numerical_col.title(), fontsize=12)

    if hue_col:
        plt.legend(
            title=hue_col.replace("_", " ").title(),
            fontsize=10,
            title_fontsize=11,
            loc=legend_loc,
        )
    else:
        plt.legend_.remove()

    plt.show()


def plot_binary_strip_and_countplot_rate(
    df: pd.DataFrame,
    binary_cols: List[str],
    age_col: str = "age",
    stroke_col: str = "stroke",
    palette: Optional[str] = "magma",
    title: Optional[str] = (
        "Hypertension and Heart Disease Analysis\n"
        "Top: Age Stripplots by Stroke  |  Bottom: Stroke Rates Within Group"
    ),
) -> None:
    """
    Plot a 2x2 grid with:
    - Top: Stripplots of Age vs Binary Features colored by Stroke Status
    - Bottom: Barplots of Stroke Rates within each Binary Feature Group

    Args:
        df (pd.DataFrame): The data.
        binary_cols (List[str]): List of binary columns to plot.
        age_col (str): Age column for y-axis in stripplots.
        stroke_col (str): Column indicating stroke status.
        palette (Optional[str]): Color palette for plots.
        title (Optional[str]): Overall plot title.
    """
    palette_colors = sns.color_palette(palette, n_colors=2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(binary_cols):
        ax = axes[i]
        sns.stripplot(
            x=col,
            y=age_col,
            data=df,
            hue=stroke_col,
            dodge=True,
            jitter=0.25,
            alpha=0.7,
            palette=palette_colors,
            ax=ax,
        )
        ax.set_title(
            f"{col.replace('_', ' ').title()} vs {age_col.title()} by Stroke Status",
            fontsize=14,
        )
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(age_col.title(), fontsize=12)
        ax.legend(title=stroke_col.title(), fontsize=10, title_fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i, col in enumerate(binary_cols):
        ax = axes[i + 2]
        prop_df = (
            df.groupby(col)[stroke_col]
            .mean()
            .reset_index()
            .rename(columns={stroke_col: "Stroke Rate"})
        )
        sns.barplot(x=col, y="Stroke Rate", data=prop_df, palette=palette_colors, ax=ax)
        ax.set_title(f"Stroke Rate by {col.replace('_', ' ').title()}", fontsize=14)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Stroke Rate", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 0.005,
                    f"{height*100:.1f}%",
                    ha="center",
                    fontsize=10,
                )

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, weight="bold", y=1.05)
    plt.show()


def plot_pearson_correlation_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
    title: Optional[str] = "Pearson Correlation Heatmap (Numerical Features)",
    cmap: Optional[str] = "magma_r",
    figsize: Optional[tuple] = (8, 6),
) -> None:
    """
    Plot a heatmap of Pearson correlations between numerical features.

    Args:
        df (pd.DataFrame): The DataFrame containing your data.
        numerical_cols (List[str]): List of numerical column names to include.
        title (Optional[str]): Title of the heatmap.
        cmap (Optional[str]): Colormap to use.
        figsize (Optional[tuple]): Figure size.
    """
    corr_matrix = df[numerical_cols].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_phik_heatmap(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    title: Optional[str] = "Phi-k Correlation Heatmap (Selected Features)",
    cmap: Optional[str] = "magma_r",
    figsize: Optional[tuple] = (10, 8),
) -> None:
    """
    Plot a Phi-k correlation heatmap for the DataFrame after excluding specified columns.

    Args:
        df (pd.DataFrame): The DataFrame with your data.
        exclude_cols (Optional[List[str]]): Columns to exclude before computing Phi-k.
        title (Optional[str]): Title of the plot.
        cmap (Optional[str]): Colormap for heatmap.
        figsize (Optional[tuple]): Size of the figure.
    """
    if exclude_cols:
        df = df.drop(columns=exclude_cols, errors="ignore")

    phik_corr = df.phik_matrix()

    mask = np.triu(np.ones_like(phik_corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        phik_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_median_differences(
    diff_data: pd.DataFrame,
    hypothesis_col: str = "Hypothesis",
    ci_lower_col: str = "CI_lower",
    ci_upper_col: str = "CI_upper",
    title: Optional[str] = "Confidence Intervals for Median Differences",
    figsize: Optional[Tuple[int, int]] = (10, 3),
    palette_name: Optional[str] = "magma",
) -> None:
    """
    Plot error bars for median differences with confidence intervals.

    Args:
        diff_data (pd.DataFrame): DataFrame with hypothesis names and confidence intervals.
        hypothesis_col (str): Column with hypothesis labels.
        ci_lower_col (str): Column with lower CI bounds.
        ci_upper_col (str): Column with upper CI bounds.
        title (Optional[str]): Title for the plot.
        figsize (Optional[Tuple[int, int]]): Figure size.
        palette_name (Optional[str]): Color palette name for seaborn.
    """
    plt.figure(figsize=figsize)
    palette = sns.color_palette(palette_name, n_colors=len(diff_data))

    for i, (idx, row) in enumerate(diff_data.iterrows()):
        center = (row[ci_lower_col] + row[ci_upper_col]) / 2
        lower_error = center - row[ci_lower_col]
        upper_error = row[ci_upper_col] - center
        errors = np.array([[lower_error], [upper_error]])

        plt.errorbar(
            x=center,
            y=i,
            xerr=errors,
            fmt="o",
            color=palette[i],
            ecolor=palette[i],
            elinewidth=2,
            capsize=5,
        )

    plt.yticks(range(len(diff_data)), diff_data[hypothesis_col])
    plt.xlabel("Median Difference")
    plt.title(title, fontsize=14, weight="bold")
    plt.axvline(x=0, color="gray", linestyle="--", label="Null (Diff=0)")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.xlim(diff_data[ci_lower_col].min() - 2, diff_data[ci_upper_col].max() + 2)
    plt.tight_layout()
    plt.show()


def plot_odds_ratios(
    or_data: pd.DataFrame,
    hypothesis_col: str = "Hypothesis",
    estimate_col: str = "Estimate",
    ci_lower_col: str = "CI_lower",
    ci_upper_col: str = "CI_upper",
    title: Optional[str] = "Confidence Intervals for Odds Ratios",
    figsize: Optional[Tuple[int, int]] = (10, 4),
    palette_name: Optional[str] = "magma",
    xlim: Optional[Tuple[float, float]] = (0.4, 5),
) -> None:
    """
    Plot error bars for odds ratios with confidence intervals.

    Args:
        or_data (pd.DataFrame): DataFrame with odds ratio estimates and confidence intervals.
        hypothesis_col (str): Column with hypothesis labels.
        estimate_col (str): Column with OR estimates.
        ci_lower_col (str): Column with lower CI bounds.
        ci_upper_col (str): Column with upper CI bounds.
        title (Optional[str]): Title for the plot.
        figsize (Optional[Tuple[int, int]]): Figure size.
        palette_name (Optional[str]): Color palette name for seaborn.
        xlim (Optional[Tuple[float, float]]): X-axis limits.
    """
    plt.figure(figsize=figsize)
    palette = sns.color_palette(palette_name, n_colors=len(or_data))

    for i, (idx, row) in enumerate(or_data.iterrows()):
        center = row[estimate_col]
        lower_error = center - row[ci_lower_col]
        upper_error = row[ci_upper_col] - center
        errors = np.array([[lower_error], [upper_error]])

        plt.errorbar(
            x=center,
            y=i,
            xerr=errors,
            fmt="o",
            color=palette[i],
            ecolor=palette[i],
            elinewidth=2,
            capsize=5,
        )

    plt.yticks(range(len(or_data)), or_data[hypothesis_col])
    plt.xlabel("Odds Ratio")
    plt.title(title, fontsize=14, weight="bold")
    plt.axvline(x=1, color="gray", linestyle="--", label="Null (OR=1)")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.xlim(xlim)
    plt.tight_layout()
    plt.show()


def plot_model_f2_scores(
    models: List[str],
    scores: List[float],
    title: Optional[str] = "F2 Scores of Trained Models",
    ylabel: Optional[str] = "F2 Score (Class 1)",
    ylim: Optional[Tuple[float, float]] = (0.2, 0.6),
) -> None:
    """
    Plot a sorted bar chart of F2 Scores for different models.

    Args:
        models (List[str]): List of model names.
        scores (List[float]): Corresponding list of F2 scores.
        title (Optional[str]): Title of the plot.
        ylabel (Optional[str]): Label for the y-axis.
        ylim (Optional[Tuple[float, float]]): Limits for the y-axis.
    """
    df = pd.DataFrame({"Model": models, "F2 Score": scores})
    df_sorted = df.sort_values(by="F2 Score", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="Model",
        y="F2 Score",
        data=df_sorted,
        palette="magma",
        hue="Model",
        legend=False,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim:
        plt.ylim(*ylim)
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            color="black",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_model_f1_macro_f2_comparison(
    models: List[str],
    f1_macro_scores: List[float],
    f2_scores: List[float],
    title: Optional[str] = "CV Results of F1 Macro and F2 Scores by Model",
    xlabel: Optional[str] = "Score",
    xlim: Optional[Tuple[float, float]] = (0.2, 0.8),
) -> None:
    """
    Plot side-by-side barplots comparing F1 Macro and F2 scores for multiple models.

    Args:
        models (List[str]): List of model names.
        f1_macro_scores (List[float]): List of F1 Macro scores.
        f2_scores (List[float]): List of F2 scores.
        title (Optional[str]): Plot title.
        xlabel (Optional[str]): Label for the x-axis.
        xlim (Optional[Tuple[float, float]]): Limits for the x-axis.
    """
    data = {
        "Model": models * 2,
        "Metric": ["F1 Macro"] * len(models) + ["F2 Score"] * len(models),
        "Score": f1_macro_scores + f2_scores,
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        y="Model",
        x="Score",
        hue="Metric",
        data=df,
        palette="magma",
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Model", fontsize=12)
    if xlim:
        plt.xlim(*xlim)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    for p in ax.patches:
        width = p.get_width()
        if width > 0.01:
            ax.annotate(
                f"{width:.4f}",
                (width + 0.01, p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=9,
                color="black",
            )

    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()


def compare_two_models_scores(
    model_names,
    f1_macro_scores,
    f2_scores,
    recall_scores,
    precision_scores,
    title="Comparison of Model Metrics",
):
    metrics = ["F1 Macro", "F2 (Class 1)", "Recall (Class 1)", "Precision (Class 1)"]

    model1_values = [
        f1_macro_scores[0],
        f2_scores[0],
        recall_scores[0],
        precision_scores[0],
    ]
    model2_values = [
        f1_macro_scores[1],
        f2_scores[1],
        recall_scores[1],
        precision_scores[1],
    ]

    y_pos = np.arange(len(metrics))
    height = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.barh(
        y_pos - height / 2,
        model1_values,
        height,
        label=model_names[0],
        color=plt.cm.magma(0.4),
    )
    bars2 = plt.barh(
        y_pos + height / 2,
        model2_values,
        height,
        label=model_names[1],
        color=plt.cm.magma(0.7),
    )

    plt.xlabel("Score", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.yticks(y_pos, metrics)
    plt.xlim(0, 1.0)
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars1:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=10,
        )

    for bar in bars2:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_threshold_tuning(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: Sequence[float] = np.arange(0.1, 0.9, 0.05),
    title: str = "Threshold Tuning Metrics",
) -> None:
    """
    Plots F1, F2, Recall, and Precision scores vs. thresholds.

    Parameters:
    - probs: array-like of shape (n_samples,), predicted probabilities for class 1
    - y_true: array-like of shape (n_samples,), true binary labels
    - thresholds: iterable of thresholds to evaluate
    - title: plot title
    """

    f1_scores = []
    f2_scores = []
    recall_scores = []
    precision_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, preds, zero_division=0))
        f2_scores.append(fbeta_score(y_true, preds, beta=2, zero_division=0))
        recall_scores.append(recall_score(y_true, preds, zero_division=0))
        precision_scores.append(precision_score(y_true, preds, zero_division=0))

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds, f1_scores, marker="o", color=plt.cm.magma(0.2), label="F1 Macro"
    )
    plt.plot(thresholds, f2_scores, marker="s", color=plt.cm.magma(0.4), label="F2")
    plt.plot(
        thresholds, recall_scores, marker="^", color=plt.cm.magma(0.6), label="Recall"
    )
    plt.plot(
        thresholds,
        precision_scores,
        marker="d",
        color=plt.cm.magma(0.8),
        label="Precision",
    )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=16, weight="bold")
    plt.legend(title="Metrics")
    plt.grid(True)
    plt.show()


def plot_roc_pr_curves(
    y_true: Union[np.ndarray, list],
    y_proba: Union[np.ndarray, list],
    model_name: str = "Model",
) -> None:
    """
    Plots ROC Curve and Precision-Recall Curve side by side for given true labels and predicted probabilities.

    Parameters:
    - y_true: array-like of shape (n_samples,), true binary labels
    - y_proba: array-like of shape (n_samples,), predicted probabilities for class 1
    - model_name: str, label for the model in plot titles
    """

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color=plt.cm.magma(0.6), lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title(f"ROC Curve - {model_name}", fontsize=14, weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        recall, precision, color=plt.cm.magma(0.4), lw=2, label=f"PR AUC = {pr_auc:.4f}"
    )
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, weight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_binary_strip_and_countplot_rate(
    df: pd.DataFrame,
    binary_cols: List[str],
    hue_col: str = "stroke",
    palette_name: str = "magma",
) -> None:
    palette = sns.color_palette(palette_name, n_colors=2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, col in enumerate(binary_cols):
        ax = axes[0, i]
        sns.stripplot(
            x=col,
            y="age",
            data=df,
            hue=hue_col,
            dodge=True,
            jitter=0.25,
            alpha=0.7,
            palette=palette,
            ax=ax,
        )
        ax.set_title(
            f"{col.replace('_', ' ').title()} vs Age by {hue_col.title()}", fontsize=14
        )
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Age", fontsize=12)
        ax.legend(title=hue_col.title(), fontsize=10, title_fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i, col in enumerate(binary_cols):
        ax = axes[1, i]
        prop_df = (
            df.groupby(col)[hue_col]
            .mean()
            .reset_index()
            .rename(columns={hue_col: f"{hue_col.title()} Rate"})
        )
        sns.barplot(
            x=col, y=f"{hue_col.title()} Rate", data=prop_df, palette=palette, ax=ax
        )
        ax.set_title(
            f"{hue_col.title()} Rate by {col.replace('_', ' ').title()}", fontsize=14
        )
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(f"{hue_col.title()} Rate", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    height + 0.005,
                    f"{height*100:.1f}%",
                    ha="center",
                    fontsize=10,
                )

    plt.tight_layout()
    plt.suptitle(
        f"{binary_cols[0].replace('_', ' ').title()} and {binary_cols[1].replace('_', ' ').title()} Analysis\nTop: Age Stripplots by {hue_col.title()}  |  Bottom: {hue_col.title()} Rates Within Group",
        fontsize=18,
        weight="bold",
        y=1.05,
    )
    plt.show()


def plot_permutation_importance(
    pipeline,
    X_val,
    y_val,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    title="Permutation Importance (Grouped by Feature)",
) -> None:
    """
    Computes and plots grouped permutation importances for a given pipeline.
    It works by permuting original features from X_val.

    Parameters:
    - pipeline: The trained scikit-learn or imblearn pipeline.
    - X_val: The validation (or test) features, in their original, untransformed format.
    - y_val: The true labels for X_val.
    - scoring: The metric to use for permutation importance (e.g., 'roc_auc', 'f1').
    - n_repeats: Number of times to permute each feature.
    - random_state: Seed for reproducibility.
    - title: Title for the plot.
    """

    original_feature_names = X_val.columns.tolist()

    print(
        f"\nComputing permutation importance using '{scoring}' (this may take a minute with {n_repeats} repeats)..."
    )
    result = permutation_importance(
        estimator=pipeline,
        X=X_val,
        y=y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    print("Permutation importance complete.")

    if len(original_feature_names) != len(result.importances_mean):
        raise ValueError(
            f"Mismatch between number of original features ({len(original_feature_names)}) "
            f"and permutation importance results ({len(result.importances_mean)}). "
            f"This should not happen if X is original and estimator is pipeline."
        )

    df_perm_imp = pd.DataFrame(
        {
            "feature": original_feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    grouped_importance = df_perm_imp.set_index("feature").sort_values(
        by="importance_mean", ascending=False
    )

    plt.figure(figsize=(10, 8))
    bars = plt.barh(
        grouped_importance.index,
        grouped_importance["importance_mean"],
        xerr=grouped_importance["importance_std"],
        color=plt.cm.magma_r(np.linspace(0, 1, len(grouped_importance))),
        alpha=0.9,
    )

    plt.xlabel(f"Mean Decrease in {scoring} (± Std Dev)", fontsize=12)
    plt.ylabel("Original Feature", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n--- Grouped Permutation Importance (Mean ± Std Dev) ---")
    print(grouped_importance)


def plot_ever_married_stripplot(df: pd.DataFrame) -> None:
    """
    Creates a stripplot of Age vs Ever Married, colored by Stroke status.

    Parameters:
    - df: pandas DataFrame containing 'ever_married', 'age', and 'stroke' columns.
    """
    palette = sns.color_palette("magma", n_colors=2)

    plt.figure(figsize=(8, 6))
    sns.stripplot(
        x="ever_married",
        y="age",
        hue="stroke",
        data=df,
        dodge=True,
        jitter=0.25,
        alpha=0.7,
        palette=palette,
    )

    plt.title("Ever Married vs Age by Stroke Status", fontsize=14, weight="bold")
    plt.xlabel("Ever Married", fontsize=12)
    plt.ylabel("Age", fontsize=12)
    plt.legend(title="Stroke", fontsize=10, title_fontsize=11, loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
