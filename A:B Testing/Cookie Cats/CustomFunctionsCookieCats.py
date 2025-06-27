import pandas as pd
from scipy.stats import chisquare
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import chi2_contingency


def plot_game_rounds_boxplot(dataframe, hue_column=None, title="Game Rounds"):
    plt.figure(figsize=(10, 4))

    if hue_column:
        sns.boxplot(
            x=dataframe["sum_gamerounds"], hue=dataframe[hue_column], palette="viridis"
        )
        plt.legend().set_visible(False)
    else:

        sns.boxplot(
            x=dataframe["sum_gamerounds"],
            color=sns.color_palette("viridis")[2],
        )
    sns.despine(top=True, right=True)
    plt.title(title)
    plt.show()


def plot_1_day_retention_barplot(
    dataframe, x_col="version", y_col="retention_1", title="1-Day Retention Rate"
):
    plt.figure(figsize=(6, 5))

    viridis_colors = sns.color_palette("viridis", len(dataframe[x_col].unique()))

    ax = sns.barplot(
        data=dataframe,
        x=x_col,
        y=y_col,
        hue=x_col,
        palette=viridis_colors,
        errorbar=None,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2%}",
            (bar.get_x() + bar.get_width() / 2, height + 0.0005),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    sns.despine(top=True, right=True)
    plt.title(title)
    plt.ylabel("Retention Rate")
    plt.xlabel("Gate Version")
    plt.ylim(0.79, 0.7985)
    plt.tight_layout()
    plt.show()


def plot_7_day_retention_barplot(
    dataframe,
    x_col="version",
    y_col="retention_7",
    title="7-Day Retention Rate by Gate Version",
):
    plt.figure(figsize=(6, 5))

    viridis_colors = sns.color_palette("viridis", len(dataframe[x_col].unique()))
    ax = sns.barplot(
        data=dataframe,
        x=x_col,
        y=y_col,
        hue=x_col,
        palette=viridis_colors,
        errorbar=None,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    max_height = dataframe[y_col].max()
    buffer = max_height * 0.05

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2%}",
            (bar.get_x() + bar.get_width() / 2, height + buffer / 4),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    sns.despine(top=True, right=True)
    plt.title(title)
    plt.ylabel("Retention Rate")
    plt.xlabel("Gate Version")
    plt.ylim(0, max_height + buffer)
    plt.tight_layout()
    plt.show()


def plot_retention_comparison_barplot(
    dataframe, x_col="version", title="Comparison of 1-Day and 7-Day Retention Rates"
):
    melted_df = dataframe.melt(
        id_vars=x_col,
        value_vars=["retention_1", "retention_7"],
        var_name="retention_period",
        value_name="retention_rate",
    )

    melted_df["retention_period"] = melted_df["retention_period"].map(
        {"retention_1": "1-Day Retention", "retention_7": "7-Day Retention"}
    )

    plt.figure(figsize=(8, 5))
    viridis_colors = sns.color_palette(
        "viridis", melted_df["retention_period"].nunique()
    )

    ax = sns.barplot(
        data=melted_df,
        x=x_col,
        y="retention_rate",
        hue="retention_period",
        palette=viridis_colors,
        errorbar=None,
    )

    sns.despine(top=True, right=True)

    max_height = melted_df["retention_rate"].max()
    buffer = max_height * 0.05

    for bar in ax.patches:
        height = bar.get_height()
        if height > 0.001:
            ax.annotate(
                f"{height:.2%}",
                (bar.get_x() + bar.get_width() / 2, height + buffer / 4),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.title(title)
    plt.xlabel("Gate Version")
    plt.ylabel("Retention Rate")
    plt.ylim(0, max_height + buffer)
    plt.legend(title="Retention Period")
    plt.tight_layout()
    plt.show()


def bootstrap_ci(data1, data2, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        boot1 = np.random.choice(data1, size=len(data1), replace=True)
        boot2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(np.mean(boot2) - np.mean(boot1))
    return np.percentile(diffs, [2.5, 97.5])
