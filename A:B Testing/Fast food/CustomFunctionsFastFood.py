import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap


def plot_sales_boxplot(df, column="Sales_K"):
    """
    Plots a boxplot of weekly sales using a magma color palette.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: str, the name of the sales column to plot (default is "Sales_K")
    """
    sns.set_palette("magma", 1)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title("Boxplot of Weekly Sales (in Thousands)")
    plt.xlabel("Sales (k$)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_promotion_sales_boxplot(df, sales_col="Sales_K"):
    """
    Plots a boxplot showing the distribution of sales by promotion type,
    with a magma color palette and correctly placed legend.

    Parameters:
    - df: pandas DataFrame with 'Promotion' and a sales column
    - sales_col: str, the column name for sales values
    """
    df = df.copy()
    df["Promotion"] = df["Promotion"].astype(str)

    colors = sns.color_palette("magma", len(df["Promotion"].unique()))
    promotion_levels = sorted(df["Promotion"].unique())
    custom_palette = {level: colors[i] for i, level in enumerate(promotion_levels)}

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df,
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        palette=custom_palette,
        legend=False,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=custom_palette[level])
        for level in promotion_levels
    ]
    legend_labels = promotion_levels

    plt.legend(
        legend_handles,
        legend_labels,
        title="Promotion Type",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )

    plt.title("Sales Distribution by Promotion Type")
    plt.xlabel("Promotion Type")
    plt.ylabel("Weekly Sales (in Thousands)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = {
        "Promotion": ["1"] * 50 + ["2"] * 60 + ["3"] * 70,
        "Sales_K": [10 + i * 0.5 for i in range(50)]
        + [15 + i * 0.7 for i in range(60)]
        + [20 + i * 0.9 for i in range(70)],
    }
    df_sample = pd.DataFrame(data)

    plot_promotion_sales_boxplot(df_sample)

    data_alt = {
        "Promotion": ["A"] * 30 + ["B"] * 40 + ["C"] * 50,
        "Sales_K": [5 + i * 0.3 for i in range(30)]
        + [12 + i * 0.6 for i in range(40)]
        + [18 + i * 0.8 for i in range(50)],
    }
    df_sample_alt = pd.DataFrame(data_alt)
    plot_promotion_sales_boxplot(df_sample_alt)


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for all numeric features in the dataframe,
    using a light magma-style color palette and black diagonal for self-correlation.

    Parameters:
    - df: pandas DataFrame containing numeric columns
    """
    numeric_features = df.select_dtypes(include="number")
    correlation_matrix = numeric_features.corr()

    light_magma = LinearSegmentedColormap.from_list(
        "light_magma", ["#fbe9d8", "#e0827c", "#7e1e9c"]
    )

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap=light_magma,
        linewidths=0.5,
        cbar=True,
        annot_kws={"color": "black"},
    )

    n = len(correlation_matrix.columns)
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (i, i), 1, 1, fill=True, facecolor="black", edgecolor="black", lw=0
            )
        )

    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.show()


def plot_promotion_violinplot(df, sales_col="Sales_K"):
    """
    Plots a violin plot of sales distribution by promotion type using a custom magma color palette.

    Parameters:
    - df: pandas DataFrame containing 'Promotion' and sales column
    - sales_col: str, name of the sales column (default = 'Sales_K')
    """
    df = df.copy()
    df["Promotion"] = df["Promotion"].astype(str)

    colors = sns.color_palette("magma", 3)
    custom_palette = {"1": colors[0], "2": colors[1], "3": colors[2]}

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df,
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        palette=custom_palette,
        inner="quartile",
        dodge=False,
    )

    legend_elements = [
        Patch(facecolor=colors[0], label="1"),
        Patch(facecolor=colors[1], label="2"),
        Patch(facecolor=colors[2], label="3"),
    ]
    plt.legend(
        handles=legend_elements,
        title="Promotion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.title("Sales Distribution by Promotion Type (Violin Plot)")
    plt.xlabel("Promotion Type")
    plt.ylabel("Weekly Sales (in Thousands)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_promotion_pointplot(df, sales_col="Sales_K"):
    """
    Plots a point plot showing mean sales with 95% confidence intervals by promotion type.

    Parameters:
    - df: pandas DataFrame containing 'Promotion' and sales column
    - sales_col: str, name of the sales column (default = 'Sales_K')
    """
    df = df.copy()
    df["Promotion"] = df["Promotion"].astype(str)

    colors = sns.color_palette("magma", 3)
    custom_palette = {"1": colors[0], "2": colors[1], "3": colors[2]}

    plt.figure(figsize=(10, 6))
    sns.pointplot(
        data=df,
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        palette=custom_palette,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        linestyle="none",
        dodge=False,
    )

    legend_elements = [
        Patch(facecolor=colors[0], label="1"),
        Patch(facecolor=colors[1], label="2"),
        Patch(facecolor=colors[2], label="3"),
    ]
    plt.legend(
        handles=legend_elements,
        title="Promotion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.title("Mean Weekly Sales with 95% Confidence Intervals")
    plt.xlabel("Promotion Type")
    plt.ylabel("Average Weekly Sales (in Thousands)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_overall_and_faceted_pointplots(
    df, sales_col="Sales_K", facet_col="MarketSize"
):
    """
    Plots overall and faceted point plots showing mean weekly sales by promotion type.

    Parameters:
    - df: pandas DataFrame containing 'Promotion', sales column, and facet column
    - sales_col: str, name of the sales column (default = 'Sales_K')
    - facet_col: str, column used for faceting in second plot (default = 'MarketSize')
    """
    df = df.copy()
    df["Promotion"] = df["Promotion"].astype(str)

    colors = sns.color_palette("magma", 3)
    custom_palette = {"1": colors[0], "2": colors[1], "3": colors[2]}

    plt.figure(figsize=(14, 6))
    sns.pointplot(
        data=df,
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        palette=custom_palette,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        dodge=True,
    )

    legend_elements = [
        Patch(facecolor=colors[0], label="1"),
        Patch(facecolor=colors[1], label="2"),
        Patch(facecolor=colors[2], label="3"),
    ]
    plt.legend(
        handles=legend_elements,
        title="Promotion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.title("Overall: Mean Weekly Sales by Promotion Type")
    plt.xlabel("Promotion Type")
    plt.ylabel("Weekly Sales (in Thousands)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    g = sns.catplot(
        data=df,
        kind="point",
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        col=facet_col,
        palette=custom_palette,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        height=5,
        aspect=1.2,
        legend=False,
    )

    g.set_titles("Market Size: {col_name}")
    g.set_axis_labels("Promotion Type", "Weekly Sales (in Thousands)")

    plt.legend(
        handles=legend_elements,
        title="Promotion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


def plot_promotion_by_agegroup(df, sales_col="Sales_K", age_col="StoreAge"):
    """
    Faceted point plot showing weekly sales by promotion type across store age groups.

    Parameters:
    - df: pandas DataFrame containing 'Promotion', 'StoreAge', and sales column
    - sales_col: str, name of the sales column (default = 'Sales_K')
    - age_col: str, name of the store age column (default = 'StoreAge')
    """
    df = df.copy()

    bins = [0, 5, 10, df[age_col].max()]
    labels = ["Young (≤5)", "Mid (6–10)", "Old (>10)"]
    df["AgeGroup"] = pd.cut(df[age_col], bins=bins, labels=labels, right=True)

    df["Promotion"] = df["Promotion"].astype(str)

    colors = sns.color_palette("magma", 3)
    custom_palette = {"1": colors[0], "2": colors[1], "3": colors[2]}

    g = sns.catplot(
        data=df,
        kind="point",
        x="Promotion",
        y=sales_col,
        hue="Promotion",
        col="AgeGroup",
        palette=custom_palette,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        height=5,
        aspect=1.2,
        legend=False,
    )

    legend_elements = [
        Patch(facecolor=colors[0], label="1"),
        Patch(facecolor=colors[1], label="2"),
        Patch(facecolor=colors[2], label="3"),
    ]
    plt.legend(
        handles=legend_elements,
        title="Promotion Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    g.set_titles("Store Age: {col_name}")
    g.set_axis_labels("Promotion Type", "Weekly Sales (in Thousands)")
    plt.tight_layout()
    plt.show()
