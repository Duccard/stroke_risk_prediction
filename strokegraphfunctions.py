import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.lines as mlines
from phik import phik_matrix
import warnings
from typing import List, Optional


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
