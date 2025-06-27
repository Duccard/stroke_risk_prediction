import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CourseraCustomFunctions import convert_enrollment

df = pd.read_csv("coursea_data.csv")
df.columns = df.columns.str.lower()

# Convert enrollments to numeric values
df["course_students_enrolled"] = df["course_students_enrolled"].apply(
    convert_enrollment
)
df["course_students_enrolled"] = pd.to_numeric(
    df["course_students_enrolled"], errors="coerce"
)
df = df.dropna(subset=["course_students_enrolled"])


def plot_histogram(
    df, column, bins=20, title=None, xlabel=None, ylabel="Number of Entries"
):
    """Plots a histogram with a custom color scheme based on value ranges."""

    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(df[column], bins=bins, edgecolor="black")

    # Define thresholds for color ranges (adjust as needed)
    high_threshold = df[column].quantile(0.75)
    mid_threshold_upper = df[column].quantile(0.75)
    mid_threshold_lower = df[column].quantile(0.25)
    low_threshold = df[column].quantile(0.25)

    # Color patches based on value range
    for i, patch in enumerate(patches):
        bin_midpoint = (bins[i] + bins[i + 1]) / 2
        if bin_midpoint >= high_threshold:
            patch.set_facecolor("red")
        elif (
            bin_midpoint >= mid_threshold_lower and bin_midpoint <= mid_threshold_upper
        ):
            patch.set_facecolor("blue")
        else:
            patch.set_facecolor("gray")

    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.title(title if title else f"Distribution of {column}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Function to plot scatter with top 3 outliers annotated
def plot_scatter_with_top3_outliers_annotated(
    df,
    column,
    outliers=3,
    title=None,
    xlabel="Index",
    ylabel=None,
    annotate_column="course_title",
):
    """Plots scatter with color scheme and black annotations."""

    plt.figure(figsize=(10, 5))

    # Define thresholds for color ranges (adjust as needed)
    high_threshold = df[column].quantile(0.75)  # Top 25%
    mid_threshold_upper = df[column].quantile(0.75)
    mid_threshold_lower = df[column].quantile(0.25)  # Middle 50%
    low_threshold = df[column].quantile(0.25)  # Bottom 25%

    # Color points based on value range
    colors = []
    for value in df[column]:
        if value >= high_threshold:
            colors.append("red")
        elif value >= mid_threshold_lower and value <= mid_threshold_upper:
            colors.append("blue")
        else:
            colors.append("gray")

    plt.scatter(range(len(df)), df[column], alpha=0.6, c=colors)  # Use list of colors

    top_outliers = df.nlargest(outliers, column)
    for i, row in top_outliers.iterrows():
        plt.annotate(
            row[annotate_column],
            (i, row[column]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="black",  # Black annotations
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.title(title if title else f"Scatter Plot of {column} with Outliers Annotated")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Function to plot a bar chart
def plot_bar_chart(
    df,
    category_column,
    value_column,
    title=None,
    xlabel=None,
    ylabel=None,
    highlight_max=True,
    color_palette=("blue", "red"),
):
    plt.figure(figsize=(8, 5))
    colors = [
        color_palette[1] if val == df[value_column].max() else color_palette[0]
        for val in df[value_column]
    ]
    sns.barplot(
        x=category_column,
        y=value_column,
        data=df,
        hue=category_column,
        palette=colors,
        legend=False,
    )
    plt.xlabel(xlabel if xlabel else category_column)
    plt.ylabel(ylabel if ylabel else value_column)
    plt.title(title if title else f"{value_column} Across {category_column}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.show()


# Function to plot horizontal bar chart for top 10 popular categories
def plot_horizontal_bar_chart_top10(
    df,
    category_column,
    value_column,
    title="Top 10 Most Popular",
    xlabel="Number of Students Enrolled",
    ylabel="Category",
    color_palette=("red", "blue"),
):
    plt.figure(figsize=(10, 6))
    colors = [
        color_palette[0] if val == df[value_column].max() else color_palette[1]
        for val in df[value_column]
    ]
    sns.barplot(
        y=df[category_column],
        x=df[value_column],
        hue=df[category_column],
        palette=colors,
        legend=False,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


# Function to plot certificate type enrollment
def plot_certificate_enrollment(df):
    """Plots enrollment by certificate type with predefined colors."""
    custom_palette = {
        "COURSE": "red",
        "SPECIALIZATION": "blue",
        "PROFESSIONAL CERTIFICATE": "gray",
    }
    enrollment = df.groupby("course_certificate_type", as_index=False)[
        "course_students_enrolled"
    ].sum()
    enrollment["course_certificate_type"] = enrollment[
        "course_certificate_type"
    ].str.upper()
    colors = [
        custom_palette.get(cert, "black")
        for cert in enrollment["course_certificate_type"]
    ]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="course_certificate_type",
        y="course_students_enrolled",
        data=enrollment,
        hue="course_certificate_type",
        palette=colors,
        legend=False,
    )
    plt.xlabel("Certificate Type")
    plt.ylabel("Total Enrollments")
    plt.title("Popularity of Each Certificate Type")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Compute enrollment by difficulty
enrollment_by_difficulty = df.groupby("course_difficulty", as_index=False)[
    "course_students_enrolled"
].sum()


# Function to plot average and maximum values
def plot_avg_max(df, value_column, title=None, xlabel=None, ylabel=None):
    """
    Plots a bar chart showing average and maximum values (minimum removed).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): Column name with numerical values.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
    """
    avg_value = df[value_column].mean()
    max_value = df[value_column].max()

    values = [avg_value, max_value]
    labels = ["Average", "Maximum"]
    colors = ["blue", "red"]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, hue=labels, palette=colors, legend=False)
    plt.xlabel(xlabel if xlabel else "Metric")
    plt.ylabel(ylabel if ylabel else value_column)
    plt.title(title if title else f"Min, Avg, and Max of {value_column}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
