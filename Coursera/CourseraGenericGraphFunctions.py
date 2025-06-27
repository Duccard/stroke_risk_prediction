import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from CourseraCustomFunctions import convert_enrollment, color_threshold

df = pd.read_csv("coursea_data.csv")


def plot_histogram(
    df,
    column,
    bins=20,
    title="Ratings Histogram",
    xlabel=None,
    ylabel="Number of Entries",
):
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(df[column], bins=bins, edgecolor="black")

    high_threshold = df[column].quantile(0.75)
    mid_threshold_upper = df[column].quantile(0.75)
    mid_threshold_lower = df[column].quantile(0.25)

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

    average = df[column].mean()
    plt.axvline(
        average, color="black", linestyle="dashed", linewidth=2, label=f"Average"
    )

    # Calculate x-coordinate for text placement (left side)
    x_coord = (
        plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
    )  # 2% from the left edge

    plt.text(
        x_coord,  # Use calculated x-coordinate
        average + (plt.ylim()[1] - plt.ylim()[0]) * 0.02,  # Slightly above the line
        f"{average:.2f}",
        color="black",
        ha="left",  # Align left
        va="bottom",  # Align bottom
    )

    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.title(title if title else f"Distribution of {column}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


def plot_scatter_with_top3_outliers_annotated(
    df,
    column,
    outliers=3,
    title="Scatter Plot of Enrollment",
    xlabel="Index",
    ylabel=None,
    annotate_column="course_title",
):
    plt.figure(figsize=(10, 5))

    high_threshold = df[column].quantile(0.75)
    mid_threshold_upper = df[column].quantile(0.75)
    mid_threshold_lower = df[column].quantile(0.25)

    colors = []
    for value in df[column]:
        if value >= high_threshold:
            colors.append("red")
        elif value >= mid_threshold_lower and value <= mid_threshold_upper:
            colors.append("blue")
        else:
            colors.append("gray")

    plt.scatter(range(len(df)), df[column], alpha=0.6, c=colors)

    top_outliers = df.nlargest(outliers, column)
    for i, row in top_outliers.iterrows():
        plt.annotate(
            row[annotate_column],
            (i, row[column]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="black",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.title(title if title else f"Scatter Plot of {column} with Outliers Annotated")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_bar_chart(
    df,
    category_column,
    value_column,
    title=None,
    xlabel=None,
    ylabel=None,
    highlight_max=True,
):
    plt.figure(figsize=(8, 5))

    high_threshold = df[value_column].quantile(0.75)
    mid_threshold_upper = df[value_column].quantile(0.75)
    mid_threshold_lower = df[value_column].quantile(0.25)

    colors = []
    for value in df[value_column]:
        if value >= high_threshold:
            colors.append("red")
        elif value >= mid_threshold_lower and value <= mid_threshold_upper:
            colors.append("blue")
        else:
            colors.append("gray")

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


def plot_horizontal_bar_chart_top10(
    df,
    category_column,
    value_column,
    title="Top 10 Most Popular",
    xlabel="Number of Students Enrolled",
    ylabel="Category",
):
    plt.figure(figsize=(10, 6))

    colors = []
    min_val = df[value_column].min()
    max_val = df[value_column].max()

    if "Top" in title and "Least" not in title:
        for val in df[value_column]:
            if val == max_val:
                colors.append("red")
            else:
                colors.append("blue")

    elif "Bottom" in title or "Least" in title:
        for val in df[value_column]:
            if val == min_val:
                colors.append("gray")
            else:
                colors.append("blue")

    else:
        for val in df[value_column]:
            if val == max_val:
                colors.append("red")
            else:
                colors.append("blue")

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


def plot_certificate_enrollment(df):
    enrollment = df.groupby("course_certificate_type", as_index=False)[
        "course_students_enrolled"
    ].sum()
    enrollment["course_certificate_type"] = enrollment[
        "course_certificate_type"
    ].str.upper()

    high_threshold = enrollment["course_students_enrolled"].quantile(0.75)
    mid_threshold_upper = enrollment["course_students_enrolled"].quantile(0.75)
    mid_threshold_lower = enrollment["course_students_enrolled"].quantile(0.25)

    colors = []
    for value in enrollment["course_students_enrolled"]:
        if value >= high_threshold:
            colors.append("red")
        elif value >= mid_threshold_lower and value <= mid_threshold_upper:
            colors.append("blue")
        else:
            colors.append("gray")

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


def plot_avg_max(df, value_column, title=None, xlabel=None, ylabel=None):
    """Plots the average and maximum values of a column."""

    avg_value = df[value_column].mean()
    max_value = df[value_column].max()

    values = [avg_value, max_value]
    labels = ["Average", "Maximum"]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, hue=labels, palette=["blue", "red"], legend=False)
    plt.xlabel(xlabel if xlabel else "Metric")
    plt.ylabel(ylabel if ylabel else value_column)
    plt.title(title if title else f"Average and Maximum of {value_column}")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_scatter_difficulty_enrollment(
    df,
    column,
    title="Impact of Course Difficulty on Student Enrollment",
    xlabel="Index",
    ylabel=None,
    annotate_column="course_title",
    top_n_annotations=4,
    figsize=(16, 9),
):
    plt.figure(figsize=figsize)

    difficulty_shapes = {
        "Beginner": "o",
        "Intermediate": "s",
        "Mixed": "D",
        "Advanced": "^",
    }

    thresholds = {
        "high": df[column].quantile(0.75),
        "mid_upper": df[column].quantile(0.75),
        "mid_lower": df[column].quantile(0.25),
    }

    all_outliers = df.nlargest(top_n_annotations, "course_students_enrolled")

    for difficulty, marker in difficulty_shapes.items():
        subset = df[df["course_difficulty"].str.strip().str.capitalize() == difficulty]

        if not subset.empty:
            colors = subset[column].apply(
                lambda x: (
                    "red"
                    if x >= thresholds["high"]
                    else "blue" if x >= thresholds["mid_lower"] else "gray"
                )
            )
            plt.scatter(
                subset.index,
                subset[column],
                c=colors,
                label=difficulty,
                marker=marker,
                alpha=0.6,
            )

    for i, row in all_outliers.iterrows():
        annotation_text = f"{row[annotate_column]} ({row['course_organization']})"
        plt.annotate(
            annotation_text,
            (i, row[column]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="black",
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor="black",
            markersize=8,
            label=label,
        )
        for label, marker in difficulty_shapes.items()
    ]
    plt.legend(handles=handles, title="Difficulty Level", loc="upper left")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.title(title if title else f"Scatter Plot of {column} by Difficulty Level")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_scatter_certificate_difficulty_type(
    df,
    column,
    title="Impact of Course Certificate, Difficulty Type on Student Enrollment",
    xlabel="Index",
    ylabel=None,
    annotate_column="course_title",
    top_n_annotations=4,
    figsize=(16, 9),
):
    plt.figure(figsize=figsize)

    difficulty_shapes = {
        "Beginner": "o",
        "Intermediate": "s",
        "Mixed": "D",
        "Advanced": "^",
    }

    certificate_colors = {
        "SPECIALIZATION": "blue",
        "COURSE": "red",
        "PROFESSIONAL CERTIFICATE": "green",
    }

    all_outliers = df.nlargest(top_n_annotations, "course_students_enrolled")

    for difficulty, marker in difficulty_shapes.items():
        subset = df[df["course_difficulty"].str.strip().str.capitalize() == difficulty]

        if not subset.empty:
            colors = (
                subset["course_certificate_type"]
                .str.strip()
                .str.upper()
                .map(certificate_colors)
                .fillna("gray")
            )
            plt.scatter(
                subset.index,
                subset[column],
                c=colors,
                label=difficulty,
                marker=marker,
                alpha=0.6,
            )

    for i, row in all_outliers.iterrows():
        annotation_text = f"{row[annotate_column]} ({row['course_organization']})"
        plt.annotate(
            annotation_text,
            (i, row[column]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=9,
            color="black",
        )

    certificate_handles = [
        plt.Line2D([0], [0], linestyle="-", linewidth=6, color=color, label=cert_type)
        for cert_type, color in certificate_colors.items()
    ]
    difficulty_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor="black",
            markersize=8,
            label=label,
        )
        for label, marker in difficulty_shapes.items()
    ]

    legend1 = plt.legend(
        handles=certificate_handles, title="Certificate Type (Colors)", loc="upper left"
    )
    plt.gca().add_artist(legend1)
    plt.legend(
        handles=difficulty_handles,
        title="Difficulty Level (Icons)",
        bbox_to_anchor=(0, 0.85),
        loc="upper left",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column)
    plt.title(title if title else f"Scatter Plot of {column} by Certificate Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
