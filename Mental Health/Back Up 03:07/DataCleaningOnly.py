import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import geopandas as gpd
import statsmodels.stats.proportion as smp
from phik import phik_matrix

from MentalHealthCustomFunctionsDictionaries import (
    filter_unrealistic_ages,
    question_mapping,
    gender_mapping,
    country_mapping,
    condition_mapping,
    plot_age_boxplot,
    plot_geographic_distribution,
    plot_state_distribution,
    get_state_data,
    plot_age_distribution,
    plot_gender_distribution,
    countries_distribution_percentages,
    plot_cleaned_state_distribution_horizontal_bar_chart,
    plot_mental_health_conditions,
    plot_mental_health_conditions_with_CI,
    categorize_condition,
    plot_mental_health_conditions_by_gender_raw,
    generate_correlation_heatmap,
)


db_path = "mental_health.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [table[0] for table in tables]
print("Tables in database:", table_names)

for table in table_names:
    print(f"\nSchema of {table}:")
    cursor.execute(f"PRAGMA table_info({table});")
    schema = cursor.fetchall()
    for column in schema:
        print(column)

answer_df = pd.read_sql("SELECT * FROM Answer", conn)
question_df = pd.read_sql("SELECT * FROM Question", conn)
survey_df = pd.read_sql("SELECT * FROM Survey", conn)

merged_df = answer_df.merge(
    question_df, left_on="QuestionID", right_on="questionid", how="left"
).merge(survey_df, on="SurveyID", how="left")

merged_df["questiontext"] = merged_df["questiontext"].replace(question_mapping)

merged_df["AnswerText"] = merged_df["AnswerText"].fillna("Not Provided")

merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = (
    merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"]
    .str.strip()
    .str.title()
)
merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = merged_df.loc[
    merged_df["questiontext"] == "Gender", "AnswerText"
].replace(gender_mapping)

merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"] = pd.to_numeric(
    merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"], errors="coerce"
)

plot_age_boxplot(merged_df)

temp_age_numeric = pd.to_numeric(
    merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"], errors="coerce"
)

temp_age_numeric_cleaned = filter_unrealistic_ages(
    temp_age_numeric, min_age=0, max_age=100
)
merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"] = (
    temp_age_numeric_cleaned.fillna("Not Provided")
)

age_series = merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"].dropna()

age_series = pd.to_numeric(age_series, errors="coerce").dropna()

if not age_series.empty:
    merged_df.loc[merged_df["questiontext"] == "Age", "Age Group"] = pd.cut(
        age_series,
        bins=[0, 19, 29, 39, 49, 59, 100],
        labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )
else:
    merged_df.loc[merged_df["questiontext"] == "Age", "Age Group"] = "Not Provided"

print(age_series.unique())

raw_genders = merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"]
print("Raw Unique Genders Before Mapping:", raw_genders.unique())

if "Gender" in merged_df["questiontext"].values:

    merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = (
        merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace(gender_mapping)
    )

    merged_df = merged_df[
        (merged_df["questiontext"] != "-1") & (merged_df["AnswerText"] != "-1")
    ]

    valid_genders = ["Male", "Female", "Other"]
    merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = merged_df.loc[
        merged_df["questiontext"] == "Gender", "AnswerText"
    ].apply(lambda x: x if x in valid_genders else "Other")

print(
    "Unique Genders after fixing:",
    merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"].unique(),
)

merged_df.loc[merged_df["questiontext"] == "Country", "AnswerText"] = (
    merged_df.loc[merged_df["questiontext"] == "Country", "AnswerText"]
    .str.strip()
    .str.title()
)

merged_df.loc[merged_df["questiontext"] == "Country", "AnswerText"] = merged_df.loc[
    merged_df["questiontext"] == "Country", "AnswerText"
].replace(country_mapping)

state_data = get_state_data(db_path)

print(state_data.head())

print(merged_df.tail())
