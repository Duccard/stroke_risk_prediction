import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from MentalHealthCustomFunctionsDictionaries import detect_outliers, question_mapping

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

gender_map = {
    "M": "Male",
    "F": "Female",
    "Male": "Male",
    "Female": "Female",
    "Other": "Other",
}
merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = (
    merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"]
    .str.strip()
    .str.title()
)
merged_df.loc[merged_df["questiontext"] == "Gender", "AnswerText"] = merged_df.loc[
    merged_df["questiontext"] == "Gender", "AnswerText"
].replace(gender_map)

merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"] = pd.to_numeric(
    merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"], errors="coerce"
)

temp_age_numeric = pd.to_numeric(
    merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"], errors="coerce"
)

temp_age_numeric_cleaned = detect_outliers(
    temp_age_numeric, min_valid_age=0, max_valid_age=100
)

merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"] = (
    temp_age_numeric_cleaned.fillna("Not Provided")
)

merged_df.loc[merged_df["questiontext"] == "Country", "AnswerText"] = (
    merged_df.loc[merged_df["questiontext"] == "Country", "AnswerText"]
    .str.strip()
    .str.title()
)

merged_df = (
    merged_df.groupby(["SurveyID", "questiontext"])
    .agg(
        {"AnswerText": lambda x: x.mode()[0] if not x.mode().empty else "Not Provided"}
    )
    .reset_index()
)

full_survey_list = merged_df["SurveyID"].unique()
full_question_list = merged_df["questiontext"].unique()
full_index = pd.MultiIndex.from_product(
    [full_survey_list, full_question_list], names=["SurveyID", "questiontext"]
)
expanded_df = (
    merged_df.set_index(["SurveyID", "questiontext"]).reindex(full_index).reset_index()
)

expanded_df["AnswerText"] = expanded_df["AnswerText"].fillna("Not Asked")

print(expanded_df.head())

# Print column names to check if "Age" exists
print(merged_df.columns)


# Define age bins and labels
bins = [0, 19, 29, 39, 49, 59, 100]
labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
merged_df["Age Group"] = pd.cut(merged_df["Age"], bins=bins, labels=labels, right=False)

# Plot histogram for age groups
plt.figure(figsize=(10, 5))
sns.histplot(merged_df["Age Group"], bins=len(labels), kde=False, discrete=True)
plt.title("Age Distribution of Respondents")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Plot boxplot to confirm outlier handling
plt.figure(figsize=(8, 5))
sns.boxplot(y=merged_df["Age"])
plt.title("Boxplot of Age Distribution")
plt.ylabel("Age")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
