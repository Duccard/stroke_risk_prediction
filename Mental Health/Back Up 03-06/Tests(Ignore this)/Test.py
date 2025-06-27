import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

db_path = "mental_health.sqlite"

# SQL Query for Age Distribution
sql_query = """
SELECT
    CASE
        WHEN CAST(Answer.AnswerText AS INTEGER) < 20 THEN '<20'
        WHEN CAST(Answer.AnswerText AS INTEGER) BETWEEN 20 AND 29 THEN '20-29'
        WHEN CAST(Answer.AnswerText AS INTEGER) BETWEEN 30 AND 39 THEN '30-39'
        WHEN CAST(Answer.AnswerText AS INTEGER) BETWEEN 40 AND 49 THEN '40-49'
        WHEN CAST(Answer.AnswerText AS INTEGER) BETWEEN 50 AND 59 THEN '50-59'
        WHEN CAST(Answer.AnswerText AS INTEGER) >= 60 THEN '60+'
        ELSE 'Not Provided'
    END AS AgeGroup,
    COUNT(*) AS RespondentCount
FROM
    Answer
JOIN
    Question ON Answer.QuestionID = Question.questionid
WHERE
    Question.questiontext = 'What is your age?'
    AND Answer.AnswerText IS NOT NULL
    AND Answer.AnswerText != ''
    AND CAST(Answer.AnswerText AS INTEGER) BETWEEN 0 and 100
GROUP BY
    AgeGroup
ORDER BY
    AgeGroup;
"""

# Fetch Data Using SQL
with sqlite3.connect(db_path) as conn:
    age_df = pd.read_sql_query(sql_query, conn)

# Plotting Age Distribution (SQL Results)
plt.figure(figsize=(10, 5))
sns.barplot(x="AgeGroup", y="RespondentCount", data=age_df)
plt.title("Age Distribution of Respondents (SQL)")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Pandas Approach (for comparison, but using SQL method is recommended)

conn = sqlite3.connect(db_path)

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

# Define age bins and labels
bins = [0, 19, 29, 39, 49, 59, 100]
labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]
merged_df["Age Group"] = pd.cut(merged_df["Age"], bins=bins, labels=labels, right=False)

# Plot histogram for age groups
plt.figure(figsize=(10, 5))
sns.histplot(
    merged_df["Age Group"].dropna(), bins=len(labels), kde=False, discrete=True
)
plt.title("Age Distribution of Respondents (Pandas)")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Plot boxplot to confirm outlier handling
plt.figure(figsize=(8, 5))
sns.boxplot(
    y=merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"]
    .dropna()
    .astype(float)
)
plt.title("Boxplot of Age Distribution (Pandas)")
plt.ylabel("Age")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

conn.close()


def detect_outliers(
    series: pd.Series, min_valid_age: int = 0, max_valid_age: int = 100
) -> pd.Series:
    """
    Detects and replaces outliers in a numerical Series using the IQR method.
    Additionally, filters out values below `min_valid_age` and above `max_valid_age`.

    Args:
        series (pd.Series): The input Series containing numerical data.
        min_valid_age (int): Minimum acceptable age value.
        max_valid_age (int): Maximum acceptable age value.

    Returns:
        pd.Series: The cleaned Series with outliers replaced by NaN.
    """
    series_cleaned = series.copy()

    # Remove unrealistic ages first
    series_cleaned[
        (series_cleaned < min_valid_age) | (series_cleaned > max_valid_age)
    ] = np.nan

    # Compute IQR (Interquartile Range)
    Q1 = series_cleaned.quantile(0.25)
    Q3 = series_cleaned.quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    series_cleaned[(series_cleaned < lower_bound) | (series_cleaned > upper_bound)] = (
        np.nan
    )

    return series_cleaned

def plot_survey_participation_over_time(df, survey_column="SurveyID"):
    """
    Plots the number of survey respondents over time.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing survey responses.
    - survey_column (str): The column representing survey participation.

    Returns:
    - Displays a line chart showing the number of respondents per survey.
    """
    # Count the number of responses per survey
    survey_counts = df[survey_column].value_counts().sort_index()

    # Ensure there is data to plot
    if survey_counts.empty:
        print("No valid survey participation data found for plotting.")
        return

    # Plot the survey participation trend
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=survey_counts.index, y=survey_counts.values, marker="o", linewidth=2)
    plt.title("Survey Participation Over Time")
    plt.xlabel("Survey ID")
    plt.ylabel("Number of Respondents")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
