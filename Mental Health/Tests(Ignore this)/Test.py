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


def count_mental_health_conditions_by_gender2(db_path):
    """
    Categorizes mental health condition counts into three gender groups: Male, Female, and Other.
    Avoids uniform ratio issues by directly linking respondents to their gender category.
    Removes invalid entries such as '-1' or empty responses.
    """

    sql_query = """
    WITH GenderedResponses AS (
    SELECT 
        CASE 
            WHEN LOWER(a1.AnswerText) IN ('male', 'm') THEN 'Male'
            WHEN LOWER(a1.AnswerText) IN ('female', 'f') THEN 'Female'
            ELSE 'Other' 
        END AS Gender,
        a1.SurveyID, 
        LOWER(a2.AnswerText) AS Condition
    FROM Answer a1
    JOIN Question q1 ON a1.QuestionID = q1.questionid
    JOIN Answer a2 ON a1.SurveyID = a2.SurveyID
    JOIN Question q2 ON a2.QuestionID = q2.questionid
    WHERE q1.questiontext LIKE 'What is your gender%'
          AND q2.questiontext LIKE 'If yes, what condition(s) have you been diagnosed with%'
          AND a1.AnswerText NOT IN ('-1', '')
)

SELECT 
    Gender,
    COUNT(DISTINCT SurveyID) AS Total_Respondents,
    ROUND(SUM(ConditionCount) / COUNT(DISTINCT SurveyID), 2) AS Avg_Conditions_Per_Person,
    SUM(CASE WHEN Condition LIKE '%anxiety%' OR Condition LIKE '%panic attack%' THEN 1 ELSE 0 END) AS Anxiety,
    SUM(CASE WHEN Condition LIKE '%depression%' OR Condition LIKE '%mood disorder%' THEN 1 ELSE 0 END) AS Depression,
    SUM(CASE WHEN Condition LIKE '%obsessive-compulsive disorder%' OR Condition LIKE '%ocd%' OR Condition LIKE '%intrusive thoughts%' THEN 1 ELSE 0 END) AS OCD,
    SUM(CASE WHEN Condition LIKE '%post-traumatic stress disorder%' OR Condition LIKE '%ptsd%' OR Condition LIKE '%trauma disorder%' THEN 1 ELSE 0 END) AS PTSD,
    SUM(CASE WHEN Condition LIKE '%eating disorder%' OR Condition LIKE '%anorexia%' OR Condition LIKE '%bulimia%' OR Condition LIKE '%binge eating%' THEN 1 ELSE 0 END) AS Eating_Disorder,
    SUM(CASE WHEN Condition LIKE '%stress%' OR Condition LIKE '%burnout%' THEN 1 ELSE 0 END) AS Stress_Disorder,
    SUM(CASE WHEN Condition NOT LIKE '%anxiety%'
                        AND Condition NOT LIKE '%panic attack%'
                        AND Condition NOT LIKE '%depression%'
                        AND Condition NOT LIKE '%mood disorder%'
                        AND Condition NOT LIKE '%ocd%'
                        AND Condition NOT LIKE '%obsessive-compulsive disorder%'
                        AND Condition NOT LIKE '%intrusive thoughts%'
                        AND Condition NOT LIKE '%ptsd%'
                        AND Condition NOT LIKE '%post-traumatic stress disorder%'
                        AND Condition NOT LIKE '%trauma disorder%'
                        AND Condition NOT LIKE '%eating disorder%'
                        AND Condition NOT LIKE '%anorexia%'
                        AND Condition NOT LIKE '%bulimia%'
                        AND Condition NOT LIKE '%binge eating%'
                        AND Condition NOT LIKE '%stress%'
                        AND Condition NOT LIKE '%burnout%'
        THEN 1 ELSE 0 END) AS Other
FROM (
    SELECT Gender, SurveyID, Condition, COUNT(*) AS ConditionCount
    FROM GenderedResponses
    GROUP BY Gender, SurveyID
) 
GROUP BY Gender;
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

    print("Mental Health Condition Counts by Gender:")
    for row in results:
        gender = row[0]
        counts = row[1:]
        print(f"\nGender: {gender}")
        for condition, count in zip(
            [
                "Anxiety",
                "Depression",
                "OCD",
                "PTSD",
                "Eating Disorder",
                "Stress Disorder",
                "Other",
            ],
            counts,
        ):
            print(f"{condition}: {count}")

    return results


def count_mental_health_conditions_by_gender(db_path):
    """
    Categorizes mental health condition counts into three gender groups: Male, Female, and Other.
    Avoids uniform ratio issues by directly linking respondents to their gender category.
    Removes invalid entries such as '-1' or empty responses.
    Expands condition keyword matching to avoid missing alternative phrasing.
    Also counts total respondents per gender and number of conditions reported per respondent.
    """

    sql_query = """
    WITH GenderedResponses AS (
        SELECT 
            CASE 
                WHEN LOWER(a1.AnswerText) IN ('male', 'm') THEN 'Male'
                WHEN LOWER(a1.AnswerText) IN ('female', 'f') THEN 'Female'
                ELSE 'Other' 
            END AS Gender,
            a1.SurveyID, 
            LOWER(a2.AnswerText) AS Condition
        FROM Answer a1
        JOIN Question q1 ON a1.QuestionID = q1.questionid
        JOIN Answer a2 ON a1.SurveyID = a2.SurveyID
        JOIN Question q2 ON a2.QuestionID = q2.questionid
        WHERE q1.questiontext LIKE 'What is your gender%'
              AND q2.questiontext LIKE 'If yes, what condition(s) have you been diagnosed with%'
              AND a1.AnswerText NOT IN ('-1', '')
    )
    
    SELECT 
        Gender,
        COUNT(DISTINCT SurveyID) AS Total_Respondents,
        AVG(ConditionsReported) AS Avg_Conditions_Per_Person,
        SUM(CASE WHEN Condition LIKE '%anxiety%' OR Condition LIKE '%panic attack%' THEN 1 ELSE 0 END) AS Anxiety,
        SUM(CASE WHEN Condition LIKE '%depression%' OR Condition LIKE '%mood disorder%' THEN 1 ELSE 0 END) AS Depression,
        SUM(CASE WHEN Condition LIKE '%obsessive-compulsive disorder%' OR Condition LIKE '%ocd%' OR Condition LIKE '%intrusive thoughts%' THEN 1 ELSE 0 END) AS OCD,
        SUM(CASE WHEN Condition LIKE '%post-traumatic stress disorder%' OR Condition LIKE '%ptsd%' OR Condition LIKE '%trauma disorder%' THEN 1 ELSE 0 END) AS PTSD,
        SUM(CASE WHEN Condition LIKE '%eating disorder%' OR Condition LIKE '%anorexia%' OR Condition LIKE '%bulimia%' OR Condition LIKE '%binge eating%' THEN 1 ELSE 0 END) AS Eating_Disorder,
        SUM(CASE WHEN Condition LIKE '%stress%' OR Condition LIKE '%burnout%' THEN 1 ELSE 0 END) AS Stress_Disorder,
        SUM(CASE WHEN Condition NOT LIKE '%anxiety%'
                            AND Condition NOT LIKE '%panic attack%'
                            AND Condition NOT LIKE '%depression%'
                            AND Condition NOT LIKE '%mood disorder%'
                            AND Condition NOT LIKE '%ocd%'
                            AND Condition NOT LIKE '%obsessive-compulsive disorder%'
                            AND Condition NOT LIKE '%intrusive thoughts%'
                            AND Condition NOT LIKE '%ptsd%'
                            AND Condition NOT LIKE '%post-traumatic stress disorder%'
                            AND Condition NOT LIKE '%trauma disorder%'
                            AND Condition NOT LIKE '%eating disorder%'
                            AND Condition NOT LIKE '%anorexia%'
                            AND Condition NOT LIKE '%bulimia%'
                            AND Condition NOT LIKE '%binge eating%'
                            AND Condition NOT LIKE '%stress%'
                            AND Condition NOT LIKE '%burnout%'
            THEN 1 ELSE 0 END) AS Other
    FROM (
        SELECT Gender, SurveyID, Condition, COUNT(Condition) OVER (PARTITION BY SurveyID) AS ConditionsReported
        FROM GenderedResponses
    )
    GROUP BY Gender;
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()

    print("Mental Health Condition Counts by Gender:")
    for row in results:
        gender, total_respondents, avg_conditions, *counts = row
        print(f"\nGender: {gender}")
        print(f"Total Respondents: {total_respondents}")
        print(f"Avg Conditions Per Person: {avg_conditions:.2f}")
        for condition, count in zip(
            [
                "Anxiety",
                "Depression",
                "OCD",
                "PTSD",
                "Eating Disorder",
                "Stress Disorder",
                "Other",
            ],
            counts,
        ):
            print(f"{condition}: {count}")

    return results


import sqlite3
import pandas as pd


def count_depression_males(db_path):
    """
    Counts the number of male respondents (those who answered 'M' or 'Male' to the gender question)
    who reported having depression or a mood disorder.

    Parameters:
    - db_path (str): The path to the SQLite database.

    Returns:
    - int: The count of male respondents who reported depression or a mood disorder.
    """
    sql_query = """
    SELECT COUNT(T1.SurveyID)
    FROM Answer AS T1
    INNER JOIN Question AS T2
    ON T1.QuestionID = T2.questionid
    INNER JOIN Answer AS T3
    ON T1.SurveyID = T3.SurveyID
    INNER JOIN Question AS T4
    ON T3.QuestionID = T4.questionid
    WHERE T2.questiontext LIKE 'What is your gender%'
    AND (
        LOWER(T1.AnswerText) = 'm'
        OR LOWER(T1.AnswerText) = 'male'
    )
    AND T4.questiontext LIKE 'If yes, what condition(s) have you been diagnosed with%'
    AND (
        LOWER(T3.AnswerText) LIKE '%mood disorder%'
        OR LOWER(T3.AnswerText) LIKE '%depression%'
    );
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchone()[0]

    return result


def count_depression_females(db_path):
    """
    Counts the number of female respondents (those who answered 'F' or 'Female' to the gender question)
    who reported having depression or a mood disorder.

    Parameters:
    - db_path (str): The path to the SQLite database.

    Returns:
    - int: The count of female respondents who reported depression or a mood disorder.
    """
    sql_query = """
    SELECT COUNT(T1.SurveyID)
    FROM Answer AS T1
    INNER JOIN Question AS T2
    ON T1.QuestionID = T2.questionid
    INNER JOIN Answer AS T3
    ON T1.SurveyID = T3.SurveyID
    INNER JOIN Question AS T4
    ON T3.QuestionID = T4.questionid
    WHERE T2.questiontext LIKE 'What is your gender%'
    AND (
        LOWER(T1.AnswerText) = 'f'
        OR LOWER(T1.AnswerText) = 'female'
    )
    AND T4.questiontext LIKE 'If yes, what condition(s) have you been diagnosed with%'
    AND (
        LOWER(T3.AnswerText) LIKE '%mood disorder%'
        OR LOWER(T3.AnswerText) LIKE '%depression%'
    );
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchone()[0]

    return result


def count_men(db_path):
    """
    Counts the total number of men (M or Male) in the dataset.

    Parameters:
    - db_path (str): Path to the SQLite database.

    Returns:
    - int: The count of male respondents in the dataset.
    """
    sql_query = """
    SELECT COUNT(DISTINCT a1.SurveyID) AS Male_Count
    FROM Answer a1
    JOIN Question q1 ON a1.QuestionID = q1.questionid
    WHERE q1.questiontext LIKE 'What is your gender%'
          AND (LOWER(a1.AnswerText) = 'm' OR LOWER(a1.AnswerText) = 'male')
    """

    with sqlite3.connect(db_path) as conn:
        result = pd.read_sql_query(sql_query, conn)

    return result.iloc[0, 0]


def plot_mental_health_conditions_by_gender_raw(
    db_path, condition_mapping, gender_mapping
):
    """
    Extracts, cleans, categorizes, and visualizes diagnosed mental health conditions by gender.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - condition_mapping (dict): Mapping of conditions to categories.
    - gender_mapping (dict): Mapping of gender responses to standardized categories.

    Returns:
    - A grouped bar chart showing the distribution of selected mental health conditions by gender.
    - A DataFrame with raw counts and percentages.
    - A text-based output summarizing the condition counts.
    """
    import sqlite3
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sql_query = """
    SELECT LOWER(a2.AnswerText) AS condition, a1.AnswerText AS gender
    FROM Answer a1
    JOIN Question q1 ON a1.QuestionID = q1.questionid
    JOIN Answer a2 ON a1.SurveyID = a2.SurveyID
    JOIN Question q2 ON a2.QuestionID = q2.questionid
    WHERE q1.questiontext LIKE 'What is your gender%'
    AND q2.questiontext LIKE 'If yes, what condition(s) have you been diagnosed with%';
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql_query, conn)

    # **Remove invalid entries (-1, empty, or unclear responses)**
    df = df[~df["condition"].isin(["-1", "", "none", "n/a"])]

    # **Apply gender mapping and cleaning**
    df["gender"] = (
        df["gender"].astype(str).str.strip().str.title().replace(gender_mapping)
    )

    # **Restrict genders to valid categories**
    valid_genders = ["Male", "Female", "Other"]
    df["gender"] = df["gender"].apply(lambda x: x if x in valid_genders else "Other")

    # **Apply categorization**
    df["Category"] = df["condition"].apply(
        lambda x: next(
            (
                cat
                for cat, keywords in condition_mapping.items()
                if any(keyword in x for keyword in keywords)
            ),
            "Other",
        )
    )

    # **Filter for selected conditions**
    selected_conditions = ["Depression", "OCD", "Anxiety", "PTSD"]
    df = df[df["Category"].isin(selected_conditions)]

    # **Count occurrences**
    condition_counts = df.groupby(["Category", "gender"]).size().unstack(fill_value=0)

    # **Get Rocket palette colors**
    rocket_palette = sns.color_palette("rocket", len(valid_genders))
    gender_colors = {
        "Male": rocket_palette[0],
        "Female": rocket_palette[1],
        "Other": rocket_palette[2],
    }

    # **Plot Grouped Bar Chart with custom colors**
    plt.figure(figsize=(10, 6))
    ax = condition_counts.plot(kind="bar", width=0.8)

    # Apply colors per gender
    for i, bar_group in enumerate(ax.containers):
        gender = condition_counts.columns[i]
        for bar in bar_group:
            bar.set_color(
                gender_colors.get(gender, "gray")  # Default to gray if not specified
            )

    plt.xlabel("Mental Health Condition")
    plt.ylabel("Number of Respondents")
    plt.title("Prevalence of Selected Mental Health Conditions by Gender")
    plt.legend(title="Gender")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # **Display text-based summary**
    print("Mental Health Condition Distribution by Gender:")
    for category in condition_counts.index:
        print(f"\n{category}:")
        for gender in condition_counts.columns:
            print(f"  {gender}: {condition_counts.loc[category, gender]}")

    return condition_counts


def plot_mental_health_conditions(db_path):
    """
    Extracts, cleans, categorizes, and visualizes diagnosed mental health conditions.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A bar chart showing the prevalence of different diagnosed mental health conditions.
    """
    sql_query = """
    SELECT LOWER(Answer.AnswerText) AS condition
    FROM Answer
    WHERE Answer.QuestionID = 115
    """

    with sqlite3.connect(db_path) as conn:
        conditions_df = pd.read_sql_query(sql_query, conn)

    # **Remove invalid entries (-1, empty, or unclear responses)**
    conditions_df = conditions_df[
        ~conditions_df["condition"].isin(["-1", "", "none", "n/a"])
    ]

    # **Function to assign categories based on keywords**
    def categorize_condition(condition):
        for category, keywords in condition_mapping.items():
            if any(keyword in condition for keyword in keywords):
                return category
        return "Other"  # Default category for unclassified conditions

    # **Apply categorization**
    conditions_df["Category"] = conditions_df["condition"].apply(categorize_condition)

    # **Count occurrences of each category**
    condition_counts = conditions_df["Category"].value_counts().reset_index()
    condition_counts.columns = ["Mental Health Condition", "Frequency"]

    # **Plot the data with hue to fix Seaborn's warning**
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Frequency",
        y="Mental Health Condition",
        hue="Mental Health Condition",
        data=condition_counts,
        palette="rocket",
        legend=False,  # Hide redundant legend
    )

    plt.title("Prevalence of Diagnosed Mental Health Conditions", fontsize=14)
    plt.xlabel("Number of Respondents", fontsize=12)
    plt.ylabel("Mental Health Condition", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

    return condition_counts
