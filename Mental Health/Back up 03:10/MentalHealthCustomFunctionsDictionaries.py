import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import geopandas as gpd
import statsmodels.stats.proportion as smp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, Optional


def filter_unrealistic_ages(
    age_series: pd.Series, min_age: int = 0, max_age: int = 100
) -> pd.Series:
    """
    Filters out unrealistic age values from a Pandas Series, replacing them with NaN.

    Args:
        age_series (pd.Series): The input Series containing age data.
        min_age (int): The minimum acceptable age.  Values below this are replaced with NaN.
        max_age (int): The maximum acceptable age.  Values above this are replaced with NaN.

    Returns:
        pd.Series: A new Series with unrealistic ages replaced by NaN.  The original series
                   is not modified.
    """

    cleaned_series = age_series.copy()

    cleaned_series[(cleaned_series < min_age) | (cleaned_series > max_age)] = np.nan

    return cleaned_series


question_mapping: Dict[str, str] = {
    "What is your age?": "Age",
    "How old are you?": "Age",
    "What is your gender?": "Gender",
    "Are you male or female?": "Gender",
    "What country do you live in?": "Country",
    "If you live in the United States, which state or territory do you live in?": "State",
    "Are you self-employed?": "Self-Employed",
    "Do you have a family history of mental illness?": "Family History of Mental Illness",
    "Have you ever sought treatment for a mental health disorder from a mental health professional?": "Sought Mental Health Treatment",
    "How many employees does your company or organization have?": "Company Size",
    "Is your employer primarily a tech company/organization?": "Tech Employer",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "Mental Health Benefits Provided",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "Anonymity Protection for Mental Health Treatment",
    "Would you bring up a mental health issue with a potential employer in an interview?": "Discuss Mental Health in Interview",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "Negative Consequences of Discussing Physical Health",
    "Do you think that discussing a mental health issue with your employer would have negative consequences?": "Negative Consequences of Discussing Mental Health",
    "Do you work remotely (outside of an office) at least 50% of the time?": "Remote Work",
    "Do you know the options for mental health care your employer provides?": "Awareness of Mental Health Care Options",
}

gender_mapping: Dict[str, str] = {
    "M": "Male",
    "F": "Female",
    "Male": "Male",
    "Female": "Female",
    "Trans-female": "Other",
    "Trans male": "Other",
    "Non-binary": "Other",
    "Genderqueer": "Other",
    "Agender": "Other",
    "Other": "Other",
}

country_mapping: Dict[str, str] = {
    "United States Of America": "United States",
    "Usa": "United States",
    "U.S.": "United States",
    "Uk": "United Kingdom",
    "England": "United Kingdom",
    "Deutschland": "Germany",
    "Czech Republic": "Czechia",
    "South Korea": "Korea, Republic Of",
}

condition_mapping: Dict[str, list] = {
    "Anxiety": [
        "anxiety disorder",
        "generalized anxiety",
        "social anxiety",
        "phobia",
    ],
    "Depression": ["mood disorder", "depression"],
    "OCD": ["obsessive-compulsive disorder", "ocd"],
    "PTSD": ["post-traumatic stress disorder", "ptsd"],
    "Eating Disorder": ["eating disorder", "anorexia", "bulimia"],
    "Stress Disorder": ["stress response syndromes"],
}


def plot_age_boxplot(
    df: pd.DataFrame,
    question_column: str = "questiontext",
    answer_column: str = "AnswerText",
) -> None:
    age_series = df.loc[df[question_column] == "Age", answer_column].dropna()
    age_series = pd.to_numeric(age_series, errors="coerce").dropna()
    if age_series.empty:
        print("No valid age data found for plotting.")
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=age_series, color=sns.color_palette("rocket")[2])
    plt.title("Box Plot of Age Outliers")
    plt.ylabel("Age")
    plt.show()


def plot_age_distribution_2016(merged_df: pd.DataFrame) -> pd.Series:
    merged_df_2016 = merged_df[
        merged_df["SurveyID"].astype(str).str.startswith("2016")
    ].copy()

    age_df = merged_df_2016[merged_df_2016["questiontext"] == "Age"][
        ["UserID", "AnswerText"]
    ].copy()
    age_df["Age"] = pd.to_numeric(age_df["AnswerText"], errors="coerce")

    unique_age_df = age_df.drop_duplicates(subset=["UserID"])

    unique_age_df["Age Group"] = pd.cut(
        unique_age_df["Age"],
        bins=[0, 19, 29, 39, 49, 59, 100],
        labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )

    age_group_counts = unique_age_df["Age Group"].value_counts().sort_index()

    plot_df = age_group_counts.reset_index()
    plot_df.columns = ["Age Group", "Count"]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="Age Group",
        y="Count",
        hue="Age Group",
        palette="rocket",
        legend=False,
    )

    plt.title("Age Distribution of Unique Respondents (2016)", fontsize=14)
    plt.xlabel("Age Group", fontsize=12)
    plt.ylabel("Number of Unique Respondents", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return age_group_counts


def plot_gender_distribution_2016(merged_df: pd.DataFrame) -> pd.Series:
    merged_df_2016 = merged_df[
        merged_df["SurveyID"].astype(str).str.startswith("2016")
    ].copy()

    gender_df = merged_df_2016[merged_df_2016["questiontext"] == "Gender"][
        ["UserID", "AnswerText"]
    ].copy()

    gender_df = gender_df.drop_duplicates(subset=["UserID"])
    gender_df.rename(columns={"AnswerText": "Gender Group"}, inplace=True)

    gender_counts = gender_df["Gender Group"].value_counts().sort_index()

    plot_df = gender_counts.reset_index()
    plot_df.columns = ["Gender Group", "Count"]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=plot_df,
        x="Gender Group",
        y="Count",
        hue="Gender Group",
        palette="rocket",
        legend=False,
    )

    plt.title("Gender Distribution of Unique Respondents (2016)", fontsize=14)
    plt.xlabel("Gender Group", fontsize=12)
    plt.ylabel("Number of Unique Respondents", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return gender_counts


def plot_global_country_distribution_2016(
    merged_df: pd.DataFrame, shapefile_path: str
) -> pd.DataFrame:
    try:
        merged_df_2016 = merged_df[
            merged_df["SurveyID"].astype(str).str.startswith("2016")
        ].copy()

        country_df = merged_df_2016[merged_df_2016["questiontext"] == "Country"][
            ["UserID", "AnswerText"]
        ].copy()

        country_df = country_df.drop_duplicates(subset=["UserID"])
        country_df.rename(columns={"AnswerText": "Country"}, inplace=True)

        country_df["Country"] = country_df["Country"].str.strip().str.title()

        country_counts = country_df["Country"].value_counts().reset_index()
        country_counts.columns = ["Country", "respondent_count"]

        country_counts["Country"] = country_counts["Country"].replace(
            {
                "United States": "United States Of America",
                "Uk": "United Kingdom",
                "Russia": "Russia",
                "South Korea": "South Korea",
                "North Korea": "North Korea",
                "Czech Republic": "Czechia",
            }
        )

        world = gpd.read_file(shapefile_path)
        world["ADMIN"] = world["ADMIN"].str.strip().str.title()

        world = world[world["ADMIN"] != "Antarctica"]

        merged_world = world.merge(
            country_counts, how="left", left_on="ADMIN", right_on="Country"
        )

        merged_world["respondent_count"] = merged_world["respondent_count"].fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.set_facecolor("lightblue")

        merged_world.boundary.plot(ax=ax, linewidth=0.5, color="white")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)

        merged_world.plot(
            column="respondent_count",
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            linewidth=0.3,
            edgecolor="white",
            legend=True,
            ax=ax,
            cax=cax,
        )

        ax.set_title(
            "Global Distribution of Unique Respondents by Country (2016)", fontsize=16
        )
        ax.axis("off")
        ax.set_aspect("auto")

        plt.tight_layout()
        plt.show()

        return country_counts.sort_values(by="respondent_count", ascending=False)

    except Exception as e:
        print(f"Error: {e}")


def plot_us_state_distribution_map_2016(
    merged_df: pd.DataFrame, shapefile_path: str
) -> pd.DataFrame:
    try:
        merged_df_2016 = merged_df[
            (merged_df["SurveyID"].astype(str).str.startswith("2016"))
            & (merged_df["questiontext"] == "State")
        ].copy()

        state_df = merged_df_2016[["UserID", "AnswerText"]].copy()

        state_df = state_df.drop_duplicates(subset=["UserID"])

        state_df.rename(columns={"AnswerText": "State"}, inplace=True)

        state_counts = state_df["State"].value_counts().reset_index()
        state_counts.columns = ["State", "respondent_count"]

        states = gpd.read_file(shapefile_path)

        us_states = states[states["admin"] == "United States of America"].copy()

        contiguous_states = us_states[
            ~us_states["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])
        ].copy()

        contiguous_states = contiguous_states.merge(
            state_counts, how="left", left_on="name", right_on="State"
        )

        contiguous_states["respondent_count"] = contiguous_states[
            "respondent_count"
        ].fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.set_facecolor("lightblue")

        contiguous_states.boundary.plot(ax=ax, linewidth=0.8, color="white")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        contiguous_states.plot(
            column="respondent_count",
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            linewidth=0.8,
            edgecolor="white",
            legend=True,
            ax=ax,
            cax=cax,
        )

        xmin, ymin, xmax, ymax = contiguous_states.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_title(
            "U.S. State Distribution of Survey Respondents (2016)", fontsize=16
        )
        ax.axis("off")
        ax.set_aspect(1.3)

        plt.tight_layout()
        plt.show()

        return state_counts.sort_values(by="respondent_count", ascending=False)

    except Exception as e:
        print(f"Error: {e}")


def get_state_data(db_path: str) -> pd.DataFrame:
    """
    Retrieves state-wise respondent counts from the SQLite database.
    Automatically closes the connection using 'with'.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - pd.DataFrame: DataFrame containing state names and respondent counts.
    """
    query = """ 
    SELECT AnswerText AS state, COUNT(*) AS respondent_count 
    FROM Answer 
    JOIN Question ON Answer.QuestionID = Question.QuestionID 
    WHERE QuestionText = 'If you live in the United States, which state or territory do you live in?'
    AND AnswerText IS NOT NULL
    AND AnswerText != '-1'  -- Excludes invalid responses
    GROUP BY AnswerText 
    ORDER BY respondent_count DESC; 
    """

    with sqlite3.connect(db_path) as conn:
        state_data = pd.read_sql(query, conn)  # Connection auto-closes after this line

    return state_data


def categorize_condition(condition: str, condition_mapping: Dict[str, list]) -> str:
    """
    Categorizes a given mental health condition into predefined groups.

    Parameters:
    - condition (str): The text response of a diagnosed condition.
    - condition_mapping (dict): Dictionary mapping categories to keywords.

    Returns:
    - (str): The assigned category or 'Other' if no match is found.
    """
    for category, keywords in condition_mapping.items():
        if any(keyword in condition for keyword in keywords):
            return category
    return "Other"


def plot_mental_health_conditions_with_CI_2016(
    db_path: str,
    condition_mapping: Dict[str, list],
    confidence_level: float = 0.95,
    brightness: float = 0.2,
) -> pd.DataFrame:
    """
    Plots mental health condition percentages with confidence intervals for 2016 survey respondents.

    Parameters:
    - db_path (str): SQLite database path.
    - condition_mapping (dict): Predefined mapping for condition categories.
    - confidence_level (float): Confidence level for CI (default 95%).
    - brightness (float): How much to brighten the colors (0.0 - no change, 1.0 - full white).

    Returns:
    - pd.DataFrame: Table with category, counts, percentages, and CI bounds.
    """

    query = """
    WITH USRespondents AS (
        SELECT DISTINCT
            A.UserID
        FROM Answer A
        JOIN Question Q
            ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'If you live in the United States, which state or territory do you live in%'
          AND A.AnswerText NOT IN ('-1', '', 'n/a', 'none')
          AND SUBSTR(A.SurveyID, 1, 4) = '2016'
    ),

    DiagnosedConditions AS (
        SELECT DISTINCT
            B.UserID,
            B.AnswerText AS RawCondition
        FROM Answer B
        JOIN Question Q2
            ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
          AND B.AnswerText NOT IN ('-1', '', 'n/a', 'none')
          AND SUBSTR(B.SurveyID, 1, 4) = '2016'
    )

    SELECT
        DC.UserID,
        LOWER(DC.RawCondition) AS RawCondition
    FROM USRespondents UR
    JOIN DiagnosedConditions DC
        ON UR.UserID = DC.UserID;
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No data found for 2016 respondents.")
        return pd.DataFrame()

    df["Category"] = df["RawCondition"].apply(
        lambda condition: categorize_condition(condition, condition_mapping)
    )

    df_unique = df.drop_duplicates(subset=["UserID", "Category"])

    grouped_df = (
        df_unique.groupby("Category").agg(TotalUsers=("UserID", "count")).reset_index()
    )

    total_respondents = df_unique["UserID"].nunique()

    grouped_df["Percentage"] = (grouped_df["TotalUsers"] / total_respondents) * 100

    ci_bounds = smp.proportion_confint(
        count=grouped_df["TotalUsers"],
        nobs=total_respondents,
        alpha=(1 - confidence_level),
        method="wilson",
    )

    grouped_df["CI Lower"] = ci_bounds[0] * 100
    grouped_df["CI Upper"] = ci_bounds[1] * 100

    grouped_df_sorted = grouped_df.sort_values(
        by="Percentage", ascending=True
    ).reset_index(drop=True)

    base_palette = sns.color_palette("rocket_r", len(grouped_df_sorted))

    def lighten_color(color: tuple, amount: float = 0.2) -> tuple:
        return tuple([min(1, c + (1 - c) * amount) for c in color])

    brightened_colors = [lighten_color(c, brightness) for c in base_palette]

    plt.figure(figsize=(10, 6))
    y_positions = np.arange(len(grouped_df_sorted))

    bars = plt.barh(
        y=y_positions,
        width=grouped_df_sorted["Percentage"],
        color=brightened_colors,
        edgecolor="black",
    )

    for i, (bar, lower, upper) in enumerate(
        zip(bars, grouped_df_sorted["CI Lower"], grouped_df_sorted["CI Upper"])
    ):
        plt.errorbar(
            x=grouped_df_sorted["Percentage"][i],
            y=i,
            xerr=[
                [grouped_df_sorted["Percentage"][i] - lower],
                [upper - grouped_df_sorted["Percentage"][i]],
            ],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=5,
        )

    for i, (bar, pct) in enumerate(zip(bars, grouped_df_sorted["Percentage"])):
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2 + 0.1,
            f"{pct:.1f}%",
            va="bottom",
            ha="left",
            fontsize=9,
        )

    plt.yticks(y_positions, grouped_df_sorted["Category"])
    plt.xlabel("Percentage of Respondents (%)")
    plt.title(
        f"Mental Health Conditions with {int(confidence_level * 100)}% Confidence Intervals (2016 Respondents)"
    )
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return grouped_df_sorted


import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict


def generate_age_conditions_contingency_heatmap_2016(
    db_path: str, merged_df: pd.DataFrame, condition_mapping: Dict[str, list]
) -> pd.DataFrame:
    """
    Generates a contingency table and heatmap for unique respondents in 2016,
    showing the distribution of diagnosed mental health conditions across age groups.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - merged_df (pd.DataFrame): Pre-cleaned DataFrame containing survey responses.
    - condition_mapping (dict): Mapping of condition categories to keywords.

    Returns:
    - pd.DataFrame: Contingency table (cross-tabulation) of Age Groups vs Conditions.
    """

    def categorize_condition_2016(condition: str) -> str:
        for category, keywords in condition_mapping.items():
            if any(keyword.lower() in condition.lower() for keyword in keywords):
                return category
        return "Other"

    merged_df_2016 = merged_df[
        merged_df["SurveyID"].astype(str).str.startswith("2016")
    ].copy()

    merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "Age"] = pd.to_numeric(
        merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "AnswerText"],
        errors="coerce",
    )

    merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "Age Group"] = pd.cut(
        merged_df_2016["Age"],
        bins=[0, 19, 29, 39, 49, 59, 100],
        labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )

    conditions_df = merged_df_2016[
        merged_df_2016["questiontext"].str.contains(
            "If yes, what condition", case=False, na=False
        )
    ].copy()

    conditions_df = conditions_df.rename(columns={"AnswerText": "condition"})

    conditions_df = conditions_df[
        ~conditions_df["condition"].isin(["-1", "", "none", "n/a"])
    ]

    conditions_df["Category"] = conditions_df["condition"].apply(
        categorize_condition_2016
    )

    age_df = merged_df_2016[merged_df_2016["questiontext"] == "Age"][
        ["UserID", "Age Group"]
    ]
    merged_data = age_df.merge(
        conditions_df[["UserID", "Category"]], on="UserID", how="inner"
    ).drop_duplicates(subset=["UserID", "Category"])

    contingency_table = pd.crosstab(merged_data["Age Group"], merged_data["Category"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="d", linewidths=0.5)
    plt.title(
        "Contingency Heatmap of Mental Health Conditions Across Age Groups (2016)"
    )
    plt.xlabel("Mental Health Conditions")
    plt.ylabel("Age Groups")
    plt.tight_layout()
    plt.show()

    return contingency_table


def generate_normalized_age_conditions_heatmap_2016(
    db_path: str, merged_df: pd.DataFrame, condition_mapping: Dict[str, list]
) -> pd.DataFrame:
    """
    Generates a normalized contingency table and heatmap for unique respondents in 2016,
    showing the percentage distribution of diagnosed mental health conditions across age groups.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - merged_df (pd.DataFrame): Pre-cleaned DataFrame containing survey responses.
    - condition_mapping (dict): Mapping of condition categories to keywords.

    Returns:
    - pd.DataFrame: Normalized contingency table (percentages) of Age Groups vs Conditions.
    """

    def categorize_condition_2016(condition: str) -> str:
        for category, keywords in condition_mapping.items():
            if any(keyword.lower() in condition.lower() for keyword in keywords):
                return category
        return "Other"

    merged_df_2016 = merged_df[
        merged_df["SurveyID"].astype(str).str.startswith("2016")
    ].copy()

    merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "Age"] = pd.to_numeric(
        merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "AnswerText"],
        errors="coerce",
    )

    merged_df_2016.loc[merged_df_2016["questiontext"] == "Age", "Age Group"] = pd.cut(
        merged_df_2016["Age"],
        bins=[0, 19, 29, 39, 49, 59, 100],
        labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"],
        right=False,
    )

    conditions_df = merged_df_2016[
        merged_df_2016["questiontext"].str.contains(
            "If yes, what condition", case=False, na=False
        )
    ].copy()

    conditions_df = conditions_df.rename(columns={"AnswerText": "condition"})

    conditions_df = conditions_df[
        ~conditions_df["condition"].isin(["-1", "", "none", "n/a"])
    ]

    conditions_df["Category"] = conditions_df["condition"].apply(
        categorize_condition_2016
    )

    age_df = merged_df_2016[merged_df_2016["questiontext"] == "Age"][
        ["UserID", "Age Group"]
    ]
    merged_data = age_df.merge(
        conditions_df[["UserID", "Category"]], on="UserID", how="inner"
    ).drop_duplicates(subset=["UserID", "Category"])

    contingency_table = pd.crosstab(merged_data["Age Group"], merged_data["Category"])

    normalized_table = (
        contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    )

    plt.figure(figsize=(12, 7))

    sns.heatmap(
        normalized_table,
        annot=True,
        fmt=".1f",
        cmap="rocket_r",
        linewidths=0.5,
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    plt.title(
        "Normalized Percentage Heatmap of Mental Health Conditions Across Age Groups (2016)",
        fontsize=14,
    )
    plt.xlabel("Mental Health Conditions", fontsize=12)
    plt.ylabel("Age Groups", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return normalized_table


def rank_top_20_categorized_conditions_2016(db_path: str) -> pd.DataFrame:
    """
    Retrieves and ranks the top 20 mental health conditions by count for the 2016 dataset,
    counting unique users (UserID) and categorizing them into major groups.

    Parameters:
    - db_path (str): Path to the SQLite database.

    Returns:
    - pd.DataFrame: Top 20 ranked conditions with counts and total users.
    """

    query = """
    WITH CountrySurvey AS (
        SELECT 
            A.SurveyID,
            A.UserID,
            A.AnswerText AS Country
        FROM Answer A
        JOIN Question Q 
            ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'What country do you live in%'
    ),

    DiagnosedConditions AS (
        SELECT
            B.SurveyID,
            B.UserID,
            B.AnswerText AS RawCondition
        FROM Answer B
        JOIN Question Q2 
            ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
        AND B.AnswerText NOT IN ('-1', '', 'none', 'n/a')
    ),

    Filtered2016 AS (
        SELECT
            CS.Country,
            DC.UserID,
            DC.RawCondition
        FROM CountrySurvey CS
        JOIN DiagnosedConditions DC
            ON CS.UserID = DC.UserID
        WHERE CS.SurveyID LIKE '2016%'
    ),

    CategorizedConditions AS (
        SELECT
            Country,
            UserID,
            CASE 
                WHEN LOWER(RawCondition) LIKE '%depression%' THEN 'Depression'
                WHEN LOWER(RawCondition) LIKE '%bipolar%' THEN 'Depression'
                WHEN LOWER(RawCondition) LIKE '%anxiety%' THEN 'Anxiety'
                WHEN LOWER(RawCondition) LIKE '%panic%' THEN 'Anxiety'
                WHEN LOWER(RawCondition) LIKE '%ptsd%' THEN 'PTSD'
                WHEN LOWER(RawCondition) LIKE '%post-traumatic%' THEN 'PTSD'
                WHEN LOWER(RawCondition) LIKE '%ocd%' THEN 'OCD'
                WHEN LOWER(RawCondition) LIKE '%obsessive%' THEN 'OCD'
                WHEN LOWER(RawCondition) LIKE '%stress%' THEN 'Stress Disorder'
                WHEN LOWER(RawCondition) LIKE '%eating%' THEN 'Eating Disorder'
                WHEN LOWER(RawCondition) LIKE '%anorexia%' THEN 'Eating Disorder'
                WHEN LOWER(RawCondition) LIKE '%bulimia%' THEN 'Eating Disorder'
                ELSE 'Other'
            END AS Category
        FROM Filtered2016
    ),

    UniqueUsersPerCategory AS (
        SELECT DISTINCT
            Country,
            UserID,
            Category
        FROM CategorizedConditions
    ),

    CountryCategoryCounts AS (
        SELECT
            Country,
            Category,
            COUNT(*) AS TotalUsers
        FROM UniqueUsersPerCategory
        GROUP BY Country, Category
    ),

    RankedConditions AS (
        SELECT
            Country,
            Category,
            TotalUsers,
            RANK() OVER (ORDER BY TotalUsers DESC) AS Rank
        FROM CountryCategoryCounts
    )

    SELECT
        Country,
        Category,
        TotalUsers,
        Rank
    FROM RankedConditions
    WHERE Rank <= 20
    ORDER BY Rank;
    """

    with sqlite3.connect(db_path) as conn:
        result_df = pd.read_sql_query(query, conn)

    print(result_df)
    return result_df


def rank_top_20_conditions_across_states_2016(
    db_path: str, condition_mapping: Dict[str, list]
) -> pd.DataFrame:
    """
    Retrieves the strict top 20 mental health condition counts across U.S. states for 2016,
    counting unique users (UserID). Same condition can appear multiple times if from different states.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - condition_mapping (dict): Predefined mapping for mental health condition categories.

    Returns:
    - pd.DataFrame: Top 20 rows with columns ['State', 'Category', 'TotalUsers', 'Rank'].
    """

    query = """
    WITH StateRespondents AS (
        SELECT DISTINCT
            A.UserID,
            A.AnswerText AS State
        FROM Answer A
        JOIN Question Q
            ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'If you live in the United States, which state or territory do you live in%'
          AND A.AnswerText NOT IN ('-1', '', 'n/a', 'none')
          AND SUBSTR(A.SurveyID, 1, 4) = '2016'
    ),

    DiagnosedConditions AS (
        SELECT DISTINCT
            B.UserID,
            B.AnswerText AS RawCondition
        FROM Answer B
        JOIN Question Q2
            ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
          AND B.AnswerText NOT IN ('-1', '', 'n/a', 'none')
          AND SUBSTR(B.SurveyID, 1, 4) = '2016'
    )

    SELECT
        SR.State,
        DC.UserID,
        LOWER(DC.RawCondition) AS RawCondition
    FROM StateRespondents SR
    JOIN DiagnosedConditions DC
        ON SR.UserID = DC.UserID;
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No data found for 2016 respondents by state.")
        return pd.DataFrame()

    df["Category"] = df["RawCondition"].apply(
        lambda condition: categorize_condition(condition, condition_mapping)
    )

    df_unique = df.drop_duplicates(subset=["UserID", "State", "Category"])

    grouped_df = (
        df_unique.groupby(["State", "Category"])
        .agg(TotalUsers=("UserID", "count"))
        .reset_index()
    )

    grouped_df_sorted = grouped_df.sort_values(
        by="TotalUsers", ascending=False
    ).reset_index(drop=True)

    grouped_df_sorted["Rank"] = grouped_df_sorted.index + 1

    top_20_df = grouped_df_sorted.head(20)

    print(top_20_df)

    return top_20_df


def plot_answers_by_year(db_path):
    """
    Retrieves the count of answers recorded each year from the SQLite database and visualizes it using a bar plot.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A bar chart displaying the number of answers recorded per year.
    """
    sql_query = """
    SELECT SUBSTR(SurveyID, 1, 4) AS Year, COUNT(*) AS AnswerCount
    FROM Answer
    GROUP BY Year
    ORDER BY Year;
    """

    with sqlite3.connect(db_path) as conn:
        year_df = pd.read_sql_query(sql_query, conn)

    year_df["Year"] = pd.to_numeric(year_df["Year"], errors="coerce")

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Year", y="AnswerCount", hue="Year", data=year_df, palette="rocket_r")
    plt.legend([], [], frameon=False)

    plt.title("Number of Answers Recorded Per Year")
    plt.xlabel("Year")
    plt.ylabel("Count of Answers")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

    return year_df


def unique_questions_per_year(db_path: str) -> pd.DataFrame:
    sql_query = """
    SELECT SUBSTR(SurveyID, 1, 4) AS Year, COUNT(DISTINCT QuestionID) AS UniqueQuestions
    FROM Answer
    GROUP BY Year
    ORDER BY Year;
    """

    with sqlite3.connect(db_path) as conn:
        unique_questions_df = pd.read_sql_query(sql_query, conn)

    unique_questions_df["Year"] = pd.to_numeric(
        unique_questions_df["Year"], errors="coerce"
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="Year",
        y="UniqueQuestions",
        hue="Year",
        data=unique_questions_df,
        palette="rocket",
    )
    plt.legend([], [], frameon=False)

    plt.title("Number of Unique Questions Per Year")
    plt.xlabel("Year")
    plt.ylabel("Unique Questions")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

    return unique_questions_df


def plot_unique_respondents_by_year(db_path: str) -> pd.DataFrame:
    query = """
    SELECT SUBSTR(SurveyID, 1, 4) AS Year, COUNT(DISTINCT UserID) AS UniqueRespondents
    FROM Answer
    GROUP BY Year
    ORDER BY Year;
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="Year", y="UniqueRespondents", hue="Year", data=df, palette="rocket_r"
    )
    plt.legend([], [], frameon=False)

    plt.title("Unique Respondents by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Unique Respondents")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

    return df


def count_unique_genders_2016(db_path):
    """
    Counts the number of unique respondents (UserID) by gender in 2016:
    - Male (M, Male)
    - Female (F, Female, Woman)
    - Other (nonbinary, agender, etc.)

    Parameters:
    - db_path (str): Path to the SQLite database.

    Returns:
    - dict: Counts of unique respondents by gender category.
    """

    queries = {
        "Male": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN ('m', 'male')
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
        "Female": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN ('f', 'female', 'woman')
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
        "Other": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN (
                      'nonbinary', 'non-binary', 'genderqueer', 'agender', 'other',
                      'trans', 'gender fluid', 'genderfluid', 'bigender', 'pangender',
                      'gender nonconforming', 'androgyne'
                  )
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
    }

    gender_counts = {}

    with sqlite3.connect(db_path) as conn:
        for gender_category, query in queries.items():
            result = pd.read_sql_query(query, conn)
            gender_counts[gender_category] = result.iloc[0, 0]

    return gender_counts


def count_unique_genders_2016(db_path: str) -> Dict[str, int]:
    queries = {
        "Male": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN ('m', 'male')
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
        "Female": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN ('f', 'female', 'woman')
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
        "Other": """
            SELECT COUNT(DISTINCT A.UserID) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            WHERE Q.QuestionText LIKE 'What is your gender%'
                  AND LOWER(A.AnswerText) IN (
                      'nonbinary', 'non-binary', 'genderqueer', 'agender', 'other',
                      'trans', 'gender fluid', 'genderfluid', 'bigender', 'pangender',
                      'gender nonconforming', 'androgyne'
                  )
                  AND SUBSTR(A.SurveyID, 1, 4) = '2016';
        """,
    }

    gender_counts: Dict[str, int] = {}

    with sqlite3.connect(db_path) as conn:
        for gender_category, query in queries.items():
            result = pd.read_sql_query(query, conn)
            gender_counts[gender_category] = result.iloc[0, 0]

    return gender_counts


def count_men_conditions_2016(
    db_path: str, condition_mapping: Dict[str, list]
) -> Dict[str, int]:
    query = """
    WITH MaleRespondents AS (
        SELECT DISTINCT A.UserID
        FROM Answer A
        JOIN Question Q ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'What is your gender%'
              AND LOWER(A.AnswerText) IN ('m', 'male')
              AND SUBSTR(A.SurveyID, 1, 4) = '2016'
    ),

    MaleConditions AS (
        SELECT DISTINCT MR.UserID, LOWER(B.AnswerText) AS raw_condition
        FROM MaleRespondents MR
        JOIN Answer B ON MR.UserID = B.UserID
        JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
              AND LOWER(B.AnswerText) NOT IN ('-1', '', 'none', 'n/a')
    )

    SELECT raw_condition, UserID
    FROM MaleConditions;
    """

    with sqlite3.connect(db_path) as conn:
        condition_df = pd.read_sql_query(query, conn)

    condition_counts: Dict[str, set] = {key: set() for key in condition_mapping.keys()}
    condition_counts["Other"] = set()

    for _, row in condition_df.iterrows():
        user_id = row["UserID"]
        raw_condition = row["raw_condition"]

        categorized = False
        for category, keywords in condition_mapping.items():
            if any(keyword in raw_condition for keyword in keywords):
                condition_counts[category].add(user_id)
                categorized = True
                break

        if not categorized:
            condition_counts["Other"].add(user_id)

    return {category: len(user_ids) for category, user_ids in condition_counts.items()}


def count_women_conditions_2016(
    db_path: str, condition_mapping: Dict[str, list]
) -> Dict[str, int]:
    query = """
    WITH FemaleRespondents AS (
        SELECT DISTINCT A.UserID
        FROM Answer A
        JOIN Question Q ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'What is your gender%'
              AND LOWER(A.AnswerText) IN ('f', 'female', 'woman')
              AND SUBSTR(A.SurveyID, 1, 4) = '2016'
    ),

    FemaleConditions AS (
        SELECT DISTINCT FR.UserID, LOWER(B.AnswerText) AS raw_condition
        FROM FemaleRespondents FR
        JOIN Answer B ON FR.UserID = B.UserID
        JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
              AND LOWER(B.AnswerText) NOT IN ('-1', '', 'none', 'n/a')
    )

    SELECT raw_condition, UserID
    FROM FemaleConditions;
    """

    with sqlite3.connect(db_path) as conn:
        condition_df = pd.read_sql_query(query, conn)

    condition_counts: Dict[str, set] = {key: set() for key in condition_mapping.keys()}
    condition_counts["Other"] = set()

    for _, row in condition_df.iterrows():
        user_id = row["UserID"]
        raw_condition = row["raw_condition"]

        categorized = False
        for category, keywords in condition_mapping.items():
            if any(keyword in raw_condition for keyword in keywords):
                condition_counts[category].add(user_id)
                categorized = True
                break

        if not categorized:
            condition_counts["Other"].add(user_id)

    return {category: len(user_ids) for category, user_ids in condition_counts.items()}


import sqlite3
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns


def count_other_gender_conditions_2016(
    db_path: str, condition_mapping: Dict[str, list]
) -> Dict[str, int]:
    query = """
    WITH OtherGenderRespondents AS (
        SELECT DISTINCT A.UserID
        FROM Answer A
        JOIN Question Q ON A.QuestionID = Q.QuestionID
        WHERE Q.QuestionText LIKE 'What is your gender%'
              AND LOWER(A.AnswerText) IN (
                  'nonbinary', 'non-binary', 'genderqueer', 'agender', 'other',
                  'trans', 'gender fluid', 'genderfluid', 'bigender', 'pangender',
                  'gender nonconforming', 'androgyne'
              )
              AND SUBSTR(A.SurveyID, 1, 4) = '2016'
    ),

    OtherGenderConditions AS (
        SELECT DISTINCT ORG.UserID, LOWER(B.AnswerText) AS raw_condition
        FROM OtherGenderRespondents ORG
        JOIN Answer B ON ORG.UserID = B.UserID
        JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
        WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
              AND LOWER(B.AnswerText) NOT IN ('-1', '', 'none', 'n/a')
    )

    SELECT raw_condition, UserID
    FROM OtherGenderConditions;
    """

    with sqlite3.connect(db_path) as conn:
        condition_df = pd.read_sql_query(query, conn)

    condition_counts: Dict[str, set] = {key: set() for key in condition_mapping.keys()}
    condition_counts["Other"] = set()

    for _, row in condition_df.iterrows():
        user_id = row["UserID"]
        raw_condition = row["raw_condition"]

        categorized = False
        for category, keywords in condition_mapping.items():
            if any(keyword in raw_condition for keyword in keywords):
                condition_counts[category].add(user_id)
                categorized = True
                break

        if not categorized:
            condition_counts["Other"].add(user_id)

    return {category: len(user_ids) for category, user_ids in condition_counts.items()}


def plot_multibar_conditions_2016(
    men_conditions: Dict[str, int],
    women_conditions: Dict[str, int],
    other_gender_conditions: Dict[str, int],
) -> None:
    df_men = pd.DataFrame(list(men_conditions.items()), columns=["Condition", "Count"])
    df_men["Gender"] = "Men"

    df_women = pd.DataFrame(
        list(women_conditions.items()), columns=["Condition", "Count"]
    )
    df_women["Gender"] = "Women"

    df_other = pd.DataFrame(
        list(other_gender_conditions.items()), columns=["Condition", "Count"]
    )
    df_other["Gender"] = "Other Gender"

    combined_df = pd.concat([df_men, df_women, df_other])

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Condition", y="Count", hue="Gender", data=combined_df, palette="rocket"
    )

    plt.title("Unique Respondents by Condition and Gender (2016)")
    plt.xlabel("Mental Health Condition")
    plt.ylabel("Number of Unique Respondents")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()


def plot_multibar_conditions_percentage_2016(
    men_conditions: Dict[str, int],
    women_conditions: Dict[str, int],
    other_gender_conditions: Dict[str, int],
) -> None:
    """
    Plots a multi-bar chart comparing normalized percentages of mental health condition counts
    by gender group in 2016 and prints the percentages.

    Parameters:
    - men_conditions: Condition counts for men.
    - women_conditions: Condition counts for women.
    - other_gender_conditions: Condition counts for other genders.

    Returns:
    - None: Displays the plot and prints percentages.
    """

    df_men = pd.DataFrame(list(men_conditions.items()), columns=["Condition", "Count"])
    df_men["Gender"] = "Men"

    df_women = pd.DataFrame(
        list(women_conditions.items()), columns=["Condition", "Count"]
    )
    df_women["Gender"] = "Women"

    df_other = pd.DataFrame(
        list(other_gender_conditions.items()), columns=["Condition", "Count"]
    )
    df_other["Gender"] = "Other Gender"

    combined_df = pd.concat([df_men, df_women, df_other])

    combined_df["Percentage"] = combined_df.groupby("Gender")["Count"].transform(
        lambda x: (x / x.sum()) * 100
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="Condition",
        y="Percentage",
        hue="Gender",
        data=combined_df,
        palette="rocket",
    )

    plt.title("Normalized Percentages of Respondents by Condition and Gender (2016)")
    plt.xlabel("Mental Health Condition")
    plt.ylabel("Percentage of Respondents (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

    print("\nNormalized Percentages by Condition and Gender (2016):\n")
    for gender in combined_df["Gender"].unique():
        gender_df = combined_df[combined_df["Gender"] == gender]
        print(f"{gender}:")
        for _, row in gender_df.iterrows():
            print(f"  {row['Condition']}: {row['Percentage']:.2f}%")
        print()


def plot_condition_breakdown_2016_horizontal(
    db_path: str, condition_mapping: Dict[str, list]
) -> pd.DataFrame:
    """
    Plots a horizontal bar chart showing the proportion of 2016 survey respondents
    who reported no mental health condition, one condition, or multiple conditions.

    Parameters:
    - db_path: SQLite database path.
    - condition_mapping: Predefined mapping for condition categories.

    Returns:
    - pd.DataFrame: Respondent counts and percentages for each category.
    """

    query_respondents = """
    SELECT DISTINCT
        A.UserID
    FROM Answer A
    JOIN Question Q
        ON A.QuestionID = Q.QuestionID
    WHERE Q.QuestionText LIKE 'If you live in the United States, which state or territory do you live in%'
      AND A.AnswerText NOT IN ('-1', '', 'n/a', 'none')
      AND SUBSTR(A.SurveyID, 1, 4) = '2016';
    """

    query_conditions = """
    SELECT DISTINCT
        B.UserID,
        LOWER(B.AnswerText) AS RawCondition
    FROM Answer B
    JOIN Question Q2
        ON B.QuestionID = Q2.QuestionID
    WHERE Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
      AND B.AnswerText NOT IN ('-1', '', 'n/a', 'none')
      AND SUBSTR(B.SurveyID, 1, 4) = '2016';
    """

    with sqlite3.connect(db_path) as conn:
        respondents_df = pd.read_sql_query(query_respondents, conn)
        conditions_df = pd.read_sql_query(query_conditions, conn)

    all_respondents = set(respondents_df["UserID"])

    condition_counts = conditions_df.groupby("UserID").nunique()["RawCondition"]

    respondents_with_multiple_conditions = condition_counts[condition_counts >= 2].index
    respondents_with_single_condition = condition_counts[condition_counts == 1].index
    respondents_with_no_conditions = all_respondents - set(condition_counts.index)

    data = {
        "Reported Multiple Conditions": len(respondents_with_multiple_conditions),
        "Reported One Condition": len(respondents_with_single_condition),
        "Reported No Mental Health Condition": len(respondents_with_no_conditions),
    }

    labels = list(data.keys())
    counts = list(data.values())
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    result_df = pd.DataFrame(
        {
            "Group": labels,
            "Count": counts,
            "Percentage": [f"{pct:.1f}%" for pct in percentages],
        }
    )

    palette = sns.color_palette("rocket_r", len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(labels))

    bars = ax.barh(y=y_positions, width=percentages, color=palette, edgecolor="black")

    for i, (bar, pct, count) in enumerate(zip(bars, percentages, counts)):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({count})",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of Respondents (%)")
    ax.set_title("2016 Respondents by Mental Health Condition Status")

    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return result_df
