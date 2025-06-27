import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import geopandas as gpd
import statsmodels.stats.proportion as smp


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
    # Create a copy to avoid modifying the original Series
    cleaned_series = age_series.copy()

    # Use boolean indexing for efficient filtering and replacement
    cleaned_series[(cleaned_series < min_age) | (cleaned_series > max_age)] = np.nan

    return cleaned_series


# Dictionary for standardizing question text
question_mapping = {
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

gender_mapping = {
    "M": "Male",
    "F": "Female",
    "Male": "Male",
    "Female": "Female",
    "Other": "Other",
    "Non-binary": "Other",
    "Nonbinary": "Other",
    "Prefer not to say": "Other",
    "Prefer Not to Say": "Other",
}


# Dictionary to unify country names (fix duplicates)
country_mapping = {
    "United States Of America": "United States",
    "Usa": "United States",
    "U.S.": "United States",
    "Uk": "United Kingdom",
    "England": "United Kingdom",
    "Deutschland": "Germany",
    "Czech Republic": "Czechia",
    "South Korea": "Korea, Republic Of",
}

condition_mapping = {
    "Anxiety": [
        "anxiety disorder",
        "generalized anxiety",
        "social anxiety",
        "phobia",
    ],
    "Depression": ["mood disorder", "depression"],
    "Bipolar Disorder": ["bipolar disorder"],
    "OCD": ["obsessive-compulsive disorder", "ocd"],
    "PTSD": ["post-traumatic stress disorder", "ptsd"],
    "Eating Disorder": ["eating disorder", "anorexia", "bulimia"],
    "Stress Disorder": ["stress response syndromes"],
}


def plot_age_boxplot(df, question_column="questiontext", answer_column="AnswerText"):
    """
    Plots a boxplot for age distribution from a survey dataset.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing survey responses.
    - question_column (str): The column name where the question text is stored.
    - answer_column (str): The column name where the answer values are stored.

    Returns:
    - Displays a boxplot showing the age distribution.
    """

    # Filter the dataset to extract only age responses
    age_series = df.loc[df[question_column] == "Age", answer_column].dropna()

    # Convert to numeric values
    age_series = pd.to_numeric(age_series, errors="coerce").dropna()

    # Ensure we have valid data before plotting
    if age_series.empty:
        print("No valid age data found for plotting.")
        return

    # Plot the boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=age_series, color=sns.color_palette("rocket")[2])
    plt.title("Box Plot of Age Outliers")
    plt.ylabel("Age")
    plt.show()


def plot_age_distribution(db_path):
    """
    Retrieves age distribution data from the SQLite database and visualizes it using a bar plot.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A bar plot displaying the number of respondents in each age group.
    """
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

    # Retrieve data using SQLite connection
    with sqlite3.connect(db_path) as conn:
        age_df = pd.read_sql_query(sql_query, conn)

    # Create the bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="AgeGroup",
        y="RespondentCount",
        hue="AgeGroup",  # Fix for Seaborn warning
        data=age_df,
        palette="rocket",  # Use original "rocket" palette
        legend=False,  # Disable legend
    )

    plt.title("Age Distribution of Respondents (SQL)")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


def plot_gender_distribution(db_path):
    """
    Retrieves gender distribution data from the SQLite database and visualizes it using a pie chart.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A pie chart displaying the percentage of respondents by gender.
    """
    sql_query = """
    SELECT
      CASE
        WHEN Answer.AnswerText = 'Male' THEN 'Male'
        WHEN Answer.AnswerText = 'Female' THEN 'Female'
        ELSE 'Other' -- Consider adding more specific categories if available
      END AS GenderCategory,
      CAST(COUNT(*) AS REAL) * 100 / (
        SELECT COUNT(*)
        FROM Answer
        JOIN Question ON Answer.QuestionID = Question.questionid
        WHERE Question.questiontext LIKE 'What is your gender?'
      ) AS Percentage
    FROM Answer
    JOIN Question ON Answer.QuestionID = Question.questionid
    WHERE Question.questiontext LIKE 'What is your gender?'
    GROUP BY GenderCategory
    ORDER BY Percentage DESC;
    """

    # Retrieve data using SQLite connection
    with sqlite3.connect(db_path) as conn:
        gender_df = pd.read_sql_query(sql_query, conn)

    # Extract values for plotting
    labels = gender_df["GenderCategory"]
    sizes = gender_df["Percentage"]

    # Get "rocket" color palette
    colors = sns.color_palette("rocket", len(labels))

    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 12},
    )

    # Style the annotations
    for autotext in autotexts:
        autotext.set_color("white")  # Make percentage text white
        autotext.set_fontweight("bold")

    # Title and display
    plt.title("Gender Breakdown of Respondents", fontsize=14)
    plt.show()


def plot_geographic_distribution(db_path, shapefile_path):
    """
    Plots a choropleth map showing the geographic distribution of survey respondents.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - shapefile_path (str): Path to the Natural Earth shapefile (.shp).

    Returns:
    - A choropleth map displaying the number of survey respondents per country.
    """

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # SQL query to get country-wise respondent count
        query = """
        SELECT AnswerText AS country, COUNT(*) AS respondent_count
        FROM Answer
        JOIN Question ON Answer.QuestionID = Question.QuestionID
        WHERE QuestionText IN ('What country do you live in?', 'Country')
        GROUP BY AnswerText
        ORDER BY respondent_count DESC;
        """

        # Load data into a DataFrame
        country_data = pd.read_sql(query, conn)

        # Close the database connection
        conn.close()

        # Load the world map from the provided shapefile path
        world = gpd.read_file(shapefile_path)

        # Ensure geometries are valid and simplify MultiPolygons
        world["geometry"] = world["geometry"].apply(
            lambda geom: geom.simplify(0.1) if geom is not None else None
        )

        # Merge survey data with world map data
        world = world.merge(
            country_data, how="left", left_on="NAME", right_on="country"
        )

        # Replace NaN values (countries with no respondents) with 0
        world["respondent_count"] = world["respondent_count"].fillna(0)

        # Get the "rocket" colormap from Seaborn
        rocket_palette = sns.color_palette("rocket_r", as_cmap=True)

        # Plot the choropleth map
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        world.boundary.plot(ax=ax, linewidth=1, color="black")  # Country borders
        world.plot(
            column="respondent_count",
            cmap=rocket_palette,
            linewidth=0.8,
            edgecolor="black",
            legend=True,
            ax=ax,
        )

        # Customize the plot
        plt.title("Geographic Distribution of Survey Respondents", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(False)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


def plot_state_distribution(db_path, shapefile_path):
    """
    Plots a choropleth map showing the geographic distribution of survey respondents by state.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - shapefile_path (str): Path to the state-level shapefile (.shp).

    Returns:
    - A choropleth map displaying the number of survey respondents per state.
    """
    # Retrieve state-level survey data
    state_data = get_state_data(db_path)

    # Load the state map from the provided shapefile path
    states = gpd.read_file(shapefile_path)

    # Print available columns in the shapefile for debugging
    print("Shapefile Columns:", states.columns)

    # Standardize 'DC' to 'District of Columbia'
    state_data["state"] = state_data["state"].replace({"dc": "district of columbia"})

    # Convert state names to lowercase & remove extra spaces
    state_data["state"] = state_data["state"].str.lower().str.strip()
    states["name"] = states["name"].str.lower().str.strip()

    # Merge survey data with shapefile using cleaned names
    states = states.merge(state_data, how="left", left_on="name", right_on="state")

    # Replace NaN values (states with no respondents) with 0
    states["respondent_count"] = states["respondent_count"].fillna(0)

    # Plot the choropleth map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    states.boundary.plot(ax=ax, linewidth=1, color="black")  # State borders
    states.plot(
        column="respondent_count",
        cmap="Oranges",
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax,
    )

    # Customize the plot
    plt.title("State-Level Geographic Distribution of Survey Respondents", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)
    plt.show()


def get_state_data(db_path):
    """
    Retrieves state-wise respondent counts from the SQLite database.
    Automatically closes the connection using 'with'.
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


def states_distribution_percentages(db_path):
    """
    Retrieves U.S. state-wise respondent percentages from SQLite, ensuring duplicate state names
    are merged and applying additional cleaning using `get_state_data()`.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A DataFrame with cleaned state names and response percentages.
    """
    sql_query = """
    SELECT AnswerText AS State, COUNT(*) AS Count
    FROM Answer
    WHERE QuestionID IN (
        SELECT QuestionID FROM Question
        WHERE QuestionText LIKE 'If you live in the United States, which state or territory do you live in%'
    )
    GROUP BY AnswerText
    ORDER BY Count DESC;
    """

    with sqlite3.connect(db_path) as conn:
        state_df = pd.read_sql_query(sql_query, conn)

    # Apply state cleaning using get_state_data()
    state_df = get_state_data(state_df)

    # Remove invalid state values like '-1'
    state_df = state_df[state_df["State"] != "-1"]

    # Calculate percentages based on total valid responses
    state_df["Percentage"] = (state_df["Count"] / state_df["Count"].sum()) * 100

    # Drop raw count column (optional)
    state_df = state_df.drop(columns=["Count"])

    return state_df


def clean_state_distribution(db_path):
    """
    Retrieves and cleans U.S. state-wise respondent percentages directly from SQLite.
    Cleans state names, merges duplicates, and removes invalid responses efficiently.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A DataFrame with cleaned U.S. state names and response percentages.
    """
    sql_query = """
    SELECT 
        CASE
            WHEN UPPER(TRIM(AnswerText)) IN ('CA', 'CALIF.') THEN 'California'
            WHEN UPPER(TRIM(AnswerText)) IN ('NY', 'N.Y.') THEN 'New York'
            WHEN UPPER(TRIM(AnswerText)) IN ('TX', 'TEX.') THEN 'Texas'
            WHEN UPPER(TRIM(AnswerText)) IN ('FL', 'FLA.') THEN 'Florida'
            WHEN UPPER(TRIM(AnswerText)) IN ('IL', 'ILL.') THEN 'Illinois'
            ELSE UPPER(TRIM(AnswerText)) -- Keeps other states as they are
        END AS State, 
        COUNT(*) * 100.0 / (
            SELECT COUNT(*) FROM Answer 
            WHERE QuestionID IN 
            (SELECT QuestionID FROM Question 
             WHERE QuestionText LIKE 'If you live in the United States, which state or territory do you live in%')
            AND AnswerText NOT IN ('-1', 'Unknown', 'N/A', 'None')  -- Exclude invalid responses
        ) AS Percentage
    FROM Answer
    WHERE QuestionID IN 
        (SELECT QuestionID FROM Question 
         WHERE QuestionText LIKE 'If you live in the United States, which state or territory do you live in%')
    AND AnswerText NOT IN ('-1', 'Unknown', 'N/A', 'None')  -- Exclude invalid responses
    GROUP BY State
    ORDER BY Percentage DESC;
    """

    with sqlite3.connect(db_path) as conn:
        state_df = pd.read_sql_query(sql_query, conn)

    return state_df


def plot_state_distribution(db_path, shapefile_path):
    """
    Plots a choropleth map showing the geographic distribution of survey respondents by state.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - shapefile_path (str): Path to the state-level shapefile (.shp).

    Returns:
    - A choropleth map displaying the number of survey respondents per state.
    """
    # Retrieve state-level survey data
    state_data = get_state_data(db_path)

    # Load the state map from the provided shapefile path
    states = gpd.read_file(shapefile_path)

    # Standardize 'DC' to 'District of Columbia'
    state_data["state"] = state_data["state"].replace({"dc": "district of columbia"})

    # Convert state names to lowercase & remove extra spaces
    state_data["state"] = state_data["state"].str.lower().str.strip()
    states["name"] = states["name"].str.lower().str.strip()

    # Merge survey data with shapefile using cleaned names
    states = states.merge(state_data, how="left", left_on="name", right_on="state")

    # Replace NaN values (states with no respondents) with 0
    states["respondent_count"] = states["respondent_count"].fillna(0)

    # Get the "rocket" colormap from Seaborn
    rocket_palette = sns.color_palette("rocket_r", as_cmap=True)

    # Plot the choropleth map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    states.boundary.plot(ax=ax, linewidth=1, color="black")  # State borders
    states.plot(
        column="respondent_count",
        cmap=rocket_palette,  # Apply the "rocket" color palette
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax,
    )

    # Customize the plot
    plt.title("State-Level Geographic Distribution of Survey Respondents", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(False)
    plt.show()


def countries_distribution_percentages(db_path):
    """
    Retrieves country-wise respondent percentages from SQLite with country name cleaning
    inside SQL, ensuring duplicate country names are merged.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A DataFrame with cleaned country names and response percentages.
    """
    sql_query = """
    SELECT
    CASE
        WHEN UPPER(TRIM(AnswerText)) IN ('UNITED STATES OF AMERICA', 'USA', 'U.S.', 'U.S.A.') THEN 'United States'
        WHEN UPPER(TRIM(AnswerText)) IN ('UK', 'ENGLAND', 'UNITED KINGDOM') THEN 'United Kingdom'
        WHEN UPPER(TRIM(AnswerText)) = 'DEUTSCHLAND' THEN 'Germany'
        WHEN UPPER(TRIM(AnswerText)) = 'CZECH REPUBLIC' THEN 'Czechia'
        WHEN UPPER(TRIM(AnswerText)) IN ('SOUTH KOREA', 'KOREA, REPUBLIC OF') THEN 'Korea, Republic Of'
        ELSE AnswerText
    END AS Country,
    SUM(Count) * 100.0 / (SELECT SUM(Count) FROM (
        SELECT COUNT(*) AS Count FROM Answer
        WHERE QuestionID IN
            (SELECT QuestionID FROM Question
             WHERE QuestionText LIKE 'What country do you live in%') -- Use LIKE with %
    )) AS Percentage
FROM (
    SELECT AnswerText, COUNT(*) AS Count FROM Answer
    WHERE QuestionID IN
        (SELECT QuestionID FROM Question
         WHERE QuestionText LIKE 'What country do you live in%') -- Use LIKE with %
    GROUP BY AnswerText
) AS SubQuery
GROUP BY Country
ORDER BY Percentage DESC;
    """

    with sqlite3.connect(db_path) as conn:
        country_df = pd.read_sql_query(sql_query, conn)

    return country_df


def plot_cleaned_state_distribution_horizontal_bar_chart(db_path):
    """
    Retrieves, cleans, and visualizes U.S. state-wise respondent percentages directly from SQLite.
    Cleans state names, merges duplicates, and removes invalid responses efficiently.
    Also generates a horizontal bar chart for better state comparison.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A DataFrame with cleaned U.S. state names and response percentages.
    - Displays a horizontal bar chart with the cleaned data.
    """
    sql_query = """
    SELECT 
        CASE
            WHEN UPPER(TRIM(AnswerText)) IN ('CA', 'CALIF.') THEN 'California'
            WHEN UPPER(TRIM(AnswerText)) IN ('NY', 'N.Y.') THEN 'New York'
            WHEN UPPER(TRIM(AnswerText)) IN ('TX', 'TEX.') THEN 'Texas'
            WHEN UPPER(TRIM(AnswerText)) IN ('FL', 'FLA.') THEN 'Florida'
            WHEN UPPER(TRIM(AnswerText)) IN ('IL', 'ILL.') THEN 'Illinois'
            ELSE UPPER(TRIM(AnswerText)) -- Keeps other states as they are
        END AS State, 
        COUNT(*) * 100.0 / (
            SELECT COUNT(*) FROM Answer 
            WHERE QuestionID IN 
            (SELECT QuestionID FROM Question 
             WHERE QuestionText LIKE 'If you live in the United States, which state or territory do you live in%')
            AND AnswerText NOT IN ('-1', 'Unknown', 'N/A', 'None')  -- Exclude invalid responses
        ) AS Percentage
    FROM Answer
    WHERE QuestionID IN 
        (SELECT QuestionID FROM Question 
         WHERE QuestionText LIKE 'If you live in the United States, which state or territory do you live in%')
    AND AnswerText NOT IN ('-1', 'Unknown', 'N/A', 'None')  -- Exclude invalid responses
    GROUP BY State
    ORDER BY Percentage DESC;
    """

    with sqlite3.connect(db_path) as conn:
        state_df = pd.read_sql_query(sql_query, conn)

    # Plot a horizontal bar chart
    plt.figure(figsize=(12, 7))
    sns.barplot(
        y="State",
        x="Percentage",
        hue="State",
        data=state_df,
        palette="rocket",
        legend=False,
    )

    plt.title("U.S. State-Level Survey Response Distribution", fontsize=14)
    plt.xlabel("Percentage of Respondents (%)", fontsize=12)
    plt.ylabel("State", fontsize=12)

    # Reduce font size of state labels
    plt.yticks(fontsize=9)

    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Show the graph
    plt.show()

    return state_df


def get_mental_health_conditions(db_path):
    """
    Extracts self-reported diagnosed mental health conditions from QuestionID 115.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A DataFrame with diagnosed mental health conditions and their frequency.
    """
    sql_query = """
    SELECT LOWER(Answer.AnswerText) AS condition
    FROM Answer
    WHERE Answer.QuestionID = 115
    """

    with sqlite3.connect(db_path) as conn:
        conditions_df = pd.read_sql_query(sql_query, conn)

    return conditions_df


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


def categorize_condition(condition, condition_mapping):
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
    return "Other"  # Default category for unclassified conditions


def plot_mental_health_conditions_with_CI(
    db_path,
    condition_mapping,
    question_id=115,
    table_name="Answer",
    condition_col="AnswerText",
):
    """
    Extracts, cleans, categorizes, and visualizes diagnosed mental health conditions
    with percentage rates and 95% confidence intervals using the Wilson score interval.

    Parameters:
    - db_path (str): Path to the SQLite database file.
    - condition_mapping (dict): Dictionary mapping categories to keywords.
    - question_id (int): The QuestionID to filter by. Default: 115
    - table_name (str): The name of the table. Default: "Answer"
    - condition_col (str): Name of the column containing condition data. Default: "AnswerText"

    Returns:
    - A horizontal bar chart and a DataFrame with prevalence rates and confidence intervals.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            sql_query = f"""
            SELECT LOWER({condition_col}) AS condition
            FROM {table_name}
            WHERE QuestionID = ?
            """
            conditions_df = pd.read_sql_query(sql_query, conn, params=(question_id,))
            if conditions_df.empty:
                print(f"No data for QuestionID {question_id}.")
                return pd.DataFrame()
    except sqlite3.Error as e:
        print(f"Database Error: {e}")
        return pd.DataFrame()

    # **Remove invalid responses (-1, empty, "none", "n/a")**
    conditions_df = conditions_df[
        ~conditions_df["condition"].isin(["-1", "", "none", "n/a"])
    ]
    if conditions_df.empty:
        print("No valid data after cleaning.")
        return pd.DataFrame()

    # **Apply Categorization Using External Function**
    conditions_df["Category"] = conditions_df["condition"].apply(
        lambda x: categorize_condition(x, condition_mapping)
    )

    # **Group and Calculate Statistics**
    condition_counts = (
        conditions_df.groupby("Category").size().reset_index(name="Count")
    )
    total_responses = condition_counts["Count"].sum()  # Total respondents

    # **Compute prevalence rate (%)**
    condition_counts["Prevalence (%)"] = (
        condition_counts["Count"] / total_responses
    ) * 100

    # **Compute 95% confidence interval using Wilson Score Interval**
    ci_lower, ci_upper = smp.proportion_confint(
        count=condition_counts["Count"],
        nobs=total_responses,
        alpha=0.05,
        method="wilson",
    )

    # Convert CI to percentages
    condition_counts["CI_lower"] = ci_lower * 100
    condition_counts["CI_upper"] = ci_upper * 100

    # Compute CI width for error bars
    condition_counts["CI_width"] = (
        condition_counts["CI_upper"] - condition_counts["CI_lower"]
    ) / 2

    # **Filter out zero-count categories (optional)**
    condition_counts = condition_counts[condition_counts["Count"] > 0]

    # **Plotting with Matplotlib**
    if not condition_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Reverse order for better readability
        condition_counts = condition_counts.sort_values(
            "Prevalence (%)", ascending=True
        )

        y_labels = condition_counts["Category"]
        x_values = condition_counts["Prevalence (%)"]
        xerr_values = condition_counts["CI_width"]

        ax.barh(
            y=y_labels,
            width=x_values,
            xerr=xerr_values,
            capsize=5,
            color=plt.get_cmap("rocket")(np.linspace(0.9, 0.3, len(y_labels))),
        )

        ax.set_xlabel("Prevalence Rate (%)", fontsize=12)
        ax.set_ylabel("Mental Health Condition", fontsize=12)
        ax.set_title(
            "Prevalence of Diagnosed Mental Health Conditions with Confidence Intervals",
            fontsize=14,
        )

        # Remove unnecessary borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.show()
    else:
        print("No data to plot after filtering.")
        return pd.DataFrame()

    return condition_counts.rename(columns={"Category": "Mental Health Condition"})


def plot_mental_health_conditions_by_gender_raw(merged_df, condition_mapping):
    """
    Extracts, cleans, categorizes, and visualizes diagnosed mental health conditions **without** normalization.

    Parameters:
    - merged_df (pd.DataFrame): The cleaned dataset with grouped responses.
    - condition_mapping (dict): Dictionary mapping mental health conditions to keywords.

    Returns:
    - A horizontal stacked bar chart showing **raw counts** of mental health conditions by gender (using the 'rocket' colormap).
    - A DataFrame with raw condition counts per gender.
    """
    print(f"merged_df shape before filtering: {merged_df.shape}")

    # ✅ **Filter relevant responses: conditions (QuestionID 115) and gender ('What is your gender?')**
    mh_df = merged_df[
        merged_df["questiontext"].str.contains("condition", case=False, na=False)
    ]
    gender_df = merged_df[
        merged_df["questiontext"].str.contains("gender", case=False, na=False)
    ]

    # ✅ **Ensure valid responses**
    mh_df = mh_df[~mh_df["AnswerText"].isin(["-1", "", "none", "n/a"])]
    gender_df = gender_df[~gender_df["AnswerText"].isin(["-1", "", "none", "n/a"])]

    print(f"mh_df shape after filtering by QuestionID 115: {mh_df.shape}")
    print(
        f"gender_df shape after filtering ('What is your gender?'): {gender_df.shape}"
    )

    # ✅ **Merge cleaned mental health conditions with gender (on SurveyID)**
    merged_conditions = mh_df.merge(
        gender_df, on="SurveyID", how="inner", suffixes=("_condition", "_gender")
    )

    print(f"merged_conditions shape after merge: {merged_conditions.shape}")

    # ✅ **Normalize condition text to lowercase for matching**
    merged_conditions["AnswerText_condition"] = merged_conditions[
        "AnswerText_condition"
    ].str.lower()

    # ✅ **Categorize conditions**
    merged_conditions["Category"] = merged_conditions["AnswerText_condition"].apply(
        lambda x: categorize_condition(x, condition_mapping)
    )

    # ✅ **Apply Gender Mapping (normalize text and map to categories)**
    merged_conditions["Gender"] = (
        merged_conditions["AnswerText_gender"]
        .str.strip()
        .str.lower()
        .map({k.lower(): v for k, v in gender_mapping.items()})
        .fillna("Other")
    )

    # ✅ **Count unique respondents for each mental health condition per gender**
    condition_counts = (
        merged_conditions.groupby(["Category", "Gender"])["SurveyID"]
        .nunique()
        .unstack(fill_value=0)
    )

    # **Ensure all values are numeric**
    condition_counts = condition_counts.astype(int)

    # **Debugging Output**
    print(condition_counts.dtypes)  # Ensure values are numeric
    print(condition_counts)  # Raw counts of conditions per gender

    # **Plot: Stacked Bar Chart (Raw Counts)**
    fig, ax = plt.subplots(figsize=(12, 6))
    condition_counts.plot(kind="barh", stacked=True, ax=ax, colormap="rocket")

    ax.set_xlabel("Number of Respondents")
    ax.set_ylabel("Mental Health Condition")
    ax.set_title("Mental Health Conditions by Gender (Raw Counts)")
    ax.legend(title="Gender")

    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

    return condition_counts
