import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import geopandas as gpd
import statsmodels.stats.proportion as smp
from scipy.stats import pointbiserialr, chi2_contingency
from phik import phik_matrix  # Importing Phi_K correlation


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
    "Trans-female": "Other",
    "Trans male": "Other",
    "Non-binary": "Other",
    "Genderqueer": "Other",
    "Agender": "Other",
    "Genderfluid": "Other",
    "Other": "Other",
    "Male-ish": "Other",
    "Something Kinda Male?": "Other",
    "Queer/She/They": "Other",
    "Nah": "Other",
    "All": "Other",
    "Enby": "Other",
    "Fluid": "Other",
    "Androgyne": "Other",
    "Guy (-Ish) ^_^": "Other",
    "Male Leaning Androgynous": "Other",
    "Trans Woman": "Other",
    "Neuter": "Other",
    "Female (Trans)": "Other",
    "Queer": "Other",
    "A Little About You": "Other",
    "P": "Other",
    "Ostensibly Male, Unsure What That Really Means": "Other",
    "Bigender": "Other",
    "Female Assigned At Birth": "Other",
    "Fm": "Other",
    "Transitioned, M2F": "Other",
    "Genderfluid (Born Female)": "Other",
    "Other/Transfeminine": "Other",
    "Female Or Multi-Gender Femme": "Other",
    "Androgynous": "Other",
    "Male 9:1 Female, Roughly": "Other",
    "-1": "Other",
    "Nb Masculine": "Other",
    "None Of Your Business": "Other",
    "Human": "Other",
    "Genderqueer Woman": "Other",
    "Mtf": "Other",
    "Male/Genderqueer": "Other",
    "Nonbinary": "Other",
    "Unicorn": "Other",
    "Male (Trans, Ftm)": "Other",
    "Genderflux Demi-Girl": "Other",
    "Female-Bodied; No Feelings About Gender": "Other",
    "Afab": "Other",
    "Transgender Woman": "Other",
    "Male/Androgynous": "Other",
    "Uhhhhhhhhh Fem Genderqueer?": "Other",
    "God King Of The Valajar": "Other",
    "Agender/Genderfluid": "Other",
    "Sometimes": "Other",
    "Woman-Identified": "Other",
    "Contextual": "Other",
    "Non Binary": "Other",
    "Genderqueer Demigirl": "Other",
    "Genderqueer/Non-Binary": "Other",
    "Female-Ish": "Other",
    "\\-": "Other",
    "Transfeminine": "Other",
    "None": "Other",
    "Ostensibly Male": "Other",
    "Male (Or Female, Or Both)": "Other",
    "Trans Man": "Other",
    "Transgender": "Other",
    "Female/Gender Non-Binary.": "Other",
    "Demiguy": "Other",
    "Trans Female": "Other",
    "She/Her/They/Them": "Other",
    "Swm": "Other",
    "Nb": "Other",
    "Nonbinary/Femme": "Other",
    "Gender Non-Conforming Woman": "Other",
    "Masculine": "Other",
    "Cishet Male": "Male",
    "Female-Identified": "Female",
    "Questioning": "Other",
    "I Have A Penis": "Male",
    "Rr": "Other",
    "Agender Trans Woman": "Other",
    "Femmina": "Female",
    "43": "Other",
    "Masculino": "Male",
    "I Am A Wookie": "Other",
    "Trans Non-Binary/Genderfluid": "Other",
    "Non-Binary And Gender Fluid": "Other",
    "Not Provided": "Other",
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


db_path = "mental_health.sqlite"


def generate_age_conditions_contingency_heatmap(db_path, merged_df):
    """
    Extracts, cleans, and generates a contingency table between age groups and diagnosed mental health conditions.
    Also visualizes the contingency table as a heatmap.

    Parameters:
    db_path (str): Path to the SQLite database.
    merged_df (pd.DataFrame): Pre-cleaned DataFrame containing survey responses.

    Returns:
    - A heatmap showing the distribution of mental health conditions across age groups.
    """
    # Extract and clean age data
    merged_df.loc[merged_df["questiontext"] == "Age", "Age"] = pd.to_numeric(
        merged_df.loc[merged_df["questiontext"] == "Age", "AnswerText"], errors="coerce"
    )

    if not merged_df["Age"].dropna().empty:
        merged_df.loc[merged_df["questiontext"] == "Age", "Age Group"] = pd.cut(
            merged_df["Age"],
            bins=[0, 19, 29, 39, 49, 59, 100],
            labels=["<20", "20-29", "30-39", "40-49", "50-59", "60+"],
            right=False,
        )
    else:
        merged_df.loc[merged_df["questiontext"] == "Age", "Age Group"] = "Not Provided"

    # Extract relevant condition data
    conditions_df = merged_df[
        merged_df["questiontext"].str.contains(
            "If yes, what condition", case=False, na=False
        )
    ]
    conditions_df = conditions_df.rename(columns={"AnswerText": "condition"})

    # Remove invalid entries (-1, empty, or unclear responses)
    conditions_df = conditions_df[
        ~conditions_df["condition"].isin(["-1", "", "none", "n/a"])
    ]

    # Apply categorization
    def categorize_condition(condition):
        for category, keywords in condition_mapping.items():
            if any(keyword.lower() in condition.lower() for keyword in keywords):
                return category
        return "Other"  # Default category for unclassified conditions

    conditions_df["Category"] = conditions_df["condition"].apply(categorize_condition)

    # Merge age groups with conditions
    df = merged_df[merged_df["questiontext"] == "Age"][["SurveyID", "Age Group"]].merge(
        conditions_df[["SurveyID", "Category"]], on="SurveyID", how="inner"
    )

    # Create a contingency table (cross-tabulation)
    contingency_table = pd.crosstab(df["Age Group"], df["Category"])

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap="coolwarm", fmt="d", linewidths=0.5)
    plt.title("Contingency Heatmap of Mental Health Conditions Across Age Groups")
    plt.xlabel("Mental Health Conditions")
    plt.ylabel("Age Groups")
    plt.show()

    return contingency_table


def rank_top_conditions_sql(db_path):
    """
    Retrieves the top 5 countries by respondent count and ranks the top 3 conditions per country.
    """
    query = """
    WITH TopCountries AS (
        SELECT AnswerText AS country, COUNT(*) AS total_respondents
        FROM Answer
        JOIN Question ON Answer.QuestionID = Question.QuestionID
        WHERE QuestionText LIKE 'What country do you live in%'
        GROUP BY AnswerText
        ORDER BY total_respondents DESC
        LIMIT 5
    ),
    CategorizedConditions AS (
        SELECT A.AnswerText AS country, 
               CASE
                   WHEN B.AnswerText LIKE '%depression%' THEN 'Depression'
                   WHEN B.AnswerText LIKE '%anxiety%' THEN 'Anxiety'
                   WHEN B.AnswerText LIKE '%ptsd%' THEN 'PTSD'
                   WHEN B.AnswerText LIKE '%ocd%' THEN 'OCD'
                   WHEN B.AnswerText LIKE '%stress%' THEN 'Stress Disorder'
                   WHEN B.AnswerText LIKE '%eating%' THEN 'Eating Disorder'
                   ELSE 'Other'
               END AS Category,
               COUNT(*) AS count
        FROM Answer A
        JOIN Question Q ON A.QuestionID = Q.QuestionID
        LEFT JOIN Answer B ON A.SurveyID = B.SurveyID
        LEFT JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
        WHERE Q.QuestionText LIKE 'What country do you live in%'
        AND Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
        GROUP BY A.AnswerText, Category
    ),
    RankedConditions AS (
        SELECT C.country, C.Category, C.count,
               RANK() OVER (PARTITION BY C.country ORDER BY C.count DESC) AS rank
        FROM CategorizedConditions C
        JOIN TopCountries T ON C.country = T.country
    )
    SELECT country, Category, count
    FROM RankedConditions
    WHERE rank <= 3
    ORDER BY country, rank;
    """

    with sqlite3.connect(db_path) as conn:
        result_df = pd.read_sql(query, conn)

    print(result_df)

    with sqlite3.connect(db_path) as conn:
        result_df = pd.read_sql(query, conn)

    print(result_df)


def rank_top_20_categorized_conditions(db_path):
    """
    Retrieves and ranks the top 20 mental health conditions by count across all countries,
    categorizing them into 7 major groups.
    """
    query = """
    WITH ConditionCounts AS (
        SELECT 
            A.AnswerText AS country,
            B.AnswerText AS raw_condition,
            COUNT(*) AS count
        FROM Answer A
        JOIN Question Q ON A.QuestionID = Q.QuestionID
        JOIN Answer B ON A.SurveyID = B.SurveyID
        JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
        WHERE Q.QuestionText LIKE 'What country do you live in%'
        AND Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
        AND B.AnswerText NOT IN ('-1', '', 'none', 'n/a')  -- Exclude invalid entries
        AND SUBSTR(A.SurveyID, 1, 4) = '2016'
        GROUP BY A.AnswerText, B.AnswerText
    ),
    CategorizedConditions AS (
        SELECT 
            country,
            count,
            CASE 
                WHEN raw_condition LIKE '%depression%' OR raw_condition LIKE '%bipolar%' THEN 'Depression'
                WHEN raw_condition LIKE '%anxiety%' OR raw_condition LIKE '%panic%' THEN 'Anxiety'
                WHEN raw_condition LIKE '%ptsd%' OR raw_condition LIKE '%post-traumatic%' THEN 'PTSD'
                WHEN raw_condition LIKE '%ocd%' OR raw_condition LIKE '%obsessive%' THEN 'OCD'
                WHEN raw_condition LIKE '%stress%' THEN 'Stress Disorder'
                WHEN raw_condition LIKE '%eating%' OR raw_condition LIKE '%anorexia%' OR raw_condition LIKE '%bulimia%' THEN 'Eating Disorder'
                ELSE 'Other'
            END AS category
        FROM ConditionCounts
    ),
    RankedConditions AS (
        SELECT 
            country,
            category,
            SUM(count) AS total_count,
            RANK() OVER (ORDER BY SUM(count) DESC) AS rank
        FROM CategorizedConditions
        GROUP BY country, category
    )
    SELECT country, category, total_count, rank
    FROM RankedConditions
    WHERE rank <= 20
    ORDER BY rank;
    """

    with sqlite3.connect(db_path) as conn:
        result_df = pd.read_sql(query, conn)

    print(result_df)


def rank_top_20_conditions_by_state(db_path):
    """
    Retrieves and ranks the top 20 most reported mental health conditions in U.S. states,
    categorizing them into 7 major groups.

    Returns:
    - A DataFrame with Rank, State, Category, and Count.
    """
    with sqlite3.connect(db_path) as conn:
        query = """
        WITH ConditionCounts AS (
            SELECT 
                A.AnswerText AS state, 
                B.AnswerText AS condition, 
                COUNT(*) AS count
            FROM Answer A
            JOIN Question Q ON A.QuestionID = Q.QuestionID
            LEFT JOIN Answer B ON A.SurveyID = B.SurveyID
            LEFT JOIN Question Q2 ON B.QuestionID = Q2.QuestionID
            WHERE 
                Q.QuestionText LIKE 'If you live in the United States, which state or territory do you live in%'
                AND Q2.QuestionText LIKE 'If yes, what condition(s) have you been diagnosed with%'
                AND B.AnswerText NOT IN ('-1', '', 'none', 'n/a')  -- Remove invalid responses
                AND A.AnswerText NOT IN ('-1', '', 'none', 'n/a')  -- Remove invalid states
            GROUP BY A.AnswerText, B.AnswerText
        ),
        RankedConditions AS (
            SELECT 
                state, 
                condition, 
                count,
                RANK() OVER (ORDER BY count DESC) AS rank
            FROM ConditionCounts
        )
        SELECT state, condition, count, rank 
        FROM RankedConditions
        WHERE rank <= 20
        ORDER BY count DESC;
        """

        state_data = pd.read_sql(query, conn)

    # **Apply Categorization to Group Conditions into 7 Categories**
    def categorize_condition(condition):
        condition_mapping = {
            "Depression": ["depression", "bipolar"],
            "Anxiety": ["anxiety", "panic disorder"],
            "PTSD": ["ptsd", "post-traumatic stress"],
            "OCD": ["ocd", "obsessive-compulsive"],
            "Stress Disorder": ["stress"],
            "Eating Disorder": ["anorexia", "bulimia", "binge eating"],
            "Other": ["substance use", "addictive disorder", "personality disorder"],
        }

        for category, keywords in condition_mapping.items():
            if any(keyword in condition.lower() for keyword in keywords):
                return category
        return "Other"

    # **Ensure Correct Categorization**
    state_data["Category"] = state_data["condition"].apply(categorize_condition)

    # **Sort by Rank and Reset Index**
    state_data = (
        state_data[["rank", "state", "Category", "count"]]
        .sort_values(["rank"])
        .reset_index(drop=True)
    )

    return state_data.iloc[:, 1:].reset_index(drop=True)


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

    # Convert year to numeric format
    year_df["Year"] = pd.to_numeric(year_df["Year"], errors="coerce")

    # Plot the bar chart with hue assigned and legend disabled
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Year", y="AnswerCount", hue="Year", data=year_df, palette="rocket_r")
    plt.legend([], [], frameon=False)  # Disable legend manually

    plt.title("Number of Answers Recorded Per Year")
    plt.xlabel("Year")
    plt.ylabel("Count of Answers")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

    return year_df


def unique_questions_per_year(db_path):
    """
    Retrieves the number of unique questions asked in each year from the SQLite database
    and visualizes it using a bar plot.

    Parameters:
    - db_path (str): Path to the SQLite database file.

    Returns:
    - A bar chart displaying the count of unique questions per year.
    - A DataFrame with the retrieved data.
    """
    sql_query = """
    SELECT SUBSTR(SurveyID, 1, 4) AS Year, COUNT(DISTINCT QuestionID) AS UniqueQuestions
    FROM Answer
    GROUP BY Year
    ORDER BY Year;
    """

    with sqlite3.connect(db_path) as conn:
        unique_questions_df = pd.read_sql_query(sql_query, conn)

    # Convert year to numeric format
    unique_questions_df["Year"] = pd.to_numeric(
        unique_questions_df["Year"], errors="coerce"
    )

    # Plot the bar chart
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="Year",
        y="UniqueQuestions",
        hue="Year",
        data=unique_questions_df,
        palette="rocket",
    )
    plt.legend([], [], frameon=False)  # Disable legend

    plt.title("Number of Unique Questions Per Year")
    plt.xlabel("Year")
    plt.ylabel("Unique Questions")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

    return unique_questions_df


def plot_unique_respondents_by_year(db_path):
    """
    Queries unique respondents by year from the database and plots them.

    Parameters:
    - db_path (str): Path to the SQLite database.

    Returns:
    - DataFrame: Unique respondents by year.
    """
    # Query to get unique UserIDs per year
    query = """
    SELECT SUBSTR(SurveyID, 1, 4) AS Year, COUNT(DISTINCT UserID) AS UniqueRespondents
    FROM Answer
    GROUP BY Year
    ORDER BY Year;
    """

    # Execute query and load into DataFrame
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    # Plot
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


def count_men_conditions_2016(db_path, condition_mapping):
    """
    Counts unique male respondents in 2016 by their diagnosed mental health condition categories.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - condition_mapping (dict): External mapping of condition categories to keywords.

    Returns:
    - dict: Counts of unique male respondents per mental health condition category.
    """

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

    # Initialize result containers
    condition_counts = {key: set() for key in condition_mapping.keys()}
    condition_counts["Other"] = set()

    # Categorize each respondent
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


def count_women_conditions_2016(db_path, condition_mapping):
    """
    Counts unique female respondents in 2016 by their diagnosed mental health condition categories.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - condition_mapping (dict): External mapping of condition categories to keywords.

    Returns:
    - dict: Counts of unique female respondents per mental health condition category.
    """

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

    # Initialize result containers
    condition_counts = {key: set() for key in condition_mapping.keys()}
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


def count_other_gender_conditions_2016(db_path, condition_mapping):
    """
    Counts unique other-gender respondents in 2016 by their diagnosed mental health condition categories.

    Parameters:
    - db_path (str): Path to the SQLite database.
    - condition_mapping (dict): External mapping of condition categories to keywords.

    Returns:
    - dict: Counts of unique other-gender respondents per mental health condition category.
    """

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

    # Initialize result containers
    condition_counts = {key: set() for key in condition_mapping.keys()}
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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_multibar_conditions_2016(
    men_conditions, women_conditions, other_gender_conditions
):
    """
    Plots a multi-bar chart comparing mental health condition counts by gender group in 2016.

    Parameters:
    - men_conditions (dict): Condition counts for men.
    - women_conditions (dict): Condition counts for women.
    - other_gender_conditions (dict): Condition counts for other genders.

    Returns:
    - None: Displays the plot.
    """

    # Convert dictionaries to a combined DataFrame
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

    # Combine all data
    combined_df = pd.concat([df_men, df_women, df_other])

    # Plot
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
    men_conditions, women_conditions, other_gender_conditions
):
    """
    Plots a multi-bar chart comparing normalized percentages of mental health condition counts
    by gender group in 2016 and prints the percentages.

    Parameters:
    - men_conditions (dict): Condition counts for men.
    - women_conditions (dict): Condition counts for women.
    - other_gender_conditions (dict): Condition counts for other genders.

    Returns:
    - None: Displays the plot and prints percentages.
    """

    # Convert dictionaries to DataFrames
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

    # Combine all data
    combined_df = pd.concat([df_men, df_women, df_other])

    # Normalize counts to percentages within each gender group
    combined_df["Percentage"] = combined_df.groupby("Gender")["Count"].transform(
        lambda x: (x / x.sum()) * 100
    )

    # Plot
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

    # Print percentages
    print("\nNormalized Percentages by Condition and Gender (2016):\n")
    for gender in combined_df["Gender"].unique():
        gender_df = combined_df[combined_df["Gender"] == gender]
        print(f"{gender}:")
        for _, row in gender_df.iterrows():
            print(f"  {row['Condition']}: {row['Percentage']:.2f}%")
        print()
