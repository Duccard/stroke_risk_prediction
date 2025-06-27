import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_feature_across_genres(df, genres, feature, title):
    """
    Compares a given feature across selected genres.

    Parameters:
        df (DataFrame): The dataset containing the data.
        genres (list): List of genres to compare.
        feature (str): The feature column to analyze (e.g., 'danceability', 'loudness').
        title (str): The title for the graph and the analysis.

    Returns:
        None: Displays the graph and prints the summary.
    """
    # Filter only the selected genres
    filtered_df = df[df["genre"].isin(genres)]
    
    if filtered_df.empty:
        print(f"No data found for the genres: {genres}")
        return

    # Group by genre and calculate feature statistics
    feature_stats = (
        filtered_df.groupby("genre")[feature]
        .agg(mean="mean", median="median", min="min", max="max")
        .reset_index()
    )

    # Determine the genre with the highest mean and median
    most_mean = feature_stats.loc[feature_stats["mean"].idxmax()]
    most_median = feature_stats.loc[feature_stats["median"].idxmax()]

    # Plot the results in a grouped bar chart
    plt.figure(figsize=(12, 6))
    x = feature_stats["genre"]
    plt.bar(x, feature_stats["mean"], width=0.4, label="Mean", align="center", color="skyblue")
    plt.bar(x, feature_stats["median"], width=0.4, label="Median", align="edge", color="orange")

    # Customize the chart
    plt.title(f"{title} Across Genres", fontsize=14)
    plt.ylabel(f"{feature.capitalize()} Score", fontsize=12)
    plt.xlabel("Genre", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the summary
    print(f"\n{title} Summary:")
    print(f"  - The genre with the highest mean {feature} is '{most_mean['genre']}' with {most_mean['mean']:.2f}.")
    print(f"  - The genre with the highest median {feature} is '{most_median['genre']}' with {most_median['median']:.2f}.")
    print("  - Comparison for all genres:")
    for _, row in feature_stats.iterrows():
        print(
            f"    * {row['genre']}: mean = {row['mean']:.2f}, "
            f"median = {row['median']:.2f}, range = {row['min']:.2f} to {row['max']:.2f}."
        )


def analyze_correlations(correlation_matrix, threshold, description, condition=">"):
    """
    Analyzes correlations based on a threshold and condition, eliminating duplicates.

    Parameters:
        correlation_matrix (DataFrame): The correlation matrix.
        threshold (float or tuple): The correlation threshold. If condition is "range", this should be a tuple (lower, upper).
        description (str): Description of the type of correlation (e.g., "Strong Positive").
        condition (str): The condition for filtering correlations ("<", ">", "range").

    Returns:
        None: Prints the result of the analysis.
    """
    if condition == ">":
        filtered = correlation_matrix[(correlation_matrix > threshold) & (correlation_matrix != 1.0)]
    elif condition == "<":
        filtered = correlation_matrix[(correlation_matrix < threshold)]
    elif condition == "range":
        lower, upper = threshold
        filtered = correlation_matrix[(correlation_matrix > lower) & (correlation_matrix < upper)]
    else:
        raise ValueError("Invalid condition. Use '>', '<', or 'range'.")

    # Extract pairs of correlated features, avoiding duplicates
    pairs = []
    seen = set()
    for row in filtered.index:
        for col in filtered.columns:
            if row != col and not pd.isnull(filtered.loc[row, col]):
                sorted_pair = tuple(sorted([row, col]))  # Ensure consistent order
                if sorted_pair not in seen:
                    pairs.append((row, col, filtered.loc[row, col]))
                    seen.add(sorted_pair)

    # Print results
    print(f"\nConclusion: {description}")
    if len(pairs) > 10:  # If too many pairs, summarize compactly
        pair_list = ", ".join([f"{row}-{col}" for row, col, _ in pairs[:10]])
        print(f"  Too many correlations to display ({len(pairs)} found). Examples: {pair_list}...")
    elif pairs:
        compact_output = ", ".join([f"{row}-{col} (r={value:.2f})" for row, col, value in pairs])
        print(f"  {compact_output}")
    else:
        print("  No correlations found.")
