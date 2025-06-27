import sqlite3
import pandas as pd


def query_sqlite_to_dataframe(database_path, query):
    """
    Executes an SQL query on an SQLite database and returns the result as a pandas DataFrame.

    Args:
        database_path (str): The path to the SQLite database file.
        query (str): The SQL query to execute.

    Returns:
        pandas.DataFrame: A DataFrame containing the query results, or None if an error occurs.
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(database_path)

        # Execute the SQL query and load the result into a DataFrame
        df = pd.read_sql_query(query, conn)

        # Close the database connection
        conn.close()

        return df

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None


# Example usage (replace with your actual database path and query)
database_path = "mental_health.sqlite"  # Your database file path
query = "SELECT * FROM YourTableName;"  # Replace with your SQL query and table name

# Execute the query and load the DataFrame
result_df = query_sqlite_to_dataframe(database_path, query)

# Check if the DataFrame was loaded successfully
if result_df is not None:
    print(result_df)
    # Further processing of the DataFrame can be done here.
else:
    print("Failed to load DataFrame.")

# Example of a more complex query.
# Replace YourTableName, and column names with your actual table and column names.
query2 = """
SELECT column1, column2, AVG(column3)
FROM YourTableName
WHERE column4 > 10
GROUP BY column1, column2
ORDER BY AVG(column3) DESC;
"""

result_df2 = query_sqlite_to_dataframe(database_path, query2)

if result_df2 is not None:
    print(result_df2)
else:
    print("Failed to load DataFrame.")
