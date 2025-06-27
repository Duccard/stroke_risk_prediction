import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def convert_enrollment(value: str | int | float) -> float:
    """
    Convert enrollment numbers from formatted strings (e.g., '10k', '2.5M') to numeric values.

    Args:
        value (str | int | float): The enrollment value in raw format.

    Returns:
        float: The converted numeric enrollment value.
    """
    value_str = str(value).lower().replace(",", "")
    if "k" in value_str:
        return float(value_str.replace("k", "")) * 1_000
    elif "m" in value_str:
        return float(value_str.replace("m", "")) * 1_000_000
    else:
        return pd.to_numeric(value_str, errors="coerce")


def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    """
    Detects outliers in numerical columns of a DataFrame using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The DataFrame containing numerical data.

    Returns:
        pd.Series: A series with column names as index and the count of outliers in each column.
    """
    numerical_cols = df.select_dtypes(include=["number"])
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = numerical_cols.quantile(0.25)
    Q3 = numerical_cols.quantile(0.75)
    # Interquartile range
    IQR = Q3 - Q1
    # Define outliers as values outside the 1.5*IQR range
    outlier_mask = (numerical_cols < (Q1 - 1.5 * IQR)) | (
        numerical_cols > (Q3 + 1.5 * IQR)
    )

    outlier_counts = outlier_mask.sum()

    return outlier_counts
