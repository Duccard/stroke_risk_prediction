import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def convert_enrollment(value: str | int | float) -> float:
    """Converts enrollment strings to numeric values."""
    value_str = str(value).lower().replace(",", "")
    if "k" in value_str:
        return float(value_str.replace("k", "")) * 1_000
    elif "m" in value_str:
        return float(value_str.replace("m", "")) * 1_000_000
    else:
        return pd.to_numeric(value_str, errors="coerce")


def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    """Detects outliers using IQR method."""
    numerical_cols = df.select_dtypes(include=["number"])
    Q1 = numerical_cols.quantile(0.25)
    Q3 = numerical_cols.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (numerical_cols < (Q1 - 1.5 * IQR)) | (
        numerical_cols > (Q3 + 1.5 * IQR)
    )
    return outlier_mask.sum()


def color_threshold(value, thresholds):
    """Applies color based on value and thresholds."""
    if value >= thresholds["high"]:
        return "red"
    elif value >= thresholds["mid_lower"] and value <= thresholds["mid_upper"]:
        return "blue"
    else:
        return "gray"
