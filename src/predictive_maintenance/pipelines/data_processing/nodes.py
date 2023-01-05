""" Nodes containing the preprocessing steps used in the preprocessing pipeline:
"""
import pandas as pd


def sanitize_col_names(raw: pd.DataFrame) -> pd.DataFrame:
    """Used to clean column names

    Args:
        raw (pd.DataFrame): raw dataframe

    Returns:
        pd.DataFrame: dataframe with proper column names
    """

    names = raw.columns.to_series().replace("[ ]{1,}", "_", regex=True)
    names = names.str.lower().str.strip()
    names = (
        names.replace("[k]", "c")
        .replace(r"\[", "", regex=True)
        .replace(r"\]", "", regex=True)
    )
    raw.columns = names
    return raw


def drop_cols(raw_sanitized: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that wont be used

    Args:
        raw_sanitized (pd.DataFrame): raw data with clean col names

    Returns:
        pd.DataFrame: clean data
    """
    return raw_sanitized.drop(["udi", "product_id", "target"], axis=1)
