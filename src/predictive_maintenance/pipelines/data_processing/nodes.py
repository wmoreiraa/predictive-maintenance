"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""
import pandas as pd


def sanitize_col_names(raw: pd.DataFrame) -> pd.DataFrame:

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
    return raw_sanitized.drop(["udi", "product_id", "target"], axis=1)
