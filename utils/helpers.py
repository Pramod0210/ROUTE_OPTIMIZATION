import pandas as pd

def is_text_dtype(series: pd.Series) -> bool:
    """
    Checks if a Pandas Series is of textual type.

    :param series: Pandas Series to check
    :return: True if the series is of textual type (object or string), False otherwise
    """
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
