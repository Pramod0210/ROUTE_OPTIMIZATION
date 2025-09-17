import pandas as pd
import os
import shutil


def is_text_dtype(series: pd.Series) -> bool:
    """
    Checks if a Pandas Series is of textual type.

    :param series: Pandas Series to check
    :return: True if the series is of textual type (object or string), False otherwise
    """
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)



def empty_directory(directory_path):
    """
    Empties a directory by removing all its files and subdirectories.

    Args:
        directory_path (str): The path to the directory to be emptied.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    # Iterate over all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                # Remove files and symbolic links
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # Remove subdirectories and all their contents
                shutil.rmtree(item_path)
        except OSError as e:
            print(f"Error: Failed to remove {item_path}. Reason: {e}")

