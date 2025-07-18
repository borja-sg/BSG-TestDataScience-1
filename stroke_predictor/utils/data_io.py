from pathlib import Path
import pandas as pd
from typing import Literal, Tuple


def load_csv_dataset(path: Path) -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame. Raises an error if the file does not exist.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the path is not a file (e.g., a directory).
    """

    # Check if the path exists and is a file
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file (may be a directory).")

    # Load and return the DataFrame
    return pd.read_csv(path)


def save_to_hdf(
    df: pd.DataFrame, path: Path, key: str, mode: Literal["a", "w", "r+"]
) -> None:
    """
    Save a DataFrame to an HDF5 file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Destination HDF5 file path.
    key : str
        Key under which the DataFrame is stored.
    mode : str, optional
        Mode to open the file ('w' for write, 'a' for append), by default 'a'.

    Raises
    ------
    ValueError
        If the mode is not 'w' or 'a'.
    TypeError
        If the key is not a string.
    """

    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate mode
    if mode not in ["w", "a", "r+"]:
        raise ValueError("Mode must be either 'a' or 'w', 'r+'")

    # Validate key
    if not isinstance(key, str):
        raise TypeError("Key must be a string")

    # Check if the path folder exists
    if not path.parent.exists():
        raise FileNotFoundError(f"The directory '{path.parent}' does not exist.")

    # Save the DataFrame to HDF5
    df.to_hdf(path, key=key, mode=mode, format="table")


def load_dataset(path: Path, key: str, target: str) -> Tuple:
    """
    Load a DataFrame from an HDF5 file.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file.
    key : str
        Key under which the DataFrame is stored.
    target : str
        Name of the target variable column in the DataFrame.

    Returns
    -------
    x : np.ndarray
        Features from the DataFrame.
    y : np.ndarray
        Target variable from the DataFrame."""

    # Check if the path exists and is a file
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file (may be a directory).")

    # Load and return the DataFrame from HDF5
    df = pd.read_hdf(path, key=key)

    # Check if the target column exists in the DataFrame
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    # Separate features and target variable
    x = df.drop(columns=[target]).values
    y = df[target].values
    return x, y
