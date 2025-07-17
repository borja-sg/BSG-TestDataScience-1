from pathlib import Path
import pandas as pd


def load_csv_dataset(path):
    """
    Load a CSV file as a pandas DataFrame. Raises an error if the file does not exist.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    # Convert to path
    path = Path(path)
    # Check if the path exists and is a file
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    # Check if the path is a file
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file (may be a directory).")
    # Return the DataFrame
    return pd.read_csv(path)


def save_to_hdf(df, path, key, mode="a"):
    """
    Save a DataFrame to an HDF5 file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination HDF5 file path.
    key : str
        Key under which the DataFrame is stored.
    mode : str, optional
        Mode to open the file ('w' for write, 'a' for append), by default 'a'.
    """
    # Convert to path
    path = Path(path)
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Raise ValueError if the mode is not 'w' or 'a'
    if mode not in ["w", "a"]:
        raise ValueError("Mode must be either 'a' or 'w'")
    # Ensure the key is valid
    if not isinstance(key, str):
        raise TypeError("Key must be a string")
    # Save it
    df.to_hdf(path, key=key, mode=mode, format="table")
