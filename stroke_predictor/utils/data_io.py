from pathlib import Path
import pandas as pd
from typing import Literal, Tuple, Any
import yaml
import os
import joblib
import mlflow


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


def load_column_names(path: Path, key: str) -> list:
    """
    Load column names from an HDF5 file.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file.
    key : str
        Key under which the DataFrame is stored.

    Returns
    -------
    list
        List of column names in the DataFrame.
    """

    # Check if the path exists and is a file
    if not path.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file (may be a directory).")

    # Load the DataFrame and return its columns
    df = pd.read_hdf(path, key=key)

    # Drop stroke
    if "stroke" in df.columns:
        df = df.drop("stroke", axis=1)  # Exclude stroke column if it exists
    return df.columns.tolist()


def load_config(model_path: str) -> dict:
    """Load model configuration and related resources."""
    config_path = Path(model_path) / "model_config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    variable_encoder_path = config["storage"].get("variable_encoder_path", None)
    if variable_encoder_path and os.path.exists(variable_encoder_path):
        variable_encoder = joblib.load(variable_encoder_path)
    else:
        raise FileNotFoundError(
            f"Variable encoder not found at {variable_encoder_path}"
        )

    path_data = Path(config["storage"]["data_path"])
    if not path_data.exists():
        raise FileNotFoundError(f"Data file not found at {path_data}")

    key = config["storage"]["hdf5_key_train"]
    column_names = load_column_names(path=path_data, key=key)

    return {
        "model_path": model_path,
        "variable_encoder": variable_encoder,
        "column_names": column_names,
    }


def load_model_and_config(model_path: str) -> Tuple[Any, dict, list[str], Any]:
    """
    Load the ML model, configuration file, variable encoder, and column names.

    Parameters
    ----------
    model_path : str
        Path to the MLflow model directory.

    Returns
    -------
    tuple
        A tuple containing the model, model config, column names, and variable encoder.

    Raises
    ------
    FileNotFoundError
        If any of the required files (config, encoder, data) are missing.
    """
    config_path = Path(model_path) / "model_config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    variable_encoder_path = model_config["storage"].get("variable_encoder_path")
    if not variable_encoder_path or not os.path.exists(variable_encoder_path):
        raise FileNotFoundError(
            f"Variable encoder not found at {variable_encoder_path}"
        )

    path_data = Path(model_config["storage"]["data_path"])
    if not path_data.exists():
        raise FileNotFoundError(f"Data file not found at {path_data}")

    key = model_config["storage"]["hdf5_key_train"]
    column_names = load_column_names(path=path_data, key=key)

    # Now it's safe to load the model
    model = mlflow.sklearn.load_model(model_path)
    variable_encoder = joblib.load(variable_encoder_path)

    return model, model_config, column_names, variable_encoder
