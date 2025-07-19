import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from unittest.mock import patch
from sklearn.preprocessing import OrdinalEncoder
from typing import Any

from stroke_predictor.utils.data_io import (
    load_csv_dataset,
    save_to_hdf,
    load_dataset,
    load_column_names,
    load_config,
    load_model_and_config,
)


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_csv_dataset_success(temp_csv: Path) -> None:
    """Test successful loading of a valid CSV file."""
    df = load_csv_dataset(temp_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)  # Check expected dimensions


def test_load_csv_dataset_nonexistent_file() -> None:
    """Test FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_csv_dataset(Path("nonexistent_file.csv"))


def test_load_csv_dataset_not_a_file(tmp_path: Path) -> None:
    """Test ValueError if path is a directory (not a file)."""
    dir_path = tmp_path / "empty_dir"
    os.makedirs(dir_path, exist_ok=True)  # Create an empty directory
    with pytest.raises(ValueError):
        load_csv_dataset(dir_path)


# Saving dataframes to HDF5
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Generate a sample DataFrame for testing."""
    return pd.DataFrame({"id": [1, 2, 3], "value": np.random.rand(3)})


@pytest.fixture
def temp_hdf_path(tmp_path: Path) -> Path:
    """Return a temporary HDF5 file path."""
    return tmp_path / "test.h5"


def test_save_to_hdf_success(
    sample_dataframe: pd.DataFrame, temp_hdf_path: Path
) -> None:
    """Test saving a DataFrame to HDF5 with default mode ('a')."""
    key = "test_data"
    save_to_hdf(df=sample_dataframe, path=temp_hdf_path, key=key, mode="w")
    # Verify the file exists and contains the key
    assert temp_hdf_path.exists()
    with pd.HDFStore(temp_hdf_path) as store:
        assert f"/{key}" in store.keys()
        loaded_df = pd.DataFrame(store.get(key))
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)


def test_save_to_hdf_overwrite_mode(
    sample_dataframe: pd.DataFrame, temp_hdf_path: Path
) -> None:
    """Test overwriting an existing key with mode='w'."""
    key = "test_data"

    # First save (creates file)
    save_to_hdf(df=sample_dataframe, path=temp_hdf_path, key=key, mode="w")

    # New DataFrame to overwrite
    new_df = pd.DataFrame({"id": [4, 5], "value": [0.99, 0.42]})
    save_to_hdf(new_df, temp_hdf_path, key, mode="w")

    # Verify only the new data exists
    with pd.HDFStore(temp_hdf_path) as store:
        assert store.get(key).shape == new_df.shape


def test_save_to_hdf_invalid_mode(
    sample_dataframe: pd.DataFrame, temp_hdf_path: Path
) -> None:
    """Test invalid mode raises ValueError."""
    # invalid_mode = "x"  # Explicitly declare the invalid mode as a Literal
    # Use an invalid mode
    invalid_mode = "x"
    with pytest.raises(ValueError, match=r"Mode must be either 'a' or 'w', 'r\+'"):
        save_to_hdf(df=sample_dataframe, path=temp_hdf_path, key="test_data", mode=invalid_mode)  # type: ignore[arg-type]


def test_save_to_hdf_nonexistent_dir(
    sample_dataframe: pd.DataFrame, tmp_path: Path
) -> None:
    """Test saving to a non-existent directory (should create parent dirs)."""
    non_existent_path = tmp_path / "nonexistent_dir" / "test.h5"
    # Check that it gives FileNotFoundError if the directory does not exist
    assert not non_existent_path.exists()
    # Attempt to save the DataFrame
    # This should create the directory and save the file
    save_to_hdf(df=sample_dataframe, path=non_existent_path, key="test_data", mode="w")


def test_save_to_hdf_invalid_key(
    sample_dataframe: pd.DataFrame, temp_hdf_path: Path
) -> None:
    """Test invalid key type raises TypeError."""
    with pytest.raises(TypeError, match="Key must be a string"):
        save_to_hdf(df=sample_dataframe, path=temp_hdf_path, key=123, mode="w")  # type: ignore


@pytest.fixture
def temp_hdf(tmp_path: Path) -> Path:
    """Create a temporary HDF5 file for testing."""
    hdf_path = tmp_path / "test_data.h5"
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_hdf(hdf_path, key="test_key", mode="w")
    return hdf_path


def test_load_dataset_valid(temp_hdf: Path) -> None:
    """Test loading a valid dataset."""
    x, y = load_dataset(path=temp_hdf, key="test_key", target="target")
    assert isinstance(x, np.ndarray), "Features (x) are not a numpy array."
    assert isinstance(y, np.ndarray), "Target (y) is not a numpy array."
    assert x.shape == (4, 2), "Features (x) shape is incorrect."
    assert y.shape == (4,), "Target (y) shape is incorrect."
    assert y.tolist() == [0, 1, 0, 1], "Target values are incorrect."


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    """Test loading a dataset from a missing file."""
    missing_path = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError, match="The file '.*' does not exist."):
        load_dataset(path=missing_path, key="test_key", target="target")


def test_load_dataset_invalid_file(tmp_path: Path) -> None:
    """Test loading a dataset from an invalid file."""
    invalid_path = tmp_path / "invalid_dir"
    invalid_path.mkdir()  # Create a directory instead of a file
    with pytest.raises(ValueError, match="'.*' is not a file .*"):
        load_dataset(path=invalid_path, key="test_key", target="target")


def test_load_dataset_missing_key(temp_hdf: Path) -> None:
    """Test loading a dataset with a missing key."""
    # Attempt to load a key that does not exist in the HDF5 file
    missing_key = "missing_key"
    with pytest.raises(
        KeyError, match="No object named " + missing_key + " in the file"
    ):
        load_dataset(path=temp_hdf, key=missing_key, target="target")


def test_load_dataset_missing_target(temp_hdf: Path) -> None:
    """Test loading a dataset with a missing target column."""
    with pytest.raises(ValueError, match="Target column 'missing_target' not found .*"):
        load_dataset(path=temp_hdf, key="test_key", target="missing_target")


@pytest.fixture
def temp_hdf_without_target(tmp_path: Path) -> Path:
    """Create a temporary HDF5 file for testing without the 'stroke' column."""
    hdf_path = tmp_path / "test_data_no_target.h5"
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8],
        }
    )
    df.to_hdf(hdf_path, key="test_key", mode="w")
    return hdf_path


def test_load_column_names_valid(temp_hdf_without_target: Path) -> None:
    """Test loading column names from a valid HDF5 file."""
    column_names = load_column_names(path=temp_hdf_without_target, key="test_key")
    expected_columns = ["feature_1", "feature_2"]  # No 'stroke' column in this fixture
    assert (
        column_names == expected_columns
    ), "Column names do not match expected values."


def test_load_column_names_missing_file(tmp_path: Path) -> None:
    """Test loading column names from a missing file."""
    missing_path = tmp_path / "missing.h5"
    with pytest.raises(FileNotFoundError, match="The file '.*' does not exist."):
        load_column_names(path=missing_path, key="test_key")


def test_load_column_names_invalid_file(tmp_path: Path) -> None:
    """Test loading column names from an invalid file."""
    invalid_path = tmp_path / "invalid_dir"
    invalid_path.mkdir()  # Create a directory instead of a file
    with pytest.raises(ValueError, match="'.*' is not a file .*"):
        load_column_names(path=invalid_path, key="test_key")


def test_load_column_names_missing_key(temp_hdf_without_target: Path) -> None:
    """Test loading column names with a missing key."""
    with pytest.raises(KeyError, match="No object named missing_key in the file"):
        load_column_names(path=temp_hdf_without_target, key="missing_key")


@pytest.fixture
def config_file_path(tmp_path: Path) -> Path:
    """Create a temporary model_config.yml file for testing."""
    config_path = tmp_path / "model_config.yml"
    config = {
        "storage": {
            "variable_encoder_path": str(tmp_path / "encoder.pkl"),
            "data_path": str(tmp_path / "data.h5"),
            "hdf5_key_train": "test_key",
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def encoder_file_path(tmp_path: Path) -> Path:
    """Create a temporary encoder.pkl file for testing."""
    encoder_path = tmp_path / "encoder.pkl"
    encoder = OrdinalEncoder()
    encoder.fit([["A"], ["B"]])  # Fit with dummy data so it's a valid encoder
    joblib.dump(encoder, encoder_path)
    return encoder_path


@pytest.fixture
def hdf_file_path(tmp_path: Path) -> Path:
    """Create a temporary HDF5 file for testing."""
    hdf_path = tmp_path / "data.h5"
    df = pd.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]})
    df.to_hdf(hdf_path, key="test_key", mode="w")
    return hdf_path


@patch("stroke_predictor.utils.data_io.load_column_names")
def test_load_config_success(
    mock_load_column_names: Any,
    tmp_path: Path,
    config_file_path: Path,
    encoder_file_path: Path,
    hdf_file_path: Path,
) -> None:
    """Test successful loading of config and resources."""
    mock_load_column_names.return_value = ["feature_1", "feature_2"]
    model_path = str(tmp_path)
    result = load_config(model_path)
    assert "model_path" in result
    assert "variable_encoder" in result
    assert "column_names" in result
    assert result["column_names"] == ["feature_1", "feature_2"]


def test_load_config_missing_config(tmp_path: Path) -> None:
    """Test missing config file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Model configuration file not found"):
        load_config(model_path)


def test_load_config_missing_encoder(
    tmp_path: Path, config_file_path: Path, hdf_file_path: Path
) -> None:
    """Test missing encoder file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Variable encoder not found"):
        load_config(model_path)


def test_load_config_missing_data(
    tmp_path: Path, config_file_path: Path, encoder_file_path: Path
) -> None:
    """Test missing data file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_config(model_path)


@patch("stroke_predictor.utils.data_io.mlflow.sklearn.load_model")
def test_load_model_and_config_missing_config(
    mock_load_model: Any, tmp_path: Path
) -> None:
    """Test missing config file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Model configuration file not found"):
        load_model_and_config(model_path)


@patch("stroke_predictor.utils.data_io.mlflow.sklearn.load_model")
def test_load_model_and_config_missing_encoder(
    mock_load_model: Any, tmp_path: Path, config_file_path: Path, hdf_file_path: Path
) -> None:
    """Test missing encoder file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Variable encoder not found"):
        load_model_and_config(model_path)


@patch("stroke_predictor.utils.data_io.mlflow.sklearn.load_model")
def test_load_model_and_config_missing_data(
    mock_load_model: Any,
    tmp_path: Path,
    config_file_path: Path,
    encoder_file_path: Path,
) -> None:
    """Test missing data file raises FileNotFoundError."""
    model_path = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_model_and_config(model_path)
