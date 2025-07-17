import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from stroke_predictor.utils.data_io import load_csv_dataset, save_to_hdf


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
