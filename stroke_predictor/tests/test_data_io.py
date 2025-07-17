import pytest
import pandas as pd
import numpy as np
from stroke_predictor.utils.data_io import load_csv_dataset, save_to_hdf


@pytest.fixture
def temp_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_csv_dataset_success(temp_csv):
    """Test successful loading of a valid CSV file."""
    df = load_csv_dataset(temp_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)  # Check expected dimensions


def test_load_csv_dataset_nonexistent_file():
    """Test FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_csv_dataset("nonexistent_file.csv")


def test_load_csv_dataset_not_a_file(tmp_path):
    """Test ValueError if path is a directory (not a file)."""
    dir_path = tmp_path / "empty_dir"
    dir_path.mkdir()
    with pytest.raises(ValueError):
        load_csv_dataset(dir_path)


# Saving dataframes to HDF5
@pytest.fixture
def sample_dataframe():
    """Generate a sample DataFrame for testing."""
    return pd.DataFrame({"id": [1, 2, 3], "value": np.random.rand(3)})


@pytest.fixture
def temp_hdf_path(tmp_path):
    """Return a temporary HDF5 file path."""
    return tmp_path / "test.h5"


def test_save_to_hdf_success(sample_dataframe, temp_hdf_path):
    """Test saving a DataFrame to HDF5 with default mode ('a')."""
    key = "test_data"
    save_to_hdf(sample_dataframe, temp_hdf_path, key)

    # Verify the file exists and contains the key
    assert temp_hdf_path.exists()
    with pd.HDFStore(temp_hdf_path) as store:
        assert f"/{key}" in store.keys()
        loaded_df = store.get(key)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)


def test_save_to_hdf_overwrite_mode(sample_dataframe, temp_hdf_path):
    """Test overwriting an existing key with mode='w'."""
    key = "test_data"

    # First save (creates file)
    save_to_hdf(sample_dataframe, temp_hdf_path, key, mode="a")

    # New DataFrame to overwrite
    new_df = pd.DataFrame({"id": [4, 5], "value": [0.99, 0.42]})
    save_to_hdf(new_df, temp_hdf_path, key, mode="w")

    # Verify only the new data exists
    with pd.HDFStore(temp_hdf_path) as store:
        assert store.get(key).shape == new_df.shape


def test_save_to_hdf_invalid_mode(sample_dataframe, temp_hdf_path):
    """Test invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Mode must be either 'a' or 'w'"):
        save_to_hdf(sample_dataframe, temp_hdf_path, "test_data", mode="x")


def test_save_to_hdf_nonexistent_dir(sample_dataframe, tmp_path):
    """Test saving to a non-existent directory (should create parent dirs)."""
    non_existent_path = tmp_path / "nonexistent_dir" / "test.h5"
    save_to_hdf(sample_dataframe, non_existent_path, "test_data")
    assert non_existent_path.exists()


def test_save_to_hdf_invalid_key(sample_dataframe, temp_hdf_path):
    """Test invalid key type raises TypeError."""
    with pytest.raises(TypeError, match="Key must be a string"):
        save_to_hdf(sample_dataframe, temp_hdf_path, key=123)  # type: ignore
