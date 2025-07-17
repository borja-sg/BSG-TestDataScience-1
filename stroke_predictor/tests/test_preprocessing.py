import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from stroke_predictor.utils.preprocessing import (
    drop_rows_with_missing,
    replace_missing_to_value,
    encode_categorical,
    impute_missing_knn,
    split_and_rebuild_dataframe,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature_1": [1, 2, np.nan, 4],
            "feature_2": ["A", "B", "C", np.nan],
            "target": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def categorical_dataframe() -> pd.DataFrame:
    """Fixture to provide a sample DataFrame with categorical data."""
    return pd.DataFrame({"cat_1": ["A", "B", "C", "A"], "cat_2": ["X", "Y", "Z", "X"]})


def test_drop_rows_with_missing(sample_dataframe: pd.DataFrame) -> None:
    """Test the drop_rows_with_missing function."""
    result = drop_rows_with_missing(
        sample_dataframe, column="feature_1", missing_value=np.nan
    )
    assert result.shape[1] == 3, "Rows with missing values were not dropped correctly."


def test_replace_missing_to_value(sample_dataframe: pd.DataFrame) -> None:
    """Test the replace_missing_to_value function."""
    result = replace_missing_to_value(
        sample_dataframe, column="feature_1", missing_value=np.nan, new_value=0
    )
    assert (
        result["feature_1"].isna().sum() == 0
    ), "Missing values were not replaced correctly."
    assert (result["feature_1"] == 0).sum() == 1, "Replacement value is incorrect."


def test_encode_categorical(categorical_dataframe: pd.DataFrame) -> None:
    """Test the encode_categorical function."""
    result, encoder = encode_categorical(
        categorical_dataframe, columns=["cat_1", "cat_2"]
    )
    assert isinstance(result, pd.DataFrame), "Result is not a DataFrame."
    assert isinstance(encoder, OrdinalEncoder), "Encoder is not an OrdinalEncoder."
    assert result["cat_1"].max() <= 2, "Encoding is incorrect."
    assert result["cat_2"].max() <= 2, "Encoding is incorrect."


def test_impute_missing_knn(sample_dataframe: pd.DataFrame) -> None:
    """Test the impute_missing_knn function."""
    result = impute_missing_knn(sample_dataframe, columns=["feature_1"], n_neighbors=2)
    assert (
        result["feature_1"].isna().sum() == 0
    ), "Missing values were not imputed correctly."


def test_split_and_rebuild_dataframe(sample_dataframe: pd.DataFrame) -> None:
    """Test the split_and_rebuild_dataframe function."""
    df_train, df_test = split_and_rebuild_dataframe(
        sample_dataframe, target_column="target", test_size=0.5
    )
    assert df_train.shape[0] == 2, "Training set size is incorrect."
    assert df_test.shape[0] == 2, "Test set size is incorrect."
    assert "target" in df_train.columns, "Target column is missing in training set."
    assert "target" in df_test.columns, "Target column is missing in test set."
