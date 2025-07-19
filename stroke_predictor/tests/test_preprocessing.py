import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
from pathlib import Path

from stroke_predictor.utils.preprocessing import (
    drop_rows_with_missing,
    replace_missing_to_value,
    encode_categorical,
    impute_missing_knn,
    split_and_rebuild_dataframe,
    compute_health_score,
    categorize_risk,
    preprocess_dataframe,
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


@pytest.fixture
def sample_complete_dataframe() -> pd.DataFrame:
    """Fixture to provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 65, 45, 75],  # float
            "hypertension": [0, 1, 0, 1],  # int
            "heart_disease": [0, 1, 0, 1],  # int
            "avg_glucose_level": [90.0, 210.0, 120.0, 300.0],  # float
            "bmi": [18.0, 28.0, 22.0, 35.0],  # float
            "gender": [
                "Male",
                "Female",
                "Male",
                "Male",
            ],  # Literal["Male", "Female", "Other"]
            "ever_married": ["Yes", "No", "Yes", "No"],  # Literal["Yes", "No"]
            "work_type": [
                "Private",
                "Self-employed",
                "Govt_job",
                "Never_worked",
            ],  # Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
            "Residence_type": [
                "Urban",
                "Rural",
                "Urban",
                "Rural",
            ],  # Literal["Urban", "Rural"]
            "smoking_status": [
                "never smoked",
                "smokes",
                "formerly smoked",
                "smokes",
            ],  # Literal["formerly smoked", "never smoked", "smokes", "Unknown"]
        }
    ).astype(
        {
            "age": float,
            "hypertension": int,
            "heart_disease": int,
            "avg_glucose_level": float,
            "bmi": float,
            "gender": str,
            "ever_married": str,
            "work_type": str,
            "Residence_type": str,
            "smoking_status": str,
        }
    )


@pytest.fixture
def fitted_encoder() -> OrdinalEncoder:
    """Fixture to provide a fitted OrdinalEncoder for testing."""
    variable_encoder_path = Path("./data/stroke_data_encoder.pkl")
    variable_encoder = joblib.load(variable_encoder_path)
    return variable_encoder


def test_compute_health_score(sample_complete_dataframe: pd.DataFrame) -> None:
    """Test the compute_health_score function."""
    row = sample_complete_dataframe.iloc[1]  # Select a row for testing
    score = compute_health_score(row)
    assert score == 5, "Health score computation failed."


def test_categorize_risk() -> None:
    """Test the categorize_risk function."""
    assert categorize_risk(0) == "low risk", "Risk categorization failed for score 0."
    assert (
        categorize_risk(2) == "moderate risk"
    ), "Risk categorization failed for score 2."
    assert categorize_risk(4) == "high risk", "Risk categorization failed for score 4."


def test_preprocess_dataframe(
    sample_complete_dataframe: pd.DataFrame, fitted_encoder: OrdinalEncoder
) -> None:
    """Test the preprocess_dataframe function."""
    columns = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "bmi_category",
        "age_group",
        "glucose_category",
        "health_score",
        "risk_category",
    ]

    preprocessed_df = preprocess_dataframe(
        sample_complete_dataframe, columns, fitted_encoder
    )

    # Assertions
    assert not preprocessed_df.empty, "Preprocessed DataFrame is empty."
    assert "health_score" in preprocessed_df.columns, "Health score column is missing."
    assert (
        "risk_category" in preprocessed_df.columns
    ), "Risk category column is missing."
    assert (
        preprocessed_df["health_score"].iloc[1] == 5
    ), "Health score computation failed."
    assert preprocessed_df.shape[1] == len(
        columns
    ), "Column count mismatch after preprocessing."
