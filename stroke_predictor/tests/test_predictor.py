import pytest
import pandas as pd
from unittest.mock import MagicMock
from stroke_predictor.utils.predictor import make_prediction


def test_make_prediction_valid() -> None:
    """Test make_prediction with valid input."""
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]

    # Input DataFrame
    input_df = pd.DataFrame({"feature_1": [1], "feature_2": [2]})

    # Run prediction
    prediction, probability = make_prediction(input_df, mock_model)

    # Assertions
    assert prediction == 1, "Prediction is incorrect."
    assert probability == 0.8, "Probability is incorrect."


def test_make_prediction_zero_probability() -> None:
    """Test make_prediction with zero probability."""
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[1.0, 0.0]]

    # Input DataFrame
    input_df = pd.DataFrame({"feature_1": [3], "feature_2": [4]})

    # Run prediction
    prediction, probability = make_prediction(input_df, mock_model)

    # Assertions
    assert prediction == 0, "Prediction is incorrect."
    assert probability == 0.0, "Probability is incorrect."


def test_make_prediction_invalid_input() -> None:
    """Test make_prediction with invalid input."""
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("Invalid input")
    mock_model.predict_proba.side_effect = ValueError("Invalid input")

    # Input DataFrame
    input_df = pd.DataFrame({"feature_1": ["invalid"], "feature_2": ["invalid"]})

    # Run prediction and assert exception
    with pytest.raises(ValueError, match="Invalid input"):
        make_prediction(input_df, mock_model)
