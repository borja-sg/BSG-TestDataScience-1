import pytest
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any

from stroke_predictor.optuna_optimization.runner import (
    run_optimization,
    train_algorithm,
)
from stroke_predictor.optuna_optimization.objective import suggest_param, objective


# Tests for the run_optimization
@pytest.fixture
def mock_config(tmp_path: Path) -> Path:
    """Create a mock configuration file for testing."""
    config_path = tmp_path / "config.yaml"
    config = {
        "global": {
            "seed": 42,
            "metric": "f1",
            "direction": "maximize",
            "verbose": True,
        },
        "storage": {
            "data_path": str(tmp_path / "data.h5"),
            "hdf5_key_train": "train_data",
            "target": "target",
            "model_output": str(tmp_path / "model.pkl"),
        },
        "models": {
            "SVC": {
                "svc_C": {"low": 0.0001, "high": 100.0, "log": True},
                "kernel": {"values": ["linear", "rbf"]},
            },
        },
        "optuna": {
            "n_trials": 1,
            "study_name": "test_study",
            "storage": "sqlite:///test.db",
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
            "output_csv": "test.csv",
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def mock_data(tmp_path: Path) -> Path:
    """Create a mock HDF5 file with sample data."""
    data_path = tmp_path / "data.h5"
    df = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4],
            "feature_2": [5, 6, 7, 8],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_hdf(data_path, key="train_data", mode="w")
    return data_path


@patch("pandas.DataFrame.to_csv")
@patch("optuna.importance.get_param_importances")
@patch("stroke_predictor.optuna_optimization.runner.SMOTE")
@patch("stroke_predictor.optuna_optimization.runner.optuna.create_study")
def test_run_optimization(
    mock_create_study: MagicMock,
    mock_smote: MagicMock,
    mock_get_param_importances: MagicMock,
    mock_to_csv: MagicMock,
    mock_config: str,
    mock_data: Path,
) -> None:
    """
    Test the run_optimization function.

    Parameters
    ----------
    mock_create_study : MagicMock
        Mocked Optuna study creation function.
    mock_smote : MagicMock
        Mocked SMOTE class for handling class imbalance.
    mock_get_param_importances : MagicMock
        Mocked Optuna function for getting parameter importances.
    mock_to_csv : MagicMock
        Mocked pandas function for saving results to CSV.
    mock_config : str
        Path to the mocked configuration file.
    mock_data : Path
        Path to the mocked HDF5 data file.

    Returns
    -------
    None
    """
    # Mock SMOTE behavior
    mock_smote_instance = MagicMock()
    mock_smote.return_value = mock_smote_instance
    mock_smote_instance.fit_resample.return_value = (
        pd.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]}),
        pd.Series([0, 1]),
    )

    # Mock Optuna study behavior
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    mock_study.best_trial.value = 0.95
    mock_study.best_trial.params = {"param_1": 0.1, "param_2": 0.2}
    mock_study.trials_dataframe.return_value = pd.DataFrame(
        {
            "state": ["COMPLETE", "COMPLETE"],
            "value": [0.9, 0.95],
            "param_1": [0.1, 0.2],
            "param_2": [0.3, 0.4],
        }
    )

    # Run the optimization function
    study = run_optimization(config_path=mock_config)

    # Assertions
    assert study == mock_study, "Returned study object is incorrect."
    mock_smote_instance.fit_resample.assert_called_once()
    mock_create_study.assert_called_once()
    mock_study.optimize.assert_called_once()
    mock_study.trials_dataframe.assert_called_once()


# Train algorithm test
@patch("stroke_predictor.optuna_optimization.runner.load_dataset")
@patch("optuna.load_study")
@patch("stroke_predictor.optuna_optimization.runner.SMOTE")
@patch("stroke_predictor.optuna_optimization.runner.get_model")
@patch("stroke_predictor.optuna_optimization.runner.mlflow")
def test_train_algorithm(
    mock_mlflow: MagicMock,
    mock_get_model: MagicMock,
    mock_smote: MagicMock,
    mock_load_study: MagicMock,
    mock_load_dataset: MagicMock,  # Mock load_dataset
    mock_config: str,
    mock_data: Path,
) -> None:
    """
    Test the train_algorithm function.

    Parameters
    ----------
    mock_mlflow : MagicMock
        Mocked MLflow module for logging and saving models.
    mock_get_model : MagicMock
        Mocked function for retrieving the model based on classifier name and parameters.
    mock_smote : MagicMock
        Mocked SMOTE class for handling class imbalance.
    mock_load_study : MagicMock
        Mocked Optuna function for loading the study.
    mock_load_dataset : MagicMock
        Mocked function for loading the dataset.
    mock_config : str
        Path to the mocked configuration file.
    mock_data : Path
        Path to the mocked HDF5 data file.

    Returns
    -------
    None
    """
    # Mock load_dataset behavior
    mock_load_dataset.return_value = (
        pd.DataFrame({"feature_1": [1, 2, 3, 4], "feature_2": [5, 6, 7, 8]}),
        pd.Series([0, 1, 0, 1]),
    )

    # Mock SMOTE behavior
    mock_smote_instance = MagicMock()
    mock_smote.return_value = mock_smote_instance
    mock_smote_instance.fit_resample.return_value = (
        pd.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]}),
        pd.Series([0, 1]),
    )

    # Mock model behavior
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    # Mock MLflow behavior
    mock_mlflow.start_run.return_value = MagicMock()
    mock_mlflow.active_run.return_value.info.run_id = "test_run_id"

    # Mock Optuna study
    mock_study = MagicMock()
    mock_study.best_trial.params = {
        "classifier": "model_1",
        "param_1": 0.1,
        "param_2": 0.2,
    }
    mock_load_study.return_value = mock_study  # Simulate the presence of the study

    # Run the train_algorithm function
    train_algorithm(config_path=mock_config, study=mock_study)

    # Assertions
    mock_load_dataset.assert_called_once_with(
        path=Path(mock_config).parent / "data.h5",
        key="train_data",
        target="target",
    )
    mock_smote_instance.fit_resample.assert_called_once()
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.sklearn.log_model.assert_called_once()
    mock_mlflow.sklearn.save_model.assert_called_once()
    mock_get_model.assert_called_once_with(
        "model_1", {"classifier": "model_1", "param_1": 0.1, "param_2": 0.2}, seed=42
    )


# Test objective functions


def test_suggest_param_categorical() -> None:
    """Test suggest_param with categorical values."""
    trial = MagicMock()
    param_config = {"values": ["A", "B", "C"]}
    param_name = "param_categorical"
    trial.suggest_categorical.return_value = "B"

    result = suggest_param(trial, param_config, param_name)
    trial.suggest_categorical.assert_called_once_with(
        param_name, param_config["values"]
    )
    assert result == "B", "Categorical parameter suggestion failed."


def test_suggest_param_float() -> None:
    """Test suggest_param with float values."""
    trial = MagicMock()
    param_config = {"low": 0.1, "high": 1.0}
    param_name = "param_float"
    trial.suggest_float.return_value = 0.5

    result = suggest_param(trial, param_config, param_name)
    trial.suggest_float.assert_called_once_with(
        param_name, param_config["low"], param_config["high"]
    )
    assert result == 0.5, "Float parameter suggestion failed."


def test_suggest_param_int() -> None:
    """Test suggest_param with integer values."""
    trial = MagicMock()
    param_config = {"low": 1, "high": 10}
    param_name = "param_int"
    trial.suggest_int.return_value = 5

    result = suggest_param(trial, param_config, param_name)
    trial.suggest_int.assert_called_once_with(
        param_name, param_config["low"], param_config["high"]
    )
    assert result == 5, "Integer parameter suggestion failed."


def test_suggest_param_log_float() -> None:
    """Test suggest_param with log-scaled float values."""
    trial = MagicMock()
    param_config = {"low": 0.001, "high": 1.0, "log": True}
    param_name = "param_log_float"
    trial.suggest_float.return_value = 0.01

    result = suggest_param(trial, param_config, param_name)
    trial.suggest_float.assert_called_once_with(
        param_name, param_config["low"], param_config["high"], log=True
    )
    assert result == 0.01, "Log-scaled float parameter suggestion failed."


@patch("stroke_predictor.optuna_optimization.objective.cross_val_score")
@patch("stroke_predictor.optuna_optimization.objective.mlflow")
def test_objective(
    mock_mlflow: MagicMock,
    mock_cross_val_score: MagicMock,
) -> None:
    """
    Test the objective function.

    Parameters
    ----------
    mock_mlflow : MagicMock
        Mocked MLflow module for logging and saving models.
    mock_cross_val_score : MagicMock
        Mocked function for performing cross-validation.

    Returns
    -------
    None
    """
    trial = MagicMock()
    trial.suggest_categorical.return_value = "RandomForest"
    trial.params = {"n_estimators": 100, "max_depth": 10}

    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    model_configs = {
        "RandomForest": {
            "n_estimators": {"low": 10, "high": 200},
            "max_depth": {"low": 1, "high": 20},
        }
    }
    global_config = {
        "seed": 42,
        "metric": "accuracy",
    }
    data_config: dict[str, Any] = {}  # Added type annotation

    mock_cross_val_score.return_value = np.array([0.8, 0.85, 0.9])

    result = objective(trial, X, y, model_configs, global_config, data_config)

    # Assertions
    trial.suggest_categorical.assert_called_once_with(
        "classifier", list(model_configs.keys())
    )
    mock_cross_val_score.assert_called_once()
    assert result == 0.85, "Objective function returned incorrect mean score."
    mock_mlflow.start_run.assert_called_once()
