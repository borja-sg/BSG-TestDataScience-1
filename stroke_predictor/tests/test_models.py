import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from stroke_predictor.utils.models import get_model
from typing import Any


def test_get_model_random_forest() -> None:
    """Test instantiation of RandomForestClassifier."""
    best_params = {"n_estimators": 100, "max_depth": 10}
    model = get_model("RandomForest", best_params, seed=42)
    assert isinstance(
        model, RandomForestClassifier
    ), "Model is not a RandomForestClassifier."
    assert model.n_estimators == 100, "n_estimators parameter is incorrect."
    assert model.max_depth == 10, "max_depth parameter is incorrect."
    assert model.random_state == 42, "random_state parameter is incorrect."


def test_get_model_logistic_regression() -> None:
    """Test instantiation of LogisticRegression."""
    best_params = {"C": 1.0, "max_iter": 200}
    model = get_model("LogisticRegression", best_params, seed=42)
    assert isinstance(model, LogisticRegression), "Model is not a LogisticRegression."
    assert model.C == 1.0, "C parameter is incorrect."
    assert model.max_iter == 200, "max_iter parameter is incorrect."
    assert model.random_state == 42, "random_state parameter is incorrect."


def test_get_model_svc() -> None:
    """Test instantiation of SVC."""
    best_params = {"svc_C": 1.0, "kernel": "linear"}
    model = get_model("SVC", best_params, seed=42)
    assert isinstance(model, SVC), "Model is not an SVC."
    assert model.C == 1.0, "C parameter is incorrect."
    assert model.kernel == "linear", "kernel parameter is incorrect."
    assert model.probability is True, "probability parameter is incorrect."
    assert model.random_state == 42, "random_state parameter is incorrect."


def test_get_model_mlp() -> None:
    """Test instantiation of MLPClassifier."""
    best_params = {
        "n_layers": 2,
        "n_neurons_layer_0": 64,
        "n_neurons_layer_1": 32,
        "alpha": 0.001,
        "max_iter": 300,
    }
    model = get_model("MLP", best_params, seed=42)
    assert isinstance(model, MLPClassifier), "Model is not an MLPClassifier."
    assert model.hidden_layer_sizes == (
        64,
        32,
    ), "hidden_layer_sizes parameter is incorrect."
    assert model.alpha == 0.001, "alpha parameter is incorrect."
    assert model.max_iter == 300, "max_iter parameter is incorrect."
    assert model.random_state == 42, "random_state parameter is incorrect."


def test_get_model_xgboost() -> None:
    """Test instantiation of XGBClassifier."""
    best_params = {"xgb_n_estimators": 100, "xgb_max_depth": 5, "xgb_lr": 0.1}
    model = get_model("XGBoost", best_params, seed=42)
    assert isinstance(model, XGBClassifier), "Model is not an XGBClassifier."
    assert model.n_estimators == 100, "n_estimators parameter is incorrect."
    assert model.max_depth == 5, "max_depth parameter is incorrect."
    assert model.learning_rate == 0.1, "learning_rate parameter is incorrect."
    assert model.random_state == 42, "random_state parameter is incorrect."


def test_get_model_unsupported_classifier() -> None:
    """Test unsupported classifier raises ValueError."""
    best_params: dict[str, Any] = {}
    with pytest.raises(ValueError, match="Unsupported classifier: UnsupportedModel"):
        get_model("UnsupportedModel", best_params, seed=42)
