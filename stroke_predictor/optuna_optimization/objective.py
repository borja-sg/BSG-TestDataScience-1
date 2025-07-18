# stroke_predictor/tuning/objective.py

import optuna
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np
from typing import Any


def suggest_param(trial: optuna.Trial, param_config: dict, param_name: str) -> Any:
    """
    Suggest a parameter value based on the configuration provided.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object used for suggesting parameters.
    param_config : dict
        The configuration dictionary for the parameter, which can include ranges or categorical values.
    param_name : str
        The name of the parameter to suggest.

    Returns
    -------
    Any
        The suggested parameter value.
    """
    if isinstance(param_config, dict):
        if "values" in param_config:
            # Categorical
            return trial.suggest_categorical(param_name, param_config["values"])
        elif "low" in param_config and "high" in param_config:
            if param_config.get("log", False):
                return trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=True
                )
            elif all(
                isinstance(i, int) for i in (param_config["low"], param_config["high"])
            ):
                return trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            else:
                return trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
    else:
        return param_config


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    model_configs: dict,
    global_config: dict,
    data_config: dict,
) -> float:
    """
    Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object used for suggesting parameters.
    """

    classifier_name = trial.suggest_categorical(
        "classifier", list(model_configs.keys())
    )
    config = model_configs[classifier_name]

    if classifier_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=suggest_param(trial, config["n_estimators"], "n_estimators"),
            max_depth=suggest_param(trial, config["max_depth"], "max_depth"),
        )

    elif classifier_name == "LogisticRegression":
        model = LogisticRegression(
            C=suggest_param(trial, config["C"], "C"),
            max_iter=suggest_param(trial, config["max_iter"], "max_iter"),
        )

    elif classifier_name == "SVC":
        model = SVC(
            C=suggest_param(trial, config["svc_C"], "svc_C"),
            kernel=suggest_param(trial, config["kernel"], "kernel"),
        )

    elif classifier_name == "MLP":
        n_layers = suggest_param(trial, config["n_layers"], "n_layers")
        n_neurons = suggest_param(trial, config["n_neurons"], "n_neurons")

        hidden_layer_sizes = tuple([n_neurons] * n_layers)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=suggest_param(trial, config["alpha"], "alpha"),
            max_iter=config.get("max_iter", 500),
            random_state=global_config["seed"],
        )

    elif classifier_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=suggest_param(
                trial, config["xgb_n_estimators"], "xgb_n_estimators"
            ),
            max_depth=suggest_param(trial, config["xgb_max_depth"], "xgb_max_depth"),
            learning_rate=suggest_param(trial, config["xgb_lr"], "xgb_lr"),
            eval_metric="logloss",
            random_state=global_config["seed"],
        )

    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        scores = cross_val_score(pipeline, X, y, cv=3, scoring=global_config["metric"])
        mean_score = scores.mean()
        # Log the things
        # mlflow.log_metric(global_config["metric"], mean_score)
        # mlflow.sklearn.log_model(pipeline, input_example=X, name="model", registered_model_name=classifier_name)
        mlflow.log_params(
            {
                "classifier": classifier_name,
                "cv_folds": 3,
                "params": trial.params,
                "metric": global_config["metric"],
                "mean_score": mean_score,
            }
        )
        print(
            f"Trial {trial.number}: {classifier_name} - {global_config['metric']}: {mean_score}"
        )

    return mean_score
