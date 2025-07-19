from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Import Any from typing import Any
from typing import Any


def get_model(classifier_name: str, best_params: dict, seed: int = 42) -> Any:
    """
    Get the model based on the classifier name and best parameters.

    Parameters
    ----------
    classifier_name : str
        The name of the classifier to instantiate.
    best_params : dict
        The best parameters obtained from the Optuna study.

    seed : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    Model
        An instance of the specified classifier with the best parameters.
    """
    if classifier_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=seed,
        )
    elif classifier_name == "LogisticRegression":
        return LogisticRegression(
            C=best_params["C"], max_iter=best_params["max_iter"], random_state=seed
        )
    elif classifier_name == "SVC":
        return SVC(
            C=best_params["svc_C"],
            kernel=best_params["kernel"],
            probability=True,
            random_state=seed,
        )
    elif classifier_name == "MLP":
        # Extract the number of layers and neurons per layer from best_params
        hidden_layer_sizes = tuple(
            best_params[f"n_neurons_layer_{i}"] for i in range(best_params["n_layers"])
        )
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=best_params["alpha"],
            max_iter=best_params.get("max_iter", 500),
            random_state=seed,
        )
    elif classifier_name == "XGBoost":
        return XGBClassifier(
            n_estimators=best_params["xgb_n_estimators"],
            max_depth=best_params["xgb_max_depth"],
            learning_rate=best_params["xgb_lr"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
