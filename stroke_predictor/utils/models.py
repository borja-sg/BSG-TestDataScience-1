from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from typing import Any, Literal
from pydantic import BaseModel, validator


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


class PatientFeatures(BaseModel):
    """
    Schema for patient features used in stroke prediction.

    Attributes
    ----------
    age : float
        Age of the patient.
    hypertension : int
        Hypertension status (0 or 1).
    heart_disease : int
        Heart disease status (0 or 1).
    avg_glucose_level : float
        Average glucose level of the patient.
    bmi : float
        Body Mass Index of the patient.
    gender : Literal["Male", "Female", "Other"]:
        Gender of the patient.
    ever_married : Literal["Yes", "No"]
        Marital status of the patient.
    work_type : Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
        Type of work the patient is engaged in.
    Residence_type : Literal["Urban", "Rural"]
        Type of residence (Urban or Rural).
    smoking_status : Literal["formerly smoked", "never smoked", "smokes", "Unknown"]
        Smoking status of the patient.

    Methods
    -------
    cap_age(cls, v)
        Cap age between 0 and 120.
    cap_bmi(cls, v)
        Cap BMI between 10 and 70.
    cap_glucose(cls, v)
        Cap average glucose level between 50 and 500.


    """

    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: Literal["Male", "Female", "Other"]
    ever_married: Literal["Yes", "No"]
    work_type: Literal[
        "Private", "Self-employed", "Govt_job", "children", "Never_worked"
    ]
    Residence_type: Literal["Urban", "Rural"]
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

    @validator("age")  # noqa
    def cap_age(cls, v: float) -> float:  # noqa
        """
        Cap age between 0 and 120.

        Parameters
        ----------
        v : float
            The age value to validate.

        Returns
        -------
        float
            The capped age value.
        """
        return max(0, min(v, 120))  # Cap age between 0 and 120

    @validator("bmi")  # noqa
    def cap_bmi(cls, v: float) -> float:  # noqa
        """
        Cap BMI between 10 and 70.

        Parameters
        ----------
        v : float
            The BMI value to validate.

        Returns
        -------
        float
            The capped BMI value.
        """
        return max(10.0, min(v, 70.0))  # Cap BMI between 10 and 70

    @validator("avg_glucose_level")  # noqa
    def cap_glucose(cls, v: float) -> float:  # noqa
        """
        Cap average glucose level between 50 and 500.

        Parameters
        ----------
        v : float
            The average glucose level to validate.

        Returns
        -------
        float
            The capped glucose level value.
        """
        return max(50.0, min(v, 500.0))  # Cap glucose between 50 and 500
