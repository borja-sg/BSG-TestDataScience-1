import mlflow
import optuna
from optuna.trial import TrialState
import yaml
import sys
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stroke_predictor.utils.data_io import load_dataset
from stroke_predictor.optuna_optimization.objective import objective
from stroke_predictor.utils.models import get_model


def run_optimization(config_path: str) -> optuna.Study:
    """
    Optimize the stroke prediction model using Optuna.

    Parameters
    ----------
    config_path : str
        Path to the configuration file containing model and optimization parameters.

    Returns
    -------
    optuna.Study
        The Optuna study object containing the optimization results.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_config = config["global"]
    data_config = config["storage"]
    model_configs = config["models"]
    optuna_config = config["optuna"]

    # Load data
    X, y = load_dataset(
        path=Path(data_config["data_path"]),
        key=data_config["hdf5_key_train"],
        target=data_config["target"],
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=global_config["seed"])
    X, y = smote.fit_resample(X, y)

    # Set up MLflow experiment
    mlflow.set_experiment(optuna_config["study_name"])

    sampler = getattr(optuna.samplers, optuna_config["sampler"])(
        seed=global_config["seed"]
    )
    pruner = getattr(optuna.pruners, optuna_config["pruner"])()

    study = optuna.create_study(
        study_name=optuna_config["study_name"],
        direction=global_config["direction"],
        sampler=sampler,
        pruner=pruner,
        storage=optuna_config["storage"],
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, X, y, model_configs, global_config, data_config),
        n_trials=optuna_config["n_trials"],
        timeout=optuna_config.get("timeout", None),
        catch=(RuntimeError, TypeError, ValueError),
    )

    print("Best trial:")
    print(study.best_trial)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe()
    df = df.loc[df["state"] == "COMPLETE"]  # Keep only results that did not prune
    df = df.drop("state", axis=1)  # Exclude state column
    df = df.sort_values("value")  # Sort based on accuracy
    df.to_csv(
        Path(optuna_config["output_csv"]),
        index=False,
    )  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None
    )

    # Display the most important hyperparameters
    print("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        print("  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100))

    return study


def train_algorithm(config_path: str, study: optuna.Study) -> None:
    """
    Train the best model based on the Optuna study results.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    study : optuna.Study
        The Optuna study containing the best trial parameters.

    Returns
    -------
    None
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load its sections
    global_config = config["global"]
    data_config = config["storage"]

    # Load Optuna study
    if not isinstance(study, optuna.study.Study):
        study = optuna.load_study(
            study_name=config["optuna"]["study_name"],
            storage=config["optuna"]["storage"],
        )

    # Get the best parameters from the study
    best_params = study.best_trial.params
    classifier_name = best_params["classifier"]

    print(f"Best parameters for {classifier_name}: {best_params}")

    # Load data
    X_train, y_train = load_dataset(
        path=Path(data_config["data_path"]),
        key=data_config["hdf5_key_train"],
        target=data_config["target"],
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=global_config["seed"])
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Build and train model
    model = get_model(classifier_name, best_params, seed=global_config["seed"])
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Save model with MLflow
    output_path = Path(config["storage"]["model_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        # Log model
        mlflow.sklearn.log_model(
            pipeline, name=classifier_name, input_example=X_train[:1]
        )

        # Log params or metrics
        mlflow.log_params(best_params)
        mlflow.log_param("classifier", classifier_name)

        # Save the model locally as well
        mlflow.sklearn.save_model(pipeline, output_path)

        print(f"Model saved to {output_path}")


if __name__ == "__main__":
    # Parse command line arguments for configuration file path
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "stroke_predictor/configs/optuna.yml"
    )
    # Run the optimization process
    study = run_optimization(config_path)
    # Train the best model based on the optimization results
    train_algorithm(config_path, study)
