import mlflow
import optuna
from optuna.trial import TrialState
import yaml
import sys
from pathlib import Path
from imblearn.over_sampling import SMOTE

from stroke_predictor.utils.data_io import load_dataset
from stroke_predictor.optuna_optimization.objective import objective


def run_optimization(config_path: str) -> optuna.Study:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    global_config = config["global"]
    data_config = config["storage"]
    model_configs = config["models"]
    optuna_config = config["optuna"]

    # Load data
    X, y = load_dataset(
        path=Path(data_config["data_path"]),
        key=data_config["hdf5_key_test"],
        target=data_config["target"],
    )

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=global_config["seed"])
    X, y = smote.fit_resample(X, y)

    mlflow.set_experiment("stroke_classification")

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


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "stroke_predictor/configs/optuna.yml"
    )
    run_optimization(config_path)
