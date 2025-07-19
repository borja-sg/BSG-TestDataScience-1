import pandas as pd
from typing import Any


def make_prediction(input_df: pd.DataFrame, model: Any) -> tuple[int, float]:
    """
    Run stroke prediction on a preprocessed input DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        The preprocessed input data.

    Returns
    -------
    tuple
        The predicted class and probability.
    """
    X_test = input_df.to_numpy().reshape(1, -1)
    prediction = model.predict(X_test)[0]
    probability = model.predict_proba(X_test)[0][1]
    return int(prediction), float(probability)
