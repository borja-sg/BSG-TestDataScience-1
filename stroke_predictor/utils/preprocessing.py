from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from typing import Any, List, Tuple
import pandas as pd
import sklearn


def drop_rows_with_missing(
    df: pd.DataFrame, column: str, missing_value: Any
) -> pd.DataFrame:
    """
    Drop rows with missing values in specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to check for missing values.
    missing_value : Any
        Value to consider as missing (e.g., np.nan).

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing missing values in specified columns removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Drop rows where the specified column has the missing value
    df_dropped = df[df[column] != missing_value].copy()
    return df_dropped


def replace_missing_to_value(
    df: pd.DataFrame, column: str, missing_value: Any, new_value: Any
) -> pd.DataFrame:
    """
    Replace missing values in specified columns with a given value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to process.
    missing_value : Any
        Value to consider as missing (e.g., np.nan).
    new_value : Any
        Value to replace missing values with.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified missing values replaced.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Replace missing values in the specified column
    df_replaced = df.copy()
    df_replaced[column] = df_replaced[column].replace(missing_value, new_value)
    return df_replaced


def encode_categorical(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Apply Ordinal Encoding to specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        List of categorical columns to encode.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded columns.
    OrdinalEncoder
        Fitted encoder object.
    """
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    encoder = OrdinalEncoder()
    df_encoded = df.copy()
    df_encoded[columns] = encoder.fit_transform(df[columns])
    return df_encoded, encoder


def impute_missing_knn(
    df: pd.DataFrame, columns: List[str], n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Impute missing values using KNN imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        Columns with missing values to impute.
    n_neighbors : int, optional
        Number of neighbors, by default 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = df.copy()
    df_imputed[columns] = imputer.fit_transform(df[columns])
    return df_imputed


def compute_health_score(row: pd.Series) -> int:
    """
    Compute a health score based on various health indicators.

    Parameters
    ----------
    row : pd.Series
        A row of the DataFrame containing patient data.

    Returns
    -------
    int
        The computed health score.
    """
    if not isinstance(row, pd.Series):
        raise ValueError(
            "Input must be a pandas Series representing a row of the DataFrame."
        )
    if (
        "age" not in row
        or "avg_glucose_level" not in row
        or "smoking_status" not in row
    ):
        raise ValueError(
            "Row must contain 'age', 'avg_glucose_level', and 'smoking_status' columns."
        )
    if not isinstance(row["age"], (int, float)) or not isinstance(
        row["avg_glucose_level"], (int, float)
    ):
        raise ValueError("'age' and 'avg_glucose_level' must be numeric values.")
    if not isinstance(row["smoking_status"], str):
        raise ValueError("'smoking_status' must be a string.")
    # Initialize score based on health indicators
    score = 0
    score += int(row["age"] > 60)
    score += int(row["avg_glucose_level"] >= 200)
    score += int(row["smoking_status"] == "smokes")
    score += int(row["hypertension"] == 1)
    score += int(row["heart_disease"] == 1)
    return score


def categorize_risk(score: int) -> str:
    """
    Categorize health score into risk levels.

    Parameters
    ----------
    score : int
        The computed health score.

    Returns
    -------
    str
        The risk category based on the health score."""
    if not isinstance(score, int):
        raise ValueError("Score must be an integer.")
    if score < 0:
        raise ValueError("Score cannot be negative.")
    # Define risk categories based on the score
    if score <= 1:
        return "low risk"
    elif score <= 3:
        return "moderate risk"
    else:
        return "high risk"


def preprocess_dataframe(
    df: pd.DataFrame,
    columns: list,
    variable_encoder: sklearn.preprocessing.OrdinalEncoder,
) -> pd.DataFrame:
    """
    Preprocess the DataFrame by creating engineered features, computing health scores, and encoding categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing patient data.
    columns : list
        The list of columns to include in the final DataFrame.
    variable_encoder : sklearn.preprocessing.OrdinalEncoder
        The fitted OrdinalEncoder to transform categorical variables.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with engineered features and encoded categorical variables.
    """
    # Check if required columns are present
    required_columns = [
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
        "smoking_status",
    ]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")
    # Check if the variable_encoder is fitted
    if not hasattr(variable_encoder, "categories_"):
        raise ValueError("The variable_encoder must be a fitted OrdinalEncoder.")
    # Ensure the DataFrame has the correct data types
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' is required but not found in the DataFrame."
            )
        if col == "smoking_status":
            df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Create the engineered features
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 75, 100],
        labels=["young", "adult", "middle_aged", "senior", "elderly"],
    )
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 99, 125, 300],
        labels=["normal", "prediabetic", "diabetic"],
    )
    # Compute health score for each individual
    df["health_score"] = df.apply(compute_health_score, axis=1)
    # Categorize score into risk levels
    df["risk_category"] = df["health_score"].apply(categorize_risk)
    # Apply the loaded encoder
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    print(categorical_cols)
    df_encoded = df.copy()
    df_encoded[categorical_cols] = variable_encoder.transform(df[categorical_cols])
    # Reorder columns to match the original DataFrame
    df_encoded = df_encoded[columns]
    return df_encoded


def split_and_rebuild_dataframe(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets, and return both with the target column included.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame including the target column.
    target_column : str
        The name of the target column.
    test_size : float, optional
        Fraction of the dataset to include in the test split. Default is 0.3.
    random_state : int, optional
        Random state for reproducibility. Default is 42.
    stratify : bool, optional
        Whether to stratify based on the target column. Default is True.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and test sets including the target column.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    df_train = X_train.copy()
    df_train[target_column] = y_train

    df_test = X_test.copy()
    df_test[target_column] = y_test

    # Ensure columns are in the same order as original DataFrame
    df_train = df_train[df.columns]
    df_test = df_test[df.columns]

    return df_train, df_test
