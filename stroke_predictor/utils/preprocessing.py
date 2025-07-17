from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from typing import Any, List, Tuple
import pandas as pd


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
