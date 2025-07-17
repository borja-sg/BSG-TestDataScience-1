from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def drop_rows_with_missing(df, column, missing_value):
    """
    Drop rows with missing values in specified columns.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to check for missing values.
    missing_value : any
        Value to consider as missing (e.g., np.nan).

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing missing values in specified columns removed.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_dropped = df.copy()
    # Drop rows where the specified column has the missing value
    df_dropped.drop(df_dropped[df_dropped[column] == missing_value].index, inplace=True)
    return df_dropped


def replace_missing_to_value(df, column, missing_value, new_value):
    """
    Replace missing values in specified columns with a given value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to process.
    missing_value : any
        Value to consider as missing (e.g., np.nan).
    new_value : any
        Value to replace missing values with.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified missing values replaced.
    """
    df_replaced = df.copy()
    df_replaced[column] = df_replaced[column].replace(missing_value, new_value)
    return df_replaced


def encode_categorical(df, columns):
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
    encoder = OrdinalEncoder()
    df_encoded = df.copy()
    df_encoded[columns] = encoder.fit_transform(df[columns])
    return df_encoded, encoder


def impute_missing_knn(df, columns, n_neighbors=5):
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
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = df.copy()
    df_imputed[columns] = imputer.fit_transform(df[columns])
    return df_imputed


def split_and_rebuild_dataframe(
    df, target_column, test_size=0.3, random_state=42, stratify=True
):
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
    df_train : pd.DataFrame
        Training set including the target column.
    df_test : pd.DataFrame
        Test set including the target column.
    """
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
