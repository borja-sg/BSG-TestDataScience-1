import pytest
import pandas as pd
import numpy as np
from stroke_predictor.utils.plotting import (
    plot_distribution,
    plot_pairwise,
    plot_categorical_counts,
    plot_variable_distributions,
    plot_correlation_heatmap,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "continuous_1": np.random.normal(0, 1, 100),
            "continuous_2": np.random.normal(5, 2, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
            "binary": np.random.choice([0, 1], 100),
        }
    )


@pytest.fixture
def sample_series_list() -> list:
    """Fixture to provide a list of sample Series for testing."""
    return [
        pd.Series(np.random.normal(0, 1, 100)),
        pd.Series(np.random.normal(5, 2, 100)),
    ]


@pytest.fixture
def sample_labels() -> list:
    """Fixture to provide labels for testing."""
    return ["Series 1", "Series 2"]


@pytest.fixture
def sample_colors() -> list:
    """Fixture to provide colors for testing."""
    return ["blue", "green"]


def test_plot_distribution(sample_dataframe: pd.DataFrame) -> None:
    """Test the plot_distribution function."""
    try:
        plot_distribution(
            df=sample_dataframe,
            column="continuous_1",
            hue="binary",
            stat="density",
            bins=20,
            title="Test Distribution",
        )
    except Exception as e:
        pytest.fail(f"plot_distribution failed with error: {e}")


def test_plot_pairwise(sample_dataframe: pd.DataFrame) -> None:
    """Test the plot_pairwise function."""
    try:
        plot_pairwise(df=sample_dataframe, hue="categorical", height=2.5)
    except Exception as e:
        pytest.fail(f"plot_pairwise failed with error: {e}")


def test_plot_categorical_counts(sample_dataframe: pd.DataFrame) -> None:
    """Test the plot_categorical_counts function."""
    try:
        plot_categorical_counts(
            df=sample_dataframe,
            column="categorical",
            hue="binary",
            normalize=True,
            title="Test Categorical Counts",
        )
    except Exception as e:
        pytest.fail(f"plot_categorical_counts failed with error: {e}")


def test_plot_variable_distributions(
    sample_series_list: list, sample_labels: list, sample_colors: list
) -> None:
    """Test the plot_variable_distributions function."""
    try:
        plot_variable_distributions(
            series_list=sample_series_list,
            labels=sample_labels,
            colors=sample_colors,
            bins=15,
            kde=True,
            xlabel="Test Variable",
            ylabel="Test Counts",
            title="Test Variable Distributions",
        )
    except Exception as e:
        pytest.fail(f"plot_variable_distributions failed with error: {e}")


def test_plot_correlation_heatmap(sample_dataframe: pd.DataFrame) -> None:
    """Test the plot_correlation_heatmap function."""
    try:
        corr_matrix = plot_correlation_heatmap(
            df=sample_dataframe,
            title="Test Correlation Heatmap",
            figsize=(8, 6),
            cmap="viridis",
        )
        assert isinstance(
            corr_matrix, pd.DataFrame
        ), "Returned correlation matrix is not a DataFrame."
        assert not corr_matrix.empty, "Returned correlation matrix is empty."
    except Exception as e:
        pytest.fail(f"plot_correlation_heatmap failed with error: {e}")
