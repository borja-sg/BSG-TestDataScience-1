import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_distribution(df, column, hue=None, stat="count", bins=30, title=None):
    """
    Plot the distribution of a continuous variable using seaborn.histplot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    column : str
        Column name to plot.
    hue : str, optional
        Column to use for color encoding.
    stat : str, optional
        Statistic to plot ('count', 'density', 'probability'), by default 'count'.
    bins : int, optional
        Number of histogram bins, by default 30.
    title : str, optional
        Title of the plot.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        stat=stat,
        bins=bins,
        element="step",
        common_norm=False,
    )
    plt.title(title or f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def plot_pairwise(df, hue=None, height=2.5):
    """
    Create a pairplot for visualizing pairwise relationships in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with continuous variables.
    hue : str, optional
        Column to use for color encoding.
    height : float, optional
        Height (in inches) of each facet, by default 2.5.
    """
    sns.pairplot(df, hue=hue, height=height)
    plt.tight_layout()
    plt.show()


def plot_categorical_counts(
    df, column, hue=None, normalize=False, title=None, y_log=False, output_path=None
):
    """
    Plot the counts of a categorical column using seaborn.countplot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to count.
    hue : str, optional
        Column to use for bar color separation.
    normalize : bool, optional
        Whether to normalize counts per group (hue-level).
    title : str, optional
        Title of the plot.
    output_path : str, optional
        Path to save the plot image. If None, the plot is shown interactively.
    """
    plt.figure(figsize=(8, 4))
    data = df.copy()
    if normalize and hue:
        norm_df = (
            data.groupby([column, hue], observed=False)
            .size()
            .div(data.groupby(column, observed=False).size(), level=column)
            .reset_index(name="proportion")
        )
        sns.barplot(data=norm_df, x=column, y="proportion", hue=hue)
        plt.ylabel("Proportion")
    else:
        sns.countplot(data=data, x=column, hue=hue)
        plt.ylabel("Counts")
    plt.title(title or f"Distribution of {column}")
    if y_log:
        plt.yscale("log")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_variable_distributions(
    series_list,
    labels,
    colors,
    alpha_list=None,
    bins="auto",
    kde=True,
    xlabel=None,
    ylabel="Normalized counts",
    title=None,
    output_path=None,
    stat="density",
):
    """
    Plot overlaid distributions of several variable series using histograms with optional KDE.

    Parameters
    ----------
    series_list : list of pd.Series
        List of pandas Series to plot.
    labels : list of str
        Labels for each series, used in the legend.
    colors : list of str
        Colors to use for each series.
    alpha_list : list of float, optional
        Transparency levels for each histogram. Defaults to 0.5 for all if not specified.
    bins : int or sequence or str, optional
        Binning strategy for histograms (e.g., "auto", int, np.linspace, etc.). Default is "auto".
    kde : bool, optional
        Whether to overlay a KDE curve. Default is True.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Default is "Normalized counts".
    title : str, optional
        Plot title.
    output_path : str, optional
        Path to save the plot. If None, shows the plot interactively.
    stat : str, optional
        Normalization strategy. Options: "count", "frequency", "density", "probability".
        Default is "density".
    """
    if alpha_list is None:
        alpha_list = [0.5] * len(series_list)

    plt.figure(figsize=(10, 6))

    # Compute unified bins if bins is a string like "auto" or a single int
    if isinstance(bins, (int, str)):
        combined = pd.concat(series_list)
        min_val, max_val = combined.min(), combined.max()
        bins = (
            np.linspace(min_val, max_val, 30)
            if isinstance(bins, str) or bins == "auto"
            else bins
        )

    for series, label, color, alpha in zip(series_list, labels, colors, alpha_list):
        sns.histplot(
            series.dropna(),
            kde=kde,
            bins=bins,
            color=color,
            alpha=alpha,
            stat=stat,
            label=label,
        )

    plt.xlabel(xlabel or "Variable")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_correlation_heatmap(
    df,
    title="Correlation Heatmap of Continuous Variables",
    figsize=(10, 8),
    cmap="coolwarm",
    output_path=None,
):
    """
    Plot a heatmap showing the correlation matrix between continuous (numeric) variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    title : str, optional
        Title of the heatmap plot.
    figsize : tuple of int, optional
        Size of the figure (width, height).
    cmap : str, optional
        Colormap to use for the heatmap.
    output_path : str, optional
        Path to save the plot image. If None, the plot is shown interactively.

    Returns
    -------
    pd.DataFrame
        The correlation matrix computed from the numeric columns.
    """
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return corr_matrix
