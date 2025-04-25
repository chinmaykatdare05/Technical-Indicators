import pandas as pd


def spearman_rank_correlation(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Spearman's Rank Correlation Coefficient between two variables.

    The Spearman rank correlation coefficient measures the strength and direction
    of the monotonic relationship between two variables, based on the ranks of the data.

    Parameters:
    x (pd.Series): The first variable (data series).
    y (pd.Series): The second variable (data series).

    Returns:
    float: Spearman's Rank Correlation coefficient (ρ) between x and y.
    """
    # Rank the values in both series
    rank_x = x.rank()
    rank_y = y.rank()

    # Calculate the difference between the ranks
    d = rank_x - rank_y

    # Calculate Spearman's rank correlation coefficient
    n = len(x)
    rho = 1 - (6 * (d**2).sum()) / (n * (n**2 - 1))

    return rho
