import pandas as pd


def roc(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.Series:
    """
    Calculate the Rate of Change (ROC) for a specified column over a given window.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column name to calculate ROC on (default is 'Close').
    - window (int): The window size for calculating the percentage change (default is 14).

    Returns:
    - pd.Series: The Rate of Change (ROC) as a percentage.
    """
    roc = df[column].pct_change(periods=window) * 100

    return roc
