import pandas as pd


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Williams %R (Williams Percent Range) for a given DataFrame.

    Williams %R is a momentum indicator that measures overbought and oversold levels,
    with values ranging from 0 to -100. A value above -20 indicates overbought conditions,
    and a value below -80 indicates oversold conditions.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    window (int): The window size for calculating the highest high and lowest low (default is 14).

    Returns:
    pd.Series: A Series containing the Williams %R values.
    """
    # Calculate the highest high and lowest low over the window period
    highest_high = df["High"].rolling(window=window, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=window, min_periods=1).min()

    # Calculate Williams %R
    williams_r = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))

    return williams_r
