import pandas as pd
from pandas import DataFrame, Series


def atr(df: DataFrame, window_size: int) -> Series:
    """
    Calculate the Average True Range (ATR) for a given DataFrame.

    Parameters:
        df (DataFrame): A DataFrame containing 'High', 'Low', and 'Close' columns.
        window_size (int): The lookback period for the ATR calculation.

    Returns:
        Series: ATR values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window_size, min_periods=1).mean()

    return atr
