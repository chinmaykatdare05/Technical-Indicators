import numpy as np
from pandas import DataFrame, Series


def cci(df: DataFrame, window_size: int) -> Series:
    """
    Calculate the Commodity Channel Index (CCI).

    Parameters:
        df (DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        window_size (int): The number of periods to use for the CCI calculation.

    Returns:
        Series: CCI values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    typical_price = (high + low + close) / 3
    rolling_mean = typical_price.rolling(window=window_size, min_periods=1).mean()

    # Mean deviation
    mean_deviation = typical_price.rolling(window=window_size, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    cci = (typical_price - rolling_mean) / (0.015 * mean_deviation)

    return cci
