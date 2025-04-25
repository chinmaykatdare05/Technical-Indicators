import pandas as pd
from pandas import DataFrame, Series


def chaikin_money_flow(df: DataFrame, window_size: int) -> Series:
    """
    Calculate the Chaikin Money Flow (CMF) for a given DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.
        window_size (int): The lookback period for the CMF calculation.

    Returns:
        Series: Chaikin Money Flow values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Avoid division by zero by replacing zero range with NaN
    hl_range = high - low
    hl_range = hl_range.replace(0, pd.NA)

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / hl_range

    # Money Flow Volume
    mfv = mfm * volume

    # Chaikin Money Flow
    cmf = (
        mfv.rolling(window=window_size, min_periods=1).sum()
        / volume.rolling(window=window_size, min_periods=1).sum()
    )

    return cmf
