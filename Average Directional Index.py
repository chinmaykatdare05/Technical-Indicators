import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def adx(df: DataFrame, window_size: int) -> Series:
    """
    Calculate the Average Directional Index (ADX) for a given DataFrame.

    Parameters:
        df (DataFrame): A DataFrame containing 'High', 'Low', and 'Close' columns.
        window_size (int): The lookback period for the ADX calculation.

    Returns:
        Series: ADX values.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range (TR)
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth the TR and directional movements
    atr = tr.rolling(window=window_size, min_periods=1).mean()
    smoothed_plus_dm = (
        pd.Series(plus_dm, index=df.index)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    smoothed_minus_dm = (
        pd.Series(minus_dm, index=df.index)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # Directional Index (DI)
    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=window_size, min_periods=1).mean()

    return adx
