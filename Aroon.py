import numpy as np
from pandas import DataFrame
from typing import Tuple


def aroon(df: DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Aroon Up and Aroon Down indicators.

    Parameters:
        df (DataFrame): A DataFrame containing 'High' and 'Low' columns.
        window_size (int): The period over which to calculate the Aroon indicators.

    Returns:
        Tuple of np.ndarrays: (Aroon Up, Aroon Down)
    """
    n = len(df)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)

    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    for i in range(window_size, n):
        high_window = highs[i - window_size : i]
        low_window = lows[i - window_size : i]

        days_since_high = window_size - np.argmax(high_window)
        days_since_low = window_size - np.argmin(low_window)

        aroon_up[i] = (days_since_high / window_size) * 100
        aroon_down[i] = (days_since_low / window_size) * 100

    return aroon_up, aroon_down


def aroon_oscillator(df: DataFrame, window_size: int) -> np.ndarray:
    """
    Calculate the Aroon Oscillator.

    Parameters:
        df (DataFrame): A DataFrame containing 'High' and 'Low' columns.
        window_size (int): The period over which to calculate the Aroon oscillator.

    Returns:
        np.ndarray: Aroon Oscillator values.
    """
    aroon_up, aroon_down = aroon(df, window_size)
    return aroon_up - aroon_down
