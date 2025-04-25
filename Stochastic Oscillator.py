import pandas as pd
from typing import Tuple


def stochastic_oscillator(
    df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Stochastic Oscillator (%K and %D) for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'High', 'Low', and 'Close' columns.
    - window (int): The window size for calculating the highest high and lowest low (default is 14).
    - smooth_k (int): The window size for smoothing %K (default is 3).
    - smooth_d (int): The window size for smoothing %D (default is 3).

    Returns:
    - Tuple[pd.Series, pd.Series]: A tuple containing the smoothed %K and %D values as pandas Series.
    """
    # Calculate the highest high and lowest low over the window period
    highest_high = df["High"].rolling(window=window, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=window, min_periods=1).min()

    # Calculate %K
    percent_k = 100 * ((df["Close"] - lowest_low) / (highest_high - lowest_low))

    # Smooth %K (using simple moving average)
    smooth_percent_k = percent_k.rolling(window=smooth_k, min_periods=1).mean()

    # Calculate %D (moving average of %K)
    percent_d = smooth_percent_k.rolling(window=smooth_d, min_periods=1).mean()

    return smooth_percent_k, percent_d
