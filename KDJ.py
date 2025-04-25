import pandas as pd


def kdj(prices: pd.DataFrame, period: int = 14, smooth_period: int = 3) -> pd.DataFrame:
    """
    Calculate the KDJ indicator for a given DataFrame of stock prices.

    The KDJ is based on the Stochastic Oscillator, with three lines:
    - %K: The current position relative to the high-low range.
    - %D: A 3-period moving average of %K.
    - %J: The difference between 3 * %K and 2 * %D.

    Parameters:
    prices (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    period (int): The period for calculating %K (default is 14).
    smooth_period (int): The smoothing period for the %D line (default is 3).

    Returns:
    pd.DataFrame: A DataFrame containing the %K, %D, and %J values.
    """
    # Calculate the lowest low and highest high for the given period
    low_min = prices["Low"].rolling(window=period).min()
    high_max = prices["High"].rolling(window=period).max()

    # Calculate %K
    k = 100 * (prices["Close"] - low_min) / (high_max - low_min)

    # Calculate %D (3-period SMA of %K)
    d = k.rolling(window=smooth_period).mean()

    # Calculate %J
    j = 3 * k - 2 * d

    # Return the result as a DataFrame
    return pd.DataFrame({"%K": k, "%D": d, "%J": j})
