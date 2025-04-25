import pandas as pd


def price_oscillator(
    prices: pd.Series, short_period: int, long_period: int
) -> pd.Series:
    """
    Calculate the Price Oscillator (PO) indicator for a given price series.

    The Price Oscillator (PO) is the difference between the short-term EMA and long-term EMA
    expressed as a percentage of the long-term EMA.

    Parameters:
    prices (pd.Series): Series of stock prices (usually closing prices).
    short_period (int): The period for the short-term EMA (e.g., 12).
    long_period (int): The period for the long-term EMA (e.g., 26).

    Returns:
    pd.Series: The Price Oscillator (PO) values.
    """
    # Calculate the short-term and long-term EMAs
    short_ema = prices.ewm(span=short_period, adjust=False).mean()
    long_ema = prices.ewm(span=long_period, adjust=False).mean()

    # Calculate the Price Oscillator (PO)
    po = ((short_ema - long_ema) / long_ema) * 100

    return po
