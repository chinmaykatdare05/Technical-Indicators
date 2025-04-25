import pandas as pd


def chaikin_volatility(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate the Chaikin Volatility (CV) for a given DataFrame of stock prices.

    The Chaikin Volatility is computed as the difference between the 3-day and
    10-day exponential moving averages (EMAs) of the difference between the high and low prices.

    Parameters:
    prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

    Returns:
    pd.Series: The Chaikin Volatility values.
    """
    # Calculate the difference between the High and Low prices
    price_range = prices["High"] - prices["Low"]

    # Compute the 3-day and 10-day EMAs of the price range
    ema_3 = price_range.ewm(span=3, adjust=False).mean()
    ema_10 = price_range.ewm(span=10, adjust=False).mean()

    # Calculate the Chaikin Volatility
    cv = ema_3 - ema_10

    return cv
