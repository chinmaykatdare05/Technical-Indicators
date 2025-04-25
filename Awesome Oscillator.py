import pandas as pd


def awesome_oscillator(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate the Awesome Oscillator (AO) for a given DataFrame of stock prices.

    The Awesome Oscillator is computed as the difference between the 5-period
    and 34-period simple moving averages of the median price, where the median
    price is defined as (High + Low) / 2.

    Parameters:
    prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

    Returns:
    pd.Series: The Awesome Oscillator values.
    """
    # Calculate the median price
    median_price = (prices["High"] + prices["Low"]) / 2

    # Compute the 5-period and 34-period simple moving averages
    sma_5 = median_price.rolling(window=5).mean()
    sma_34 = median_price.rolling(window=34).mean()

    # Calculate the Awesome Oscillator
    ao = sma_5 - sma_34

    return ao
