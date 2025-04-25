import pandas as pd


def envelope(prices: pd.Series, period: int, percentage: float) -> pd.DataFrame:
    """
    Calculate the Envelope indicator for a given price series.

    The Envelope consists of two lines: the upper and lower envelopes, which are
    a fixed percentage away from the Simple Moving Average (SMA) of the prices.

    Parameters:
    prices (pd.Series): Series of stock prices (usually closing prices).
    period (int): The period for the Simple Moving Average (SMA).
    percentage (float): The percentage distance of the envelope lines from the SMA.

    Returns:
    pd.DataFrame: A DataFrame containing the Upper and Lower Envelope values.
    """
    # Calculate the Simple Moving Average (SMA) of the prices
    sma = prices.rolling(window=period).mean()

    # Calculate the Upper and Lower Envelopes
    upper_envelope = sma * (1 + percentage / 100)
    lower_envelope = sma * (1 - percentage / 100)

    # Return the envelopes as a DataFrame
    return pd.DataFrame(
        {"Upper Envelope": upper_envelope, "Lower Envelope": lower_envelope}
    )
