import pandas as pd


def adaptive_moving_average(
    prices: pd.Series, fast_period: int, slow_period: int
) -> pd.Series:
    """
    Calculate the Adaptive Moving Average (AMA) of stock prices.

    The AMA adjusts the smoothing constant based on the volatility of the price changes.

    Parameters:
    prices (pd.Series): Series of stock prices.
    fast_period (int): The fast moving average period.
    slow_period (int): The slow moving average period.

    Returns:
    pd.Series: The computed Adaptive Moving Average for the given prices.
    """
    # Calculate price changes
    price_changes = prices.diff()

    # Calculate volatility as the sum of absolute price changes
    volatility = price_changes.abs().rolling(window=slow_period).sum()

    # Calculate the smoothing constant (scaling factor)
    scaling_factor = (volatility / volatility.rolling(window=fast_period).sum()).fillna(
        0
    )

    # Compute the Adaptive Moving Average
    ama = prices.ewm(span=slow_period, adjust=False).mean()
    ama = ama + scaling_factor * (prices - ama)

    return ama
