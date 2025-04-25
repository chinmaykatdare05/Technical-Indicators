import pandas as pd


def modified_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Modified Moving Average (MMA) for a given price series.

    The Modified Moving Average (MMA) gives more weight to recent prices and is
    computed similarly to an Exponential Moving Average (EMA).

    Parameters:
    prices (pd.Series): Series of stock prices (usually closing prices).
    period (int): The period for calculating the MMA.

    Returns:
    pd.Series: The Modified Moving Average (MMA) values.
    """
    # Calculate the alpha (smoothing factor)
    alpha = 2 / (period + 1)

    # Initialize the MMA with the first price (can also be the first SMA)
    mma = pd.Series(index=prices.index, dtype=float)
    mma.iloc[0] = prices.iloc[0]  # Set the first value of MMA as the first price

    # Calculate the MMA using the recursive formula
    for t in range(1, len(prices)):
        mma.iloc[t] = mma.iloc[t - 1] + alpha * (prices.iloc[t] - mma.iloc[t - 1])

    return mma
