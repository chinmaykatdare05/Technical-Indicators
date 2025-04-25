import pandas as pd


def psychological_line(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Psychological Line (PL) for a given price series.

    The Psychological Line (PL) is calculated based on the current closing price relative
    to the highest high and lowest low over the last n periods.

    Parameters:
    prices (pd.Series): Series of stock prices (usually closing prices).
    period (int): The period for calculating the highest high and lowest low.

    Returns:
    pd.Series: The Psychological Line (PL) values, expressed as percentages.
    """
    # Calculate the highest high and lowest low over the given period
    highest_high = prices.rolling(window=period).max()
    lowest_low = prices.rolling(window=period).min()

    # Calculate the Psychological Line (PL)
    pl = ((prices - lowest_low) / (highest_high - lowest_low)) * 100

    return pl
