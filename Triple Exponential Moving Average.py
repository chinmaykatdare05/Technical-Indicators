import pandas as pd


def tema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Triple Exponential Moving Average (TEMA) for a given price series.

    The TEMA is an advanced version of the Exponential Moving Average (EMA) that is
    more responsive to price changes. It is calculated by applying multiple EMAs to the
    data.

    Parameters:
    prices (pd.Series): Series of stock prices (usually closing prices).
    period (int): The period for calculating the EMA (e.g., 14).

    Returns:
    pd.Series: The TEMA values.
    """
    # Calculate the first EMA (EMA_1)
    ema_1 = prices.ewm(span=period, adjust=False).mean()

    # Calculate the second EMA (EMA_2) of the first EMA (EMA_1)
    ema_2 = ema_1.ewm(span=period, adjust=False).mean()

    # Calculate the third EMA (EMA_3) of the second EMA (EMA_2)
    ema_3 = ema_2.ewm(span=period, adjust=False).mean()

    # Calculate the TEMA
    tema = 3 * ema_1 - 3 * ema_2 + ema_3

    return tema
