import pandas as pd


def chaikin_oscillator(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate the Chaikin Oscillator (CO) for a given DataFrame of stock prices.

    The Chaikin Oscillator is computed as the difference between the 3-day and
    10-day exponential moving averages (EMAs) of the Accumulation/Distribution Line (ADL).

    Parameters:
    prices (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.

    Returns:
    pd.Series: The Chaikin Oscillator values.
    """
    # Calculate the Money Flow Multiplier (MFM)
    mfm = ((prices["Close"] - prices["Low"]) - (prices["High"] - prices["Close"])) / (
        prices["High"] - prices["Low"]
    )

    # Compute the Money Flow Volume (MFV)
    mfv = mfm * prices["Volume"]

    # Calculate the Accumulation/Distribution Line (ADL)
    adl = mfv.cumsum()

    # Compute the 3-day and 10-day EMAs of the ADL
    ema_3 = adl.ewm(span=3, adjust=False).mean()
    ema_10 = adl.ewm(span=10, adjust=False).mean()

    # Calculate the Chaikin Oscillator
    co = ema_3 - ema_10

    return co
