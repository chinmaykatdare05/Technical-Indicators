import pandas as pd


def rvi(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    Calculate the Relative Volatility Index (RVI) for the given DataFrame over a specified window.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'Close', 'High', and 'Low' columns.
    - n (int): The window size for the simple moving averages (default is 10).

    Returns:
    - pd.Series: The calculated RVI values.
    """
    # Calculate the numerator (close - low)
    numerator = df["Close"] - df["Low"]

    # Calculate the denominator (high - low)
    denominator = df["High"] - df["Low"]

    # Smooth the numerator and denominator using a simple moving average
    numerator_sma = numerator.rolling(window=n).mean()
    denominator_sma = denominator.rolling(window=n).mean()

    # Calculate the RVI
    rvi = numerator_sma / denominator_sma
    return rvi
