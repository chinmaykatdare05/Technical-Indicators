import pandas as pd
import numpy as np


def rsi(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for the specified column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing stock price data.
    - column (str): The column name to calculate RSI on (default is 'Close').
    - window (int): The window size for calculating RSI (default is 14).

    Returns:
    - pd.Series: The calculated RSI values.
    """
    # Calculate price differences
    delta = df[column].diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate average gains and losses using exponential moving average
    avg_gain = gain.ewm(span=window, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=window, min_periods=1, adjust=False).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Identify RSI divergence (bullish or bearish) in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing stock price data and RSI.
    - window (int): The window size for RSI calculation (default is 14).

    Returns:
    - pd.DataFrame: The original DataFrame with an added 'RSI Divergence' column.
    """
    # Calculate RSI for the DataFrame
    df["RSI"] = rsi(df, window)

    # Initialize 'RSI Divergence' column
    df["RSI Divergence"] = "None"

    # Detect peaks and troughs in price and RSI
    df["Price Peak"] = (df["Close"] > df["Close"].shift(1)) & (
        df["Close"] > df["Close"].shift(-1)
    )
    df["RSI Peak"] = (df["RSI"] > df["RSI"].shift(1)) & (
        df["RSI"] > df["RSI"].shift(-1)
    )
    df["Price Trough"] = (df["Close"] < df["Close"].shift(1)) & (
        df["Close"] < df["Close"].shift(-1)
    )
    df["RSI Trough"] = (df["RSI"] < df["RSI"].shift(1)) & (
        df["RSI"] < df["RSI"].shift(-1)
    )

    # Forward-fill to align peaks and troughs for divergence detection
    df[["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"]] = (
        df[["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"]]
        .replace(False, np.nan)
        .ffill()
    )

    # Detect bearish divergence
    bearish_divergence = (df["Price Peak"] > df["Price Peak"].shift(1)) & (
        df["RSI Peak"] < df["RSI Peak"].shift(1)
    )
    df.loc[bearish_divergence, "RSI Divergence"] = "Bearish"

    # Detect bullish divergence
    bullish_divergence = (df["Price Trough"] < df["Price Trough"].shift(1)) & (
        df["RSI Trough"] > df["RSI Trough"].shift(1)
    )
    df.loc[bullish_divergence, "RSI Divergence"] = "Bullish"

    # Clean up temporary columns
    df.drop(
        ["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"], axis=1, inplace=True
    )

    return df
