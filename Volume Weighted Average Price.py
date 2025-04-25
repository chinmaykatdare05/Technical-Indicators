import pandas as pd


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Volume Weighted Average Price (VWAP).

    The VWAP is the average price of an asset, weighted by volume, over a specified time period.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close', 'High', 'Low', and 'Volume' columns.

    Returns:
    pd.DataFrame: The original DataFrame with an additional 'VWAP' column containing the VWAP values.
    """
    # Calculate the Typical Price (TP)
    df["Typical Price"] = (df["Close"] + df["High"] + df["Low"]) / 3

    # Calculate the cumulative sum of the Typical Price * Volume (TPV) and Volume
    df["Cumulative TPV"] = (df["Typical Price"] * df["Volume"]).cumsum()
    df["Cumulative Volume"] = df["Volume"].cumsum()

    # Calculate the VWAP
    df["VWAP"] = df["Cumulative TPV"] / df["Cumulative Volume"]

    return df
