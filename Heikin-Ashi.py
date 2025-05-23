import pandas as pd


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Heikin-Ashi candlesticks for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
        pd.DataFrame: DataFrame with Heikin-Ashi 'Open', 'High', 'Low', and 'Close' columns.
    """
    # Calculate Heikin-Ashi close
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # Calculate Heikin-Ashi open
    df["HA_Open"] = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
    df["HA_Open"].fillna((df["Open"] + df["Close"]) / 2, inplace=True)

    # Calculate Heikin-Ashi high and low
    df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
    df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)

    return df[["HA_Open", "HA_High", "HA_Low", "HA_Close"]]
