import pandas as pd
import numpy as np


def keltner_channels(
    df: pd.DataFrame, window_size: int = 20, multiplier: float = 2, ema_window: int = 20
) -> pd.DataFrame:
    """
    Calculate the Keltner Channels for the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window_size (int): The window size for the ATR calculation. Default is 20.
        multiplier (float): The multiplier for the ATR to calculate the upper and lower bands. Default is 2.
        ema_window (int): The window size for the EMA of the Typical Price. Default is 20.

    Returns:
        pd.DataFrame: DataFrame with Keltner Channel components: 'Middle_Line', 'Upper_Band', 'Lower_Band', 'Typical_Price', 'ATR'.
    """
    # Calculate Typical Price
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Calculate Middle Line (EMA of Typical Price)
    df["Middle_Line"] = (
        df["Typical_Price"].ewm(span=ema_window, min_periods=ema_window).mean()
    )

    # Calculate True Range (TR)
    df["TR"] = np.maximum.reduce(
        [
            df["High"] - df["Low"],
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift()),
        ]
    )

    # Calculate Average True Range (ATR)
    df["ATR"] = df["TR"].ewm(span=window_size, min_periods=window_size).mean()

    # Calculate Upper and Lower Bands
    df["Upper_Band"] = df["Middle_Line"] + multiplier * df["ATR"]
    df["Lower_Band"] = df["Middle_Line"] - multiplier * df["ATR"]

    return df[["Typical_Price", "Middle_Line", "Upper_Band", "Lower_Band", "ATR"]]
