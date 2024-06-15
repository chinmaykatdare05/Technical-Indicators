import numpy as np


def keltner_channels(df, window_size=20, multiplier=2, ema_window=20):
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

    return df


# df = keltner_channels(df)
