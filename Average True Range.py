import numpy as np


def atr(df, window_size):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window_size, min_periods=1).mean()

    return atr


# df['ATR'] = atr(df, window_size = 14)
