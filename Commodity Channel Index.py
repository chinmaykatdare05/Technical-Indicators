import numpy as np


def CCI(df, window_size):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mean_deviation = typical_price.rolling(window=window_size).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    CCI = (typical_price - typical_price.rolling(window=window_size).mean()) / (
        0.015 * mean_deviation
    )
    return CCI


# df['CCI'] = CCI(df, window_size=14)
