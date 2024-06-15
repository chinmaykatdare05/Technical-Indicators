import numpy as np


def sma(data, window_swing_indexze):
    weights = np.repeat(1.0, window_swing_indexze) / window_swing_indexze
    moving_averages = np.convolve(data, weights, mode="valid")
    return moving_averages


# sma(df["Close"], 20)
