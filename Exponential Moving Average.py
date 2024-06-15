import numpy as np


def ema(data, window_swing_indexze):
    weights = np.exp(np.linspace(-1.0, 0.0, window_swing_indexze))
    weights /= weights.sum()
    moving_averages = np.convolve(data, weights)[: len(data)]
    moving_averages[:window_swing_indexze] = moving_averages[window_swing_indexze]
    return moving_averages


# ema(df["Close"], 20)
