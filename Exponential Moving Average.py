import numpy as np
from typing import Union


def ema(data: Union[np.ndarray, list], window_size: int) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) for the given data.

    Parameters:
        data (Union[np.ndarray, list]): Input data (array or list).
        window_size (int): The window size for the EMA calculation.

    Returns:
        np.ndarray: Exponential Moving Average values.
    """
    data = np.asarray(data)
    weights = np.exp(np.linspace(-1.0, 0.0, window_size))
    weights /= weights.sum()

    # Compute the EMA using convolution
    moving_averages = np.convolve(data, weights, mode="full")[: len(data)]

    # Set the first `window_size` values to the first EMA value
    moving_averages[:window_size] = moving_averages[window_size]

    return moving_averages
