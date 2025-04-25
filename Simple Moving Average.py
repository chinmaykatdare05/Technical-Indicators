import numpy as np


def sma(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA) of a given data series over a specified window size.

    Parameters:
    - data (np.ndarray): The input data array (typically a column from a pandas DataFrame).
    - window_size (int): The number of periods over which to calculate the moving average.

    Returns:
    - np.ndarray: The calculated Simple Moving Averages.
    """
    weights = np.repeat(1.0, window_size) / window_size
    moving_averages = np.convolve(data, weights, mode="valid")
    return moving_averages
