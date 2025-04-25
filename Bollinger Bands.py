from pandas import DataFrame, Series
from typing import Tuple


def bollinger_bands(
    df: DataFrame, window_size: int, num_std_dev: float
) -> Tuple[Series, Series, Series]:
    """
    Calculate Bollinger Bands for a given DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing a 'Close' column.
        window_size (int): The number of periods to use for the moving average and standard deviation.
        num_std_dev (float): The number of standard deviations to determine the width of the bands.

    Returns:
        Tuple[Series, Series, Series]: (Middle Band, Upper Band, Lower Band)
    """
    close = df["Close"]
    rolling_mean = close.rolling(window=window_size, min_periods=1).mean()
    rolling_std = close.rolling(window=window_size, min_periods=1).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return rolling_mean, upper_band, lower_band
