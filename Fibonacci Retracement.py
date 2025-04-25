import pandas as pd
from typing import Dict


def fibonacci_retracement(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the Fibonacci retracement levels for a given DataFrame of stock data.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

    Returns:
        Dict[str, float]: Dictionary containing Fibonacci retracement levels and their corresponding price values.
    """
    high = data["High"].max()
    low = data["Low"].min()
    diff = high - low

    levels = {
        "0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100%": low,
    }

    return levels
