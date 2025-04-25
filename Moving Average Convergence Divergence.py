import pandas as pd


def macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD), Signal Line, and MACD Histogram.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' column.
        fast_period (int): The period for the fast (short-term) EMA. Default is 12.
        slow_period (int): The period for the slow (long-term) EMA. Default is 26.
        signal_period (int): The period for the signal line EMA. Default is 9.

    Returns:
        tuple: A tuple containing:
            - pd.Series: MACD line
            - pd.Series: Signal line
            - pd.Series: MACD histogram
    """
    # Calculate short-term (fast) EMA
    ema_fast = df["Close"].ewm(span=fast_period, adjust=False).mean()

    # Calculate long-term (slow) EMA
    ema_slow = df["Close"].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line (trigger line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram
