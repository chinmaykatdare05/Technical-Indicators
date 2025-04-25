import pandas as pd


def rsi(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a specified column over a given window.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing stock prices or other time series data.
    - column (str): The column name to calculate RSI on (default is 'Close').
    - window (int): The window size for calculating RSI (default is 14).

    Returns:
    - pd.Series: The calculated RSI values.
    """
    # Calculate price differences
    delta = df[column].diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate average gains and losses over the specified window using exponential moving average
    avg_gain = gain.ewm(span=window, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=window, min_periods=1, adjust=False).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi
