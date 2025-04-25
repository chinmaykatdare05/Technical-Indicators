import pandas as pd


def stoch_rsi(df: pd.DataFrame, n: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """
    Calculate the Stochastic Relative Strength Index (Stochastic RSI) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the 'Close' prices.
    - n (int): The window size for calculating the RSI and Stochastic RSI (default is 14).
    - k (int): The window size for smoothing %K (default is 3).
    - d (int): The window size for smoothing %D (default is 3).

    Returns:
    - pd.DataFrame: A DataFrame containing the RSI, %K, and %D values.
    """
    # Calculate the price change
    delta = df["Close"].diff(1)

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(window=n, min_periods=1).min()) / (
        rsi.rolling(window=n, min_periods=1).max()
        - rsi.rolling(window=n, min_periods=1).min()
    )

    # Smooth %K and %D
    stoch_rsi_k = stoch_rsi.rolling(window=k, min_periods=1).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d, min_periods=1).mean()

    # Add RSI, %K, and %D to the DataFrame
    df["RSI"] = rsi
    df["%K"] = stoch_rsi_k * 100
    df["%D"] = stoch_rsi_d * 100

    return df[["RSI", "%K", "%D"]]
