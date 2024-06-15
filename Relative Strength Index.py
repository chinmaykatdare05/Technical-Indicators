import numpy as np


def rsi(df, window=14):
    # Calculate price differences
    delta = df["Close"].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss.replace(to_replace=0, method="ffill").replace(0, np.nan)

    # Calculate RSI directly from RS using vectorized operations
    rsi = 100 - (100 / (1 + rs))

    return rsi


# df['RSI'] = rsi(df, window=14)
