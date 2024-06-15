def chande_momentum_oscillator(df, window_size):
    close = df["Close"]

    # Calculate the differences in the closing prices
    diff = close.diff()

    # Calculate the sum of gains and losses
    gain = diff.where(diff > 0, 0).rolling(window=window_size, min_periods=1).sum()
    loss = -diff.where(diff < 0, 0).rolling(window=window_size, min_periods=1).sum()

    # Calculate the CMO
    cmo = 100 * (gain - loss) / (gain + loss)

    return cmo


# df['CMO'] = chande_momentum_oscillator(df, window_size=10)
