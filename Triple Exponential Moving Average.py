import pandas as pd

def trix(df, window=14):
    # Calculate the single, double, and triple EMA
    single_ema = df["Close"].ewm(span=window, min_periods=1, adjust=False).mean()
    double_ema = single_ema.ewm(span=window, min_periods=1, adjust=False).mean()
    triple_ema = double_ema.ewm(span=window, min_periods=1, adjust=False).mean()

    # Calculate the TRIX as the percentage rate of change of the triple EMA
    trix = 100 * (triple_ema.diff() / triple_ema.shift(1))

    return trix

# df['TRIX'] = trix(df)
