import pandas as pd
import numpy as np

def envelope(df, window=20, percent=5):
    # Calculate the Simple Moving Average (SMA)
    sma = df['Close'].rolling(window=window).mean()

    # Calculate upper and lower bands
    upper_band = sma * (1 + percent/100)
    lower_band = sma * (1 - percent/100)
    return upper_band, lower_band

# df['Upper Band'], df['Lower Band'] = envelope(df)
