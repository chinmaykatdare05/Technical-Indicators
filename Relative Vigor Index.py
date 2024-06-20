import pandas as pd
import numpy as np

def rvi(df, n=10):
    # Calculate the numerator (close - open)
    numerator = df['Close'] - df['Low']

    # Calculate the denominator (high - low)
    denominator = df['High'] - df['Low']

    # Smooth the numerator and denominator using a simple moving average
    numerator_sma = numerator.rolling(window=n).mean()
    denominator_sma = denominator.rolling(window=n).mean()

    # Calculate the RVI
    rvi = numerator_sma / denominator_sma
    return rvi

print(rvi(df))
