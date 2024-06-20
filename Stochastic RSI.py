import pandas as pd
import numpy as np

def stoch_rsi(df, n=14, k=3, d=3):
    # Calculate the Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(window=n, min_periods=1).min()) / \
                (rsi.rolling(window=n, min_periods=1).max() - rsi.rolling(window=n, min_periods=1).min())

    stoch_rsi_k = stoch_rsi.rolling(window=k, min_periods=1).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d, min_periods=1).mean()
    
    df['RSI'] = rsi
    df['%K'] = stoch_rsi_k * 100
    df['%D'] = stoch_rsi_d * 100
    
    return df[['RSI', '%K', '%D']]

print(stoch_rsi(df))
