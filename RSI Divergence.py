import pandas as pd
import numpy as np

def rsi(df, window=14):
    # Calculate price differences
    delta = df["Close"].diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gains and losses over the specified window using exponential moving average for better performance
    avg_gain = gain.ewm(span=window, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(span=window, min_periods=1, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_divergence(df, window=14):
    df['RSI'] = rsi(df, window)
    
    # Initialize divergence column
    df['RSI Divergence'] = 'None'
    
    # Calculate peaks and troughs
    df['Price Peak'] = (df['Close'] > df['Close'].shift(1)) & (df['Close'] > df['Close'].shift(-1))
    df['RSI Peak'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'] > df['RSI'].shift(-1))
    
    df['Price Trough'] = (df['Close'] < df['Close'].shift(1)) & (df['Close'] < df['Close'].shift(-1))
    df['RSI Trough'] = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'] < df['RSI'].shift(-1))
    
    # Forward-fill peaks and troughs to align with the divergence detection logic
    df[['Price Peak', 'RSI Peak', 'Price Trough', 'RSI Trough']] = df[['Price Peak', 'RSI Peak', 'Price Trough', 'RSI Trough']].replace(False, np.nan).ffill()
    
    # Detect bearish divergence
    bearish_divergence = (df['Price Peak'] > df['Price Peak'].shift(1)) & (df['RSI Peak'] < df['RSI Peak'].shift(1))
    df.loc[bearish_divergence, 'RSI Divergence'] = 'Bearish'
    
    # Detect bullish divergence
    bullish_divergence = (df['Price Trough'] < df['Price Trough'].shift(1)) & (df['RSI Trough'] > df['RSI Trough'].shift(1))
    df.loc[bullish_divergence, 'RSI Divergence'] = 'Bullish'

    df.drop(['Price Peak', 'RSI Peak', 'Price Trough', 'RSI Trough'], inplace=True)
    return df

df['RSI'] = rsi(df)
df = rsi_divergence(df)
