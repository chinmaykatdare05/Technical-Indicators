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

# df['RSI'] = rsi(df, window=14)
