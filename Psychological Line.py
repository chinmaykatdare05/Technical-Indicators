import pandas as pd

def psy(df, window=14):
    # Calculate the number of positive changes (bullish days)
    df['Change'] = df['Close'].diff()
    df['Up'] = df['Change'].apply(lambda x: 1 if x > 0 else 0)
    
    # Calculate PSY as the percentage of bullish days over the window period
    psy = df['Up'].rolling(window=window, min_periods=1).sum() / window * 100
    
    return psy

# df['PSY'] = psy(df)
