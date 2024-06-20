import pandas as pd

def roc(df, column='Close', window=14):
    # Calculate ROC as percentage change over the specified window
    roc = df[column].pct_change(periods=window) * 100
    
    return roc

# df['ROC'] = roc(df)
