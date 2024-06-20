import pandas as pd

def smma(df, column='Close', window=14):
    # Calculate initial SMA for the first window period
    smma_values = df[column].rolling(window=window, min_periods=1).mean()
    
    # Calculate SMMA for subsequent periods
    for i in range(window, len(df)):
        smma_values.iloc[i] = (smma_values.iloc[i-1] * (window - 1) + df[column].iloc[i]) / window
    
    return smma_values

# df['SMMA'] = smma(df)
