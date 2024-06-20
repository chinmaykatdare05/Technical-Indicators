import pandas as pd

def momentum(df, period=14):
    # Calculate the difference in closing prices
    return df['Close'].diff(period)

# df['Momentum'] = momentum(df)
