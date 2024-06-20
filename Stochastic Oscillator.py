import pandas as pd

def stochastic_oscillator(df, window=14, smooth_k=3, smooth_d=3):
    # Calculate the highest high and lowest low over the window period
    highest_high = df['High'].rolling(window=window, min_periods=1).max()
    lowest_low = df['Low'].rolling(window=window, min_periods=1).min()
    
    # Calculate %K
    percent_k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # Smooth %K (using simple moving average)
    smooth_percent_k = percent_k.rolling(window=smooth_k, min_periods=1).mean()
    
    # Calculate %D (moving average of %K)
    percent_d = smooth_percent_k.rolling(window=smooth_d, min_periods=1).mean()
    
    return smooth_percent_k, percent_d

# df['%K'], df['%D'] = stochastic_oscillator(df)
