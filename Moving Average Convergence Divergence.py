import pandas as pd

def macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate short-term (fast) EMA
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()

    # Calculate long-term (slow) EMA
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line (trigger line)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram

# df['MACD Line'], df['Signal Line'], df['MACD Histogram'] = macd(df)
