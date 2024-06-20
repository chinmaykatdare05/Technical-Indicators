import pandas as pd

def fibonacci_retracement(data):
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    levels = {
        "0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100%": low
    }
    return levels

for level, price in fibonacci_retracement(df).items():
    print(f"Fibonacci Level {level}: {price:.2f}")
