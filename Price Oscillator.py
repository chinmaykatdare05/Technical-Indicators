import pandas as pd

def ppo(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate fast and slow EMAs
    fast_ema = df['Close'].ewm(span=fast_period, min_periods=1, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, min_periods=1, adjust=False).mean()
    
    # Calculate PPO and signal line
    ppo_line = (fast_ema - slow_ema) / slow_ema * 100
    signal_line = ppo_line.ewm(span=signal_period, min_periods=1, adjust=False).mean()
    
    return ppo_line, signal_line

# df['PPO'], df['PPO Signal'] = ppo(df)
