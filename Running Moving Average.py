import pandas as pd

def rma(df, column='Close', alpha=0.2):
    rma_values = pd.Series(index=df.index, dtype=float)
    
    # Initialize the first value of RMA as the first price
    rma_values.iloc[0] = df[column].iloc[0]
    
    # Calculate RMA for subsequent periods
    for i in range(1, len(df)):
        rma_values.iloc[i] = alpha * df[column].iloc[i] + (1 - alpha) * rma_values.iloc[i-1]
    
    return rma_values

# df['RMA'] = rma(df, column='Close', alpha=0.2)
