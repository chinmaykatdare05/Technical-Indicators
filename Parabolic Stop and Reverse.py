import pandas as pd

def calculate_psar(high, low, close, initial_af=0.02, max_af=0.2, increment=0.02):
    length = len(close)
    psar = close.copy()
    bull = True
    af = initial_af
    ep = low.iloc[0]
    sar = high.iloc[0]
    
    for i in range(1, length):
        previous_sar = sar
        if bull:
            sar = sar + af * (ep - sar)
            if low.iloc[i] < sar:
                bull = False
                sar = ep
                ep = low.iloc[i]
                af = initial_af
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + increment, max_af)
        else:
            sar = sar + af * (ep - sar)
            if high.iloc[i] > sar:
                bull = True
                sar = ep
                ep = high.iloc[i]
                af = initial_af
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + increment, max_af)

        psar.iloc[i] = sar if bull else previous_sar

    return psar

print(calculate_psar(data['high'], data['low'], data['close']))
