import pandas as pd

def calculate_adl(data):
    data['Money Flow Multiplier'] = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    data['Money Flow Volume'] = data['Money Flow Multiplier'] * data['Volume']
    data['ADL'] = data['Money Flow Volume'].cumsum()
    return data['ADL']

# print(calculate_adl(df))
