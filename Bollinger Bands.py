def bollinger_bands(df, window_size, num_std_dev):
    rolling_mean = df["Close"].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=window_size, min_periods=1).std()

    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)

    return rolling_mean, upper_band, lower_band


# df['Middle Band'], df['Upper Band'], df['Lower Band'] = bollinger_bands(df, window_size = 20, num_std_dev = 2)
