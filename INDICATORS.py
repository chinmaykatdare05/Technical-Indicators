import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
from typing import Tuple, Union, Dict


class Indicators:
    """
    A collection of technical indicators for stock analysis.
    """

    def calculate_adl(data: DataFrame) -> Series:
        """
        Calculate the Accumulation Distribution Line (ADL) for a given DataFrame.
        The ADL is a volume-based indicator that measures the cumulative flow of money into and out of a security.
        It is calculated using the following formula:
        ADL = Previous ADL + Money Flow Volume
        where:
        Money Flow Volume = Money Flow Multiplier * Volume
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        """
        required_cols = {"High", "Low", "Close", "Volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        high_low_range = data["High"] - data["Low"]
        high_low_range = high_low_range.replace(0, 1e-10)  # prevent division by zero

        money_flow_multiplier = (
            (data["Close"] - data["Low"]) - (data["High"] - data["Close"])
        ) / high_low_range
        money_flow_volume = money_flow_multiplier * data["Volume"]
        adl = money_flow_volume.cumsum()

        return adl

    def adaptive_moving_average(
        prices: Series, fast_period: int, slow_period: int
    ) -> Series:
        """
        Calculate the Adaptive Moving Average (AMA) of stock prices.

        The AMA adjusts the smoothing constant based on the volatility of the price changes.

        Parameters:
        prices (pd.Series): Series of stock prices.
        fast_period (int): The fast moving average period.
        slow_period (int): The slow moving average period.

        Returns:
        pd.Series: The computed Adaptive Moving Average for the given prices.
        """
        # Calculate price changes
        price_changes = prices.diff()

        # Calculate volatility as the sum of absolute price changes
        volatility = price_changes.abs().rolling(window=slow_period).sum()

        # Calculate the smoothing constant (scaling factor)
        scaling_factor = (
            volatility / volatility.rolling(window=fast_period).sum()
        ).fillna(0)

        # Compute the Adaptive Moving Average
        ama = prices.ewm(span=slow_period, adjust=False).mean()
        ama = ama + scaling_factor * (prices - ama)

        return ama

    def aroon(df: DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Aroon Up and Aroon Down indicators.

        Parameters:
            df (DataFrame): A DataFrame containing 'High' and 'Low' columns.
            window_size (int): The period over which to calculate the Aroon indicators.

        Returns:
            Tuple of np.ndarrays: (Aroon Up, Aroon Down)
        """
        n = len(df)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()

        for i in range(window_size, n):
            high_window = highs[i - window_size : i]
            low_window = lows[i - window_size : i]

            days_since_high = window_size - np.argmax(high_window)
            days_since_low = window_size - np.argmin(low_window)

            aroon_up[i] = (days_since_high / window_size) * 100
            aroon_down[i] = (days_since_low / window_size) * 100

        return aroon_up, aroon_down

    def aroon_oscillator(df: DataFrame, window_size: int) -> np.ndarray:
        """
        Calculate the Aroon Oscillator.

        Parameters:
            df (DataFrame): A DataFrame containing 'High' and 'Low' columns.
            window_size (int): The period over which to calculate the Aroon oscillator.

        Returns:
            np.ndarray: Aroon Oscillator values.
        """
        aroon_up, aroon_down = Indicators.aroon(df, window_size)
        return aroon_up - aroon_down

    def adx(df: DataFrame, window_size: int) -> Series:
        """
        Calculate the Average Directional Index (ADX) for a given DataFrame.

        Parameters:
            df (DataFrame): A DataFrame containing 'High', 'Low', and 'Close' columns.
            window_size (int): The lookback period for the ADX calculation.

        Returns:
            Series: ADX values.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range (TR)
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth the TR and directional movements
        atr = tr.rolling(window=window_size, min_periods=1).mean()
        smoothed_plus_dm = (
            pd.Series(plus_dm, index=df.index)
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        smoothed_minus_dm = (
            pd.Series(minus_dm, index=df.index)
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        # Directional Index (DI)
        plus_di = 100 * (smoothed_plus_dm / atr)
        minus_di = 100 * (smoothed_minus_dm / atr)

        # DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=window_size, min_periods=1).mean()

        return adx

    def atr(df: DataFrame, window_size: int) -> Series:
        """
        Calculate the Average True Range (ATR) for a given DataFrame.

        Parameters:
            df (DataFrame): A DataFrame containing 'High', 'Low', and 'Close' columns.
            window_size (int): The lookback period for the ATR calculation.

        Returns:
            Series: ATR values.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window_size, min_periods=1).mean()

        return atr

    def awesome_oscillator(prices: pd.DataFrame) -> pd.Series:
        """
        Calculate the Awesome Oscillator (AO) for a given DataFrame of stock prices.

        The Awesome Oscillator is computed as the difference between the 5-period
        and 34-period simple moving averages of the median price, where the median
        price is defined as (High + Low) / 2.

        Parameters:
        prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

        Returns:
        pd.Series: The Awesome Oscillator values.
        """
        # Calculate the median price
        median_price = (prices["High"] + prices["Low"]) / 2

        # Compute the 5-period and 34-period simple moving averages
        sma_5 = median_price.rolling(window=5).mean()
        sma_34 = median_price.rolling(window=34).mean()

        # Calculate the Awesome Oscillator
        ao = sma_5 - sma_34

        return ao

    def black_scholes(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> float:
        """
        Calculate the Black-Scholes option price for a European call or put.

        Parameters:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility of the underlying asset
            option_type (str): 'call' or 'put'

        Returns:
            float: Option price
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            raise ValueError(
                "Inputs must be positive and time to expiration must be > 0"
            )

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == "put":
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def get_black_scholes_params(
        ticker: str,
        start_date: str,
        end_date: str,
        strike_price: float,
        risk_free_rate: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Retrieve required parameters for Black-Scholes model using historical data.

        Parameters:
            ticker (str): Stock symbol
            start_date (str): Format 'YYYY-MM-DD'
            end_date (str): Format 'YYYY-MM-DD'
            strike_price (float): Option strike price
            risk_free_rate (float): Annual risk-free interest rate (e.g., 0.03 for 3%)

        Returns:
            Tuple: (S, K, T, r, sigma)
        """
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock_data.empty:
            raise ValueError(f"No data found for ticker '{ticker}' in the given range.")

        # Current price (last available adjusted close)
        S = stock_data["Adj Close"].iloc[-1]

        # Time to expiration (in years)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        T = (end_dt - start_dt).days / 365.0

        # Volatility (annualized std dev of daily returns)
        returns = stock_data["Adj Close"].pct_change().dropna()
        sigma = returns.std() * np.sqrt(252)

        return S, strike_price, T, risk_free_rate, sigma

    def bollinger_bands(
        df: DataFrame, window_size: int, num_std_dev: float
    ) -> Tuple[Series, Series, Series]:
        """
        Calculate Bollinger Bands for a given DataFrame.

        Parameters:
            df (DataFrame): DataFrame containing a 'Close' column.
            window_size (int): The number of periods to use for the moving average and standard deviation.
            num_std_dev (float): The number of standard deviations to determine the width of the bands.

        Returns:
            Tuple[Series, Series, Series]: (Middle Band, Upper Band, Lower Band)
        """
        close = df["Close"]
        rolling_mean = close.rolling(window=window_size, min_periods=1).mean()
        rolling_std = close.rolling(window=window_size, min_periods=1).std()

        upper_band = rolling_mean + (rolling_std * num_std_dev)
        lower_band = rolling_mean - (rolling_std * num_std_dev)

        return rolling_mean, upper_band, lower_band

    def chaikin_money_flow(df: DataFrame, window_size: int) -> Series:
        """
        Calculate the Chaikin Money Flow (CMF) for a given DataFrame.

        Parameters:
            df (DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.
            window_size (int): The lookback period for the CMF calculation.

        Returns:
            Series: Chaikin Money Flow values.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        volume = df["Volume"]

        # Avoid division by zero by replacing zero range with NaN
        hl_range = high - low
        hl_range = hl_range.replace(0, pd.NA)

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / hl_range

        # Money Flow Volume
        mfv = mfm * volume

        # Chaikin Money Flow
        cmf = (
            mfv.rolling(window=window_size, min_periods=1).sum()
            / volume.rolling(window=window_size, min_periods=1).sum()
        )

        return cmf

    def chaikin_oscillator(prices: pd.DataFrame) -> pd.Series:
        """
        Calculate the Chaikin Oscillator (CO) for a given DataFrame of stock prices.

        The Chaikin Oscillator is computed as the difference between the 3-day and
        10-day exponential moving averages (EMAs) of the Accumulation/Distribution Line (ADL).

        Parameters:
        prices (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.

        Returns:
        pd.Series: The Chaikin Oscillator values.
        """
        # Calculate the Money Flow Multiplier (MFM)
        mfm = (
            (prices["Close"] - prices["Low"]) - (prices["High"] - prices["Close"])
        ) / (prices["High"] - prices["Low"])

        # Compute the Money Flow Volume (MFV)
        mfv = mfm * prices["Volume"]

        # Calculate the Accumulation/Distribution Line (ADL)
        adl = mfv.cumsum()

        # Compute the 3-day and 10-day EMAs of the ADL
        ema_3 = adl.ewm(span=3, adjust=False).mean()
        ema_10 = adl.ewm(span=10, adjust=False).mean()

        # Calculate the Chaikin Oscillator
        co = ema_3 - ema_10

        return co

    def chaikin_volatility(prices: pd.DataFrame) -> pd.Series:
        """
        Calculate the Chaikin Volatility (CV) for a given DataFrame of stock prices.

        The Chaikin Volatility is computed as the difference between the 3-day and
        10-day exponential moving averages (EMAs) of the difference between the high and low prices.

        Parameters:
        prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

        Returns:
        pd.Series: The Chaikin Volatility values.
        """
        # Calculate the difference between the High and Low prices
        price_range = prices["High"] - prices["Low"]

        # Compute the 3-day and 10-day EMAs of the price range
        ema_3 = price_range.ewm(span=3, adjust=False).mean()
        ema_10 = price_range.ewm(span=10, adjust=False).mean()

        # Calculate the Chaikin Volatility
        cv = ema_3 - ema_10

        return cv

    def cci(df: DataFrame, window_size: int) -> Series:
        """
        Calculate the Commodity Channel Index (CCI).

        Parameters:
            df (DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
            window_size (int): The number of periods to use for the CCI calculation.

        Returns:
            Series: CCI values.
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        typical_price = (high + low + close) / 3
        rolling_mean = typical_price.rolling(window=window_size, min_periods=1).mean()

        # Mean deviation
        mean_deviation = typical_price.rolling(window=window_size, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        cci = (typical_price - rolling_mean) / (0.015 * mean_deviation)

        return cci

    def directional_movement(
        df: DataFrame, period: int = 14
    ) -> Tuple[Series, Series, Series]:
        """
        Calculate the Directional Movement (+DI, -DI) and Average Directional Index (ADX).

        Parameters:
            df (DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
            period (int): The period for calculating the smoothed ATR and ADX. Default is 14.

        Returns:
            Tuple[Series, Series, Series]: (+DI, -DI, ADX)
        """
        # Calculate differences between high and low prices
        high_diff = df["High"].diff()
        low_diff = -df["Low"].diff()

        # Calculate True Range (TR)
        TR = np.maximum(high_diff, low_diff)

        # Calculate Directional Movement (+DM, -DM)
        DMplus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        DMminus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate smoothed True Range (ATR), +DM, and -DM
        ATR = TR.rolling(window=period, min_periods=1).mean()
        ADMplus = DMplus.rolling(window=period, min_periods=1).mean()
        ADMminus = DMminus.rolling(window=period, min_periods=1).mean()

        # Calculate +DI and -DI
        plus_DI = (ADMplus / ATR) * 100
        minus_DI = (ADMminus / ATR) * 100

        # Calculate DX and ADX
        DX = np.abs(plus_DI - minus_DI) / (plus_DI + minus_DI) * 100
        ADX = DX.rolling(window=period, min_periods=1).mean()

        return plus_DI, minus_DI, ADX

    def envelope(prices: pd.Series, period: int, percentage: float) -> pd.DataFrame:
        """
        Calculate the Envelope indicator for a given price series.

        The Envelope consists of two lines: the upper and lower envelopes, which are
        a fixed percentage away from the Simple Moving Average (SMA) of the prices.

        Parameters:
        prices (pd.Series): Series of stock prices (usually closing prices).
        period (int): The period for the Simple Moving Average (SMA).
        percentage (float): The percentage distance of the envelope lines from the SMA.

        Returns:
        pd.DataFrame: A DataFrame containing the Upper and Lower Envelope values.
        """
        # Calculate the Simple Moving Average (SMA) of the prices
        sma = prices.rolling(window=period).mean()

        # Calculate the Upper and Lower Envelopes
        upper_envelope = sma * (1 + percentage / 100)
        lower_envelope = sma * (1 - percentage / 100)

        # Return the envelopes as a DataFrame
        return pd.DataFrame(
            {"Upper Envelope": upper_envelope, "Lower Envelope": lower_envelope}
        )

    def ema(data: Union[np.ndarray, list], window_size: int) -> np.ndarray:
        """
        Calculate the Exponential Moving Average (EMA) for the given data.

        Parameters:
            data (Union[np.ndarray, list]): Input data (array or list).
            window_size (int): The window size for the EMA calculation.

        Returns:
            np.ndarray: Exponential Moving Average values.
        """
        data = np.asarray(data)
        weights = np.exp(np.linspace(-1.0, 0.0, window_size))
        weights /= weights.sum()

        # Compute the EMA using convolution
        moving_averages = np.convolve(data, weights, mode="full")[: len(data)]

        # Set the first `window_size` values to the first EMA value
        moving_averages[:window_size] = moving_averages[window_size]

        return moving_averages

    def fibonacci_retracement(data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the Fibonacci retracement levels for a given DataFrame of stock data.

        Parameters:
            data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.

        Returns:
            Dict[str, float]: Dictionary containing Fibonacci retracement levels and their corresponding price values.
        """
        high = data["High"].max()
        low = data["Low"].min()
        diff = high - low

        levels = {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100%": low,
        }

        return levels

    def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candlesticks for the given DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: DataFrame with Heikin-Ashi 'Open', 'High', 'Low', and 'Close' columns.
        """
        # Calculate Heikin-Ashi close
        df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

        # Calculate Heikin-Ashi open
        df["HA_Open"] = (df["Open"].shift(1) + df["Close"].shift(1)) / 2
        df["HA_Open"].fillna((df["Open"] + df["Close"]) / 2, inplace=True)

        # Calculate Heikin-Ashi high and low
        df["HA_High"] = df[["High", "HA_Open", "HA_Close"]].max(axis=1)
        df["HA_Low"] = df[["Low", "HA_Open", "HA_Close"]].min(axis=1)

        return df[["HA_Open", "HA_High", "HA_Low", "HA_Close"]]

    def ichimoku_cloud(
        df: pd.DataFrame, n1: int = 9, n2: int = 26, n3: int = 52
    ) -> pd.DataFrame:
        """
        Calculate the Ichimoku Cloud indicator for the given DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
            n1 (int): Period for the Tenkan-sen (Conversion Line). Default is 9.
            n2 (int): Period for the Kijun-sen (Base Line) and Senkou Span A & B. Default is 26.
            n3 (int): Period for the Senkou Span B. Default is 52.

        Returns:
            pd.DataFrame: DataFrame with Ichimoku Cloud components:
                'Conversion Line', 'Base Line', 'Leading Span A', 'Leading Span B', 'Cloud Top', 'Cloud Bottom'.
        """
        # Tenkan-sen (Conversion Line)
        df["Conversion Line"] = (
            df["High"].rolling(window=n1).max() + df["Low"].rolling(window=n1).min()
        ) / 2

        # Kijun-sen (Base Line)
        df["Base Line"] = (
            df["High"].rolling(window=n2).max() + df["Low"].rolling(window=n2).min()
        ) / 2

        # Senkou Span A (Leading Span A)
        df["Leading Span A"] = ((df["Conversion Line"] + df["Base Line"]) / 2).shift(n2)

        # Senkou Span B (Leading Span B)
        df["Leading Span B"] = (
            (df["High"].rolling(window=n3).max() + df["Low"].rolling(window=n3).min())
            / 2
        ).shift(n2)

        # Kumo (Cloud)
        df["Cloud Top"] = df[["Leading Span A", "Leading Span B"]].max(axis=1)
        df["Cloud Bottom"] = df[["Leading Span A", "Leading Span B"]].min(axis=1)

        return df[
            [
                "Conversion Line",
                "Base Line",
                "Leading Span A",
                "Leading Span B",
                "Cloud Top",
                "Cloud Bottom",
            ]
        ]

    def kdj(
        prices: pd.DataFrame, period: int = 14, smooth_period: int = 3
    ) -> pd.DataFrame:
        """
        Calculate the KDJ indicator for a given DataFrame of stock prices.

        The KDJ is based on the Stochastic Oscillator, with three lines:
        - %K: The current position relative to the high-low range.
        - %D: A 3-period moving average of %K.
        - %J: The difference between 3 * %K and 2 * %D.

        Parameters:
        prices (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): The period for calculating %K (default is 14).
        smooth_period (int): The smoothing period for the %D line (default is 3).

        Returns:
        pd.DataFrame: A DataFrame containing the %K, %D, and %J values.
        """
        # Calculate the lowest low and highest high for the given period
        low_min = prices["Low"].rolling(window=period).min()
        high_max = prices["High"].rolling(window=period).max()

        # Calculate %K
        k = 100 * (prices["Close"] - low_min) / (high_max - low_min)

        # Calculate %D (3-period SMA of %K)
        d = k.rolling(window=smooth_period).mean()

        # Calculate %J
        j = 3 * k - 2 * d

        # Return the result as a DataFrame
        return pd.DataFrame({"%K": k, "%D": d, "%J": j})

    def keltner_channels(
        df: pd.DataFrame,
        window_size: int = 20,
        multiplier: float = 2,
        ema_window: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate the Keltner Channels for the given DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
            window_size (int): The window size for the ATR calculation. Default is 20.
            multiplier (float): The multiplier for the ATR to calculate the upper and lower bands. Default is 2.
            ema_window (int): The window size for the EMA of the Typical Price. Default is 20.

        Returns:
            pd.DataFrame: DataFrame with Keltner Channel components: 'Middle_Line', 'Upper_Band', 'Lower_Band', 'Typical_Price', 'ATR'.
        """
        # Calculate Typical Price
        df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3

        # Calculate Middle Line (EMA of Typical Price)
        df["Middle_Line"] = (
            df["Typical_Price"].ewm(span=ema_window, min_periods=ema_window).mean()
        )

        # Calculate True Range (TR)
        df["TR"] = np.maximum.reduce(
            [
                df["High"] - df["Low"],
                abs(df["High"] - df["Close"].shift()),
                abs(df["Low"] - df["Close"].shift()),
            ]
        )

        # Calculate Average True Range (ATR)
        df["ATR"] = df["TR"].ewm(span=window_size, min_periods=window_size).mean()

        # Calculate Upper and Lower Bands
        df["Upper_Band"] = df["Middle_Line"] + multiplier * df["ATR"]
        df["Lower_Band"] = df["Middle_Line"] - multiplier * df["ATR"]

        return df[["Typical_Price", "Middle_Line", "Upper_Band", "Lower_Band", "ATR"]]

    def modified_moving_average(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate the Modified Moving Average (MMA) for a given price series.

        The Modified Moving Average (MMA) gives more weight to recent prices and is
        computed similarly to an Exponential Moving Average (EMA).

        Parameters:
        prices (pd.Series): Series of stock prices (usually closing prices).
        period (int): The period for calculating the MMA.

        Returns:
        pd.Series: The Modified Moving Average (MMA) values.
        """
        # Calculate the alpha (smoothing factor)
        alpha = 2 / (period + 1)

        # Initialize the MMA with the first price (can also be the first SMA)
        mma = pd.Series(index=prices.index, dtype=float)
        mma.iloc[0] = prices.iloc[0]  # Set the first value of MMA as the first price

        # Calculate the MMA using the recursive formula
        for t in range(1, len(prices)):
            mma.iloc[t] = mma.iloc[t - 1] + alpha * (prices.iloc[t] - mma.iloc[t - 1])

        return mma

    def momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the momentum indicator, which measures the rate of change in price.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'Close' column.
            period (int): The number of periods over which to calculate momentum. Default is 14.

        Returns:
            pd.Series: The momentum indicator values.
        """
        # Calculate the difference in closing prices
        return df["Close"].diff(period)

    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Money Flow Index (MFI), a volume-weighted version of the Relative Strength Index (RSI).

        Parameters:
            df (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.
            period (int): The number of periods over which to calculate the MFI. Default is 14.

        Returns:
            pd.Series: The Money Flow Index (MFI) values.
        """
        # Calculate typical price
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

        # Calculate raw money flow
        raw_money_flow = typical_price * df["Volume"]

        # Determine positive and negative money flows
        positive_flow = np.where(
            typical_price > typical_price.shift(1), raw_money_flow, 0
        )
        negative_flow = np.where(
            typical_price < typical_price.shift(1), raw_money_flow, 0
        )

        # Calculate 14-period sums of positive and negative money flows
        positive_mf = (
            pd.Series(positive_flow).rolling(window=period, min_periods=1).sum()
        )
        negative_mf = (
            pd.Series(negative_flow).rolling(window=period, min_periods=1).sum()
        )

        # Calculate money flow ratio (MFR)
        mfr = positive_mf / negative_mf.replace(to_replace=0, method="ffill")

        # Calculate Money Flow Index (MFI)
        mfi = 100 - (100 / (1 + mfr))

        return mfi

    def macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate the Moving Average Convergence Divergence (MACD), Signal Line, and MACD Histogram.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'Close' column.
            fast_period (int): The period for the fast (short-term) EMA. Default is 12.
            slow_period (int): The period for the slow (long-term) EMA. Default is 26.
            signal_period (int): The period for the signal line EMA. Default is 9.

        Returns:
            tuple: A tuple containing:
                - pd.Series: MACD line
                - pd.Series: Signal line
                - pd.Series: MACD histogram
        """
        # Calculate short-term (fast) EMA
        ema_fast = df["Close"].ewm(span=fast_period, adjust=False).mean()

        # Calculate long-term (slow) EMA
        ema_slow = df["Close"].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line (trigger line)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate MACD histogram
        macd_histogram = macd_line - signal_line

        return macd_line, signal_line, macd_histogram

    def obv(df: pd.DataFrame) -> pd.Series:
        """
        Calculate the On-Balance Volume (OBV) indicator.

        OBV is a momentum indicator that measures the flow of volume in and out of a security.
        It adds volume on up days and subtracts volume on down days.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

        Returns:
            pd.Series: The On-Balance Volume (OBV) values.
        """
        # Calculate price changes
        price_diff = df["Close"].diff()

        # Initialize OBV with 0
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = 0

        # Determine OBV values based on price changes
        for i in range(1, len(df)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + df["Volume"].iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - df["Volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def psar(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        initial_af: float = 0.02,
        max_af: float = 0.2,
        increment: float = 0.02,
    ) -> pd.Series:
        """
        Calculate the Parabolic SAR (PSAR) indicator.

        Parameters:
            high (pd.Series): The high prices of the security.
            low (pd.Series): The low prices of the security.
            close (pd.Series): The closing prices of the security.
            initial_af (float): The initial acceleration factor (default is 0.02).
            max_af (float): The maximum acceleration factor (default is 0.2).
            increment (float): The increment to increase the acceleration factor (default is 0.02).

        Returns:
            pd.Series: The Parabolic SAR values for each period.
        """
        length = len(close)
        psar = close.copy()
        bull = True  # Initial trend direction (True for Bullish, False for Bearish)
        af = initial_af
        ep = low.iloc[
            0
        ]  # Extreme Point (EP) for a bullish trend (initially the first low price)
        sar = high.iloc[0]  # Start with the first high price as the SAR value

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

    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        pivot_point = (high + low + close) / 3
        range_high_low = high - low

        support1 = (2 * pivot_point) - high
        support2 = pivot_point - range_high_low
        support3 = low - 2 * (high - pivot_point)
        resistance1 = (2 * pivot_point) - low
        resistance2 = pivot_point + range_high_low
        resistance3 = high + 2 * (pivot_point - low)

        return {
            "Pivot Point": pivot_point,
            "Support 1": support1,
            "Support 2": support2,
            "Support 3": support3,
            "Resistance 1": resistance1,
            "Resistance 2": resistance2,
            "Resistance 3": resistance3,
        }

    def price_channels(prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate the Price Channel indicator, which consists of an Upper Channel Line (UCL)
        and a Lower Channel Line (LCL), based on the highest high and lowest low over the last n periods.

        Parameters:
        prices (pd.DataFrame): DataFrame containing 'High' and 'Low' columns for stock prices.
        period (int): The period for calculating the price channel (lookback period).

        Returns:
        pd.DataFrame: A DataFrame containing the Upper Channel Line (UCL) and Lower Channel Line (LCL).
        """
        # Calculate the Highest High and Lowest Low over the given period
        highest_high = prices["High"].rolling(window=period).max()
        lowest_low = prices["Low"].rolling(window=period).min()

        # Create the Price Channel with the Upper and Lower Channel Lines
        price_channel = pd.DataFrame({"UCL": highest_high, "LCL": lowest_low})

        return price_channel

    def price_oscillator(
        prices: pd.Series, short_period: int, long_period: int
    ) -> pd.Series:
        """
        Calculate the Price Oscillator (PO) indicator for a given price series.

        The Price Oscillator (PO) is the difference between the short-term EMA and long-term EMA
        expressed as a percentage of the long-term EMA.

        Parameters:
        prices (pd.Series): Series of stock prices (usually closing prices).
        short_period (int): The period for the short-term EMA (e.g., 12).
        long_period (int): The period for the long-term EMA (e.g., 26).

        Returns:
        pd.Series: The Price Oscillator (PO) values.
        """
        # Calculate the short-term and long-term EMAs
        short_ema = prices.ewm(span=short_period, adjust=False).mean()
        long_ema = prices.ewm(span=long_period, adjust=False).mean()

        # Calculate the Price Oscillator (PO)
        po = ((short_ema - long_ema) / long_ema) * 100

        return po

    def psychological_line(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate the Psychological Line (PL) for a given price series.

        The Psychological Line (PL) is calculated based on the current closing price relative
        to the highest high and lowest low over the last n periods.

        Parameters:
        prices (pd.Series): Series of stock prices (usually closing prices).
        period (int): The period for calculating the highest high and lowest low.

        Returns:
        pd.Series: The Psychological Line (PL) values, expressed as percentages.
        """
        # Calculate the highest high and lowest low over the given period
        highest_high = prices.rolling(window=period).max()
        lowest_low = prices.rolling(window=period).min()

        # Calculate the Psychological Line (PL)
        pl = ((prices - lowest_low) / (highest_high - lowest_low)) * 100

        return pl

    def spearman_rank_correlation(x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Spearman's Rank Correlation Coefficient between two variables.

        The Spearman rank correlation coefficient measures the strength and direction
        of the monotonic relationship between two variables, based on the ranks of the data.

        Parameters:
        x (pd.Series): The first variable (data series).
        y (pd.Series): The second variable (data series).

        Returns:
        float: Spearman's Rank Correlation coefficient (ρ) between x and y.
        """
        # Rank the values in both series
        rank_x = x.rank()
        rank_y = y.rank()

        # Calculate the difference between the ranks
        d = rank_x - rank_y

        # Calculate Spearman's rank correlation coefficient
        n = len(x)
        rho = 1 - (6 * (d**2).sum()) / (n * (n**2 - 1))

        return rho

    def roc(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.Series:
        """
        Calculate the Rate of Change (ROC) for a specified column over a given window.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The column name to calculate ROC on (default is 'Close').
        - window (int): The window size for calculating the percentage change (default is 14).

        Returns:
        - pd.Series: The Rate of Change (ROC) as a percentage.
        """
        roc = df[column].pct_change(periods=window) * 100

        return roc

    def rsi(df: pd.DataFrame, column: str = "Close", window: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a specified column over a given window.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing stock prices or other time series data.
        - column (str): The column name to calculate RSI on (default is 'Close').
        - window (int): The window size for calculating RSI (default is 14).

        Returns:
        - pd.Series: The calculated RSI values.
        """
        # Calculate price differences
        delta = df[column].diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gains and losses over the specified window using exponential moving average
        avg_gain = gain.ewm(span=window, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(span=window, min_periods=1, adjust=False).mean()

        # Calculate Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def rvi(df: pd.DataFrame, n: int = 10) -> pd.Series:
        """
        Calculate the Relative Volatility Index (RVI) for the given DataFrame over a specified window.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing 'Close', 'High', and 'Low' columns.
        - n (int): The window size for the simple moving averages (default is 10).

        Returns:
        - pd.Series: The calculated RVI values.
        """
        # Calculate the numerator (close - low)
        numerator = df["Close"] - df["Low"]

        # Calculate the denominator (high - low)
        denominator = df["High"] - df["Low"]

        # Smooth the numerator and denominator using a simple moving average
        numerator_sma = numerator.rolling(window=n).mean()
        denominator_sma = denominator.rolling(window=n).mean()

        # Calculate the RVI
        rvi = numerator_sma / denominator_sma
        return rvi

    def rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Identify RSI divergence (bullish or bearish) in the provided DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing stock price data and RSI.
        - window (int): The window size for RSI calculation (default is 14).

        Returns:
        - pd.DataFrame: The original DataFrame with an added 'RSI Divergence' column.
        """
        # Calculate RSI for the DataFrame
        df["RSI"] = Indicators.rsi(df, window)

        # Initialize 'RSI Divergence' column
        df["RSI Divergence"] = "None"

        # Detect peaks and troughs in price and RSI
        df["Price Peak"] = (df["Close"] > df["Close"].shift(1)) & (
            df["Close"] > df["Close"].shift(-1)
        )
        df["RSI Peak"] = (df["RSI"] > df["RSI"].shift(1)) & (
            df["RSI"] > df["RSI"].shift(-1)
        )
        df["Price Trough"] = (df["Close"] < df["Close"].shift(1)) & (
            df["Close"] < df["Close"].shift(-1)
        )
        df["RSI Trough"] = (df["RSI"] < df["RSI"].shift(1)) & (
            df["RSI"] < df["RSI"].shift(-1)
        )

        # Forward-fill to align peaks and troughs for divergence detection
        df[["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"]] = (
            df[["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"]]
            .replace(False, np.nan)
            .ffill()
        )

        # Detect bearish divergence
        bearish_divergence = (df["Price Peak"] > df["Price Peak"].shift(1)) & (
            df["RSI Peak"] < df["RSI Peak"].shift(1)
        )
        df.loc[bearish_divergence, "RSI Divergence"] = "Bearish"

        # Detect bullish divergence
        bullish_divergence = (df["Price Trough"] < df["Price Trough"].shift(1)) & (
            df["RSI Trough"] > df["RSI Trough"].shift(1)
        )
        df.loc[bullish_divergence, "RSI Divergence"] = "Bullish"

        # Clean up temporary columns
        df.drop(
            ["Price Peak", "RSI Peak", "Price Trough", "RSI Trough"],
            axis=1,
            inplace=True,
        )

        return df

    def sma(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Calculate the Simple Moving Average (SMA) of a given data series over a specified window size.

        Parameters:
        - data (np.ndarray): The input data array (typically a column from a pandas DataFrame).
        - window_size (int): The number of periods over which to calculate the moving average.

        Returns:
        - np.ndarray: The calculated Simple Moving Averages.
        """
        weights = np.repeat(1.0, window_size) / window_size
        moving_averages = np.convolve(data, weights, mode="valid")
        return moving_averages

    def stochastic_oscillator(
        df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate the Stochastic Oscillator (%K and %D) for the given DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing 'High', 'Low', and 'Close' columns.
        - window (int): The window size for calculating the highest high and lowest low (default is 14).
        - smooth_k (int): The window size for smoothing %K (default is 3).
        - smooth_d (int): The window size for smoothing %D (default is 3).

        Returns:
        - Tuple[pd.Series, pd.Series]: A tuple containing the smoothed %K and %D values as pandas Series.
        """
        # Calculate the highest high and lowest low over the window period
        highest_high = df["High"].rolling(window=window, min_periods=1).max()
        lowest_low = df["Low"].rolling(window=window, min_periods=1).min()

        # Calculate %K
        percent_k = 100 * ((df["Close"] - lowest_low) / (highest_high - lowest_low))

        # Smooth %K (using simple moving average)
        smooth_percent_k = percent_k.rolling(window=smooth_k, min_periods=1).mean()

        # Calculate %D (moving average of %K)
        percent_d = smooth_percent_k.rolling(window=smooth_d, min_periods=1).mean()

        return smooth_percent_k, percent_d

    def stoch_rsi(
        df: pd.DataFrame, n: int = 14, k: int = 3, d: int = 3
    ) -> pd.DataFrame:
        """
        Calculate the Stochastic Relative Strength Index (Stochastic RSI) for a given DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the 'Close' prices.
        - n (int): The window size for calculating the RSI and Stochastic RSI (default is 14).
        - k (int): The window size for smoothing %K (default is 3).
        - d (int): The window size for smoothing %D (default is 3).

        Returns:
        - pd.DataFrame: A DataFrame containing the RSI, %K, and %D values.
        """
        # Calculate the price change
        delta = df["Close"].diff(1)

        # Calculate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=n, min_periods=1).mean()
        avg_loss = loss.rolling(window=n, min_periods=1).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate Stochastic RSI
        stoch_rsi = (rsi - rsi.rolling(window=n, min_periods=1).min()) / (
            rsi.rolling(window=n, min_periods=1).max()
            - rsi.rolling(window=n, min_periods=1).min()
        )

        # Smooth %K and %D
        stoch_rsi_k = stoch_rsi.rolling(window=k, min_periods=1).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=d, min_periods=1).mean()

        # Add RSI, %K, and %D to the DataFrame
        df["RSI"] = rsi
        df["%K"] = stoch_rsi_k * 100
        df["%D"] = stoch_rsi_d * 100

        return df[["RSI", "%K", "%D"]]

    def tema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate the Triple Exponential Moving Average (TEMA) for a given price series.

        The TEMA is an advanced version of the Exponential Moving Average (EMA) that is
        more responsive to price changes. It is calculated by applying multiple EMAs to the
        data.

        Parameters:
        prices (pd.Series): Series of stock prices (usually closing prices).
        period (int): The period for calculating the EMA (e.g., 14).

        Returns:
        pd.Series: The TEMA values.
        """
        # Calculate the first EMA (EMA_1)
        ema_1 = prices.ewm(span=period, adjust=False).mean()

        # Calculate the second EMA (EMA_2) of the first EMA (EMA_1)
        ema_2 = ema_1.ewm(span=period, adjust=False).mean()

        # Calculate the third EMA (EMA_3) of the second EMA (EMA_2)
        ema_3 = ema_2.ewm(span=period, adjust=False).mean()

        # Calculate the TEMA
        tema = 3 * ema_1 - 3 * ema_2 + ema_3

        return tema

    def vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Volume Weighted Average Price (VWAP).

        The VWAP is the average price of an asset, weighted by volume, over a specified time period.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'Close', 'High', 'Low', and 'Volume' columns.

        Returns:
        pd.DataFrame: The original DataFrame with an additional 'VWAP' column containing the VWAP values.
        """
        # Calculate the Typical Price (TP)
        df["Typical Price"] = (df["Close"] + df["High"] + df["Low"]) / 3

        # Calculate the cumulative sum of the Typical Price * Volume (TPV) and Volume
        df["Cumulative TPV"] = (df["Typical Price"] * df["Volume"]).cumsum()
        df["Cumulative Volume"] = df["Volume"].cumsum()

        # Calculate the VWAP
        df["VWAP"] = df["Cumulative TPV"] / df["Cumulative Volume"]

        return df

    def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate the Williams %R (Williams Percent Range) for a given DataFrame.

        Williams %R is a momentum indicator that measures overbought and oversold levels,
        with values ranging from 0 to -100. A value above -20 indicates overbought conditions,
        and a value below -80 indicates oversold conditions.

        Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int): The window size for calculating the highest high and lowest low (default is 14).

        Returns:
        pd.Series: A Series containing the Williams %R values.
        """
        # Calculate the highest high and lowest low over the window period
        highest_high = df["High"].rolling(window=window, min_periods=1).max()
        lowest_low = df["Low"].rolling(window=window, min_periods=1).min()

        # Calculate Williams %R
        williams_r = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))

        return williams_r
