# Technical Indicators: Formulas, Insights, and Applications

This guide covers essential technical indicators for financial analysis and trading. It explains their significance, provides formulas, and shows how to use them to analyze price movements, identify trends, and improve trading strategies.

## Table of Contents

1. [Accumulation Distribution Line (ADL)](#1-accumulation-distribution-line-adl)
2. [Adaptive Moving Average (AMA)](#2-adaptive-moving-average-ama)
3. [Aroon](#3-aroon)
4. [Average Directional Index (ADX)](#4-average-directional-index-adx)
5. [Average True Range (ATR)](#5-average-true-range-atr)
6. [Awesome Oscillator (AO)](#6-awesome-oscillator-ao)
7. [Black-Scholes Model](#7-black-scholes-model)
8. [Bollinger Bands](#8-bollinger-bands)
9. [Chaikin Money Flow (CMF)](#9-chaikin-money-flow-cmf)
10. [Chaikin Oscillator](#10-chaikin-oscillator)
11. [Chaikin Volatility](#11-chaikin-volatility)
12. [Commodity Channel Index (CCI)](#12-commodity-channel-index-cci)
13. [Directional Movement Indicator (DMI)](#13-directional-movement-indicator-dmi)
14. [Envelope](#14-envelope)
15. [Exponential Moving Average (EMA)](#15-exponential-moving-average-ema)
16. [Fibonacci Retracement](#16-fibonacci-retracement)
17. [Heikin-Ashi](#17-heikin-ashi)
18. [Ichimoku Cloud (IKH)](#18-ichimoku-cloud-ikh)
19. [KDJ](#19-kdj)
20. [Keltner Channels](#20-keltner-channels)
21. [Modified Moving Average (MMA)](#21-modified-moving-average-mma)
22. [Momentum](#22-momentum)
23. [Money Flow Index (MFI)](#23-money-flow-index-mfi)
24. [Moving Average Convergence Divergence (MACD)](#24-moving-average-convergence-divergence-macd)
25. [On Balance Volume (OBV)](#25-on-balance-volume-obv)
26. [Parabolic SAR (PSAR)](#26-parabolic-sar-psar)
27. [Pivot Points](#27-pivot-points)
28. [Price Channels](#28-price-channels)
29. [Price Oscillator (PO)](#29-price-oscillator-po)
30. [Psychological Line (PSY)](#30-psychological-line-psy)
31. [Rank Correlation Index (RCI)](#31-rank-correlation-index-rci)
32. [Rate of Change (ROC)](#32-rate-of-change-roc)
33. [Relative Strength Index (RSI)](#33-relative-strength-index-rsi)
34. [Relative Vigor Index (RVI)](#34-relative-vigor-index-rvi)
35. [Simple Moving Average (SMA)](#35-simple-moving-average-sma)
36. [Stochastic Oscillator](#36-stochastic-oscillator)
37. [Stochastic RSI (StochRSI)](#37-stochastic-rsi-stochrsi)
38. [Triple Exponential Moving Average (TRIX)](#38-triple-exponential-moving-average-trix)
39. [Volume Weighted Average Price (VWAP)](#39-volume-weighted-average-price-vwap)
40. [Williams %R](#40-williams-r)

---

## 1. Accumulation Distribution Line (ADL)

The ADL indicator combines both price and volume to determine whether a stock is being accumulated (bought) or distributed (sold). It uses a calculation that factors in the close price relative to the high and low, multiplied by the volume. The result is a cumulative line that moves up with increased buying volume and down with selling volume, helping traders gauge the strength of price movements.

```math
ADL = \text{Previous ADL} + \text{Current Money Flow Volume}
```

---

## 2. Adaptive Moving Average (AMA)

The Adaptive Moving Average (AMA) adjusts its smoothing constant based on market volatility. In volatile markets, the AMA reacts more quickly, while in stable markets, it smooths the price data more. This helps identify trends more accurately by adapting to the changing market conditions.

```math
\text{ER} = \frac{\lvert \text{Price}_{\text{today}} - \text{Price}_{\text{n days ago}}\rvert}{\sum_{i=1}^{n} \lvert \text{Price}_i - \text{Price}_{i-1}\rvert}
```

```math
  \text{SC} = \left[\text{ER} \cdot (\alpha_{\text{fast}} - \alpha_{\text{slow}}) + \alpha_{\text{slow}}\right]^2
```

```math
\text{AMA}_{t} = \text{AMA}_{t-1} + \text{SC} \times (\text{Price}_t - \text{AMA}_{t-1})
```

---

## 3. Aroon

The Aroon indicator consists of two lines: Aroon Up and Aroon Down. The Aroon Up measures how long it's been since the highest high over a specific period, while Aroon Down does the same for the lowest low. It identifies trends by showing if the market is trending or if it's in a sideways phase.

```math
\text{Aroon Up} = \left( \frac{\text{Period} - \text{Number of Periods Since Highest High}}{\text{Period}} \right) \times 100
```

```math
\text{Aroon Down} = \left( \frac{\text{Period} - \text{Number of Periods Since Lowest Low}}{\text{Period}} \right) \times 100
```

---

## 4. Average Directional Index (ADX)

The ADX measures the strength of a trend without considering its direction. It ranges from 0 to 100, with values above 25 indicating a strong trend and values below 20 signaling a weak or non-existent trend. ADX is often paired with +DI (positive directional indicator) and -DI (negative directional indicator) to confirm trend direction.

```math
  \text{ADX} = \text{SMA}(|(+DI) - (-DI)| / ((+DI) + (-DI)), n)
```

---

## 5. Average True Range (ATR)

ATR measures the volatility of a security by calculating the average range between the high and low prices over a specified period. This is a key indicator used to gauge market uncertainty, with higher ATR values indicating increased volatility.

```math
  \text{TR} = \max(\text{High} - \text{Low}, |\text{High} - \text{Previous Close}|, |\text{Low} - \text{Previous Close}|)
```

---

## 6. Awesome Oscillator (AO)

The Awesome Oscillator is a momentum indicator that compares two simple moving averages of the median price. It helps traders spot market momentum, with readings above zero indicating positive momentum and readings below zero indicating negative momentum.

```math
\text{Median Price} = \frac{\text{High} + \text{Low}}{2}
```

```math
\text{AO} = \text{SMA}(\text{Median Price}, 5) - \text{SMA}(\text{Median Price}, 34)
```

---

## 7. Black-Scholes Model

The Black-Scholes Model is a mathematical model used to determine the fair value of options. It calculates the price of an option based on variables like the underlying asset's price, strike price, volatility, time to expiration, and the risk-free interest rate. It is essential for traders in options markets.

```math
C = S \cdot N(d_1) - X \cdot e^{-rT} \cdot N(d_2)
```

```math
P = X \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
```

Where:

```math
d_1 = \frac{\ln(S / X) + (r + \sigma^2 / 2) \cdot T}{\sigma \cdot \sqrt{T}}
```

```math
d_2 = d_1 - \sigma \cdot \sqrt{T}
```

- \( C \): Call option price
- \( P \): Put option price
- \( S \): Current stock price
- \( X \): Strike price
- \( T \): Time to expiration (in years)
- \( r \): Risk-free interest rate
- \( \sigma \): Volatility of the stock
- \( N(x) \): Cumulative distribution function of the standard normal distribution
- \( e \): Euler's number (approx. 2.718)

---

## 8. Bollinger Bands

Bollinger Bands consist of a moving average (usually the 20-day SMA) and two bands that are plotted a set number of standard deviations above and below the moving average. The bands expand when volatility is high and contract when volatility is low. Prices that touch or breach the outer bands may signal overbought or oversold conditions.

```math
\text{Upper Band} = \text{SMA}(Close, n) + (k \times \text{StdDev}(Close, n))
```

```math
\text{Lower Band} = \text{SMA}(Close, n) - (k \times \text{StdDev}(Close, n))
```

```math
\%B = \frac{(Close - \text{Lower Band})}{(\text{Upper Band} - \text{Lower Band})}
```

```math
BBW = \frac{(\text{Upper Band} - \text{Lower Band})}{\text{SMA}(Close, n)}
```

---

## 9. Chaikin Money Flow (CMF)

CMF combines price and volume to assess the buying and selling pressure in a market over a given period. A positive CMF value indicates buying pressure, while a negative value signals selling pressure. It helps identify trends and potential reversals.

```math
CMF = \frac{((Close - Low) - (High - Close))}{(High - Low)} \times \text{Volume}
```

---

## 10. Chaikin Oscillator

The Chaikin Oscillator is derived from the Accumulation/Distribution Line (ADL). It calculates the difference between a 3-day EMA and a 10-day EMA of the ADL. This helps traders identify momentum changes and potential reversals by showing shifts in buying or selling pressure.

```math
  \text{MFM}_t = \frac{(Close - Low) - (High - Close)}{High - Low}
```

```math
\text{ADL}_{t} = \text{Previous ADL} + \text{Money Flow Multiplier}_t \times \text{Volume}_t
```

```math
\text{Chaikin Oscillator} = \text{EMA}(\text{ADL}, 3) - \text{EMA}(\text{ADL}, 10)
```

---

## 11. Chaikin Volatility

The Chaikin Volatility indicator calculates the difference between the high and low prices over a period, then applies a moving average to the result. It measures the rate of change in volatility, with larger changes suggesting higher future volatility.

```math
\text{High-Low Range}_t = \text{High}_t - \text{Low}_t
```

```math
\text{Chaikin Volatility} = \frac{\text{EMA}(\text{High-Low Range}, n) - \text{EMA}(\text{High-Low Range}, m)}{\text{EMA}(\text{High-Low Range}, m)} \times 100
```

---

## 12. Commodity Channel Index (CCI)

The CCI measures a security’s price relative to its average price over a period, and it’s used to identify overbought or oversold conditions. High CCI values (above +100) suggest overbought conditions, while low CCI values (below -100) signal oversold conditions.

```math
CCI = \frac{(Typical Price - \text{SMA}(Typical Price, n))}{(0.015 \times \text{Mean Deviation})}
```

---

## 13. Directional Movement Indicator (DMI)

The DMI uses two components: +DI and -DI. The +DI indicates the strength of upward price movements, and the -DI signals downward price movements. When +DI crosses above -DI, it suggests an uptrend, and vice versa for a downtrend. The ADX is often used alongside to gauge trend strength.

```math
+DI = 100 \times \frac{(Smoothed +DM)}{ATR}
```

```math
-DI = 100 \times \frac{(Smoothed -DM)}{ATR}
```

---

## 14. Envelope

The Envelope indicator consists of two bands, set a fixed percentage above and below a moving average (usually SMA). It helps identify overbought or oversold conditions, as price movements outside the bands may signal potential reversals.

```math
\text{Middle Band} = \text{SMA}(Close, n)
```

```math
\text{Upper Band} = \text{Middle Band} \times (1 + p)
```

```math
\text{Lower Band} = \text{Middle Band} \times (1 - p)
```

---

## 15. Exponential Moving Average (EMA)

The EMA is a moving average that gives more weight to recent prices, making it more sensitive to recent price changes. It is widely used in trend-following strategies and helps traders identify price direction more quickly than a simple moving average.

```math
EMA = \alpha \times Close + (1 - \alpha) \times \text{Previous EMA}
```

---

## 16. Fibonacci Retracement

Fibonacci retracement is a technical analysis tool that uses horizontal lines to indicate potential support or resistance levels at key Fibonacci ratios (23.6%, 38.2%, 50%, 61.8%) of a price move. Traders use these levels to anticipate reversal points in the market.

```math
\text{Fibonacci Retracement Levels} = \text{High} - (\text{High} - \text{Low}) \times \{0, 0.236, 0.382, 0.5, 0.618, 1\}
```

---

## 17. Heikin-Ashi

Heikin-Ashi is a type of charting technique that averages the open, high, low, and close prices to create smoother candles. This method helps to filter out market noise and highlight prevailing trends, making it easier for traders to spot potential reversals.

```math
\text{HA Close} = \frac{(Open + High + Low + Close)}{4}
```

```math
\text{HA Open} = \frac{(\text{Previous HA Open} + \text{Previous HA Close})}{2}
```

```math
\text{HA High} = \text{Max}(Open, High, Close)
```

```math
\text{HA Low} = \text{Min}(Open, Low, Close)
```

---

## 18. Ichimoku Cloud (IKH)

The Ichimoku Cloud is a comprehensive indicator that provides information about trend direction, strength, and support/resistance levels. It consists of several components, including the Tenkan-sen, Kijun-sen, and Senkou Span, which help identify trends and reversal points.

```math
  \text{Tenkan-sen} = \frac{\text{Highest High over 9 periods} + \text{Lowest Low over 9 periods}}{2}
```

```math
\text{Kijun-sen} = \frac{\text{Highest High over 26 periods} + \text{Lowest Low over 26 periods}}{2}

```

```math
\text{Senkou Span A} = \frac{(Tenkan-sen + Kijun-sen)}{2}
```

```math
\text{Senkou Span B} = \frac{(Highest High + Lowest Low)}{2}
```

---

## 19. KDJ

The KDJ is a variation of the Stochastic Oscillator that includes a “J” line, designed to highlight potential reversals more strongly. The KDJ indicator uses three lines: K, D, and J, where J amplifies the signals generated by the K and D lines.

```math
\%K = \frac{(\text{Close} - \text{Lowest Low}_n)}{(\text{Highest High}_n - \text{Lowest Low}_n)} \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

```math
J = 3 \times \%K - 2 \times \%D
```

---

## 20. Keltner Channels

Keltner Channels are volatility-based envelopes placed above and below a moving average (typically EMA) using the Average True Range (ATR). They help traders identify overbought or oversold conditions, with price touching or breaching the outer bands often signaling reversals.

```math
\text{Middle Line} = \text{EMA}(Close, n)
```

```math
\text{Upper Channel} = \text{Middle Line} + (ATR \times \text{ATRMultiplier})
```

```math
\text{Lower Channel} = \text{Middle Line} - (ATR \times \text{ATRMultiplier})
```

---

## 21. Modified Moving Average (MMA)

The Modified Moving Average is a variant of the moving average that smooths out price data more effectively. It works similarly to other moving averages but places more emphasis on recent data, providing a clearer signal of price movement.

```math
\text{MMA}_t = \frac{\text{Price}_t + (n-1)\times \text{MMA}_{t-1}}{n}
```

---

## 22. Momentum

Momentum measures the rate of price change.The Momentum indicator measures the rate of change of a security’s price over a defined period. A high momentum indicates strong trends, while low momentum suggests weak or sideways movement. Momentum helps traders identify potential breakout points.

```math
Momentum = Close - Close(n \text{ periods ago})
```

---

## 23. Money Flow Index (MFI)

The MFI combines price and volume to measure the buying and selling pressure of an asset. It ranges from 0 to 100 and is often used to identify overbought or oversold conditions. Values above 80 suggest overbought conditions, and below 20 suggests oversold conditions.

```math
MFI = 100 - \left( \frac{100}{(1 + \text{Money Flow Ratio})} \right)
```

```math
\text{Money Flow Ratio} = \frac{(Positive Money Flow)}{(Negative Money Flow)}
```

---

## 24. Moving Average Convergence Divergence (MACD)

The MACD is a momentum indicator that shows the relationship between two EMAs (typically the 12-day and 26-day). The MACD line is calculated by subtracting the 26-day EMA from the 12-day EMA. The signal line (usually a 9-day EMA) is used for generating buy or sell signals when the MACD crosses above or below it.

```math
\text{MACD Line} = \text{EMA}(Close, Shorter Period) - \text{EMA}(Close, Longer Period)
```

```math
\text{Signal Line} = \text{EMA}(\text{MACD Line}, \text{Signal Period})
```

---

## 25. On Balance Volume (OBV)

OBV uses volume flow to determine the direction of price movements. It adds volume to the OBV line on up days and subtracts it on down days. The direction of the OBV line can confirm trends, with divergence between price and OBV indicating potential trend reversals.

```math
OBV = \text{Previous OBV} + \text{Volume if Close} > \text{Previous Close}
```

```math
OBV = \text{Previous OBV} - \text{Volume if Close} < \text{Previous Close}
```

```math
OBV = \text{Previous OBV} \text{ if Close} = \text{Previous Close}
```

---

## 26. Parabolic SAR (PSAR)

The Parabolic SAR is a trend-following indicator that places dots above or below the price chart, depending on the trend direction. When the market is trending up, the dots appear below the price; when the market is trending down, the dots appear above the price. A dot switching sides signals a potential trend reversal.

```math
PSAR = \text{Previous PSAR} + AF \times (EP - \text{Previous PSAR})
```

---

## 27. Pivot Points

Pivot points are calculated using the previous period's high, low, and close prices. They provide potential support and resistance levels for the current period. These levels are widely used by day traders to forecast market turning points.

```math
\text{Pivot Point} = \frac{\text{High} + \text{Low} + \text{Close}}{3}
```

```math
\text{Support and Resistance Levels} = \{ \text{R1, R2, R3, S1, S2, S3} \}
```

---

## 28. Price Channels

Price Channels form by plotting the highest high and the lowest low over a given period. The resulting channel helps identify trend direction and strength, with price bouncing between the upper and lower bounds in a strong trend.

```math
\text{Upper Channel} = \max(\text{High}_{t-n+1} \dots \text{High}_t)
```

```math
\text{Lower Channel} = \min(\text{Low}_{t-n+1} \dots \text{Low}_t)
```

---

## 29. Price Oscillator (PO)

The Price Oscillator measures the percentage difference between two moving averages, typically an EMA. It helps traders spot changes in momentum, with crossovers signaling potential buy or sell opportunities.

```math
\text{PO} = \frac{\text{EMA}(Close, n_{\text{fast}}) - \text{EMA}(Close, n_{\text{slow}})}{\text{EMA}(Close, n_{\text{slow}})} \times 100
```

---

## 30. Psychological Line (PSY)

The Psychological Line measures the percentage of closing prices above the previous closing price over a defined period. Values above 50% suggest bullish sentiment, while values below 50% indicate bearish sentiment.

```math
\text{PSY} = \frac{\text{Number of days with } Close_t > Close_{t-1} \text{ over } n \text{ days}}{n} \times 100
```

---

## 31. Rank Correlation Index (RCI)

The Rank Correlation Index measures the correlation between the rank of a security's price and the rank of a time series. The index ranges from -100 to +100, where values close to +100 indicate a strong positive correlation (bullish) and values close to -100 indicate a strong negative correlation (bearish).

```math
d_i = \text{Rank}(Price_t) - \text{Rank}(t)
```

```math
\text{RCI} = \left(1 - \frac{6\sum d_i^2}{n(n^2-1)}\right) \times 100
```

---

## 32. Rate of Change (ROC)

The Rate of Change (ROC) indicator measures the percentage change in price over a specific period. It highlights the strength of a trend, with a rising ROC suggesting bullish momentum and a falling ROC indicating bearish momentum.

```math
ROC = \left( \frac{(Close - \text{Close n periods ago})}{\text{Close n periods ago}} \right) \times 100
```

---

## 33. Relative Strength Index (RSI)

The RSI is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100. It identifies overbought conditions when above 70 and oversold conditions when below 30, helping traders predict potential reversals.

```math
RSI = 100 - \left( \frac{100}{(1 + RS)} \right)
```

```math
RS = \text{Average of } x \text{ days' up closes} / \text{Average of } x \text{ days' down closes}
```

---

## 34. Relative Vigor Index (RVI)

The Relative Vigor Index is a momentum oscillator that compares the closing price to the price range of the day. It indicates the strength of a trend, with positive values signaling a strong bullish trend and negative values signaling a strong bearish trend.

```math
RVI = \frac{\text{Close} - \text{Open}}{\text{High} - \text{Low}}
```

---

## 35. Simple Moving Average (SMA)

The SMA is a basic moving average that calculates the average of a security’s price over a specified period. It is widely used to smooth out price data and identify the underlying trend direction.

```math
SMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
```

---

## 36. Stochastic Oscillator

The Stochastic Oscillator measures the location of the current close relative to its price range over a defined period. Values above 80 indicate overbought conditions, while values below 20 indicate oversold conditions. It helps traders identify potential trend reversals.

```math
\%K = \left( \frac{(Current Close - Lowest Low)}{(Highest High - Lowest Low)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

---

## 37. Stochastic RSI (StochRSI)

Stochastic RSI is an indicator of the RSI, applying the Stochastic Oscillator formula to RSI values. It is more sensitive than the RSI, helping identify extreme levels of overbought and oversold conditions.

```math
\%K = \left( \frac{(Current RSI - Lowest RSI)}{(Highest RSI - Lowest RSI)} \right) \times 100
```

```math
\%D = \text{SMA}(\%K, m)
```

---

## 38. Triple Exponential Moving Average (TRIX)

TRIX is a momentum indicator that smooths the price data three times using exponential moving averages. This eliminates noise and provides a clearer signal of long-term trends.

```math
\alpha = 2 / (n + 1)
```

```math
\text{EMA1}_t = \alpha \times Close_t + (1 - \alpha)\times \text{EMA1}_{t-1}
```

```math
\text{EMA2}_t = \alpha \times \text{EMA1}_t + (1 - \alpha)\times \text{EMA2}_{t-1}
```

```math
\text{EMA3}_t = \alpha \times \text{EMA2}_t + (1 - \alpha)\times \text{EMA3}_{t-1}
```

```math
\text{TRIX} = \frac{\text{EMA3}_t - \text{EMA3}_{t-1}}{\text{EMA3}_{t-1}} \times 100
```

---

## 39. Volume Weighted Average Price (VWAP)

VWAP calculates the average price of a security, weighted by volume, over a specific period, typically used for intraday trading. It helps traders assess whether the price is over or under the average price for the day.

```math
VWAP = \frac{\sum(\text{Price} \times \text{Volume})}{\sum(\text{Volume})}
```

---

## 40. Williams %R

Williams %R is a momentum indicator that measures the overbought and oversold levels of an asset on a scale from 0 to -100. Readings above -20 indicate overbought conditions, while readings below -80 indicate oversold conditions.

```math
\%R = \left( \frac{{\text{{Highest High}} - \text{{Close}}}}{{\text{{Highest High}} - \text{{Lowest Low}}}} \right) \times -100
```
