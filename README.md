# Mastering Technical Indicators: Formulas, Insights, and Applications

---

This comprehensive guide provides descriptions and formulas for various technical indicators used in financial analysis and trading. Understanding these indicators can help you analyze price movements, identify trends, and make informed trading decisions.

Explore the descriptions and formulas below to gain insights into each indicator's significance and how it can be applied in your trading strategies.

Now, let's dive into the world of technical indicators and enhance our trading skills!

---

## Accumulation Distribution Line

The Accumulation Distribution Line (ADL) is a momentum indicator that relates price and volume. It combines price and volume to show how much money is flowing in or out of a security. It is calculated by adding the money flow volume to the previous ADL.

$$
ADL = \text{Previous ADL} + \text{Current Money Flow Volume}
$$

## Adaptive Moving Average

The Adaptive Moving Average (AMA) is a technical analysis indicator used to filter price fluctuations and determine trend direction. It adjusts its sensitivity based on market volatility. It is calculated by adding a fraction of the difference between the close price and the previous AMA to the previous AMA.

$$
AMA = \text{Previous AMA} + \alpha \times (Close - \text{Previous AMA})
$$

## Aroon

Aroon is an indicator system that determines whether a stock is trending or not and how strong the trend is. It consists of two lines: Aroon Up and Aroon Down. Aroon Up measures the number of periods since the highest high, while Aroon Down measures the number of periods since the lowest low.

$$
\text{Aroon Up} = \left( \frac{\text{Period} - \text{Number of Periods Since Highest High}}{\text{Period}} \right) \times 100
$$

$$
\text{Aroon Down} = \left( \frac{\text{Period} - \text{Number of Periods Since Lowest Low}}{\text{Period}} \right) \times 100
$$

## Average True Range (ATR)

The Average True Range (ATR) measures market volatility by calculating the average range between successive periods. It considers any gaps in trading that may occur between periods.

$$
ATR = \text{SMA}(\left| \text{High} - \text{Low} \right|, n)
$$

## Bollinger Bands

Bollinger Bands consist of a middle band (SMA) and two outer bands that are standard deviations away from the middle band. They provide a relative definition of high and low prices. The upper band indicates overbought conditions, while the lower band indicates oversold conditions.

$$
\text{Upper Band} = \text{SMA}(Close, n) + (k \times \text{StdDev}(Close, n))
$$

$$
\text{Lower Band} = \text{SMA}(Close, n) - (k \times \text{StdDev}(Close, n))
$$

## Bollinger Bands %B

%B measures a security's position relative to the bands. It quantifies a security's price relative to the upper and lower bands.

$$
%B = \frac{(Close - \text{Lower Band})}{(\text{Upper Band} - \text{Lower Band})}
$$

## Bollinger Bands Width

The Bollinger Bands Width quantifies the width of the Bollinger Bands. It is calculated as the percentage difference between the upper and lower bands relative to the middle band.

$$
BBW = \frac{(\text{Upper Band} - \text{Lower Band})}{\text{SMA}(Close, n)}
$$

## Chaikin Money Flow

Chaikin Money Flow (CMF) is an oscillator that measures the accumulation/distribution of a security over a specified period. It combines price and volume to determine buying and selling pressure.

$$
CMF = \frac{((Close - Low) - (High - Close))}{(High - Low)} \times \text{Volume}
$$

## Chaikin Oscillator

The Chaikin Oscillator is the difference between the 3-day and 10-day exponential moving averages of the Accumulation Distribution Line (ADL). It is used to identify possible trend reversals and confirm trends.

$$
\text{Chaikin Oscillator} = \text{EMA}(ADL, 3) - \text{EMA}(ADL, 10)
$$

## Commodity Channel Index

The Commodity Channel Index (CCI) measures a security's deviation from its statistical average. It indicates overbought and oversold conditions and potential trend reversals.

$$
CCI = \frac{(Typical Price - \text{SMA}(Typical Price, n))}{(0.015 \times \text{Mean Deviation})}
$$

## Directional Movement Indicator

The Directional Movement Indicator (DMI) consists of two lines: +DI and -DI. It helps identify trend direction and strength. The +DI measures bullish movement, while the -DI measures bearish movement.

$$
+DI = 100 \times \frac{(Smoothed +DM)}{ATR}
$$

$$
-DI = 100 \times \frac{(Smoothed -DM)}{ATR}
$$

## Envelope (ENV)

Envelopes consist of two SMA lines that form a channel around the price. They help identify overbought and oversold conditions.

$$
\text{Upper Envelope} = \text{SMA}(Close, n) \times (1 + \frac{\text{Percentage}}{100})
$$

$$
\text{Lower Envelope} = \text{SMA}(Close, n) \times (1 - \frac{\text{Percentage}}{100})
$$

## Exponential Moving Average (EMA)

The Exponential Moving Average (EMA) gives more weight to recent prices, making it more responsive to price changes compared to the SMA. It is calculated using a smoothing factor (α).

$$
EMA = \alpha \times Close + (1 - \alpha) \times \text{Previous EMA}
$$

## Heikin-Ashi

Heikin-Ashi charts smooth out price data, making trends and market direction easier to identify. They are useful for determining trends, potential reversal points, and entry/exit signals.

$$
\text{HA Close} = \frac{(Open + High + Low + Close)}{4}
$$

$$
\text{HA Open} = \frac{(\text{Previous HA Open} + \text{Previous HA Close})}{2}
$$

$$
\text{HA High} = \text{Max}(Open, High, Close)
$$

$$
\text{HA Low} = \text{Min}(Open, Low, Close)
$$

## Ichimoku Cloud (IKH)

The Ichimoku Cloud is a comprehensive indicator that provides information about support and resistance levels, trend direction, and momentum. It consists of several components, including the Tenkan-sen, Kijun-sen, Senkou Span A, and Senkou Span B.

$$
\text{Tenkan-sen} = \frac{(Highest High + Lowest Low)}{2}
$$

$$
\text{Kijun-sen} = \frac{(Highest High + Lowest Low)}{2}
$$

$$
\text{Senkou Span A} = \frac{(Tenkan-sen + Kijun-sen)}{2}
$$

$$
\text{Senkou Span B} = \frac{(Highest High + Lowest Low)}{2}
$$

## KDJ

The KDJ indicator is a derivative of the Stochastic Oscillator. It helps identify overbought and oversold conditions and potential trend reversals.

$$
RSV = \frac{(Close - Lowest Low)}{(Highest High - Lowest Low)} \times 100
$$

$$
K = \text{SMA}(RSV, n)
$$

$$
D = \text{SMA}(K, m)
$$

$$
J = 3 \times K - 2 \times D
$$

## Keltner Channels

Keltner Channels consist of an SMA line and two outer lines that are plotted above and below the SMA line. They help identify trend direction and potential reversal points.

$$
\text{Middle Line} = \text{EMA}(Close, n)
$$

$$
\text{Upper Channel} = \text{Middle Line} + (ATR \times \text{ATRMultiplier})
$$

$$
\text{Lower Channel} = \text{Middle Line} - (ATR \times \text{ATRMultiplier})
$$

## Modified (Smoothed) Moving Average

The Modified Moving Average (SMMA) is similar to the Exponential Moving Average (EMA) but with a smoother curve. It helps filter out noise and identify trends more accurately.

$$
SMMA = \frac{(Previous SMMA \times (n - 1) + Close)}{n}
$$

## Momentum

Momentum measures the rate of change in price and helps identify trend strength. It is calculated as the difference between the current price and the price n periods ago.

$$
Momentum = Close - Close(n \text{ periods ago})
$$

## Money Flow Index (MFI)

The Money Flow Index (MFI) measures buying and selling pressure by comparing positive and negative money flows. It helps identify overbought and oversold conditions.

$$
MFI = 100 - \left( \frac{100}{(1 + \text{Money Flow Ratio})} \right)
$$

$$
\text{Money Flow Ratio} = \frac{(Positive Money Flow)}{(Negative Money Flow)}
$$

## Moving Average Convergence Divergence (MACD)

The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. It consists of the MACD Line and the Signal Line.

$$
\text{MACD Line} = \text{EMA}(Close, Shorter Period) - \text{EMA}(Close, Longer Period)
$$

$$
\text{Signal Line} = \text{EMA}(\text{MACD Line}, \text{Signal Period})
$$

## On Balance Volume (OBV)

On Balance Volume (OBV) measures buying and selling pressure by adding volume on up days and subtracting volume on down days. It helps confirm price trends and identify potential reversals.

$$
OBV = \text{Previous OBV} + \text{Volume if Close} > \text{Previous Close}
$$

$$
OBV = \text{Previous OBV} - \text{Volume if Close} < \text{Previous Close}
$$

$$
OBV = \text{Previous OBV} \text{ if Close} = \text{Previous Close}
$$

## Parabolic SAR (PSAR)

The Parabolic SAR (PSAR) is a trend-following indicator that helps traders identify potential reversal points in price direction. It is plotted on the price chart and moves incrementally closer to the price as the trend extends.

$$
PSAR = \text{Previous PSAR} + AF \times (EP - \text{Previous PSAR})
$$

## Price Channels

Price Channels are used to identify potential breakout points and trend reversals. They consist of an upper channel line and a lower channel line based on the highest high and lowest low prices over a specified period.

$$
\text{Upper Channel} = \text{Highest High}(n)
$$

$$
\text{Lower Channel} = \text{Lowest Low}(n)
$$

## Price Oscillator (PPO)

The Price Oscillator (PPO) is a momentum oscillator that measures the difference between two moving averages as a percentage. It helps identify trend direction and potential buy/sell signals.

$$
PPO = \left( \frac{(Shorter EMA - Longer EMA)}{Longer EMA} \right) \times 100
$$

## Psychological Line (PSY)

The Psychological Line (PSY) measures market sentiment by calculating the percentage of periods closing higher than the previous period. It helps identify potential trend reversals.

$$
PSY = \left( \frac{\text{Number of periods closing higher}}{\text{Total number of periods}} \right) \times 100
$$

## Rank Correlation Index (RCI)

The Rank Correlation Index (RCI) measures the correlation between two data series. It helps identify the strength and direction of the relationship between variables.

$$
RCI = \frac{\sum (\text{rank of } x) \times (\text{rank of } y)}{n(n^2 - 1)}
$$

## Rate of Change (ROC)

The Rate of Change (ROC) measures the percentage change in price over a specified period. It helps identify momentum and potential trend reversals.

$$
ROC = \left( \frac{(Close - \text{Close n periods ago})}{\text{Close n periods ago}} \right) \times 100
$$

## Ratiocator (RAT)

The Ratiocator (RAT) compares the current price to its moving average to identify overbought and oversold conditions. It helps confirm trend direction and potential entry/exit points.

$$
RAT = \frac{Close}{\text{SMA}(Close, n)}
$$

## Relative Strength Index (RSI)

The Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate overbought or oversold conditions in a security. It ranges from 0 to 100, with values above 70 indicating overbought conditions and values below 30 indicating oversold conditions.

$$
RSI = 100 - \left( \frac{100}{(1 + RS)} \right)
$$

$$
RS = \text{Average of } x \text{ days' up closes} / \text{Average of } x \text{ days' down closes}
$$

## Running Moving Average (RMA)

The Running Moving Average (RMA) is a type of moving average that continually updates as new data becomes available. It helps identify trends and potential reversal points more accurately.

$$
RMA = \frac{(\text{Previous RMA} \times (n - 1)) + Close}{n}
$$

## Simple Moving Average (SMA)

The Simple Moving Average (SMA) is the average of a security's closing prices over a specified period. It is used to identify trends and potential support/resistance levels.

$$
SMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
$$

## Smoothed Moving Average (SMMA)

The Smoothed Moving Average (SMMA) is similar to the Simple Moving Average (SMA) but with a smoother curve. It helps filter out noise and identify trends more accurately.

$$
SMMA = \frac{\text{Sum of Close Prices for } n \text{ periods}}{n}
$$

## Stochastic Oscillator

The Stochastic Oscillator measures the relative position of a security's closing price within its price range over a specified period. It helps identify overbought and oversold conditions and potential trend reversals.

$$
\%K = \left( \frac{(Current Close - Lowest Low)}{(Highest High - Lowest Low)} \right) \times 100
$$

$$
\%D = \text{SMA}(\%K, m)
$$

## Triple Exponential Moving Average (TRIX)

The Triple Exponential Moving Average (TRIX) is a momentum oscillator that displays the percentage rate of change of a triple smoothed exponential moving average. It helps identify trend direction and potential reversal points.

$$
TRIX = \text{EMA}(\text{EMA}(\text{EMA}(Close, n), n), n)
$$

## Volume + Moving Average

The Volume + Moving Average (VMA) measures the average volume of a security over a specified period. It helps identify changes in trading activity and potential trend reversals.

$$
VMA = \text{SMA}(Volume, n)
$$

## Williams %R

The Williams %R indicator measures overbought and oversold levels in a security. It ranges from 0 to -100, with values above -20 indicating overbought conditions and values below -80 indicating oversold conditions.

$$
\%R = \left( \frac{(Highest High - Close)}{(Highest High - Lowest Low)} \right) \times -100
$$
