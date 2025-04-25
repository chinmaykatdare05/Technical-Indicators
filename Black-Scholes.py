import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
from typing import Tuple


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
        raise ValueError("Inputs must be positive and time to expiration must be > 0")

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
