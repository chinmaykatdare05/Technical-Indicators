import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    return option_price

def get_black_scholes_params(ticker, start_date, end_date, strike_price, risk_free_rate):
    # Download historical stock price data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate current stock price (S)
    S = stock_data['Adj Close'].iloc[-1]
    
    # Calculate time to expiration (T) in years
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    T = (end_dt - start_dt).days / 365.0
    
    # Calculate volatility (sigma) from historical stock prices
    returns = stock_data['Adj Close'].pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)  # Annualized volatility
    
    return S, strike_price, T, risk_free_rate, sigma

# Example usage:
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-01-01'
strike_price = 150
risk_free_rate = 0.03

# Calculate Black-Scholes parameters
S, K, T, r, sigma = get_black_scholes_params(ticker, start_date, end_date, strike_price, risk_free_rate)

# Print results
print("Black-Scholes Parameters:")
print(f"Current Stock Price (S): {S:.2f}")
print(f"Strike Price (K): {K}")
print(f"Time to Expiration (T) in years: {T:.2f}")
print(f"Risk-Free Rate (r): {r}")
print(f"Volatility (sigma): {sigma:.4f}")

print("\nOption Prices:")
print(f"Call Option Price: {black_scholes(S, K, T, r, sigma, option_type='call'):.2f}")
print(f"Put Option Price: {black_scholes(S, K, T, r, sigma, option_type='put'):.2f}")
