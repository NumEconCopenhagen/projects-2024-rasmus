import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd

class DataProject:
    def get_stock_data(self, ticker, start_date, end_date=None):
        """
        Fetches historical stock data from Yahoo Finance.

        Parameters:
        ticker (str): The ticker symbol of the stock.
        start_date (str): The start date for the data in the format 'YYYY-MM-DD'.
        end_date (str, optional): The end date for the data in the format 'YYYY-MM-DD'. If not provided, fetches data up to the present.

        Returns:
        pandas.DataFrame: A DataFrame with the historical stock data.
        """
        # Get the stock data
        stock = yf.Ticker(ticker)

        # Get the historical stock prices for the specified period
        data = stock.history(start=start_date, end=end_date)

        return data

    def get_option_data(self, ticker):
        """
        Fetches options data from Yahoo Finance for all available expiration dates.

        Parameters:
        ticker (str): The ticker symbol of the stock.

        Returns:
        pandas.DataFrame: A DataFrame with the options data.
        """
        # Get the stock data
        stock = yf.Ticker(ticker)

        # Initialize an empty DataFrame to hold the options data
        all_option_data = pd.DataFrame()

        # Loop over all available expiration dates
        for expiry_date in stock.options:
            # Get the options data for the current expiry date
            opts = stock.option_chain(expiry_date)

            # Append the options data to the DataFrame
            all_option_data = all_option_data.append(opts.calls)
            all_option_data = all_option_data.append(opts.puts)

        return all_option_data

    def black_scholes(self, S, K, r, sigma, T=1, option_type='call'):
        """
        Calculates the Black-Scholes price for a European option.

        Parameters:
        S (float): The spot price of the underlying asset.
        K (float): The strike price.
        r (float): The risk-free rate.
        sigma (float): The volatility of the underlying asset.
        T (float, optional): The time to expiration in years. Default is 1.
        option_type (str, optional): The type of the option. Can be either 'call' or 'put'. Default is 'call'.

        Returns:
        float: The Black-Scholes price for the option.
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate the option price
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        return price