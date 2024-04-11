import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
N = norm.pdf

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