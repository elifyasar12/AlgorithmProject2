import yfinance as yf
import pandas as pd

class DataFetcher:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_data(self):
      
       # Fetch historical daily returns for the specified tickers.
       
        data = yf.download(self.tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True)
        close_prices = data.loc[:, (slice(None), "Close")]
        daily_returns = close_prices.pct_change().dropna()
        return close_prices, daily_returns
