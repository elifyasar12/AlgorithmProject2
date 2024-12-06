import yfinance as yf
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename="simulation.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class DataFetcher:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_data(self):
        try:
            # Fetch historical daily returns for the specified tickers
            data = yf.download(self.tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True)
            close_prices = data.loc[:, (slice(None), "Close")]
            daily_returns = close_prices.pct_change().dropna()
            return close_prices, daily_returns
        except Exception as e:
            logging.error(f"Error fetching data for tickers {self.tickers}: {e}")
            raise RuntimeError("Data fetching failed. Check logs for details.")
