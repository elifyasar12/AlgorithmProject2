import yfinance as yf
import pandas as pd


class DataFetcher:
    def __init__(self, tickers):
        """
        Initialize the DataFetcher with a list of tickers.
        :param tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "^GSPC"])
        """
        self.tickers = tickers

    def get_data(self):
        """
        Fetch historical data and calculate daily returns.
        :return: Tuple (close_prices DataFrame, daily_returns DataFrame)
        """
        try:
            data = yf.download(self.tickers, period="1y", group_by="ticker")

            print("Fetched data structure:")
            print(data.head())  # Debug output
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {e}")

        # Handle multi-ticker or single-ticker data structure
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data.loc[:, (slice(None), 'Close')].droplevel(1, axis=1)
        elif "Close" in data.columns:
            close_prices = data["Close"]
        else:
            raise KeyError("'Close' column not found in data.")

        
        close_prices = close_prices.dropna()

        if close_prices.empty:
            raise ValueError("No valid data for tickers.")

        # Calculate daily returns
        daily_returns = close_prices.pct_change().dropna()
        return close_prices, daily_returns
