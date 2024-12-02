import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class RiskVisualizer:
    def __init__(self, daily_returns):
        """
        Initialize the RiskVisualizer with daily returns data.
        :param daily_returns: DataFrame of daily returns for assets in the portfolio
        """
        self.daily_returns = daily_returns

    def plot_correlation_heatmap(self):
        """
        Generate a heatmap of correlation between asset returns.
        """
        correlation_matrix = self.daily_returns.corr()  # Compute the correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Correlation Heatmap of Asset Returns")
        plt.show()

    def plot_sector_risk_heatmap(self, sector_map):
        """
        Generate a heatmap of sector-level risk.
        :param sector_map: Dictionary mapping tickers to sectors
        """
        # Map tickers to sectors
        daily_returns_transposed = self.daily_returns.T  # Transpose to group by columns
        daily_returns_transposed['Sector'] = daily_returns_transposed.index.map(sector_map)
    
        # Group by sector and calculate mean returns
        sector_returns = daily_returns_transposed.groupby('Sector').mean().T  # Transpose back
    
        # Generate correlation heatmap for sectors
        correlation_matrix = sector_returns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Sector-Level Risk Heatmap")
        plt.show()
        sector_returns = self.daily_returns.groupby(sector_map, axis=1).mean()
        
        
