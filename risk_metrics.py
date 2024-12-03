import numpy as np
from scipy.stats import norm


class RiskMetrics:
    def __init__(self, simulated_values): #  Initialize with the simulated portfolio values

        self.simulated_values = simulated_values

    def calculate_var(self, confidence_level=0.95): # calculate value at risk (VaR) at a given confidence level
       
        final_values = self.simulated_values[-1, :]
        var = np.percentile(final_values, (1 - confidence_level) * 100)
        return var

    def calculate_cvar(self, confidence_level=0.95): # calculate conditional value at risk (CVaR)
        
        final_values = self.simulated_values[-1, :]
        var = self.calculate_var(confidence_level)
        cvar = np.mean(final_values[final_values < var]) if len(final_values[final_values < var]) > 0 else None
        return cvar

    def calculate_sharpe_ratio(self, portfolio_returns, risk_free_rate=0.01): # calculate sharpe ratio 
       
        mean_return = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else None
        return sharpe_ratio
