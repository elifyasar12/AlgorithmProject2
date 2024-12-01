import numpy as np

class MonteCarloSimulator:
    def __init__(self, daily_returns, weights, initial_investment):
        """
        initialize the simulator
        :param daily_returns: DataFrame of daily returns for each asset
        :param weights: List of portfolio weights (summing to 1)
        :param initial_investment: Starting portfolio value
        """
        self.daily_returns = daily_returns
        self.weights = np.array(weights)
        self.initial_investment = initial_investment

        # Portfolio-level returns
        self.portfolio_returns = (self.daily_returns * self.weights).sum(axis=1)

    def simulate(self, num_simulations, time_horizon):

        """
        :param num_simulations: Number of simulations
        :param time_horizon: Number of trading days
        :return: Simulated portfolio values
        """
        mean = self.portfolio_returns.mean()
        std = self.portfolio_returns.std()

        simulated_returns = np.random.normal(mean, std, (time_horizon, num_simulations))
        simulated_values = self.initial_investment * (1 + simulated_returns).cumprod(axis=0)
        return simulated_values
