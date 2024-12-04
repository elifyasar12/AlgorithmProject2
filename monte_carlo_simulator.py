import numpy as np

class MonteCarloSimulator:
    def __init__(self, daily_returns, weights, initial_investment):
        self.daily_returns = daily_returns
        self.weights = weights
        self.initial_investment = initial_investment
        self.portfolio_returns = (self.daily_returns * self.weights).sum(axis=1)

    def simulate(self, num_simulations, time_horizon):
        """
        Run Monte Carlo simulations for the portfolio over the specified time horizon.
        """
        simulated_values = np.zeros((time_horizon, num_simulations))

        for i in range(num_simulations):
            cumulative_return = 1
            for t in range(time_horizon):
                daily_return = np.random.choice(self.portfolio_returns)
                cumulative_return *= (1 + daily_return)
                simulated_values[t, i] = self.initial_investment * cumulative_return

        return simulated_values
