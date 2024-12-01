import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, simulated_values, portfolio_returns):
        self.simulated_values = simulated_values
        self.portfolio_returns = portfolio_returns

    def plot_simulation(self, var_95=None):
        
        mean_trajectory = self.simulated_values.mean(axis=1)

        plt.figure(figsize=(10, 6))

        # Plot all simulations
        plt.plot(self.simulated_values, color='purple', alpha=0.1)
        plt.plot(mean_trajectory, color='red', linewidth=2, label='Mean Portfolio Value')

        if var_95:
            plt.axhline(y=var_95, color='green', linestyle='dashed', label=f'VaR (95%): ${var_95:.2f}')

        plt.plot([], [], color='blue', alpha=0.6, label='Simulations')

        plt.title("Monte Carlo Simulation of Portfolio Performance")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()


    def plot_var_distribution(self, var_95):
      
        final_values = self.simulated_values[-1, :]  # Get final portfolio values

        plt.figure(figsize=(10, 6))
        plt.hist(final_values, bins=50, color='blue', alpha=0.7, label="Final Portfolio Values")
        plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'VaR (95%): ${var_95:.2f}')

        plt.title("Distribution of Portfolio Values at End of Simulation")
        plt.xlabel("Portfolio Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

    def compare_to_benchmark(self, benchmark_returns):
      
        mean_trajectory = self.simulated_values.mean(axis=1)

        benchmark_cum_returns = (1 + benchmark_returns).cumprod() * self.simulated_values[0, 0]
        benchmark_cum_returns = benchmark_cum_returns.iloc[:len(mean_trajectory)]

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mean_trajectory)), mean_trajectory, label="Portfolio", color="blue")
        plt.plot(range(len(benchmark_cum_returns)), benchmark_cum_returns, label="Benchmark (S&P 500)", color="green")
        plt.title("Portfolio vs Benchmark Performance")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid()
        plt.show()

    def validate_var(self, simulated_var, analytical_var):
        """
        Compare simulated VaR to analytical VaR.
        """
        plt.figure(figsize=(8, 5))
        plt.bar(["Simulated VaR", "Analytical VaR"], [simulated_var, analytical_var], color=['blue', 'orange'])
        plt.title("Comparison of Simulated and Analytical VaR")
        plt.ylabel("Value at Risk (USD)")
        plt.grid()
        plt.show()


    def validate_var(self, simulated_var, analytical_var):
       
        plt.figure(figsize=(8, 5))
        plt.bar(["Simulated VaR", "Analytical VaR"], [simulated_var, analytical_var], color=['blue', 'orange'])
        plt.title("Comparison of Simulated and Analytical VaR")
        plt.ylabel("Value at Risk (USD)")
        plt.grid()
        plt.show()