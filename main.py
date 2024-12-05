import schedule
import time
import os
import csv
from datetime import datetime
from data_fetcher import DataFetcher
from monte_carlo_simulator import MonteCarloSimulator
from risk_metrics import RiskMetrics
import matplotlib.pyplot as plt
import numpy as np

# add headers to CSV if the file does not exist
if not os.path.exists("simulation_results.csv"):
    with open("simulation_results.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "VaR (95%)", "CVaR (95%)", "Sharpe Ratio"])

def save_results(date, var_95, cvar_95, sharpe_ratio):

    # save results to a CSV file for tracking over time
    with open("simulation_results.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([date, var_95, cvar_95, sharpe_ratio])

def run_simulation():
  
    # full workflow to fetch data, run Monte Carlo simulations, and visualize results
    print("Starting Monte Carlo Simulation...")

    # Fetch data (including S&P 500 as benchmark)
    tickers = ["AAPL", "BSX", "CAT", "DVA", "EMN", "FDX", "GRMN",
               "HLT", "IBM", "JBL", "LLY", "MAR", "^GSPC"]
    fetcher = DataFetcher(tickers)
    close_prices, daily_returns = fetcher.get_data()

    
    print(f"Daily returns shape: {daily_returns.shape}")
    
    # Set portfolio weights
    weights = [1 / (len(tickers) - 1)] * (len(tickers) - 1)  # exclude ^GSPC from weights
    print(f"Weights length: {len(weights)}")

    # run Monte Carlo simulation
    initial_investment = 10000
    simulator = MonteCarloSimulator(daily_returns.iloc[:, :-1], weights, initial_investment)  # Exclude ^GSPC
    simulated_values = simulator.simulate(num_simulations=1000, time_horizon=365)

    # calculate risk metrics
    risk_metrics = RiskMetrics(simulated_values)
    var_95 = risk_metrics.calculate_var(0.95)
    cvar_95 = risk_metrics.calculate_cvar(0.95)
    sharpe_ratio = risk_metrics.calculate_sharpe_ratio(simulator.portfolio_returns)

    print(f"Value at Risk (VaR) at 95% confidence: ${var_95:.2f}")
    print(f"Conditional Value at Risk (CVaR) at 95% confidence: ${cvar_95:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

   
    save_results(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), var_95, cvar_95, sharpe_ratio)

    # Visualize results
    plot_simulation(simulated_values, simulator.portfolio_returns, var_95)
    plot_final_value_histogram(simulated_values, var_95)
    plot_portfolio_vs_benchmark(simulator.portfolio_returns, daily_returns["^GSPC", "Close"])

    print("Monte Carlo Simulation Complete.")

def plot_simulation(simulated_values, portfolio_returns, var_95):
    """
    Plot Monte Carlo simulation results along with Value at Risk (VaR).
    """
    days = range(simulated_values.shape[0])

    plt.figure(figsize=(12, 6))
    plt.plot(days, simulated_values.mean(axis=1), label="Mean Portfolio Value", color="blue")
    plt.fill_between(days, simulated_values.min(axis=1), simulated_values.max(axis=1), color='blue', alpha=0.1, label="Portfolio Value Range")
    plt.axhline(var_95, color='red', linestyle='--', label=f"VaR (95%): ${var_95:.2f}")
    plt.title("Monte Carlo Simulation of Portfolio Performance")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

def plot_final_value_histogram(simulated_values, var_95):
    """
    Plot histogram of the final portfolio values at the end of the simulation period.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(simulated_values[-1], bins=50, color='green', edgecolor='black', alpha=0.75, label="Final Portfolio Values")
    plt.axvline(var_95, color='blue', linestyle='--', linewidth=2, label=f"VaR (95%): ${var_95:.2f}")
    plt.title("Distribution of Portfolio Values at End of Simulation")
    plt.xlabel("Portfolio Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.show()

def plot_portfolio_vs_benchmark(portfolio_returns, benchmark_returns):
    """
    Plot portfolio performance against a benchmark (e.g., S&P 500).
    """
    days = range(len(portfolio_returns))

    plt.figure(figsize=(12, 6))
    plt.plot(days, (1 + portfolio_returns).cumprod() * 10000, label="Portfolio", color='blue')
    plt.plot(days, (1 + benchmark_returns).cumprod() * 10000, label="Benchmark (S&P 500)", color='green')
    plt.title("Portfolio vs Benchmark Performance")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

schedule.every().day.at("09:00").do(run_simulation)

if __name__ == "__main__":
    np.random.seed(42)  # set seed for reproducibility
    run_simulation()

    # schedule future runs
    while True:
        schedule.run_pending()
        time.sleep(1)




