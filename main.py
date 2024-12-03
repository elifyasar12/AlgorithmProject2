import schedule
import time
from data_fetcher import DataFetcher
from monte_carlo_simulator import MonteCarloSimulator
from risk_metrics import RiskMetrics
from visualizer import Visualizer
from datetime import datetime
import csv
import os

# Add headers to CSV if the file does not exist
if not os.path.exists("simulation_results.csv"):
    with open("simulation_results.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "VaR (95%)", "CVaR (95%)", "Sharpe Ratio"])


def save_results(date, var_95, cvar_95, sharpe_ratio):  #Save results to a CSV file for tracking over time.
   
    with open("simulation_results.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([date, var_95, cvar_95, sharpe_ratio])


def run_simulation():   # Full workflow to fetch data, run Monte Carlo simulations, and visualize results.

    print("Starting Monte Carlo Simulation...")

    # Fetch data
    tickers = ["AAPL", "MSFT", "^GSPC"]
    fetcher = DataFetcher(tickers)
    close_prices, daily_returns = fetcher.get_data()

    # Set portfolio weights
    weights = [0.4, 0.3, 0.3]

    # Run Monte Carlo simulation
    initial_investment = 10000
    simulator = MonteCarloSimulator(daily_returns, weights, initial_investment)

    # Update time horizon to 365 days
    simulated_values = simulator.simulate(num_simulations=1000, time_horizon=365)

    # Calculate risk metrics
    risk_metrics = RiskMetrics(simulated_values)
    var_95 = risk_metrics.calculate_var(0.95)
    cvar_95 = risk_metrics.calculate_cvar(0.95)
    sharpe_ratio = risk_metrics.calculate_sharpe_ratio(simulator.portfolio_returns)

    print(f"Value at Risk (VaR) at 95% confidence: ${var_95:.2f}")
    print(f"Conditional Value at Risk (CVaR) at 95% confidence: ${cvar_95:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Save results for tracking
    save_results(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), var_95, cvar_95, sharpe_ratio)

    # Visualize results
    visualizer = Visualizer(simulated_values, simulator.portfolio_returns)
    visualizer.plot_simulation(var_95=var_95)
    visualizer.plot_var_distribution(var_95)
    visualizer.compare_to_benchmark(daily_returns['^GSPC'])

    print("Monte Carlo Simulation Complete.")


schedule.every().day.at("09:00").do(run_simulation)

if __name__ == "__main__":
    run_simulation()

    # Schedule future runs
    while True:
        schedule.run_pending()
        time.sleep(1)


'''
# Date: The timestamp of the simulation run
# VaR (95%): Value at Risk at a 95% confidence level (worst-case loss in the top 5% scenarios)
# CVaR (95%): Conditional Value at Risk at a 95% confidence level (average loss in the worst 5% cases)
# Sharpe Ratio: Portfolio's risk-adjusted return (higher is better)
Date,VaR (95%),CVaR (95%),Sharpe Ratio
2024-12-02 09:00:00,10588.58,10236.45,0.85

'''