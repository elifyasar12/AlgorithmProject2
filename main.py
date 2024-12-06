import schedule
import time
import os
import csv
from datetime import datetime
from data_fetcher import DataFetcher
from monte_carlo_simulator import MonteCarloSimulator
from risk_metrics import RiskMetrics, get_risk_free_rate
import matplotlib.pyplot as plt
import numpy as np

# Define save_results function
def save_results(date, var_95, cvar_95, sharpe_ratio, initial_investment, portfolio_returns, num_simulations):
  
    # save simulation results to a CSV file with properly aligned columns.

    file_exists = os.path.exists("simulation_results.csv")
    
    with open("simulation_results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        
        # write headers if the file does not exist
        if not file_exists:
            writer.writerow([
                "Date", 
                "VaR (95%)", 
                "CVaR (95%)", 
                "Sharpe Ratio", 
                "Initial Investment",
                "Portfolio Returns",
                "Simulations"
            ])
        
        writer.writerow([
            date, 
            f"${var_95:.2f}", 
            f"${cvar_95:.2f}", 
            f"{sharpe_ratio:.2f}", 
            f"${initial_investment:.2f}",
            f"{portfolio_returns:.4f}",
            num_simulations
        ])

def optimize_portfolio(daily_returns):
  
    # optimize portfolio weights using mean-variance optimization
    num_assets = daily_returns.shape[1]
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Initialize random weights
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize to sum to 1

    # Monte Carlo simulation for optimization
    num_portfolios = 50000
    results = np.zeros((3, num_portfolios))
    weight_array = []

    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_array.append(weights)

        # Portfolio return
        portfolio_return = np.sum(weights * mean_returns) * 252
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        # Sharpe Ratio
        sharpe_ratio = portfolio_return / portfolio_volatility

        # Store the results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    # Find the portfolio with the max Sharpe Ratio
    max_sharpe_idx = results[2].argmax()
    optimal_weights = weight_array[max_sharpe_idx]

    # Print named weights
    tickers = ["AAPL", "BSX", "CAT", "DVA", "EMN", "FDX", "GRMN", "HLT", "IBM", "JBL", "LLY", "MAR", "^GSPC"]
    print("\nOptimized Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    return optimal_weights


# Define run_simulation
def run_simulation():
    
    # fetch data, run Monte Carlo simulations, calculate risk metrics, and save results.
    
    print("Starting Monte Carlo Simulation...")
    tickers = ["AAPL", "BSX", "CAT", "DVA", "EMN", "FDX", "GRMN",
               "HLT", "IBM", "JBL", "LLY", "MAR", "^GSPC"]
    fetcher = DataFetcher(tickers)
    close_prices, daily_returns = fetcher.get_data()

    # optimize weights using historical returns
    weights = optimize_portfolio(daily_returns.iloc[:, :-1])  # Exclude the S&P 500 (^GSPC) for weight calculation

    initial_investment = 10000
    simulator = MonteCarloSimulator(daily_returns.iloc[:, :-1], weights, initial_investment)  # Exclude S&P 500
    simulated_values = simulator.simulate(num_simulations=1000, time_horizon=365)

    risk_metrics = RiskMetrics(simulated_values)
    var_95 = risk_metrics.calculate_var(0.95)
    cvar_95 = risk_metrics.calculate_cvar(0.95)
    sharpe_ratio = risk_metrics.calculate_sharpe_ratio(simulator.portfolio_returns)

    print(f"Value at Risk (VaR) at 95% confidence: ${var_95:.2f}")
    print(f"Conditional Value at Risk (CVaR) at 95% confidence: ${cvar_95:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # save results to CSV
    save_results(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        var_95, 
        cvar_95, 
        sharpe_ratio, 
        initial_investment, 
        simulator.portfolio_returns.mean(), 
        1000  # Number of simulations
    )

    # plot results
    plot_simulation(simulated_values, var_95)
    plot_final_value_histogram(simulated_values, var_95)
    plot_portfolio_vs_benchmark(simulator.portfolio_returns, daily_returns["^GSPC"])

# plotting functions
def plot_simulation(simulated_values, var_95):
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
    days = range(len(portfolio_returns))
    plt.figure(figsize=(12, 6))
    plt.plot(days, (1 + portfolio_returns).cumprod() * 10000, label="Portfolio", color="blue")
    plt.plot(days, (1 + benchmark_returns).cumprod() * 10000, label="Benchmark (S&P 500)", color='green')
    plt.title("Portfolio vs Benchmark Performance")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

# main execution
if __name__ == "__main__":
    np.random.seed(42)  # ensure reproducibility
    run_simulation()  
