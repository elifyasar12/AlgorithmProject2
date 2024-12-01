import schedule
import time
from data_fetcher import DataFetcher
from monte_carlo_simulator import MonteCarloSimulator
from risk_metrics import RiskMetrics
from visualizer import Visualizer
from datetime import datetime
import csv


def save_results(date, var_95, cvar_95, sharpe_ratio):
    """
    save results to a CSV file for tracking over time.
    """
    with open("simulation_results.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([date, var_95, cvar_95, sharpe_ratio])


def run_simulation():
    """
    full workflow to fetch data, run Monte Carlo simulations, and visualize results.
    """
    print("Starting Monte Carlo Simulation...")

    # fetch data
    tickers = ["AAPL", "MSFT", "^GSPC"]
    fetcher = DataFetcher(tickers)
    close_prices, daily_returns = fetcher.get_data()

    # set portfolio weights
    weights = [0.4, 0.3, 0.3]

    # run monte carlo simulation
    initial_investment = 10000
    simulator = MonteCarloSimulator(daily_returns, weights, initial_investment)
    simulated_values = simulator.simulate(num_simulations=1000, time_horizon=252)

    # calculate risk metrics
    risk_metrics = RiskMetrics(simulated_values)
    var_95 = risk_metrics.calculate_var(0.95)
    cvar_95 = risk_metrics.calculate_cvar(0.95)
    sharpe_ratio = risk_metrics.calculate_sharpe_ratio(simulator.portfolio_returns)

    print(f"Value at Risk (VaR) at 95% confidence: ${var_95:.2f}")
    print(f"Conditional Value at Risk (CVaR) at 95% confidence: ${cvar_95:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Save results for tracking
    save_results(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), var_95, cvar_95, sharpe_ratio)

    # visualize results
    visualizer = Visualizer(simulated_values, simulator.portfolio_returns)
    visualizer.plot_simulation(var_95=var_95)
    visualizer.plot_var_distribution(var_95)
    visualizer.compare_to_benchmark(daily_returns['^GSPC'])

    print("Monte Carlo Simulation Complete.")


schedule.every().day.at("09:00").do(run_simulation)

if __name__ == "__main__":
    run_simulation()

    # schedule future runs
    while True:
        schedule.run_pending()
        time.sleep(1)
