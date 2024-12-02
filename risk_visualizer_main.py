from risk_visualizer import RiskVisualizer
import pandas as pd

# Example data (Replace this with actual daily_returns from your project)
# Load or mock daily returns DataFrame
daily_returns = pd.DataFrame({
    "AAPL": [0.01, -0.02, 0.03, 0.04],
    "MSFT": [-0.01, 0.02, -0.03, 0.01],
    "^GSPC": [0.00, 0.01, -0.01, 0.02]
})

# Define the sector map
sector_map = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "^GSPC": "Index"
}

# Initialize RiskVisualizer
visualizer = RiskVisualizer(daily_returns)

# Generate heat maps
visualizer.plot_correlation_heatmap()
visualizer.plot_sector_risk_heatmap(sector_map)

