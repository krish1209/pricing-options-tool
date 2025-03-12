# Options Pricing & Risk Analysis Tool

A comprehensive Python application for pricing European options using Black-Scholes, Monte Carlo simulation, and Binomial tree models, with integrated risk metrics analysis and real-time market data visualization.

## 🚀 Features

- **Multiple Pricing Models**:
  - Black-Scholes model with Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
  - Monte Carlo simulation with configurable paths and confidence intervals
  - Binomial tree model supporting both European and American options

- **Advanced Risk Analytics**:
  - Value at Risk (VaR) using historical and Monte Carlo methods
  - Conditional Value at Risk (CVaR/Expected Shortfall)
  - Sharpe Ratio, Beta, and volatility analysis
  - Performance metrics with benchmark comparison

- **Market Data Integration**:
  - Real-time stock data from Yahoo Finance API
  - Complete option chain data retrieval and analysis
  - Implied volatility surface visualization
  - Greeks visualization and analysis

- **Interactive Web Interface**:
  - Responsive Streamlit dashboard with intuitive controls
  - Dynamic data visualization with Plotly
  - Real-time calculation and updates
  - Data export capabilities

## 📊 Key Visualizations

- **Option Pricing Models Comparison**: Visual comparison of prices from different models
- **Risk Analysis Dashboard**: Interactive VaR, CVaR and performance metrics
- **Implied Volatility Surface**: 3D visualization of volatility across strikes and expirations
- **Option Greeks Visualization**: Interactive plots of Delta, Gamma, Theta, and Vega

## 🛠️ Technologies

- **Backend**: Python 3.10, NumPy, Pandas, SciPy
- **Financial Modeling**: Black-Scholes, Monte Carlo, Binomial Tree implementations
- **Data Source**: Yahoo Finance API via yfinance
- **Visualization**: Streamlit, Plotly, Matplotlib

## 📋 Prerequisites

- Python 3.9+ (3.10 recommended)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/options-pricing-app.git
cd options-pricing-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will be available at http://localhost:8501

## 📂 Project Structure

```
options_pricing_app/
├── models/                  # Financial models
│   ├── __init__.py
│   ├── black_scholes.py     # Black-Scholes model
│   ├── monte_carlo.py       # Monte Carlo simulation
│   ├── binomial.py          # Binomial tree model
│   └── risk_metrics.py      # Risk metrics calculations
├── data/                    # Data integration
│   ├── __init__.py
│   └── yahoo_finance.py     # Yahoo Finance API integration
├── ui/                      # User interface components 
│   ├── __init__.py
│   ├── pricing_view.py      # Option pricing view
│   ├── risk_view.py         # Risk analysis view
│   ├── market_view.py       # Market data view
│   └── strategy_view.py     # Strategy builder view
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── config.py            # Configuration utilities
│   ├── styling.py           # UI styling utilities
│   └── caching.py           # Data caching utilities
├── static/                  # Static assets
│   └── css/                 # CSS styling
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
└── README.md                # You are here!
```

## 📚 Usage Guide

### Option Pricing

1. Enter a stock ticker in the sidebar (e.g., AAPL, MSFT, GOOGL)
2. Select models to compare (Black-Scholes, Monte Carlo, Binomial Tree)
3. Adjust parameters like strike price, maturity, volatility
4. Click "Calculate Option Prices" to view results
5. Optionally, view Monte Carlo paths or Binomial tree visualization

### Risk Analysis

1. Navigate to the "Risk Analysis" tab
2. Select VaR calculation parameters (confidence level, investment amount)
3. Choose between Historical or Monte Carlo VaR methods
4. Use the Performance Metrics section to compare with benchmarks
5. Explore historical volatility and return distributions

### Market Data

1. Go to the "Market Data" tab to view option chains
2. Select an expiration date to view available options
3. Use the visualization tools to examine implied volatility smiles
4. Analyze option Greeks for different strike prices
5. Export data for further analysis

### Strategy Builder

1. Navigate to the "Strategy Builder" tab
2. Select from predefined strategies or create a custom strategy
3. Adjust parameters and view the payoff diagram
4. Analyze risk metrics and break-even points
5. Understand strategy interpretations and use cases

### Advanced Models

1. Go to the "Advanced Models" tab
2. Explore the Heston stochastic volatility model
3. Adjust model parameters and view Monte Carlo simulations
4. Visualize the implied volatility surface
5. Compare with the standard Black-Scholes model

## 🔄 Deployment

For local development and use:

```bash
streamlit run app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Black-Scholes model formulation based on the original 1973 paper by Fischer Black and Myron Scholes
- Monte Carlo simulation techniques adapted from contemporary quantitative finance literature
- Risk metrics implementations follow industry standards for VaR and performance measurement
- Yahoo Finance API used for real-time market data