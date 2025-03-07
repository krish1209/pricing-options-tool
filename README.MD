# Options Pricing & Risk Analysis Tool

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-Compatible-4285F4.svg)](https://cloud.google.com/)

A comprehensive Python application for pricing European options using Black-Scholes, Monte Carlo simulation, and Binomial tree models, with integrated risk metrics analysis and real-time market data visualization.

![Options Pricing App Demo](https://via.placeholder.com/800x400?text=Options+Pricing+Dashboard)

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

- **Enterprise-Ready Deployment**:
  - Dockerized for consistent deployment
  - Google Cloud Platform integration
  - 100% compatibility with modern cloud infrastructure

## 📊 Screenshots

<details>
<summary>Click to expand screenshots</summary>

### Option Pricing Models Comparison
![Pricing Models](https://via.placeholder.com/700x350?text=Option+Pricing+Models)

### Risk Analysis Dashboard
![Risk Analysis](https://via.placeholder.com/700x350?text=Risk+Analysis+Dashboard)

### Volatility Surface
![Volatility Surface](https://via.placeholder.com/700x350?text=Implied+Volatility+Surface)

### Option Greeks Visualization
![Option Greeks](https://via.placeholder.com/700x350?text=Option+Greeks+Visualization)

</details>

## 🛠️ Technologies

- **Backend**: Python 3.10, NumPy, Pandas, SciPy
- **Financial Modeling**: Black-Scholes, Monte Carlo, Binomial Tree implementations
- **Data Source**: Yahoo Finance API via yfinance
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Deployment**: Docker, Google Cloud Platform (Cloud Run)

## 📋 Prerequisites

- Python 3.9+ (3.10 recommended)
- Docker (for containerized deployment)
- Google Cloud SDK (for GCP deployment)

## 🔧 Installation

### Method 1: Local Python Installation

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

### Method 2: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/options-pricing-app.git
cd options-pricing-app

# Build and run with Docker Compose
docker-compose up --build

# Alternatively, use Docker directly
docker build -t options-pricing-app .
docker run -p 8501:8501 options-pricing-app
```

The application will be available at http://localhost:8501

### Method 3: Google Cloud Platform Deployment

See the [GCP Deployment Guide](./docs/gcp-deployment.md) for detailed instructions.

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
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
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

## 🔄 Deployment Options

### Local Development

Run with Streamlit for local development and testing:

```bash
streamlit run app.py
```

### Docker Deployment

```bash
docker-compose up
```

### Google Cloud Run Deployment

Deploy to GCP Cloud Run for scalable, serverless hosting:

```bash
# Build and tag the image
docker build -t gcr.io/[PROJECT_ID]/options-pricing-app .

# Push to Google Container Registry
docker push gcr.io/[PROJECT_ID]/options-pricing-app

# Deploy to Cloud Run
gcloud run deploy options-pricing-app --image gcr.io/[PROJECT_ID]/options-pricing-app --platform managed
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Black-Scholes model formulation based on the original 1973 paper by Fischer Black and Myron Scholes
- Monte Carlo simulation techniques adapted from contemporary quantitative finance literature
- Risk metrics implementations follow industry standards for VaR and performance measurement
- Yahoo Finance API used for real-time market data