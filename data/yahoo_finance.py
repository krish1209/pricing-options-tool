import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period to fetch data for (e.g., '1d', '5d', '1mo', '1y')
    interval : str
        Data interval (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing stock data
    """
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Clean data
        data = data.dropna()
        
        # Add return calculations
        if not data.empty and len(data) > 1:
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_option_chain(ticker):
    """
    Fetch option chain data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    tuple
        (calls_df, puts_df) containing option chain data
    """
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        
        # Get all available expiration dates
        expirations = stock.options
        
        if not expirations:
            return pd.DataFrame(), pd.DataFrame()
        
        # Initialize DataFrames for calls and puts
        all_calls = pd.DataFrame()
        all_puts = pd.DataFrame()
        
        # Fetch option chains for each expiration
        for exp_date in expirations:
            opt_chain = stock.option_chain(exp_date)
            
            # Add expiration date column
            calls = opt_chain.calls.copy()
            calls['expirationDate'] = exp_date
            
            puts = opt_chain.puts.copy()
            puts['expirationDate'] = exp_date
            
            # Append to main DataFrames
            all_calls = pd.concat([all_calls, calls])
            all_puts = pd.concat([all_puts, puts])
        
        # Reset indices
        all_calls.reset_index(drop=True, inplace=True)
        all_puts.reset_index(drop=True, inplace=True)
        
        return all_calls, all_puts
    
    except Exception as e:
        print(f"Error fetching option chain for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_implied_volatility(ticker, period='1y'):
    """
    Calculate historical volatility from stock data
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period to fetch data for
        
    Returns:
    --------
    float
        Annualized volatility
    """
    try:
        # Fetch data
        data = get_stock_data(ticker, period=period)
        
        if data.empty or len(data) < 2:
            return None
        
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate annualized volatility
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days in a year
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility for {ticker}: {e}")
        return None

def get_risk_free_rate():
    """
    Get current risk-free rate from US Treasury yield
    
    Returns:
    --------
    float
        Current risk-free rate (annualized)
    """
    try:
        # Fetch 1-year Treasury yield as proxy for risk-free rate
        treasury = yf.Ticker("^TNX")  # 10-year Treasury yield
        data = treasury.history(period="1d")
        
        if data.empty:
            # Default to a reasonable value if fetching fails
            return 0.04  # 4%
        
        # Convert from percentage to decimal
        risk_free_rate = data['Close'].iloc[-1] / 100
        
        return risk_free_rate
    
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        # Return a default value
        return 0.04  # 4%

def get_market_data(start_date=None, end_date=None):
    """
    Get market data (S&P 500) for beta calculation
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing market data
    """
    try:
        # Default to 1 year of data if no dates specified
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
            
        if not start_date:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = pd.to_datetime(start_date)
        
        # Fetch S&P 500 data
        market = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Calculate returns
        market['Daily_Return'] = market['Close'].pct_change()
        
        return market
    
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return pd.DataFrame()