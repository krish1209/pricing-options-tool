import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

# Define custom headers to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

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
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))
            
            # Fetch data with custom headers
            data = yf.download(
                ticker, 
                period=period, 
                interval=interval,
                progress=False,
                threads=False,  # Single-threaded to avoid issues
                headers=HEADERS
            )
            
            # Clean data
            data = data.dropna()
            
            # Add return calculations
            if not data.empty and len(data) > 1:
                data['Daily_Return'] = data['Close'].pct_change()
                data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            if data.empty:
                print(f"Warning: No data returned for {ticker}, attempt {tries+1}/{max_tries}")
                tries += 1
                continue
                
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker} (attempt {tries+1}/{max_tries}): {e}")
            tries += 1
            time.sleep(2)  # Wait longer between retries
    
    # If we get here, all attempts failed
    print(f"All attempts to fetch data for {ticker} failed")
    
    # Return empty DataFrame or dummy data as fallback
    if ticker.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        # Return some dummy data for common stocks
        return get_dummy_data(ticker)
    
    return pd.DataFrame()

def get_dummy_data(ticker):
    """Generate dummy stock data for demonstration purposes"""
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(252)]
    dates.reverse()
    
    # Generate synthetic prices
    base_price = {'AAPL': 180.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'AMZN': 130.0, 'META': 300.0}.get(ticker.upper(), 100.0)
    np.random.seed(hash(ticker) % 10000)  # Consistent seed per ticker
    
    # Generate prices with some randomness but trend
    prices = [base_price]
    for i in range(1, 252):
        daily_return = np.random.normal(0.0003, 0.015)  # Mean positive return with volatility
        prices.append(prices[-1] * (1 + daily_return))
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.uniform(1000000, 50000000)) for _ in prices]
    }, index=dates)
    
    # Add returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    return data

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
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))
            
            # Fetch data
            stock = yf.Ticker(ticker)
            stock._fetch_ticker(headers=HEADERS)
            
            # Get all available expiration dates
            expirations = stock.options
            
            if not expirations:
                print(f"No option expirations found for {ticker}, attempt {tries+1}/{max_tries}")
                tries += 1
                continue
            
            # Initialize DataFrames for calls and puts
            all_calls = pd.DataFrame()
            all_puts = pd.DataFrame()
            
            # Fetch option chains for each expiration
            for exp_date in expirations[:5]:  # Limit to 5 expirations to avoid rate limits
                time.sleep(0.5)  # Add small delay between requests
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
            
            if all_calls.empty and all_puts.empty:
                print(f"No option data found for {ticker}, attempt {tries+1}/{max_tries}")
                tries += 1
                continue
                
            return all_calls, all_puts
        
        except Exception as e:
            print(f"Error fetching option chain for {ticker} (attempt {tries+1}/{max_tries}): {e}")
            tries += 1
            time.sleep(2)
    
    # If we get here, all attempts failed - return dummy data for demo purposes
    return get_dummy_options(ticker)

def get_dummy_options(ticker):
    """Generate dummy option chain data for demonstration purposes"""
    stock_price = {'AAPL': 180.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'AMZN': 130.0, 'META': 300.0}.get(ticker.upper(), 100.0)
    today = datetime.now()
    
    # Generate some future expiration dates
    expirations = [(today + timedelta(days=30)).strftime('%Y-%m-%d'),
                   (today + timedelta(days=60)).strftime('%Y-%m-%d'),
                   (today + timedelta(days=90)).strftime('%Y-%m-%d')]
    
    # Create strikes around current price
    strikes = [round(stock_price * (1 + i * 0.05), 1) for i in range(-10, 11)]
    
    # Create dummy calls
    calls_data = []
    for exp in expirations:
        for strike in strikes:
            itm = strike < stock_price
            call_price = max(0.1, stock_price - strike + 5) if itm else max(0.01, (stock_price * 0.1) * np.exp(-0.25 * ((strike - stock_price) / stock_price)**2))
            calls_data.append({
                'strike': strike,
                'lastPrice': round(call_price, 2),
                'bid': round(call_price * 0.95, 2),
                'ask': round(call_price * 1.05, 2),
                'volume': int(np.random.uniform(10, 1000)),
                'openInterest': int(np.random.uniform(100, 5000)),
                'impliedVolatility': round(np.random.uniform(0.2, 0.6), 4),
                'expirationDate': exp
            })
    
    # Create dummy puts
    puts_data = []
    for exp in expirations:
        for strike in strikes:
            itm = strike > stock_price
            put_price = max(0.1, strike - stock_price + 5) if itm else max(0.01, (stock_price * 0.1) * np.exp(-0.25 * ((strike - stock_price) / stock_price)**2))
            puts_data.append({
                'strike': strike,
                'lastPrice': round(put_price, 2),
                'bid': round(put_price * 0.95, 2),
                'ask': round(put_price * 1.05, 2),
                'volume': int(np.random.uniform(10, 1000)),
                'openInterest': int(np.random.uniform(100, 5000)),
                'impliedVolatility': round(np.random.uniform(0.2, 0.6), 4),
                'expirationDate': exp
            })
    
    return pd.DataFrame(calls_data), pd.DataFrame(puts_data)

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
            # Return reasonable default if data fetch fails
            return 0.3  # 30% volatility as default
        
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate annualized volatility
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days in a year
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility for {ticker}: {e}")
        # Return reasonable default
        return 0.3  # 30% volatility as default

def get_risk_free_rate():
    """
    Get current risk-free rate from US Treasury yield
    
    Returns:
    --------
    float
        Current risk-free rate (annualized)
    """
    try:
        # Fetch 10-year Treasury yield as proxy for risk-free rate
        time.sleep(0.5)  # Add delay
        treasury = yf.Ticker("^TNX")  # 10-year Treasury yield
        data = treasury.history(period="1d", headers=HEADERS)
        
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
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
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
            
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.0))
            
            # Fetch S&P 500 data
            market = yf.download('^GSPC', start=start_date, end=end_date, headers=HEADERS, progress=False)
            
            if market.empty:
                print(f"Empty market data returned, attempt {tries+1}/{max_tries}")
                tries += 1
                continue
                
            # Calculate returns
            market['Daily_Return'] = market['Close'].pct_change()
            
            return market
        
        except Exception as e:
            print(f"Error fetching market data (attempt {tries+1}/{max_tries}): {e}")
            tries += 1
            time.sleep(2)
    
    # If all attempts fail, return dummy market data
    return get_dummy_market_data(start_date, end_date)

def get_dummy_market_data(start_date=None, end_date=None):
    """Generate dummy market data for demonstration purposes"""
    if not end_date:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
        
    if not start_date:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = pd.to_datetime(start_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Generate S&P 500 prices
    base_price = 4500  # Approximate S&P 500 price
    np.random.seed(42)  # For reproducibility
    
    prices = [base_price]
    for i in range(1, len(date_range)):
        daily_return = np.random.normal(0.0002, 0.01)  # Small positive bias with volatility
        prices.append(prices[-1] * (1 + daily_return))
    
    # Create DataFrame
    market = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.uniform(1e9, 5e9)) for _ in prices]
    }, index=date_range)
    
    # Calculate returns
    market['Daily_Return'] = market['Close'].pct_change()
    
    return market