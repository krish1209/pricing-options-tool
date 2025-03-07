import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

# Create a session with advanced retry strategy
def create_session():
    session = requests.Session()
    retry = Retry(
        total=7,  # Increased total retries
        backoff_factor=1.0,  # Increased backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Added 429 (Too Many Requests)
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Allow POST for more flexibility
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Randomize User-Agent from a pool to avoid detection
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Edge/121.0.0.0'
    ]
    
    # Set browser-like headers with randomized User-Agent
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',  # Changed to no-cache to avoid stale data
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        # Add referer to appear more like a browser
        'Referer': 'https://finance.yahoo.com/',
    })
    return session

# Global session for reuse with periodic refresh
session = create_session()
last_session_refresh = datetime.now()

def refresh_session_if_needed():
    """Refresh the session if it's been more than 15 minutes"""
    global session, last_session_refresh
    
    if (datetime.now() - last_session_refresh).total_seconds() > 900:  # 15 minutes
        print("Refreshing session...")
        session = create_session()
        last_session_refresh = datetime.now()
        time.sleep(random.uniform(0.5, 1.5))  # Random delay after refresh

def get_stock_data(ticker, period='1y', interval='1d', use_direct_download=True):
    """
    Fetch stock data from Yahoo Finance with multiple fallback methods
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period to fetch data for (e.g., '1d', '5d', '1mo', '1y')
    interval : str
        Data interval (e.g., '1m', '5m', '1h', '1d')
    use_direct_download : bool
        Whether to try direct download approach first
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing stock data
    """
    global session
    refresh_session_if_needed()
    
    # Force period lookback to ensure fresh data
    if period == '1d':
        # For 1d, use 2d to make sure we get today's data
        lookback_period = '2d'
    elif period == '5d':
        lookback_period = '7d'
    else:
        lookback_period = period
    
    # Calculate date ranges for fallback method
    end_date = datetime.now()
    if period == '1d':
        start_date = end_date - timedelta(days=2)
    elif period == '5d':
        start_date = end_date - timedelta(days=7)
    elif period == '1mo':
        start_date = end_date - timedelta(days=31)
    elif period == '3mo':
        start_date = end_date - timedelta(days=92)
    elif period == '6mo':
        start_date = end_date - timedelta(days=183)
    elif period == '1y':
        start_date = end_date - timedelta(days=366)
    elif period == '2y':
        start_date = end_date - timedelta(days=731)
    elif period == '5y':
        start_date = end_date - timedelta(days=1827)
    else:
        start_date = end_date - timedelta(days=366)  # Default to 1 year
    
    # Method 1: Try yfinance Ticker method
    def try_ticker_method():
        tries = 0
        max_tries = 3
        
        while tries < max_tries:
            try:
                # Add delay between retries with randomization to avoid patterns
                if tries > 0:
                    delay = tries * 1.0 + random.uniform(0.2, 0.8)
                    time.sleep(delay)
                    # Refresh session for each retry
                    fresh_session = create_session()
                else:
                    fresh_session = session
                
                print(f"Method 1: Fetching data for {ticker} (attempt {tries+1}/{max_tries})")
                
                # Use Ticker object
                ticker_obj = yf.Ticker(ticker, session=fresh_session)
                data = ticker_obj.history(period=lookback_period, interval=interval)
                
                if data.empty:
                    print(f"Empty data returned for {ticker}")
                    tries += 1
                    continue
                
                print(f"Successfully fetched data for {ticker} with {len(data)} rows")
                if not data.empty:
                    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
                
                return data
            
            except Exception as e:
                print(f"Method 1 Error for {ticker} (attempt {tries+1}): {str(e)}")
                tries += 1
        
        return pd.DataFrame()
    
    # Method 2: Try yfinance download method
    def try_download_method():
        tries = 0
        max_tries = 3
        
        while tries < max_tries:
            try:
                # Add delay with randomization
                if tries > 0:
                    delay = tries * 1.0 + random.uniform(0.5, 1.0)
                    time.sleep(delay)
                    # Refresh session for each retry
                    fresh_session = create_session()
                else:
                    fresh_session = session
                
                print(f"Method 2: Downloading data for {ticker} (attempt {tries+1}/{max_tries})")
                
                # Use download function
                data = yf.download(
                    ticker, 
                    period=lookback_period,
                    interval=interval,
                    progress=False,
                    session=fresh_session
                )
                
                if data.empty:
                    print(f"Empty data returned for {ticker}")
                    tries += 1
                    continue
                
                print(f"Successfully downloaded data for {ticker} with {len(data)} rows")
                if not data.empty:
                    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
                
                return data
            
            except Exception as e:
                print(f"Method 2 Error for {ticker} (attempt {tries+1}): {str(e)}")
                tries += 1
        
        return pd.DataFrame()
    
    # Method 3: Try yfinance download with explicit dates (different API endpoint)
    def try_date_range_method():
        tries = 0
        max_tries = 3
        
        while tries < max_tries:
            try:
                # Add delay with randomization
                if tries > 0:
                    delay = tries * 1.5 + random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    # Refresh session for each retry
                    fresh_session = create_session()
                else:
                    fresh_session = session
                
                print(f"Method 3: Downloading data with date range for {ticker} (attempt {tries+1}/{max_tries})")
                
                # Format dates as strings for yfinance
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')  # Add 1 day to include today
                
                # Use download with explicit dates
                data = yf.download(
                    ticker, 
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    progress=False,
                    session=fresh_session
                )
                
                if data.empty:
                    print(f"Empty data returned for {ticker}")
                    tries += 1
                    continue
                
                print(f"Successfully downloaded data for {ticker} with {len(data)} rows")
                if not data.empty:
                    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
                
                return data
            
            except Exception as e:
                print(f"Method 3 Error for {ticker} (attempt {tries+1}): {str(e)}")
                tries += 1
        
        return pd.DataFrame()
    
    # Method 4: Try direct Yahoo API query (more reliable but complex)
    def try_direct_api_method():
        tries = 0
        max_tries = 3
        
        # Calculate Unix timestamps for API
        end_unix = int(end_date.timestamp())
        start_unix = int(start_date.timestamp())
        
        # Map interval to Yahoo's format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '60m': '60m', '1h': '60m',
            '1d': '1d', '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
        }
        yahoo_interval = interval_map.get(interval, '1d')
        
        while tries < max_tries:
            try:
                # Add delay with randomization
                if tries > 0:
                    delay = tries * 2.0 + random.uniform(1.0, 2.0)
                    time.sleep(delay)
                    # Refresh session for each retry
                    fresh_session = create_session()
                else:
                    fresh_session = session
                
                print(f"Method 4: Direct API query for {ticker} (attempt {tries+1}/{max_tries})")
                
                # Build Yahoo Finance API URL
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                params = {
                    'period1': start_unix,
                    'period2': end_unix,
                    'interval': yahoo_interval,
                    'includePrePost': 'false',
                    'events': 'div,split',
                    'corsDomain': 'finance.yahoo.com',
                }
                
                # Make request with randomized delay
                time.sleep(random.uniform(0.1, 0.3))
                response = fresh_session.get(url, params=params)
                
                if response.status_code != 200:
                    print(f"API request failed with status code {response.status_code}")
                    tries += 1
                    continue
                
                # Parse JSON response
                data_json = response.json()
                
                # Check if we have valid data
                if 'chart' not in data_json or 'result' not in data_json['chart'] or not data_json['chart']['result']:
                    print(f"Invalid API response format for {ticker}")
                    tries += 1
                    continue
                
                # Extract price data
                chart_data = data_json['chart']['result'][0]
                timestamps = chart_data['timestamp']
                quote = chart_data['indicators']['quote'][0]
                
                # Check for missing fields
                required_fields = ['open', 'high', 'low', 'close', 'volume']
                if not all(field in quote for field in required_fields):
                    print(f"Missing required fields in API response for {ticker}")
                    tries += 1
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame({
                    'Open': quote['open'],
                    'High': quote['high'],
                    'Low': quote['low'],
                    'Close': quote['close'],
                    'Volume': quote['volume']
                }, index=pd.to_datetime([datetime.fromtimestamp(x) for x in timestamps]))
                
                # Handle missing data
                df = df.dropna(subset=['Close'])
                
                if df.empty:
                    print(f"Empty data after processing for {ticker}")
                    tries += 1
                    continue
                
                print(f"Successfully fetched API data for {ticker} with {len(df)} rows")
                print(f"Latest price: ${df['Close'].iloc[-1]:.2f}")
                
                return df
            
            except Exception as e:
                print(f"Method 4 Error for {ticker} (attempt {tries+1}): {str(e)}")
                tries += 1
        
        return pd.DataFrame()
    
    # Try methods in order, starting with either direct download or Ticker based on parameter
    methods = []
    if use_direct_download:
        methods = [try_download_method, try_ticker_method, try_date_range_method, try_direct_api_method]
    else:
        methods = [try_ticker_method, try_download_method, try_date_range_method, try_direct_api_method]
    
    # Try each method until we get data
    for i, method in enumerate(methods):
        print(f"Trying method {i+1} for {ticker}...")
        data = method()
        
        if not data.empty:
            # Add return calculations
            if len(data) > 1:
                data['Daily_Return'] = data['Close'].pct_change()
                data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Truncate to requested period if needed
            if period != lookback_period and len(data) > 1:
                if period == '1d':
                    data = data.tail(1)
                elif period == '5d':
                    data = data.tail(5)
            
            return data
    
    # If all methods fail, return empty DataFrame
    print(f"All methods failed to fetch data for {ticker}")
    return pd.DataFrame()

# The rest of your functions with improved session handling

def get_option_chain(ticker):
    """Fetch option chain data with improved reliability"""
    global session
    refresh_session_if_needed()
    
    fresh_session = create_session()
    
    tries = 0
    max_tries = 4  # Increased retries
    
    while tries < max_tries:
        try:
            if tries > 0:
                delay = tries * 1.5 + random.uniform(0.5, 1.5)
                time.sleep(delay)
                fresh_session = create_session()
            
            # Create ticker object with session
            stock = yf.Ticker(ticker, session=fresh_session)
            
            # Get available expiration dates
            expirations = stock.options
            
            if not expirations:
                print(f"No option expiration dates found for {ticker}")
                tries += 1
                continue
            
            # Only process a subset of expirations to avoid rate limits
            process_expirations = expirations[:min(5, len(expirations))]
            print(f"Found {len(expirations)} expiration dates for {ticker}, processing {len(process_expirations)}")
            
            # Initialize DataFrames
            all_calls = pd.DataFrame()
            all_puts = pd.DataFrame()
            
            # Get option chains with progressive delays
            for i, exp_date in enumerate(process_expirations):
                try:
                    # Add delay between expiration requests with randomization
                    if i > 0:
                        delay = 0.5 + random.uniform(0.2, 0.8)
                        time.sleep(delay)
                    
                    # Get option chain for this expiration
                    print(f"Fetching options for {ticker} expiring {exp_date}")
                    opt_chain = stock.option_chain(exp_date)
                    
                    if not hasattr(opt_chain, 'calls') or not hasattr(opt_chain, 'puts'):
                        print(f"Invalid option chain data for {exp_date}")
                        continue
                    
                    # Add expiration date column
                    calls = opt_chain.calls.copy()
                    calls['expirationDate'] = exp_date
                    
                    puts = opt_chain.puts.copy()
                    puts['expirationDate'] = exp_date
                    
                    # Append to main DataFrames
                    all_calls = pd.concat([all_calls, calls])
                    all_puts = pd.concat([all_puts, puts])
                    
                except Exception as e:
                    print(f"Error processing {exp_date} options: {str(e)}")
                    continue
            
            # If we have data, return it
            if not all_calls.empty and not all_puts.empty:
                # Reset indices for clean DataFrames
                all_calls.reset_index(drop=True, inplace=True)
                all_puts.reset_index(drop=True, inplace=True)
                
                print(f"Successfully fetched option chain data for {ticker}")
                return all_calls, all_puts
            
            # If we reach here with no data, try again
            print(f"No valid option data found for {ticker}")
            tries += 1
            
        except Exception as e:
            print(f"Error fetching option chain for {ticker} (attempt {tries+1}): {str(e)}")
            tries += 1
    
    # If all attempts fail, return empty DataFrames
    print(f"Failed to fetch option chain for {ticker}")
    return pd.DataFrame(), pd.DataFrame()

def get_risk_free_rate():
    """Get current risk-free rate with multiple fallback methods"""
    global session
    refresh_session_if_needed()
    
    tries = 0
    max_tries = 4
    
    # Method 1: Try to get from Treasury yield
    while tries < max_tries:
        try:
            # Fresh session with delay
            if tries > 0:
                delay = tries * 1.0 + random.uniform(0.5, 1.0)
                time.sleep(delay)
                fresh_session = create_session()
            else:
                fresh_session = session
                
            print(f"Fetching current risk-free rate (attempt {tries+1})...")
            
            # Fetch 10-year Treasury yield
            treasury = yf.Ticker("^TNX", session=fresh_session)
            data = treasury.history(period="2d")
            
            if data.empty:
                print("Empty treasury data returned")
                tries += 1
                continue
            
            # Convert from percentage to decimal
            risk_free_rate = data['Close'].iloc[-1] / 100
            
            print(f"Current risk-free rate: {risk_free_rate:.4f}")
            return risk_free_rate
            
        except Exception as e:
            print(f"Error fetching risk-free rate (attempt {tries+1}): {str(e)}")
            tries += 1
    
    # Method 2: Try alternative source (3-month Treasury bill)
    tries = 0
    while tries < max_tries:
        try:
            # Fresh session with delay
            if tries > 0:
                delay = tries * 1.0 + random.uniform(0.5, 1.0)
                time.sleep(delay)
                fresh_session = create_session()
            else:
                fresh_session = session
                
            print(f"Trying alternative source for risk-free rate (attempt {tries+1})...")
            
            # Fetch 3-month Treasury bill
            treasury = yf.Ticker("^IRX", session=fresh_session)
            data = treasury.history(period="2d")
            
            if data.empty:
                print("Empty alternative treasury data returned")
                tries += 1
                continue
            
            # Convert from percentage to decimal
            risk_free_rate = data['Close'].iloc[-1] / 100
            
            print(f"Alternative risk-free rate: {risk_free_rate:.4f}")
            return risk_free_rate
            
        except Exception as e:
            print(f"Error fetching alternative risk-free rate (attempt {tries+1}): {str(e)}")
            tries += 1
    
    # Default value if all attempts fail
    print("Using default risk-free rate")
    return 0.0429  # ~4.29% (current 10Y Treasury yield)

# Example usage
if __name__ == "__main__":
    # Test the improved code
    ticker = "AAPL"
    print(f"Testing improved data fetching for {ticker}")
    
    # Try to get stock data
    stock_data = get_stock_data(ticker, period='1d')
    
    if not stock_data.empty:
        print("Successfully fetched stock data!")
        print(f"Latest price: ${stock_data['Close'].iloc[-1]:.2f}")
    else:
        print(f"Failed to fetch stock data for {ticker}")
    
    # Get current risk-free rate
    rf_rate = get_risk_free_rate()
    print(f"Current risk-free rate: {rf_rate:.4f}")