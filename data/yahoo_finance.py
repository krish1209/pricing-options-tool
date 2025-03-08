import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a session with retry strategy
def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Set browser-like headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    })
    return session

# Global session for reuse
session = create_session()

def get_stock_data(ticker, period='1y', interval='1d'):
    """
    Fetch stock data from Yahoo Finance with aggressive refresh
    
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
    global session
    
    # Force period lookback to ensure fresh data
    if period == '1d':
        # For 1d, use 2d to make sure we get today's data
        lookback_period = '2d'
    elif period == '5d':
        lookback_period = '7d'
    else:
        lookback_period = period
    
    tries = 0
    max_tries = 5
    
    while tries < max_tries:
        try:
            # Add delay between retries
            if tries > 0:
                time.sleep(tries * 0.5)  # Progressive delay
                # Refresh session after failed attempts
                session = create_session()
            
            # Print detailed info for debugging
            print(f"Fetching live data for {ticker} (attempt {tries+1}/{max_tries})")
            
            # Directly use yfinance with our session
            ticker_obj = yf.Ticker(ticker, session=session)
            data = ticker_obj.history(period=lookback_period, interval=interval)
            
            # Check if we got valid data
            if data.empty:
                print(f"Empty data returned for {ticker}")
                tries += 1
                continue
                
            print(f"Successfully fetched data for {ticker} with {len(data)} rows")
            print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
            
            # Truncate to requested period if needed
            if period != lookback_period and len(data) > 1:
                if period == '1d':
                    data = data.tail(1)
                elif period == '5d':
                    data = data.tail(5)
            
            # Add return calculations
            if len(data) > 1:
                data['Daily_Return'] = data['Close'].pct_change()
                data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            return data
        
        except Exception as e:
            print(f"Error fetching data for {ticker} (attempt {tries+1}): {str(e)}")
            tries += 1
    
    # If all attempts fail, return empty DataFrame
    print(f"Failed to fetch data for {ticker} after {max_tries} attempts")
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
    global session
    
    # Get fresh session
    fresh_session = create_session()
    
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            if tries > 0:
                time.sleep(tries * 1.0)
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
            
            # Get option chains
            for exp_date in process_expirations:
                try:
                    # Add delay between expiration requests
                    if exp_date != process_expirations[0]:
                        time.sleep(0.5)
                    
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

def calculate_implied_volatility(ticker, period='1y'):
    """
    Calculate historical volatility from stock data
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    period : str
        Time period to calculate for
        
    Returns:
    --------
    float
        Annualized volatility
    """
    try:
        # Get stock data
        data = get_stock_data(ticker, period=period)
        
        if data.empty or len(data) < 5:
            print(f"Insufficient data for volatility calculation for {ticker}")
            return 0.3  # Default volatility
        
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        
        # Calculate annualized volatility
        daily_volatility = log_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        print(f"Calculated volatility for {ticker}: {annualized_volatility:.4f}")
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility for {ticker}: {str(e)}")
        return 0.3  # Default volatility

def get_risk_free_rate():
    """
    Get current risk-free rate from US Treasury yield
    
    Returns:
    --------
    float
        Current risk-free rate (annualized)
    """
    global session
    
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            # Fresh session for Treasury data
            if tries > 0:
                time.sleep(tries * 0.5)
                session = create_session()
                
            print("Fetching current risk-free rate...")
            
            # Fetch 10-year Treasury yield
            treasury = yf.Ticker("^TNX", session=session)
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
    
    # Default value if all attempts fail
    print("Using default risk-free rate")
    return 0.0429  # ~4.29% (current 10Y Treasury yield)

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
    global session
    fresh_session = create_session()
    
    # Set date range
    if not end_date:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
        
    if not start_date:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = pd.to_datetime(start_date)
    
    tries = 0
    max_tries = 3
    
    while tries < max_tries:
        try:
            if tries > 0:
                time.sleep(tries * 0.5)
                fresh_session = create_session()
            
            print(f"Fetching S&P 500 data from {start_date.date()} to {end_date.date()}")
            
            # Fetch S&P 500 data with session
            market = yf.download(
                "^GSPC",
                start=start_date,
                end=end_date,
                session=fresh_session,
                progress=False
            )
            
            if market.empty:
                print("Empty market data returned")
                tries += 1
                continue
            
            # Calculate returns
            print(f"Successfully fetched market data with {len(market)} rows")
            market['Daily_Return'] = market['Close'].pct_change()
            
            return market
            
        except Exception as e:
            print(f"Error fetching market data (attempt {tries+1}): {str(e)}")
            tries += 1
    
    # Return empty DataFrame if all attempts fail
    print("Failed to fetch market data")
    return pd.DataFrame()