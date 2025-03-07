import yfinance as yf
import pandas as pd
import time
import random
def test_ticker(symbol="AAPL"):
    print(f"Testing data fetch for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        if hist.empty:
            print(f"Got empty data for {symbol}")
        else:
            print(f"Successfully retrieved {len(hist)} days of data:")
            print(hist.head())
            
        print("\nTesting option chain...")
        try:
            options = ticker.options
            if options:
                print(f"Available option dates: {options[:3]}...")
                # Get options for the first available date
                if options:
                    calls = ticker.option_chain(options[0]).calls
                    print(f"Found {len(calls)} call options")
            else:
                print("No options data available")
        except Exception as e:
            print(f"Error fetching options: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ticker("AAPL")
    test_ticker("MSFT")