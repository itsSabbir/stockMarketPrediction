import requests
import pandas as pd
import time
from datetime import datetime
import os
import csv
from typing import List, Dict, Optional

# Constants
API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
BASE_URL = 'https://www.alphavantage.co/query'
OUTPUT_DIR = 'stock_data'
FORTUNE_1000_FILE = 'fortune1000.csv'  # Ensure this file exists with company symbols
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 12  # Alpha Vantage has a limit of 5 API requests per minute for free tier

def load_fortune_1000_symbols() -> List[str]:
    """
    Load Fortune 1000 company symbols from a CSV file.
    """
    symbols = []
    try:
        with open(FORTUNE_1000_FILE, 'r') as f:
            reader = csv.DictReader(f)
            symbols = [row['Symbol'] for row in reader if row['Symbol']]
    except FileNotFoundError:
        print(f"Error: {FORTUNE_1000_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Error loading Fortune 1000 symbols: {e}")
    return symbols

def fetch_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetches daily adjusted stock data for a given symbol using the Alpha Vantage API.
    
    Parameters:
    symbol (str): The stock symbol to fetch data for.
    
    Returns:
    pd.DataFrame: A DataFrame containing stock data or None if an error occurs.
    """
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': API_KEY
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                print(f"No data found for {symbol}, check the symbol and API limitations.")
                return None
            
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.columns = [col.split('. ')[1] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            df = df.astype(float)
            
            if df.isnull().all(axis=1).any():
                print(f"Warning: Encountered entirely NaN rows in the DataFrame for {symbol}.")
            
            return df
        
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP Error for {symbol}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request Error for {symbol}: {req_err}")
        except ValueError as val_err:
            print(f"Value Error for {symbol}: {val_err}")
        except Exception as e:
            print(f"An unexpected error occurred for {symbol}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"Retrying {symbol} in 5 seconds...")
            time.sleep(5)
    
    print(f"Failed to fetch data for {symbol} after {MAX_RETRIES} attempts.")
    return None

def save_to_csv(symbol: str, df: pd.DataFrame) -> None:
    """
    Save the DataFrame to a CSV file.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    filename = f'{OUTPUT_DIR}/{symbol}_{datetime.now().strftime("%Y-%m-%d")}.csv'
    df.to_csv(filename)
    print(f'Saved {symbol} data to {filename}')

def main():
    symbols = load_fortune_1000_symbols()
    
    if not symbols:
        print("No symbols loaded. Exiting.")
        return
    
    stocks_data: Dict[str, pd.DataFrame] = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"Fetching data for {symbol} ({i}/{len(symbols)})...")
        df = fetch_stock_data(symbol)
        
        if df is not None:
            stocks_data[symbol] = df
            save_to_csv(symbol, df)
            print(f"Data collected and saved for {symbol}.")
        else:
            print(f"Failed to fetch data for {symbol}.")
        
        # Respect rate limits
        if i % 5 == 0:
            print(f"Pausing for {RATE_LIMIT_DELAY} seconds to respect API rate limits...")
            time.sleep(RATE_LIMIT_DELAY)
    
    print(f"Data collection completed for {len(stocks_data)} out of {len(symbols)} symbols.")

if __name__ == '__main__':
    main()