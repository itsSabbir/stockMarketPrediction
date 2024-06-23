import pandas as pd
import os
from typing import Dict, Optional
import numpy as np
from datetime import datetime

# Constants
INPUT_DIR = 'stock_data'
OUTPUT_DIR = 'processed_data'
FORTUNE_1000_FILE = 'fortune1000.csv'

def load_fortune_1000_symbols() -> list:
    """
    Load Fortune 1000 company symbols from a CSV file.
    """
    try:
        df = pd.read_csv(FORTUNE_1000_FILE)
        return df['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: {FORTUNE_1000_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Error loading Fortune 1000 symbols: {e}")
    return []

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes stock data by selecting relevant columns, calculating daily returns,
    removing rows with missing values or zero trading volume, and handling outliers.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing raw stock data.
    
    Returns:
    pd.DataFrame: A cleaned and processed DataFrame.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty and cannot be processed.")
    
    # Select necessary columns and calculate daily returns
    try:
        df = df[['adjusted close', 'open', 'high', 'low', 'close', 'volume']]
        df['daily_return'] = df['adjusted close'].pct_change()
        
        # Remove rows with any missing values
        df.dropna(inplace=True)
        
        # Remove days with no trading volume
        df = df[df['volume'] != 0]
        
        # Handling outliers in 'close' price using Interquartile Range (IQR)
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]
        
        # Calculate additional features
        df['price_range'] = df['high'] - df['low']
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['close_ma_20'] = df['close'].rolling(window=20).mean()
        
        return df
    
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame if processing fails

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads stock data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing stock data.
    
    Returns:
    pd.DataFrame: DataFrame loaded from the CSV.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        return df
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty.")
    except Exception as e:
        print(f"An error occurred while loading the data from {file_path}: {e}")
    return None

def save_processed_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Saves the processed data to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The processed DataFrame.
    symbol (str): Stock symbol to append to the file name for identification.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not df.empty:
        output_file = os.path.join(OUTPUT_DIR, f'processed_{symbol}_{datetime.now().strftime("%Y-%m-%d")}.csv')
        df.to_csv(output_file)
        print(f'Processed data for {symbol} saved successfully to {output_file}.')
    else:
        print(f'No data to save for {symbol}.')

def main():
    symbols = load_fortune_1000_symbols()
    
    if not symbols:
        print("No symbols loaded. Exiting.")
        return
    
    processed_stocks: Dict[str, pd.DataFrame] = {}
    
    for symbol in symbols:
        file_path = os.path.join(INPUT_DIR, f'{symbol}_*.csv')
        matching_files = [f for f in os.listdir(INPUT_DIR) if f.startswith(f'{symbol}_') and f.endswith('.csv')]
        
        if not matching_files:
            print(f"No data file found for {symbol}.")
            continue
        
        latest_file = max(matching_files)  # Get the most recent file if multiple exist
        file_path = os.path.join(INPUT_DIR, latest_file)
        
        print(f"Loading data for {symbol} from {file_path}...")
        df = load_data(file_path)
        
        if df is not None:
            print(f"Processing data for {symbol}...")
            try:
                processed_df = process_data(df)
                if not processed_df.empty:
                    processed_stocks[symbol] = processed_df
                    save_processed_data(processed_df, symbol)
                else:
                    print(f"Processing resulted in an empty DataFrame for {symbol}.")
            except Exception as e:
                print(f"Error processing data for {symbol}: {e}")
        else:
            print(f"Failed to load data for {symbol}.")
    
    print(f"Data processing completed for {len(processed_stocks)} out of {len(symbols)} symbols.")

if __name__ == '__main__':
    main()