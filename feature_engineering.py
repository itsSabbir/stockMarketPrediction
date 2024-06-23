import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
from datetime import datetime

# Constants
INPUT_DIR = 'processed_data'
OUTPUT_DIR = 'enhanced_data'
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


def load_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Loads processed stock data from a CSV file.

    Parameters:
    symbol (str): The stock symbol to load data for.

    Returns:
    pd.DataFrame: DataFrame loaded from the CSV, or None if loading fails.
    """
    try:
        file_pattern = f'processed_{symbol}_*.csv'
        matching_files = [f for f in os.listdir(INPUT_DIR) if
                          f.startswith(f'processed_{symbol}_') and f.endswith('.csv')]

        if not matching_files:
            print(f"No processed data file found for {symbol}.")
            return None

        latest_file = max(matching_files)
        file_path = os.path.join(INPUT_DIR, latest_file)

        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        return df
    except FileNotFoundError:
        print(f"The file for {symbol} was not found.")
    except pd.errors.EmptyDataError:
        print(f"The file for {symbol} is empty.")
    except Exception as e:
        print(f"An error occurred while loading the data for {symbol}: {e}")
    return None


def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Computes the Relative Strength Index (RSI) for the given data.

    Parameters:
    data (pd.Series): Series of stock prices.
    window (int): The period over which to calculate RSI.

    Returns:
    pd.Series: The RSI values.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> tuple:
    """
    Computes the Moving Average Convergence Divergence (MACD) line and the signal line.

    Parameters:
    data (pd.Series): Series of stock prices.
    slow (int): The period for the slow moving average.
    fast (int): The period for the fast moving average.
    signal (int): The period for the signal line.

    Returns:
    tuple of pd.Series: The MACD line and the signal line.
    """
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
    """
    Computes Bollinger Bands for the given data.

    Parameters:
    data (pd.Series): Series of stock prices.
    window (int): The rolling window period.
    num_std (float): The number of standard deviations for the bands.

    Returns:
    tuple of pd.Series: The upper band, middle band, and lower band.
    """
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame: 200-day moving average, RSI, MACD, and Bollinger Bands.

    Parameters:
    df (pd.DataFrame): The DataFrame containing stock prices.

    Returns:
    pd.DataFrame: The DataFrame with additional columns for each indicator.
    """
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['macd'], df['signal'] = compute_macd(df['close'])
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = compute_bollinger_bands(df['close'])

    # Additional features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)  # Annualized volatility
    df['price_momentum'] = df['close'] / df['close'].shift(10) - 1  # 10-day price momentum

    df.dropna(inplace=True)
    return df


def save_enhanced_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Saves the enhanced data to a CSV file.

    Parameters:
    df (pd.DataFrame): The enhanced DataFrame.
    symbol (str): Stock symbol to append to the file name for identification.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not df.empty:
        output_file = os.path.join(OUTPUT_DIR, f'enhanced_{symbol}_{datetime.now().strftime("%Y-%m-%d")}.csv')
        df.to_csv(output_file)
        print(f'Enhanced data for {symbol} saved successfully to {output_file}.')
    else:
        print(f'No data to save for {symbol}.')


def main():
    symbols = load_fortune_1000_symbols()

    if not symbols:
        print("No symbols loaded. Exiting.")
        return

    enhanced_stocks: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        print(f"Processing data for {symbol}...")
        df = load_data(symbol)

        if df is not None and not df.empty:
            try:
                enhanced_df = add_technical_indicators(df)
                enhanced_stocks[symbol] = enhanced_df
                save_enhanced_data(enhanced_df, symbol)
                print(f"Enhanced data for {symbol} processed and saved.")
            except Exception as e:
                print(f"Error processing data for {symbol}: {e}")
        else:
            print(f"No data available for {symbol}. Skipping.")

    print(f"Feature engineering completed for {len(enhanced_stocks)} out of {len(symbols)} symbols.")


if __name__ == '__main__':
    main()