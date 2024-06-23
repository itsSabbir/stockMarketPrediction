import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional
from datetime import datetime

# Constants
INPUT_DIR = 'processed_data'
OUTPUT_DIR = 'visualizations'
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


def plot_stock_price(df: pd.DataFrame, symbol: str) -> None:
    """
    Plots the stock price over time.

    Parameters:
    df (pd.DataFrame): The stock data DataFrame.
    symbol (str): The stock symbol.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['close_ma_20'], label='20-day MA')
    plt.title(f'{symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    save_plot(f'{symbol}_price')


def plot_volume(df: pd.DataFrame, symbol: str) -> None:
    """
    Plots the trading volume over time.

    Parameters:
    df (pd.DataFrame): The stock data DataFrame.
    symbol (str): The stock symbol.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df['volume'], alpha=0.7)
    plt.plot(df.index, df['volume_ma_5'], color='red', label='5-day MA')
    plt.title(f'{symbol} Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    save_plot(f'{symbol}_volume')


def plot_returns_distribution(df: pd.DataFrame, symbol: str) -> None:
    """
    Plots the distribution of daily returns.

    Parameters:
    df (pd.DataFrame): The stock data DataFrame.
    symbol (str): The stock symbol.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['daily_return'].dropna(), kde=True, bins=50)
    plt.title(f'{symbol} Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    save_plot(f'{symbol}_returns_dist')


def plot_correlation_heatmap(df: pd.DataFrame, symbol: str) -> None:
    """
    Plots a correlation heatmap of the features.

    Parameters:
    df (pd.DataFrame): The stock data DataFrame.
    symbol (str): The stock symbol.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'{symbol} Feature Correlation Heatmap')
    save_plot(f'{symbol}_correlation')


def save_plot(filename: str) -> None:
    """
    Saves the current plot to a file.

    Parameters:
    filename (str): The name of the file to save the plot to.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.close()


def main():
    symbols = load_fortune_1000_symbols()

    if not symbols:
        print("No symbols loaded. Exiting.")
        return

    for symbol in symbols:
        print(f"Processing visualizations for {symbol}...")
        df = load_data(symbol)

        if df is not None and not df.empty:
            plot_stock_price(df, symbol)
            plot_volume(df, symbol)
            plot_returns_distribution(df, symbol)
            plot_correlation_heatmap(df, symbol)
            print(f"Visualizations for {symbol} completed.")
        else:
            print(f"No data available for {symbol}. Skipping visualizations.")

    print("All visualizations completed.")


if __name__ == '__main__':
    main()