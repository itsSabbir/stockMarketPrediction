import pandas as pd
import matplotlib.pyplot as plt


def load_processed_data(file_path):
    """
    Loads processed stock data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing processed stock data.

    Returns:
        pd.DataFrame: DataFrame loaded from the CSV.
    """
    try:
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        return df
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
    return None


def plot_stock_data(df, symbol):
    """
    Plots the closing price and 50-day moving average of a stock.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the stock data.
        symbol (str): The stock symbol.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['ma50'], label='50-Day MA')
    plt.title(f'{symbol} Stock Closing Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    for symbol in symbols:
        file_path = f'processed_{symbol}.csv'
        print(f"Loading processed data for {symbol} from {file_path}...")
        df = load_processed_data(file_path)
        if df is not None:
            print(f"Plotting data for {symbol}...")
            df['ma50'] = df['close'].rolling(window=50).mean()
            plot_stock_data(df, symbol)
        else:
            print(f"Failed to load data for {symbol}.")


if __name__ == '__main__':
    main()
