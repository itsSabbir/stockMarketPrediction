import pandas as pd


def process_data(df):
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
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['daily_return'] = df['close'].pct_change()

    # Remove rows with any missing values
    df.dropna(inplace=True)

    # Remove days with no trading volume
    df = df[df['volume'] != 0]

    # Handling outliers in 'close' price
    q_low = df['close'].quantile(0.01)
    q_high = df['close'].quantile(0.99)
    df = df[(df['close'] > q_low) & (df['close'] < q_high)]

    return df


def load_data(file_path):
    """
    Loads stock data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing stock data.

    Returns:
        pd.DataFrame: DataFrame loaded from the CSV.
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
    return None


def save_processed_data(df, symbol):
    """
    Saves the processed data to a CSV file.

    Parameters:
        df (pd.DataFrame): The processed DataFrame.
        symbol (str): Stock symbol to append to the file name for identification.
    """
    if not df.empty:
        df.to_csv(f'processed_{symbol}.csv')
        print(f'Processed data for {symbol} saved successfully.')
    else:
        print(f'No data to save for {symbol}.')


def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    processed_stocks = {}

    for symbol in symbols:
        file_path = f'{symbol}_data.csv'
        print(f"Loading data for {symbol} from {file_path}...")
        df = load_data(file_path)
        if df is not None:
            print(f"Processing data for {symbol}...")
            processed_df = process_data(df)
            processed_stocks[symbol] = processed_df
            save_processed_data(processed_df, symbol)
        else:
            print(f"Failed to load data for {symbol}.")

if __name__ == '__main__':
    main()
