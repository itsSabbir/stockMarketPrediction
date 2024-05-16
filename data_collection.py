import requests
import pandas as pd
from datetime import datetime


def fetch_stock_data(symbol, api_key):
    """
    Fetches daily adjusted stock data for a given symbol using the Alpha Vantage API.

    Parameters:
        symbol (str): The stock symbol to fetch data for.
        api_key (str): API key for Alpha Vantage.

    Returns:
        pd.DataFrame: A DataFrame containing stock data or None if an error occurs.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()

        # Check if the expected data is present in the response
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No data found for {symbol}, check the symbol and API limitations.")

        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.columns = [col.split(' ')[1] for col in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        df = df.astype(float)

        # Check for any full rows of NaNs which indicate missing data
        if df.isnull().all(axis=1).any():
            raise ValueError("Encountered entirely NaN rows in the DataFrame.")
        return df
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error: {req_err}")
    except ValueError as val_err:
        print(f"Value Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def main():
    API_KEY = 'OFAECQYH7MVBYVZX'
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    stocks_data = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = fetch_stock_data(symbol, API_KEY)
        if df is not None:
            stocks_data[symbol] = df
            print(f"Data collected for {symbol}.")
        else:
            print(f"Failed to fetch data for {symbol}.")

    # Optional: save the data to CSV files
    today = datetime.now().strftime("%Y-%m-%d")
    for symbol, df in stocks_data.items():
        df.to_csv(f'{symbol}_{today}.csv')
        print(f'Saved {symbol} data to CSV.')


if __name__ == '__main__':
    main()
