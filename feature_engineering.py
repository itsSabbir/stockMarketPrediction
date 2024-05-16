import pandas as pd


def compute_rsi(data, window=14):
    """
    Computes the Relative Strength Index (RSI) for the given data.

    Parameters:
        data (pd.Series): Series of stock prices.
        window (int): The period over which to calculate RSI.

    Returns:
        pd.Series: The RSI values.
    """
    diff = data.diff(1)
    gain = (diff.where(diff > 0, 0)).fillna(0)
    loss = (-diff.where(diff < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(data, slow=26, fast=12, signal=9):
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


def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame: 200-day moving average, RSI, and MACD.

    Parameters:
        df (pd.DataFrame): The DataFrame containing stock prices.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for each indicator.
    """
    df['ma200'] = df['close'].rolling(window=200).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'], df['signal'] = compute_macd(df['close'])
    df.dropna(inplace=True)
    return df


def main():
    # Assuming processed_stocks is a dictionary of DataFrames
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    processed_stocks = {symbol: pd.read_csv(f'processed_{symbol}.csv', index_col='date', parse_dates=True) for symbol in
                        symbols}

    enhanced_stocks = {symbol: add_technical_indicators(df) for symbol, df in processed_stocks.items()}

    for symbol, df in enhanced_stocks.items():
        print(f"{symbol} head of enhanced data:")
        print(df.head())


if __name__ == '__main__':
    main()
