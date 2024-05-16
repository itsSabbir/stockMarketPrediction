import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib  # Used for model loading if models are saved on disk


def load_data(file_path):
    """
    Loads data from a CSV file into a DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
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


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model by plotting the predicted vs. actual prices.

    Parameters:
        model: The trained machine learning model.
        X_test (pd.DataFrame): DataFrame containing the features for testing.
        y_test (pd.Series): Series containing the actual closing prices.
    """
    y_pred = model.predict(X_test)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Price')
    plt.plot(y_test.index, y_pred, label='Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    # Example symbol list and feature set
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    features = ['open', 'high', 'low', 'close', 'volume', 'ma50', 'ma200', 'rsi', 'macd', 'signal']

    # Assuming models are stored on disk and data is named as 'enhanced_<symbol>.csv'
    for symbol in symbols:
        file_path = f'enhanced_{symbol}.csv'
        model_path = f'model_{symbol}.pkl'  # Adjust if models are stored with different naming
        print(f"Loading enhanced data and model for {symbol}...")

        df = load_data(file_path)
        if df is not None:
            model = joblib.load(model_path)  # Load model from disk
            X_test = df[features]
            y_test = df['close']
            print(f"Evaluating model for {symbol}...")
            evaluate_model(model, X_test, y_test)
        else:
            print(f"Failed to load data for {symbol}.")


if __name__ == '__main__':
    main()
