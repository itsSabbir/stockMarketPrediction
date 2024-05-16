import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_enhanced_data(file_path):
    """
    Loads enhanced stock data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing enhanced stock data.

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


def build_model(df):
    """
    Builds and tunes a RandomForestRegressor model using GridSearchCV.

    Parameters:
        df (pd.DataFrame): DataFrame containing the features and target variable.

    Returns:
        tuple: A tuple containing the trained model and the mean squared error of the predictions.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    features = ['open', 'high', 'low', 'close', 'volume', 'ma50', 'ma200', 'rsi', 'macd', 'signal']
    X = df[features]
    y = df['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
    model = GridSearchCV(RandomForestRegressor(), params, cv=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    models = {}

    for symbol in symbols:
        file_path = f'enhanced_{symbol}.csv'
        print(f"Loading enhanced data for {symbol} from {file_path}...")
        df = load_enhanced_data(file_path)
        if df is not None:
            print(f"Building model for {symbol}...")
            model, mse = build_model(df)
            models[symbol] = (model, mse)
            print(f"Model for {symbol} built. MSE: {mse}")
        else:
            print(f"Failed to load data for {symbol}.")


if __name__ == '__main__':
    main()
