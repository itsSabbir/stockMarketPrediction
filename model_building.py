import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from typing import Dict, Tuple, Optional

# Constants
INPUT_DIR = 'enhanced_data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'evaluation_results'
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


def load_enhanced_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Loads enhanced stock data from a CSV file.

    Parameters:
    symbol (str): The stock symbol to load data for.

    Returns:
    pd.DataFrame: DataFrame loaded from the CSV, or None if loading fails.
    """
    try:
        file_pattern = f'enhanced_{symbol}_*.csv'
        matching_files = [f for f in os.listdir(INPUT_DIR) if
                          f.startswith(f'enhanced_{symbol}_') and f.endswith('.csv')]

        if not matching_files:
            print(f"No enhanced data file found for {symbol}.")
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


def load_model(symbol: str) -> Optional[RandomForestRegressor]:
    """
    Loads a trained model for a given symbol.

    Parameters:
    symbol (str): The stock symbol for which to load the model.

    Returns:
    RandomForestRegressor: The loaded model, or None if loading fails.
    """
    model_path = os.path.join(MODEL_DIR, f'model_{symbol}.joblib')
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"No model file found for {symbol}.")
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
    return None


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model using various metrics.

    Parameters:
    model (RandomForestRegressor): The trained model to evaluate.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): True test values.

    Returns:
    Dict[str, float]: A dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, symbol: str) -> None:
    """
    Plots actual vs predicted values.

    Parameters:
    y_test (pd.Series): True test values.
    y_pred (np.ndarray): Predicted values.
    symbol (str): The stock symbol.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted Stock Prices for {symbol}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{symbol}_actual_vs_predicted.png'))
    plt.close()


def plot_feature_importance(model: RandomForestRegressor, features: list, symbol: str) -> None:
    """
    Plots feature importance.

    Parameters:
    model (RandomForestRegressor): The trained model.
    features (list): List of feature names.
    symbol (str): The stock symbol.
    """
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance for {symbol}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{symbol}_feature_importance.png'))
    plt.close()


def evaluate_symbol(symbol: str) -> Optional[Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Evaluates the model for a given symbol.

    Parameters:
    symbol (str): The stock symbol to evaluate.

    Returns:
    Optional[Tuple[Dict[str, float], pd.DataFrame]]: A tuple containing evaluation metrics and feature importance,
    or None if evaluation fails.
    """
    df = load_enhanced_data(symbol)
    if df is None:
        return None

    model = load_model(symbol)
    if model is None:
        return None

    features = ['open', 'high', 'low', 'close', 'volume', 'ma200', 'rsi', 'macd', 'signal', 'upper_bb', 'lower_bb',
                'volatility', 'price_momentum']
    X = df[features]
    y = df['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    metrics = evaluate_model(model, X_test, y_test)

    y_pred = model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred, symbol)
    plot_feature_importance(model, features, symbol)

    feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    return metrics, feature_importance


def main():
    symbols = load_fortune_1000_symbols()

    if not symbols:
        print("No symbols loaded. Exiting.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = {}

    for symbol in symbols:
        print(f"Evaluating model for {symbol}...")
        evaluation = evaluate_symbol(symbol)
        if evaluation:
            metrics, feature_importance = evaluation
            results[symbol] = {
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            print(f"Evaluation for {symbol} completed. Metrics: {metrics}")
        else:
            print(f"Failed to evaluate model for {symbol}.")

    # Save overall results
    with open(os.path.join(OUTPUT_DIR, 'evaluation_summary.txt'), 'w') as f:
        for symbol, data in results.items():
            f.write(f"Symbol: {symbol}\n")
            f.write("Metrics:\n")
            for metric, value in data['metrics'].items():
                f.write(f"  {metric}: {value}\n")
            f.write("Top 5 Important Features:\n")
            for _, row in data['feature_importance'].head().iterrows():
                f.write(f"  {row['feature']}: {row['importance']}\n")
            f.write("\n")

    print("Model evaluation completed. Results saved in the 'evaluation_results' directory.")


if __name__ == '__main__':
    main()