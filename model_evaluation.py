import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, Optional, Tuple

# Constants
INPUT_DIR = 'enhanced_data'
MODEL_DIR = 'models'
OUTPUT_DIR = 'evaluation_results'
FORTUNE_1000_FILE = 'fortune1000.csv'


def load_fortune_1000_symbols() -> list:
    """Load Fortune 1000 company symbols from a CSV file."""
    try:
        df = pd.read_csv(FORTUNE_1000_FILE)
        return df['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: {FORTUNE_1000_FILE} not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Error loading Fortune 1000 symbols: {e}")
    return []


def load_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load enhanced data for a given symbol."""
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


def load_model(symbol: str) -> Optional[object]:
    """Load the trained model for a given symbol."""
    model_path = os.path.join(MODEL_DIR, f'model_{symbol}.joblib')
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"No model file found for {symbol}.")
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
    return None


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, symbol: str) -> None:
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Price', alpha=0.7)
    plt.plot(y_test.index, y_pred, label='Predicted Price', alpha=0.7)
    plt.title(f'Actual vs Predicted Prices for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{symbol}_actual_vs_predicted.png'))
    plt.close()


def plot_feature_importance(model: object, features: list, symbol: str) -> None:
    """Plot feature importance."""
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
    """Evaluate the model for a given symbol."""
    df = load_data(symbol)
    if df is None:
        return None

    model = load_model(symbol)
    if model is None:
        return None

    features = ['open', 'high', 'low', 'close', 'volume', 'ma200', 'rsi', 'macd', 'signal']
    X_test = df[features]
    y_test = df['close']

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