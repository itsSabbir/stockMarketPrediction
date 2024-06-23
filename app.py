from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import os
from typing import Dict, Optional

app = Flask(__name__)

# Constants
MODEL_DIR = 'models'
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


def load_models() -> Dict[str, object]:
    """Load models for all Fortune 1000 companies."""
    symbols = load_fortune_1000_symbols()
    models = {}
    for symbol in symbols:
        model_path = os.path.join(MODEL_DIR, f'model_{symbol}.joblib')
        try:
            models[symbol] = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Model for {symbol} not found. Skipping.")
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
    return models


# Load models on app start
models = load_models()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a POST request with JSON containing stock data and a symbol.
    Returns a JSON response with the predicted stock price.
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symbol = data.get('symbol')
        if not symbol:
            return jsonify({'error': 'Symbol not provided'}), 400

        # Extract input features based on the expected model features
        features = ['open', 'high', 'low', 'close', 'volume', 'ma200', 'rsi', 'macd', 'signal']
        input_features = [data.get(feature) for feature in features]

        # Check for None values that indicate missing features
        if any(feature is None for feature in input_features):
            return jsonify({'error': f'Missing features. Required features are: {", ".join(features)}'}), 400

        # Predict using the pre-loaded model
        model = models.get(symbol)
        if model is not None:
            prediction = model.predict([input_features])
            return jsonify({'symbol': symbol, 'prediction': prediction[0]})
        else:
            return jsonify({'error': f'Model not found for the provided symbol: {symbol}'}), 404

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON data'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'models_loaded': len(models)}), 200


@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Return a list of supported symbols."""
    return jsonify({'symbols': list(models.keys())}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)