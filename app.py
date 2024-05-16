from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load models on app start
models = {}
symbols = ['AAPL', 'MSFT', 'GOOGL']
for symbol in symbols:
    model_path = f'model_{symbol}.pkl'
    models[symbol] = joblib.load(model_path)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a POST request with JSON containing stock data and a symbol.
    Returns a JSON response with the predicted stock price.
    """
    data = json.loads(request.data)
    symbol = data['symbol']

    # Extract input features based on the expected model features
    input_features = [data.get(feature) for feature in
                      ['open', 'high', 'low', 'close', 'volume', 'ma50', 'ma200', 'rsi', 'macd', 'signal']]

    # Check for None values that indicate missing features
    if any(feature is None for feature in input_features):
        return jsonify({'error': 'Missing features'}), 400

    # Predict using the pre-loaded model
    model = models.get(symbol)
    if model is not None:
        prediction = model.predict([input_features])
        return jsonify({'prediction': prediction[0]})
    else:
        return jsonify({'error': 'Model not found for the provided symbol'}), 404


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
