# StockTraderPipline

## Project Overview

This repository contains a series of Python scripts that are part of a stock market prediction project. Each script is responsible for a specific part of the process, from data collection to model deployment. Below is a breakdown of the key features of each script and step-by-step instructions on how to get the project up and running.

### Key Features

#### `data_collection.py`
- **Error Handling**: Robust management of HTTP errors, request issues, and invalid API responses.
- **Data Validation**: Ensures the presence of 'Time Series (Daily)' data and checks for NaN rows.
- **Data Integrity**: Correctly indexes DataFrame by date and sorts dates in ascending order.
- **Data Saving**: Optionally saves data to CSV, named by stock symbol and retrieval date.

#### `data_processing.py`
- **Data Cleaning**: Removes rows with missing values or zero trading volume.
- **Outlier Detection**: Handles outliers in the 'close' price column using quantiles.
- **Modular Functions**: Organizes functionality into separate, clear functions.
- **Error Handling**: Manages file-related errors and other exceptions.
- **Data Loading and Saving**: Demonstrates CSV file operations.

#### `data_visualization.py`
- **Data Loading**: Robust CSV file loading with comprehensive error checks.
- **Data Visualization**: Uses Matplotlib for plotting stock prices and moving averages.
- **Rolling Averages**: Calculates moving averages on-the-fly.
- **Modular Structure**: Ensures code is adaptable and easy to modify.
- **Error Handling**: Provides detailed feedback on data loading and plotting successes or failures.

#### `feature_engineering.py`
- **Technical Indicators Calculation**: Calculates RSI, MACD, and 200-Day Moving Average.
- **Modular Design**: Keeps calculations in separate functions for clarity and reuse.
- **Main Function**: Demonstrates independent usage with comprehensive examples.

#### `model_building.py`
- **Data Loading**: Ensures robust data loading.
- **Model Building and Tuning**: Uses RandomForestRegressor and GridSearchCV for optimal parameter selection.
- **Error Handling**: Includes comprehensive error management.
- **Performance Metrics**: Utilizes Mean Squared Error to evaluate model performance.
- **Modularity and Scalability**: Designed for easy expansion and modification.

#### `model_evaluation.py`
- **Data Loading**: Handles loading of enhanced stock data from CSV files.
- **Model Loading**: Assumes models are saved on disk and can be loaded directly.
- **Visual Evaluation**: Allows visual comparison of actual vs. predicted prices.
- **Modularity**: Facilitates easy adjustments and maintenance.

#### `app.py`
- **Model Pre-loading**: Loads models into memory at app start to improve response times.
- **Error Handling**: Manages missing features and unsupported stock symbols.
- **Secure and Scalable Deployment**: Configures Flask for production environments.
- **JSON Data Handling**: Parses input and ensures all required features are present.

### Deployment Guide

#### Step 1: Obtain API Key
- **Sign Up** for an account with a data provider like Alpha Vantage.
- **Request an API Key** to use in data collection scripts.

#### Step 2: Collect Data
- Run `data_collection.py` to fetch and save historical stock data.

#### Step 3: Process Data
- Execute `data_processing.py` to clean and prepare the data.

#### Step 4: Train the Model
- Use `model_building.py` to train and save your machine learning model.

#### Step 5: Set Up Deployment
- Ensure all dependencies are installed and model files are placed in the correct directory.

#### Step 6: Deploy Flask Application
- Start the Flask server using `app.py` to host your model.

#### Step 7: Test the API
- Use tools like Postman or curl to send requests to your server:
  ```bash
  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"symbol\": \"AAPL\", \"open\": 150, \"high\": 155, \"low\": 148, \"close\": 154, \"volume\": 1000000, \"ma50\": 150, \"ma200\": 145, \"rsi\": 70, \"macd\": 1.5, \"signal\": 1.2}"
  ```

### Final Notes
- **Monitoring and Maintenance**: Keep an eye on the application's performance and be ready to troubleshoot any issues that arise. This may involve updating API keys, fixing bugs, or retraining models.

This README is intended to provide clear instructions for setting up and running the stock market prediction project, ensuring anyone with basic technical knowledge can get it running successfully.


