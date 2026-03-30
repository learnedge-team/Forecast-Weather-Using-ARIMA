import kagglehub
import pandas as pd
import numpy as np
import pickle
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

warnings.filterwarnings('ignore')

def download_dataset():
    """Download weather dataset from Kaggle"""
    print("=" * 60)
    print("STEP 1: Downloading Dataset from Kaggle...")
    print("=" * 60)
    path = kagglehub.dataset_download("muthuj7/weather-dataset")
    print(f"Path to dataset files: {path}")
    return path

def load_and_explore_data(path):
    """Load and explore the weather dataset"""
    print("\n" + "=" * 60)
    print("STEP 2: Loading and Exploring Data...")
    print("=" * 60)
    
    # Find CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
                print(f"Found CSV file: {file}")
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the dataset path!")
    
    # Load the dataset
    df = pd.read_csv(csv_files[0])
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst 5 Rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    return df

def preprocess_data(df):
    """Preprocess the weather data for ARIMA modeling"""
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing Data...")
    print("=" * 60)
    
    # Print all columns to identify the right ones
    print(f"Available columns: {df.columns.tolist()}")
    
    # The Muthuj7 weather dataset typically has these columns:
    # 'Formatted Date', 'Summary', 'Precip Type', 'Temperature (C)', 
    # 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    # 'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 
    # 'Pressure (millibars)', 'Daily Summary'
    
    # Identify date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        # If no date column, create one
        print("No date column found. Creating date index...")
        df['Date'] = pd.date_range(start='2006-01-01', periods=len(df), freq='H')
        date_col = 'Date'
    
    print(f"Using date column: {date_col}")
    
    # Identify temperature column
    temp_col = None
    for col in df.columns:
        if 'temp' in col.lower() and 'apparent' not in col.lower():
            temp_col = col
            break
    
    # If no temperature column found, try other numeric columns
    if temp_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            temp_col = numeric_cols[0]
            print(f"No temperature column found. Using: {temp_col}")
        else:
            raise ValueError("No suitable numeric column found for forecasting!")
    
    print(f"Using temperature column: {temp_col}")
    
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=[date_col])
    
    # Set date as index
    df = df.set_index(date_col)
    df.index = df.index.tz_localize(None)  # Remove timezone info
    
    # Select temperature column
    ts = df[[temp_col]].copy()
    ts.columns = ['Temperature']
    
    # Handle missing values
    ts = ts.dropna()
    
    # Remove duplicates in index
    ts = ts[~ts.index.duplicated(keep='first')]
    
    # Resample to daily frequency (take mean temperature per day)
    ts_daily = ts.resample('D').mean()
    
    # Fill any remaining missing values
    ts_daily = ts_daily.fillna(method='ffill').fillna(method='bfill')
    
    print(f"\nDaily Time Series Shape: {ts_daily.shape}")
    print(f"Date Range: {ts_daily.index.min()} to {ts_daily.index.max()}")
    print(f"Total Days: {len(ts_daily)}")
    print(f"\nDaily Temperature Statistics:\n{ts_daily.describe()}")
    
    # Also extract other weather features for display
    weather_cols = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'humidity' in col_lower:
            weather_cols['Humidity'] = col
        elif 'wind' in col_lower and 'speed' in col_lower:
            weather_cols['Wind_Speed'] = col
        elif 'pressure' in col_lower:
            weather_cols['Pressure'] = col
        elif 'visibility' in col_lower:
            weather_cols['Visibility'] = col
        elif 'apparent' in col_lower:
            weather_cols['Apparent_Temperature'] = col
    
    # Create additional daily features
    additional_features = pd.DataFrame(index=ts_daily.index)
    for feature_name, col_name in weather_cols.items():
        if col_name in df.columns:
            feature_daily = df[[col_name]].resample('D').mean()
            feature_daily = feature_daily.fillna(method='ffill').fillna(method='bfill')
            additional_features[feature_name] = feature_daily[col_name]
    
    print(f"\nAdditional features extracted: {additional_features.columns.tolist()}")
    
    return ts_daily, additional_features

def train_arima_model(ts_daily):
    """Train ARIMA model on the temperature time series"""
    print("\n" + "=" * 60)
    print("STEP 4: Training ARIMA Model...")
    print("=" * 60)
    
    # Use last 2 years of data for faster training
    if len(ts_daily) > 730:
        ts_train_data = ts_daily.iloc[-730:]
        print(f"Using last 730 days for training: {ts_train_data.index.min()} to {ts_train_data.index.max()}")
    else:
        ts_train_data = ts_daily
        print(f"Using all {len(ts_train_data)} days for training")
    
    # Split into train and test
    train_size = int(len(ts_train_data) * 0.85)
    train = ts_train_data.iloc[:train_size]
    test = ts_train_data.iloc[train_size:]
    
    print(f"\nTraining set: {len(train)} days ({train.index.min()} to {train.index.max()})")
    print(f"Testing set: {len(test)} days ({test.index.min()} to {test.index.max()})")
    
    # Try different ARIMA orders using AIC
    print("\nSearching for best ARIMA parameters...")
    best_aic = float('inf')
    best_order = (1, 1, 1)
    best_model = None
    
    # Parameter grid
    p_values = [0, 1, 2, 3, 5]
    d_values = [0, 1, 2]
    q_values = [0, 1, 2, 3]
    
    results_list = []
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train['Temperature'], order=(p, d, q))
                    fitted = model.fit()
                    aic = fitted.aic
                    results_list.append({
                        'order': (p, d, q),
                        'AIC': aic,
                        'BIC': fitted.bic
                    })
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted
                        print(f"  Better model found: ARIMA{(p,d,q)} - AIC: {aic:.2f}")
                except Exception as e:
                    continue
    
    print(f"\n{'='*40}")
    print(f"Best ARIMA Order: {best_order}")
    print(f"Best AIC: {best_aic:.2f}")
    print(f"{'='*40}")
    
    # Print model summary
    print(f"\nModel Summary:")
    print(best_model.summary())
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Model on Test Set...")
    print("=" * 60)
    
    # Forecast for test period
    forecast_test = best_model.forecast(steps=len(test))
    
    # Calculate metrics
    mae = mean_absolute_error(test['Temperature'], forecast_test)
    rmse = np.sqrt(mean_squared_error(test['Temperature'], forecast_test))
    mape = np.mean(np.abs((test['Temperature'] - forecast_test) / test['Temperature'])) * 100
    
    print(f"\nTest Set Evaluation Metrics:")
    print(f"  MAE  (Mean Absolute Error):     {mae:.4f} °C")
    print(f"  RMSE (Root Mean Squared Error):  {rmse:.4f} °C")
    print(f"  MAPE (Mean Abs Percentage Error): {mape:.2f}%")
    
    # Retrain on full data with best order for final model
    print("\n" + "=" * 60)
    print("STEP 6: Retraining Final Model on Full Data...")
    print("=" * 60)
    
    final_model = ARIMA(ts_train_data['Temperature'], order=best_order)
    final_fitted = final_model.fit()
    
    print("Final model trained successfully!")
    
    # Store evaluation metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'best_order': best_order,
        'best_aic': best_aic,
        'train_size': len(train),
        'test_size': len(test),
        'test_actual': test['Temperature'].values.tolist(),
        'test_predicted': forecast_test.values.tolist(),
        'test_dates': test.index.tolist()
    }
    
    return final_fitted, best_order, metrics, ts_train_data

def save_artifacts(model, order, metrics, ts_data, additional_features):
    """Save model and data artifacts as pickle files"""
    print("\n" + "=" * 60)
    print("STEP 7: Saving Model and Artifacts...")
    print("=" * 60)
    
    # Save everything in a single pickle file
    artifacts = {
        'model': model,
        'order': order,
        'metrics': metrics,
        'historical_data': ts_data,
        'additional_features': additional_features,
        'last_date': ts_data.index.max(),
        'training_info': {
            'total_observations': len(ts_data),
            'date_range_start': str(ts_data.index.min()),
            'date_range_end': str(ts_data.index.max()),
            'arima_order': order,
            'model_aic': metrics['best_aic']
        }
    }
    
    # Save as pickle
    with open('weather_arima_model.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Model saved to: weather_arima_model.pkl")
    
    # Also save historical data separately for quick loading
    with open('historical_weather_data.pkl', 'wb') as f:
        pickle.dump({
            'temperature': ts_data,
            'features': additional_features
        }, f)
    
    print(f"Historical data saved to: historical_weather_data.pkl")
    
    # Verify saved files
    file_size = os.path.getsize('weather_arima_model.pkl') / (1024 * 1024)
    print(f"Model file size: {file_size:.2f} MB")
    
    # Verify by loading
    print("\nVerifying saved model...")
    with open('weather_arima_model.pkl', 'rb') as f:
        loaded = pickle.load(f)
    
    test_forecast = loaded['model'].forecast(steps=7)
    print(f"Verification - 7-day forecast:\n{test_forecast}")
    print("\n✅ Model saved and verified successfully!")
    
    return artifacts

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("  WEATHER FORECASTING - ARIMA MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Download dataset
    path = download_dataset()
    
    # Step 2: Load and explore
    df = load_and_explore_data(path)
    
    # Step 3: Preprocess
    ts_daily, additional_features = preprocess_data(df)
    
    # Step 4-6: Train ARIMA model
    model, order, metrics, ts_data = train_arima_model(ts_daily)
    
    # Step 7: Save artifacts
    artifacts = save_artifacts(model, order, metrics, ts_data, additional_features)
    
    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  Best ARIMA Order: {order}")
    print(f"  MAE:  {metrics['mae']:.4f} °C")
    print(f"  RMSE: {metrics['rmse']:.4f} °C")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"\n  Files created:")
    print(f"    1. weather_arima_model.pkl")
    print(f"    2. historical_weather_data.pkl")
    print(f"\n  Next step: Run 'streamlit run app.py'")
    print("=" * 60)

if __name__ == "__main__":
    main()