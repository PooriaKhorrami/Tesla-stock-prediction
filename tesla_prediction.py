Original file is located at
    https://colab.research.google.com/drive/11rLqo8_4Vq6IAP1hLGP3-7J6mfsEwFjj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

def load_data():

    print("Loading Tesla data...")

    try:
        # Updated path for Google Colab
        path = '/content/drive/MyDrive/Data_Science_Project/Tesla.csv'
        df = pd.read_csv(path)

        # Handle mixed date formats in your Tesla data
        print("Fixing date formats...")
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
        df = df.set_index('Date').sort_index()

        print(f"✓ Data loaded: {len(df)} days")
        print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"✓ Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

        # Handle missing values
        df = df.fillna(method='ffill')

        # Show sample of data
        print(f"\nSample of loaded data:")
        print(df.head())

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the file path is correct and you have mounted Google Drive")

        # Try alternative date parsing methods
        try:
            print("\nTrying alternative date parsing...")
            df = pd.read_csv(path)

            # Try different date parsing approaches
            for fmt in ['%m/%d/%Y', '%m-%d-%y', '%Y-%m-%d', 'mixed']:
                try:
                    if fmt == 'mixed':
                        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                    else:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                    print(f"✓ Success with format: {fmt}")
                    break
                except:
                    continue

            df = df.set_index('Date').sort_index()
            df = df.fillna(method='ffill')

            print(f"✓ Data loaded: {len(df)} days")
            return df

        except Exception as e2:
            print(f"All date parsing methods failed: {e2}")
            return None

def create_features(df):
    """Create simple features"""
    print("Creating features...")

    # Convert price columns to numeric (fix string values)
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("✓ Converted price columns to numeric")

    # Remove any rows with NaN values after conversion
    df = df.dropna()
    print(f"✓ Data shape after cleaning: {df.shape}")

    # Simple moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Price change (now should work with numeric data)
    df['Price_Change'] = df['Close'].pct_change()

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Volume ratio
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # High-Low spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

    print(f"✓ Features created successfully")
    print(f"✓ Final data shape: {df.shape}")

    return df

def prepare_data(df):
    """Prepare data for ML"""
    print("Preparing ML data...")

    # Create target (next day close price)
    df['Target'] = df['Close'].shift(-1)

    # Select features
    features = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20',
                'Price_Change', 'Volatility', 'Volume_Ratio', 'HL_Spread']

    # Remove rows with NaN
    df_clean = df[features + ['Target']].dropna()

    X = df_clean[features]
    y = df_clean['Target']

    print(f"✓ Dataset ready: {len(X)} samples, {len(features)} features")
    return X, y, features

def train_models(X, y):
    """Train ML models"""
    print("Training models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }

        print(f"{name}: RMSE=${rmse:.2f}, R²={r2:.4f}")

    return results

def plot_results(results):
    """Plot model results"""
    print("Creating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Tesla Stock Prediction Results')

    # Model comparison
    models = list(results.keys())
    rmse_scores = [results[m]['rmse'] for m in models]
    r2_scores = [results[m]['r2'] for m in models]

    # RMSE comparison
    axes[0, 0].bar(models, rmse_scores, color=['blue', 'green'])
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE ($)')

    # R2 comparison
    axes[0, 1].bar(models, r2_scores, color=['blue', 'green'])
    axes[0, 1].set_title('R² Score Comparison')
    axes[0, 1].set_ylabel('R² Score')

    # Best model predictions
    best_model = max(models, key=lambda x: results[x]['r2'])
    actual = results[best_model]['actual']
    predicted = results[best_model]['predictions']

    # Scatter plot
    axes[1, 0].scatter(actual, predicted, alpha=0.5)
    axes[1, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    axes[1, 0].set_xlabel('Actual Price')
    axes[1, 0].set_ylabel('Predicted Price')
    axes[1, 0].set_title(f'{best_model} - Actual vs Predicted')

    # Time series
    axes[1, 1].plot(actual.values, label='Actual', color='blue')
    axes[1, 1].plot(predicted, label='Predicted', color='red', linestyle='--')
    axes[1, 1].set_title('Price Prediction Timeline')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return best_model

def analyze_features(results, feature_names):
    """Analyze feature importance"""
    print("Feature importance analysis...")

    rf_model = results['Random Forest']['model']
    importance = rf_model.feature_importances_

    # Create importance dataframe
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    print("\nTop 5 Important Features:")
    print(feature_df.head().to_string(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['Feature'], feature_df['Importance'])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def generate_signals(df):
    """Generate trading signals"""
    print("Generating trading signals...")

    # Simple MA crossover strategy
    df['Signal'] = 0
    df.loc[df['MA_5'] > df['MA_20'], 'Signal'] = 1  # Buy
    df.loc[df['MA_5'] < df['MA_20'], 'Signal'] = -1  # Sell

    # Count signals
    buy_signals = (df['Signal'] == 1).sum()
    sell_signals = (df['Signal'] == -1).sum()

    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")

    # Plot recent signals
    recent = df.tail(200)  # Last 200 days

    plt.figure(figsize=(12, 6))
    plt.plot(recent.index, recent['Close'], label='Tesla Price', linewidth=2)

    # Mark signals
    buy_points = recent[recent['Signal'] == 1]
    sell_points = recent[recent['Signal'] == -1]

    plt.scatter(buy_points.index, buy_points['Close'],
                color='green', marker='^', s=50, label='Buy Signal')
    plt.scatter(sell_points.index, sell_points['Close'],
                color='red', marker='v', s=50, label='Sell Signal')

    plt.title('Tesla Stock with Trading Signals (Last 200 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df

def main():
    """Main function"""
    print("🚗 Tesla Stock ML Analysis")
    print("=" * 30)

    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Check if Tesla.csv exists.")
        return

    # Create features
    data = create_features(data)

    # Prepare ML data
    X, y, features = prepare_data(data)

    # Train models
    results = train_models(X, y)

    # Plot results
    best_model = plot_results(results)

    # Feature analysis
    analyze_features(results, features)

    # Trading signals
    # data = generate_signals(data)

    # Summary
    print(f"\n🎯 SUMMARY")
    print(f"✓ Best model: {best_model}")
    print(f"✓ R² score: {results[best_model]['r2']:.4f}")
    print(f"✓ RMSE: ${results[best_model]['rmse']:.2f}")
    print("✓ Analysis complete!")

if __name__ == "__main__":
    main()

