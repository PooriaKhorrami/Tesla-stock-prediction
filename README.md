# 🚗 Tesla Stock Price Prediction

A machine learning project that predicts Tesla stock prices using technical indicators and compares multiple algorithms to demonstrate realistic performance in financial forecasting.

##  Project Overview

This project analyzes Tesla stock data from 2010-2021 to predict future stock prices using machine learning techniques.
The goal is to demonstrate practical ML skills applied to financial time series data while maintaining realistic expectations about model performance.

## 📊 Key Results

| Model | R² Score | RMSE | Performance |
|-------|----------|------|-------------|
| **Random Forest** | **48.95%** | **$66.81** | **Best Model** |
| Linear Regression | 28.74% | $78.94 | Baseline |

- **Dataset**: 1,692 days of Tesla stock data (2010-2021)
- **Features**: 10 engineered technical indicators
- **Improvement**: Random Forest achieved 20% better accuracy than linear baseline

## Key Insights

- **Realistic Performance**: 48% R² score reflects the inherent difficulty of stock prediction
- **Model Comparison**: Tree-based models (Random Forest) outperform linear models for this task
- **Feature Engineering**: Technical indicators provide meaningful predictive signal
- **Market Volatility**: Financial markets are complex and partially unpredictable by nature

## 🛠️ Technical Features Created

- **Moving Averages**: 5-day and 20-day moving averages
- **Price Momentum**: Daily and 5-day price change percentages
- **Volatility**: 10-day rolling standard deviation
- **Volume Analysis**: Volume ratio and moving average
- **Price Spread**: High-low price spread normalized by closing price

## 💻 Technologies Used

- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **scikit-learn** - Machine learning models and metrics

## How to Run

1. **Install Dependencies**:
```bash
pip install pandas numpy matplotlib scikit-learn
```

2. **Run the Analysis**:
```bash
python tesla_analysis.py
```

3. **View Results**:
   - Model performance comparison charts
   - Feature importance analysis
   - Actual vs predicted price plots
   - Trading signal visualizations

## 📈 Sample Output

```
Tesla Stock ML Analysis
Loading Tesla data...
✓ Data loaded: 1692 days
✓ Date range: 2010-06-29 to 2021-04-16
Creating features...
✓ Features created successfully
Training models...
Linear Regression: RMSE=$78.94, R²=0.2874
Random Forest: RMSE=$66.81, R²=0.4895
✓ Best model: Random Forest
✓ Analysis complete!
```

## 📁 Project Structure

```
tesla-stock-prediction/
├── tesla_analysis.py    # Main ML pipeline
├── Tesla.csv           # Historical stock data
└── README.md          # Project documentation
```

## Visualizations Generated

1. **Model Performance Comparison** - Bar charts comparing RMSE and R² scores
2. **Actual vs Predicted** - Scatter plot showing prediction accuracy
3. **Feature Importance** - Analysis of which indicators matter most
4. **Trading Signals** - Price chart with buy/sell recommendations
5. **Prediction Timeline** - Time series plot of actual vs predicted prices

## 🎓 Learning Outcomes

This project demonstrates:

### Technical Skills
- **Data Preprocessing**: Handling missing values and mixed date formats
- **Feature Engineering**: Creating domain-specific financial indicators
- **Model Selection**: Comparing different ML algorithms systematically
- **Performance Evaluation**: Using appropriate metrics for regression tasks
- **Data Visualization**: Creating professional charts and plots

## Future Enhancements

Potential improvements to explore:

- **Advanced Features**: RSI, MACD, Bollinger Bands
- **Deep Learning**: LSTM networks for sequence modeling
- **External Data**: Economic indicators, news sentiment
- **Ensemble Methods**: Combining multiple models
- **Risk Management**: Portfolio optimization and drawdown analysis

## References

- Technical indicators based on standard financial analysis practices
- Model evaluation using industry-standard metrics
- Data visualization following best practices for financial data

## ⚠️ Disclaimer

This project is for educational and demonstration purposes only. It is not intended as financial advice. Stock market investments carry inherent risks, and past performance does not guarantee future results.
