# Tesla Stock Price Forecasting Using LSTM

![forceplot](https://github.com/user-attachments/assets/29c204b5-7b95-4e92-a363-6ab9ca50b777)

## 1. Project Analysis
This project aims to predict Tesla's stock price (TSLA) using the Long Short-Term Memory (LSTM) model. LSTM is chosen for its ability to capture temporal patterns in time-series data. Stock price forecasting is crucial for helping investors make informed decisions based on historical trends.

### Main Objectives:
- Build a predictive model for Tesla's stock price based on historical data.
- Evaluate the model's performance using metrics such as Mean Squared Error (MSE).
- Analyze the prediction results to understand the patterns learned by the model.

---

## 2. Data Collection (Scraping)
Tesla stock price data is sourced from **Yahoo Finance** using the Python library `yfinance`. The collected data includes:
- Opening price (`Open`)
- Closing price (`Close`)
- Highest price (`High`)
- Lowest price (`Low`)
- Trading volume (`Volume`)

### Example Code for Data Collection:
```python
import yfinance as yf

# Download Tesla stock data
data = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
data.to_csv('TSLA_2015_2025_Histogram.csv')  # Save data to a CSV file
```

---

## 3. Dataset Preprocessing
### a. **Data Cleaning**
- Missing values are either removed or filled using interpolation.
- Irrelevant columns are dropped to focus on key features.

### b. **Data Normalization**
- Data is normalized using `MinMaxScaler` to scale values between [0, 1].
- Normalization helps accelerate model convergence during training.

### c. **Sequence Creation**
- Data is transformed into sequences of a fixed length (e.g., 60 days) to capture temporal patterns.
- Each sequence is used as input to predict the price for the next day.

### d. **Dataset Splitting**
- The dataset is split into three subsets:
  - **Training Set**: 70% of the data for training the model.
  - **Validation Set**: 15% of the data for monitoring performance during training.
  - **Testing Set**: 15% of the data for final evaluation.

---

## 4. Algorithm and Model Explanation
### a. **LSTM Algorithm**
LSTM is a type of Recurrent Neural Network (RNN) designed to handle the vanishing gradient problem. LSTM has an internal memory that allows the model to learn long-term dependencies in time-series data.

### b. **Model Architecture**
The LSTM model used in this project has the following architecture:
1. **Input Layer**: Accepts data in the shape `(60, 1)` (60 timesteps, 1 feature).
2. **Hidden Layers**:
   - Two LSTM layers with 50 units each.
   - Dropout of 20% to prevent overfitting.
3. **Dense Layers**:
   - A Dense layer with 25 units and ReLU activation.
   - A final Dense layer with 1 unit for price prediction.
4. **Optimizer**: Adam with `learning_rate=0.001`.
5. **Loss Function**: Mean Squared Error (MSE).

### c. **Model Implementation Code**
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

### d. **Model Training**
- The model is trained for 20 epochs with a batch size of 16.
- Validation data is used to monitor the model's performance during training.

---

## 5. Model Evaluation
### a. **Evaluation Metrics**
- **Mean Squared Error (MSE)** is used to measure the average squared error between actual and predicted values.

### b. **Result Visualization**
- Graphs are used to compare the model's predictions with actual data.
- Training, validation, and testing data are visualized to understand the model's performance.

### Example Visualization Code:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.legend()
plt.title('Comparison of Actual and Predicted Prices')
plt.show()
```

---

## 6. Analysis of Results
- **Training Loss**: The loss graph shows that the model successfully learns patterns from the training data.
- **Testing Performance**: The model is able to predict stock price trends well, although some deviations are observed in the testing data.
- **Overfitting**: No significant overfitting is observed as the validation loss remains stable.

---

## 7. Suggestions for Future Development
1. **Model Improvement**:
   - Experiment with more complex architectures such as Bidirectional LSTM or GRU.
   - Add additional layers to increase the model's capacity.

2. **Additional Features**:
   - Include technical indicators such as RSI, MACD, or Bollinger Bands.
   - Use external data such as financial news or market sentiment.

3. **Hyperparameter Optimization**:
   - Use Grid Search or Bayesian Optimization to find the best hyperparameter combinations.

4. **Deployment**:
   - Implement the model in a web-based application or API for real-time predictions.

5. **Real-Time Data Usage**:
   - Integrate the model with real-time data from APIs like Yahoo Finance for live predictions.
