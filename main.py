import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Dataset
data = pd.read_csv("sales.csv")

print(data.head())

# Convert Date Column
data['Date'] = pd.to_datetime(data['Date'])

# Sort Data
data = data.sort_values('Date')

# Set Date as Index
data.set_index('Date', inplace=True)

# Select Sales Column
sales = data['Sales']

# Plot Sales Trend
plt.figure(figsize=(10,5))
plt.plot(sales)
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Split Dataset
train_size = int(len(sales) * 0.8)

train = sales[:train_size]
test = sales[train_size:]

# Build ARIMA Model
model = ARIMA(train, order=(5,1,0))

model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))

print("RMSE:", rmse)

# Plot Actual vs Forecast
plt.figure(figsize=(10,5))

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast')

plt.legend()
plt.title("Sales Forecasting")
plt.show()