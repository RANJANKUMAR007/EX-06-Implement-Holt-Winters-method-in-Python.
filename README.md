# Ex.No: 06 HOLT WINTERS METHOD
### Date: 05.10.2025

### AIM:
To implement the Holt Winters Method Model using Python. 

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it
   
### PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load the dataset
data = pd.read_csv("gold_price_data.csv", parse_dates=['Date'], index_col='Date')
print("First rows:\n", data.head())

# Use the first numeric column if 'Price' is not available
if 'Price' in data.columns:
    series = data['Price']
else:
    series = data.select_dtypes(include=['float64', 'int64']).iloc[:, 0]

# 2. Resample to monthly frequency and fill missing values
monthly = series.resample('MS').mean().interpolate()

# 3. Plot the time series
monthly.plot(title='Monthly Gold Price', figsize=(10, 4))
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.show()

# 4. Split data into train and test (80%-20%)
train_size = int(len(monthly) * 0.8)
train, test = monthly[:train_size], monthly[train_size:]

# 5. Fit Holt-Winters model
model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit()

# 6. Forecast for the test period
forecast = model.forecast(len(test))

# 7. Plot train, test, and forecast
plt.figure(figsize=(10, 4))
train.plot(label='Train')
test.plot(label='Test')
forecast.plot(label='Forecast', linestyle='--')
plt.title("Holt-Winters Forecast vs Actual")
plt.legend()
plt.show()

# 8. Evaluate model
rmse = np.sqrt(mean_squared_error(test, forecast))
print("RMSE:", rmse)

# 9. Forecast future 12 months
future_forecast = model.forecast(12)
print("\nFuture 12 months forecast:\n", future_forecast)

# 10. Plot final forecast
plt.figure(figsize=(10, 4))
monthly.plot(label='Historical')
future_forecast.plot(label='Future Forecast', linestyle='--')
plt.title("Gold Price Forecast - Next 12 Months")
plt.legend()
plt.show()


```
### OUTPUT:
<img width="151" height="127" alt="image" src="https://github.com/user-attachments/assets/43992f81-2ddc-4a70-97e9-314f76458337" />

<img width="827" height="352" alt="image" src="https://github.com/user-attachments/assets/ea57916d-a0c2-471f-807c-1acb36b7688e" />

<img width="822" height="366" alt="image" src="https://github.com/user-attachments/assets/49a97062-34f4-4a24-a73d-cd115da9c38b" />

<img width="206" height="281" alt="image" src="https://github.com/user-attachments/assets/a351fac7-43ab-4863-8bb9-77c20f4c3838" />

<img width="808" height="358" alt="image" src="https://github.com/user-attachments/assets/ee723717-3ca5-44c1-a3df-88572c979d76" />








### RESULT:
Thus we have successfully implemented the auto regression function using python.
