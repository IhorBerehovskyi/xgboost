import pandas as pd
import sqlite3
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the model
model = XGBRegressor()
model.load_model('python_model.json')

# Load the second model
second_model = XGBRegressor()
second_model.load_model('cpp_model.model')

# Connect to the database
conn = sqlite3.connect('co2_levels.db')
data = pd.read_sql_query("SELECT * FROM co2_levels", conn)
conn.close()

# Convert datetime and extract features
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month
data['weekday'] = data['datetime'].dt.weekday

# Define features and target
X = data[['hour', 'day', 'month', 'weekday']]
y = data['CO2Level']

# Predict and evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on the entire dataset: {mse}')

# Predict with the second model
y_pred_second_model = second_model.predict(X)
mse_second_model = mean_squared_error(y, y_pred_second_model)
print(f'Mean Squared Error on the entire dataset (second model): {mse_second_model}')

# Plotting the predictions of both models
plt.figure(figsize=(14, 7))
plt.plot(data['datetime'].values[-250:], y.values[-250:], label='Actual CO2Level')  # Use .values
plt.plot(data['datetime'].values[-250:], y_pred[-250:], label='Predicted CO2Level (Python Model)')  # Use .values
plt.plot(data['datetime'].values[-250:], y_pred_second_model[-250:], label='Predicted CO2Level (C++ Model)')  # Use .values
plt.xlabel('Date and Time')
plt.ylabel('CO2 Level')
plt.title('Actual vs. Predicted CO2 Levels (Last 250 Values)')
plt.legend()
plt.xticks(rotation=45)  # Rotate for readability
plt.tight_layout()  # Improve layout
plt.show()

# Feature importance plot
from xgboost import plot_importance
plot_importance(model)
plt.show()

