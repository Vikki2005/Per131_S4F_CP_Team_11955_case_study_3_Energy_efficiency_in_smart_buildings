import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_excel("building_energy_dataset.xlsx")

# Convert date and time to datetime object
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
df = df.drop(columns=['date', 'time'])

# Feature Selection
features = ['occupancy', 'temperature_C', 'humidity_percent']
target = 'energy_usage_kWh'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train AI Model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} kWh")

# Function to optimize HVAC based on AI predictions
def optimize_hvac(occupancy, temperature, humidity):
    input_data = pd.DataFrame([{
        'occupancy': occupancy,
        'temperature_C': temperature,
        'humidity_percent': humidity
    }])
    predicted_energy = model.predict(input_data)[0]
    optimized_temp = max(18, min(26, temperature - (predicted_energy * 0.01)))  # Adjust temperature dynamically
    print(f"Recommended Temperature: {optimized_temp:.1f}°C for Energy Efficiency")
    return optimized_temp

# Example Usage
optimize_hvac(occupancy=15, temperature=24, humidity=50)

# Plot: Actual vs Predicted Energy Usage
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Energy Usage', linewidth=2)
plt.plot(y_pred, label='Predicted Energy Usage', linestyle='--', color='orange')
plt.title('Actual vs Predicted Energy Usage')
plt.xlabel('Test Sample Index')
plt.ylabel('Energy Usage (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
