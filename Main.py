import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset Simulation
np.random.seed(42)
n_samples = 500

data = {
    'Distance_KM': np.random.uniform(10, 500, n_samples),
    'Traffic_Congestion_Index': np.random.randint(1, 11, n_samples),
    'Weather_Impact_Score': np.random.randint(1, 6, n_samples),
    'Package_Weight_KG': np.random.uniform(1, 50, n_samples),
    'Driver_Experience_Yrs': np.random.randint(1, 20, n_samples)
}

df = pd.DataFrame(data)

# Target Variable formulation
df['Delivery_Delay_Hours'] = (
    (df['Distance_KM'] * 0.05) + 
    (df['Traffic_Congestion_Index'] * 1.2) + 
    (df['Weather_Impact_Score'] * 2.1) - 
    (df['Driver_Experience_Yrs'] * 0.3) + 
    np.random.normal(0, 2, n_samples)
)
df['Delivery_Delay_Hours'] = df['Delivery_Delay_Hours'].clip(lower=0)

# 2. Model Training
X = df[['Distance_KM', 'Traffic_Congestion_Index', 'Weather_Impact_Score', 'Package_Weight_KG', 'Driver_Experience_Yrs']]
y = df['Delivery_Delay_Hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 3. Performance Metrics
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
