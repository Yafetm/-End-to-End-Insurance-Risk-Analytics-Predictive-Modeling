import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Preprocess
features = ['TotalPremium', 'Province', 'VehicleType', 'Gender']
df = df.dropna(subset=features + ['TotalClaims'])
X = pd.get_dummies(df[features], drop_first=True)
y = df['TotalClaims']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate RMSE manually
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_model.pkl')

# Save metrics
with open('scripts/model_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse:.2f}\nR2: {r2:.2f}\n")