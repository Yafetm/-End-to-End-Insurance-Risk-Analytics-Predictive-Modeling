import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Preprocess
features = ['TotalPremium', 'Province', 'VehicleType', 'Gender']
df = df.dropna(subset=features + ['TotalClaims'])
# Filter out invalid data (e.g., negative or zero TotalPremium, extreme outliers)
df = df[df['TotalPremium'] > 0]
df = df[df['TotalClaims'] >= 0]
# Remove outliers using IQR for TotalClaims
Q1 = df['TotalClaims'].quantile(0.25)
Q3 = df['TotalClaims'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['TotalClaims'] >= Q1 - 1.5 * IQR) & (df['TotalClaims'] <= Q3 + 1.5 * IQR)]

# Encode categorical features
X_cat = pd.get_dummies(df[['Province', 'VehicleType', 'Gender']], drop_first=True)
# Scale numerical feature
scaler = StandardScaler()
X_num = scaler.fit_transform(df[['TotalPremium']])
X = np.hstack([X_num, X_cat])
y = df['TotalClaims']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_model.pkl')

# Save metrics
with open('scripts/model_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse:.2f}\nR2: {r2:.4f}\n")