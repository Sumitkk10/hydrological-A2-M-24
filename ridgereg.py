import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
df = pd.read_csv('shuffled_file.csv')

# # Convert 'date' column to datetime
# df['date'] = pd.to_datetime(df['date'])

# Define features (X) and target (y)
X = df.drop(['inflow', 'date'], axis=1)  # Drop 'inflow' (target) and 'date' columns
y = df['inflow']

# Handle NaN values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)  # 70% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)    # Split remaining 30% for validation and test (15% each)

# Initialize and train the Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust alpha for regularization strength
ridge_model.fit(X_train, y_train)

# Predict inflow values on the test set
y_test_pred = ridge_model.predict(X_test)

# Plot the predicted inflow values vs their frequency (histogram)
plt.figure(figsize=(10, 6))
plt.hist(y_test_pred, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Predicted Inflow Values (Test Set) vs Frequency')
plt.xlabel('Predicted Inflow Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the actual vs predicted inflow values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Inflow', color='blue')
plt.plot(y_test_pred, label='Predicted Inflow', color='red', linestyle='--')
plt.title('Actual vs Predicted Inflow Values')
plt.xlabel('Sample Index')
plt.ylabel('Inflow')
plt.legend()
plt.grid(True)
plt.show()

# # Plot the actual inflow values from the test set vs their frequency (histogram)
# plt.figure(figsize=(10, 6))
# plt.hist(y_test, bins=30, color='green', alpha=0.7, edgecolor='black')
# plt.title('Actual Inflow Values (Test Set) vs Frequency')
# plt.xlabel('Actual Inflow Values')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# Evaluate the model on test data
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
