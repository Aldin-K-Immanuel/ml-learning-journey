import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load real data
data = fetch_california_housing()
X = data.data        # features (8 columns)
y = data.target      # house prices

# See what we're working with
print("Feature names:")
for name, value in zip(data.feature_names, X[0]):
    print(f"  {name}: {value}")
print("Dataset shape:", X.shape)
print("First house price:", y[0])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# See what the model learned
print("\nWeights for each feature:")
for name, weight in zip(data.feature_names, model.coef_):
    print(f"  {name}: {weight:.4f}")
print(f"  Intercept (c): {model.intercept_:.4f}")

# Make predictions on both sets
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Measure performance on both
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"\nTraining RMSE: {np.sqrt(mse_train):.4f}")
print(f"Testing RMSE:  {np.sqrt(mse_test):.4f}")
print(f"Difference:    {np.sqrt(mse_test) - np.sqrt(mse_train):.4f}")

# Plot predicted vs actual for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()