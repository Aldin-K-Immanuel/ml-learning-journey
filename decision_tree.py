import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data — same as before
data = fetch_california_housing()
X = data.data
y = data.target

# Split — same as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train decision tree with no depth limit
tree_full = DecisionTreeRegressor(random_state=42)
tree_full.fit(X_train, y_train)

# Evaluate on both sets
train_rmse = np.sqrt(mean_squared_error(y_train, tree_full.predict(X_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  tree_full.predict(X_test)))

print("=== Full Tree (no depth limit) ===")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE:  {test_rmse:.4f}")
print(f"Difference:    {test_rmse - train_rmse:.4f}")
print(f"Tree depth:    {tree_full.get_depth()}")

# Now train with limited depth
tree_limited = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_limited.fit(X_train, y_train)

train_rmse_l = np.sqrt(mean_squared_error(y_train, tree_limited.predict(X_train)))
test_rmse_l  = np.sqrt(mean_squared_error(y_test,  tree_limited.predict(X_test)))

print("\n=== Limited Tree (max depth=5) ===")
print(f"Training RMSE: {train_rmse_l:.4f}")
print(f"Testing RMSE:  {test_rmse_l:.4f}")
print(f"Difference:    {test_rmse_l - train_rmse_l:.4f}")
print(f"Tree depth:    {tree_limited.get_depth()}")