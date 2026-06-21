import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data=fetch_california_housing()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

forest=RandomForestRegressor(n_estimators=100,random_state=42)
forest.fit(X_train,y_train)

train_rmse=np.sqrt(mean_squared_error(y_train,forest.predict(X_train)))
test_rmse=np.sqrt(mean_squared_error(y_test,forest.predict(X_test)))

print("=== Random Forest (100 trees) ===")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE:  {test_rmse:.4f}")
print(f"Difference:    {test_rmse - train_rmse:.4f}")