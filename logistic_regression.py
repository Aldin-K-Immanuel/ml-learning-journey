import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,confusion_matrix

data=load_breast_cancer()
X=data.data
y=data.target

print("Feature names:", data.feature_names[:5], "...")
print("Target names:", data.target_names)
print("Dataset shape:", X.shape)
 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

from sklearn.metrics import precision_score, recall_score
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")