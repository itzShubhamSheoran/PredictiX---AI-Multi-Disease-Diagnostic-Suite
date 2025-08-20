# diabetes_svm.py
"""
Diabetes Prediction using Support Vector Machine (SVM)
Dataset: PIMA Indians Diabetes Dataset
Author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns

# Load dataset
data = pd.read_csv("diabetes.csv")
print("Dataset Shape:", data.shape)
print(data.head())

# Features/labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\nSVM (Diabetes Prediction)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix - Diabetes (SVM)")
plt.show()

# Save model
joblib.dump(model, "diabetes_svm_model.pkl")
joblib.dump(scaler, "diabetes_scaler.pkl")
