import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create data for 50 students
data = {
    "Student_ID": range(1, 51),

    "Study_Hours": [
        2, 5, 3, 8, 1, 6, 4, 7, 2, 9,
        3, 5, 6, 1, 4, 8, 7, 2, 5, 10,
        3, 6, 4, 8, 2, 7, 5, 9, 1, 6,
        4, 3, 8, 7, 5, 2, 10, 6, 4, 9,
        3, 7, 5, 1, 8, 6, 2, 9, 4, 7
    ],

    "Exam_Score": [
        45, 72, 55, 88, 35, 78, 65, 84, 48, 92,
        58, 70, 76, 38, 62, 89, 82, 46, 68, 95,
        54, 79, 64, 87, 43, 81, 73, 91, 32, 75,
        61, 57, 90, 85, 71, 44, 97, 77, 63, 93,
        52, 83, 69, 36, 86, 80, 41, 94, 59, 78
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create Result column
# 1 = Pass, 0 = Fail
df["Result"] = np.where(df["Exam_Score"] >= 50, 1, 0)

print(df)

# Features and target
X = df[["Study_Hours"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
