import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    sns.set_theme(style="whitegrid")

    print("Loading the California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)

    os.makedirs("Supervised-Learning/outputs", exist_ok=True)

    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary statistics:\n")
    print(df.describe().T)

    # EDA visualizations
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("Supervised-Learning/outputs/correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="MedInc", y="target", alpha=0.5)
    plt.title("Target vs Median Income")
    plt.xlabel("Median Income")
    plt.ylabel("House Value")
    plt.tight_layout()
    plt.savefig("Supervised-Learning/outputs/target_vs_income.png")
    plt.close()
    

    plt.figure(figsize=(14, 10))
    df.hist(bins=25, edgecolor="black", figsize=(14, 10))
    plt.suptitle("Feature Distribution")
    plt.tight_layout()
    plt.savefig("Supervised-Learning/outputs/feature_distributions.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="AveRooms", y="target", alpha=0.4)
    plt.title("Target vs Average Rooms")
    plt.xlabel("Average Rooms")
    plt.ylabel("House Value")
    plt.tight_layout()
    plt.savefig("Supervised-Learning/outputs/target_vs_rooms.png")
    plt.close()

    # Prepare data for linear regression
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\nLinear Regression Results")
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R^2:", round(r2, 4))

    print("\nSaved plots to:")
    print("- Supervised-Learning/outputs/correlation_heatmap.png")
    print("- Supervised-Learning/outputs/target_vs_income.png")
    print("- Supervised-Learning/outputs/feature_distributions.png")
    print("- Supervised-Learning/outputs/target_vs_rooms.png")


if __name__ == "__main__":
    main()
