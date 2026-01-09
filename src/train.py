import json
import joblib
import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/winequality-red.csv", sep=",")
df.columns = df.columns.str.strip().str.lower()

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

joblib.dump(model, "outputs/model/model.joblib")

metrics = {
    "mse": mse,
    "r2": r2
}

with open("outputs/results/metrics.json", "w") as f:
    json.dump(metrics, f)
