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

summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_path:
    with open(summary_path, "a") as f:
        f.write("## Model Evaluation Results\n\n")
        f.write("**Name:** Nissi Veronika Y  \n")
        f.write("**Roll No:** 2022BCS0138\n\n")

        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Squared Error (MSE) | {mse:.4f} |\n")
        f.write(f"| R² Score | {r2:.4f} |\n")



with open("outputs/results/metrics.json", "w") as f:
    json.dump(metrics, f)
