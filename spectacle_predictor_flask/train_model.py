# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# ---------- 1. Load dataset ----------
df = pd.read_csv("dataset.csv")  # Make sure 'dataset.csv' is in the same folder

# ---------- 2. Select predictors and target ----------
X = df[
    [
        "Sleep hours per day",
        "Screen time per day (in hours)",
        "Physical activity score",
        "Diet quality (self-rated)",
        "Family history of myopia",
        "Genetic risk score",
        "Study hours per day",
    ]
].copy()

# Convert Yes/No to 1/0
X["Family history of myopia"] = X["Family history of myopia"].map({"Yes": 1, "No": 0})

y = df["Spectacle onset age (in years)"]

# ---------- 3. Split data into training and testing sets ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 4. Train the model ----------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ---------- 5. Evaluate the model ----------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Validation MAE: {mae:.2f} years")

# ---------- 6. Save the trained model ----------
with open("onset_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'onset_model.pkl'")
