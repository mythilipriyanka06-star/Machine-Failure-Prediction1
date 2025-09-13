# train_model_vscode.py
"""
Train a RandomForest model for machine failure prediction.
Run this in VS Code's terminal:
    python train_model_vscode.py data.csv 0.2 100
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
if len(sys.argv) < 2:
    print("Usage: python train_model_vscode.py <data.csv> [test_size=0.2] [n_estimators=100]")
    sys.exit(1)

DATA_PATH = sys.argv[1]
TEST_SIZE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
N_ESTIMATORS = int(sys.argv[3]) if len(sys.argv) > 3 else 100

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ CSV file not found: {DATA_PATH}")

# -------------------- LOAD AND CLEAN --------------------
print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Convert all columns to numeric, dropping problematic rows
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna().reset_index(drop=True)

TARGET = "fail"
if TARGET not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET}' not found. Columns available: {df.columns.tolist()}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -------------------- SPLIT DATA --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(f"✅ Data split complete: {len(X_train)} train rows, {len(X_test)} test rows.")

# -------------------- TRAIN MODEL --------------------
print(f"🌲 Training RandomForest with {N_ESTIMATORS} trees...")
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
model.fit(X_train, y_train)

# -------------------- EVALUATE --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", report)

# Save metrics
with open("metrics_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)
print("💾 Metrics saved -> metrics_report.txt")

# -------------------- FEATURE IMPORTANCE --------------------
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, color="teal")
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("📈 Feature importance plot saved -> feature_importances.png")

# -------------------- SAVE MODEL --------------------
joblib.dump(model, "model.pkl")
print("✅ Trained model saved -> model.pkl")
