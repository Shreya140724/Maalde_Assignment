import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

from src.feature_extractor import extract_features


# -----------------------------
# STEP 1: Load images
# -----------------------------
def get_all_images(image_root):
    image_paths = []

    for root, _, files in os.walk(image_root):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, file))

    return sorted(image_paths)


image_paths = get_all_images("Images/")
print(f"Total images found: {len(image_paths)}")

if len(image_paths) == 0:
    raise ValueError("No images found")


# -----------------------------
# STEP 2: Load CSV
# -----------------------------
df = pd.read_csv("Data/sales.csv")

print("Columns:", df.columns)

df["code"] = df["code"].astype(int).astype(str)

# 🔥 Aggregate demand
df = df.groupby("code").agg({
    "qty": "sum",
    "rate": "mean"
}).reset_index()

print(f"Unique products: {len(df)}")


# -----------------------------
# STEP 3: Sequential mapping
# -----------------------------
min_len = min(len(df), len(image_paths))

df = df.iloc[:min_len].copy()
df["image_path"] = image_paths[:min_len]

print(f"Using {min_len} samples")


# -----------------------------
# STEP 4: Feature extraction
# -----------------------------
features = []
targets = []

print("Extracting features...")

for i, row in df.iterrows():
    feat = extract_features(row["image_path"])

    if feat is not None:
        # 🔥 Add price feature
        feat = np.append(feat, row["rate"])

        features.append(feat)
        targets.append(row["qty"])

    if i % 10 == 0:
        print(f"Processed {i}/{len(df)}")

if len(features) == 0:
    raise ValueError("No features extracted")

X = np.array(features)
y = np.array(targets)

print("Feature shape:", X.shape)


# -----------------------------
# STEP 5: Normalize target
# -----------------------------
y = np.log1p(y)


# -----------------------------
# STEP 6: Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# Convert back
preds_actual = np.expm1(preds)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, preds_actual)
print("MAE:", mae)


# -----------------------------
# STEP 7: Insights (IMPORTANT)
# -----------------------------
y_actual = np.expm1(y)

print("Min demand:", y_actual.min())
print("Max demand:", y_actual.max())
print("Mean demand:", y_actual.mean())

error_percent = (mae / y_actual.mean()) * 100
print(f"Relative Error: {error_percent:.2f}%")


# -----------------------------
# STEP 8: Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("✅ Model saved!")