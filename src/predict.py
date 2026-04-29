import sys
import os
import joblib
import numpy as np
from src.feature_extractor import extract_features

model = joblib.load("models/model.pkl")


def predict_demand(image_path):
    feat = extract_features(image_path)

    if feat is None:
        return None

    # 🔥 Add default price
    default_rate = 1000
    feat = np.append(feat, default_rate)

    feat = feat.reshape(1, -1)

    pred_log = model.predict(feat)

    # Convert back
    pred = np.expm1(pred_log)

    return float(pred[0])