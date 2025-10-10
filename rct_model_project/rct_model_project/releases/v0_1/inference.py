# Auto-generated inference helper for RCT model
import os, json, numpy as np, pandas as pd, xgboost as xgb, joblib

def _load_feature_list(model_dir: str):
    with open(os.path.join(model_dir, "feature_list.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]

def predict_batch(df: pd.DataFrame, model_dir: str = "./model") -> np.ndarray:
    feats = _load_feature_list(model_dir)
    # Prefer XGB if present; fallback to linear pipeline
    xgb_path = os.path.join(model_dir, "xgb.json")
    lin_path = os.path.join(model_dir, "linear.joblib")

    if os.path.exists(xgb_path):
        booster = xgb.Booster()
        booster.load_model(xgb_path)
        X = df[feats].astype("float32").values
        dm = xgb.DMatrix(X, feature_names=feats)
        pred = booster.predict(dm)
        return np.maximum(0.0, pred)

    # Fallback: linear baseline (includes its own preprocessor)
    pack = joblib.load(lin_path)
    pipe = pack["pipeline"]
    feats_saved = pack.get("features", feats)
    X = df[feats_saved]
    pred = pipe.predict(X)
    return np.maximum(0.0, pred)
