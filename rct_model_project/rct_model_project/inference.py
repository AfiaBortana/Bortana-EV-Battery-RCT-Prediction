# inference.py — for SageMaker Serverless Inference
import os, json, io
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES_CACHE = None

# ---------- Utility ----------
def _load_feature_list(model_dir: str):
    """Cache and load feature list for consistent column order."""
    global FEATURES_CACHE
    if FEATURES_CACHE is None:
        path = os.path.join(model_dir, "feature_list.txt")
        with open(path, "r") as f:
            FEATURES_CACHE = [ln.strip() for ln in f if ln.strip()]
    return FEATURES_CACHE


# ---------- Model load ----------
def model_fn(model_dir: str):
    """Load XGBoost model from SageMaker model directory."""
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgb.json"))
    features = _load_feature_list(model_dir)
    return {"booster": booster, "features": features}


# ---------- Input processing ----------
def input_fn(request_body, content_type="application/json"):
    """Handle JSON or CSV input from endpoint."""
    if content_type == "application/json":
        return json.loads(request_body)
    elif content_type == "text/csv":
        return request_body
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# ---------- Prediction ----------
def predict_fn(input_data, model):
    booster = model["booster"]
    features = model["features"]

    # Handle JSON input
    if isinstance(input_data, dict):
        if "instances" in input_data:
            rows = input_data["instances"]
        elif "inputs" in input_data:
            rows = input_data["inputs"]
        else:
            rows = [input_data]
        df = pd.DataFrame(rows)
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, str):
        df = pd.read_csv(io.StringIO(input_data), header=None, names=features)
    else:
        raise ValueError("Unsupported input format")

    # Ensure same feature order
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[features].astype("float32")

    dmat = xgb.DMatrix(df.values, feature_names=features)
    preds = booster.predict(dmat)
    return preds


# ---------- Output ----------
def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps({"predictions": np.asarray(prediction).tolist()}), accept
    elif accept == "text/csv":
        return "\n".join(str(float(x)) for x in prediction), "text/csv"
    else:
        raise ValueError(f"Unsupported accept: {accept}")
