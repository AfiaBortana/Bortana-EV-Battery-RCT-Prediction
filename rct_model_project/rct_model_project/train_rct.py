# ===== STEP 2: LOAD FIXED SPLITS SAFELY (numeric-only) =====
import os
import pandas as pd
import numpy as np

DATA_DIR = "./rct_train"
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL = "rct_minutes"
FORBIDDEN = {"timestamp_local", "session_id", "vehicle"}  # never use as features
FORBIDDEN_SUBSTR = ("energy", "efficiency")               # drop any columns containing these words

def read_feature_list(path: str):
    feats = [ln.strip() for ln in open(path) if ln.strip()]
    # Guardrails: remove anything forbidden if it slipped in
    feats = [f for f in feats if f not in FORBIDDEN and not any(s in f for s in FORBIDDEN_SUBSTR)]
    return feats

def load_fixed_splits(data_dir: str):
    # 1) read files
    feat_path = os.path.join(data_dir, "feature_list.txt")
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv   = os.path.join(data_dir, "val.csv")
    test_csv  = os.path.join(data_dir, "test.csv")

    assert os.path.exists(feat_path), f"Missing {feat_path}"
    assert os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv), \
        "train/val/test CSVs not found in ./rct_train/"

    feats = read_feature_list(feat_path)
    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    test  = pd.read_csv(test_csv)

    # 2) label presence
    for name, df in [("train", train), ("val", val), ("test", test)]:
        assert LABEL in df.columns, f"{LABEL} missing in {name}.csv"

    # 3) keep numeric columns only, preserving the feature_list order
    numeric_cols_train = set(train.select_dtypes(include=["number"]).columns)
    used = [c for c in feats if c in numeric_cols_train]

    # 4) warn if feature_list had non-numeric/missing columns
    dropped = [c for c in feats if c not in used]
    if dropped:
        print("[WARN] Dropped non-numeric or missing columns:", dropped)

    # 5) verify all splits have the exact same feature columns
    for name, df in [("train", train), ("val", val), ("test", test)]:
        missing_here = [c for c in used if c not in df.columns]
        assert not missing_here, f"{name}.csv is missing columns: {missing_here}"

    # 6) build matrices (float32 is fine for trees & speed)
    X_train = train[used].astype("float32")
    y_train = train[LABEL].astype("float32")

    X_val   = val[used].astype("float32")
    y_val   = val[LABEL].astype("float32")

    X_test  = test[used].astype("float32")
    y_test  = test[LABEL].astype("float32")

    # 7) quick visibility
    print(f"[OK] Split sizes → train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"[OK] Using {len(used)} numeric features (ordered): {used[:10]}{' ...' if len(used)>10 else ''}")

    return (X_train, y_train, X_val, y_val, X_test, y_test, used)

# Actually load them now:
X_train, y_train, X_val, y_val, X_test, y_test, feature_order = load_fixed_splits(DATA_DIR)

# ===== STEP 3: PREPROCESS =====
# Trees (like XGBoost) can take NaNs directly. Linear needs impute + scale.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# --- 3.1 Linear baseline preprocessor: median impute + standardize ---
linear_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # fill missing with column median
    ("scaler",  StandardScaler())                   # make features comparable in scale
])

# Fit on TRAIN only, then transform val/test
X_train_lin = linear_preprocessor.fit_transform(X_train)
X_val_lin   = linear_preprocessor.transform(X_val)
X_test_lin  = linear_preprocessor.transform(X_test)

# --- 3.2 Tree inputs: keep numeric DataFrames (float32), NaNs are OK for trees ---
X_train_tree = X_train.astype("float32").copy()
X_val_tree   = X_val.astype("float32").copy()
X_test_tree  = X_test.astype("float32").copy()

# --- 3.3 quick sanity checks (shapes + NaN counts) ---
def nan_count(arr_or_df):
    if isinstance(arr_or_df, np.ndarray):
        return int(np.isnan(arr_or_df).sum())
    else:
        return int(np.isnan(arr_or_df.to_numpy()).sum())

print("[Linear] shapes:", X_train_lin.shape, X_val_lin.shape, X_test_lin.shape,
      "| NaNs after preprocess:", nan_count(X_train_lin))
print("[Trees ] shapes:",  X_train_tree.shape, X_val_tree.shape, X_test_tree.shape,
      "| NaNs kept (OK for trees):", nan_count(X_train_tree))

# We'll save `linear_preprocessor` later with the model so inference can apply the same steps.

# ===== STEP 4: BASELINE (Ridge Regression) =====
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np, os, json, joblib

def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true >= 1.0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))*100) if mask.sum() else None
    return {"rmse": rmse, "mae": mae, "mape": mape}

# 4.1 fit the linear model on the preprocessed TRAIN data
linear_model = Ridge(alpha=1.0)             # keep it simple; no random_state here
linear_model.fit(X_train_lin, y_train)

# 4.2 predict (clip negatives to 0 so "minutes" never go below zero)
yhat_train_lin = np.maximum(0.0, linear_model.predict(X_train_lin))
yhat_val_lin   = np.maximum(0.0, linear_model.predict(X_val_lin))
yhat_test_lin  = np.maximum(0.0, linear_model.predict(X_test_lin))

# 4.3 compute metrics
baseline_metrics = {
    "train": metrics_dict(y_train, yhat_train_lin),
    "val":   metrics_dict(y_val,   yhat_val_lin),
    "test":  metrics_dict(y_test,  yhat_test_lin),
    "kind":  "Ridge(alpha=1.0)"
}

print(f"[LINEAR] Val RMSE: {baseline_metrics['val']['rmse']:.3f} | "
      f"Test RMSE: {baseline_metrics['test']['rmse']:.3f} | "
      f"Val MAPE: {baseline_metrics['val']['mape']:.2f}%")

# 4.4 save a single artifact that includes BOTH the preprocessor and the model
#     so we can reuse it later for inference or comparison
baseline_pipeline = Pipeline([("prep", linear_preprocessor), ("ridge", linear_model)])
joblib.dump({"pipeline": baseline_pipeline, "features": feature_order},
            os.path.join(MODEL_DIR, "linear.joblib"))

# ===== STEP 5: PRIMARY MODEL (XGBoost) — robust to xgboost versions =====
import xgboost as xgb
import numpy as np, json, os

SEED = 42

# Build DMatrices with feature names (faster & consistent)
# Build DMatrices with feature names (faster & consistent)
# --- Optional: emphasize low-SOC rows with sample weights
USE_SOC_WEIGHTS = True  # set False to disable weighting

if USE_SOC_WEIGHTS and ("soc" in X_train_tree.columns):
    weights = np.ones(len(X_train_tree), dtype="float32")
    soc_vals = X_train_tree["soc"].values
    # 0–20% SOC gets 1.5× weight; 20–40% gets 1.2×; others 1.0×
    weights[(soc_vals >= 0) & (soc_vals < 20)]  = 1.5
    weights[(soc_vals >= 20) & (soc_vals < 40)] = 1.2
    print("[WEIGHTS] Using SOC-based sample weights (low SOC emphasized).")
else:
    weights = None
    if "soc" not in X_train_tree.columns:
        print("[WEIGHTS] 'soc' not found; training unweighted.")
    else:
        print("[WEIGHTS] Weighting disabled (USE_SOC_WEIGHTS=False).")

# dtrain gets weights; dval/dtest stay unweighted
dtrain = xgb.DMatrix(
    X_train_tree.values, label=y_train.values, weight=weights, feature_names=feature_order
)
dval   = xgb.DMatrix(
    X_val_tree.values,   label=y_val.values,   feature_names=feature_order
)
dtest  = xgb.DMatrix(
    X_test_tree.values,  label=y_test.values,  feature_names=feature_order
)


def get_best_rounds(booster: xgb.Booster) -> int:
    """
    Return the number of boosting rounds to use for prediction,
    compatible with old/new xgboost versions.
    """
    # Newer xgboost exposes best_iteration (0-based)
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        return int(booster.best_iteration) + 1
    # Fallback: use number of boosted rounds
    try:
        return int(booster.num_boosted_rounds())
    except Exception:
        return None  # predict without slicing

rng = np.random.default_rng(SEED)

def sample_params():
    return {
        "eta":              float(rng.uniform(0.01, 0.20)),   # learning_rate
        "max_depth":        int(rng.integers(4, 13)),
        "min_child_weight": int(rng.integers(1, 21)),
        "subsample":        float(rng.uniform(0.60, 1.00)),
        "colsample_bytree": float(rng.uniform(0.60, 1.00)),
        "lambda":           float(rng.uniform(0.0, 10.0)),    # L2
        "alpha":            float(rng.uniform(0.0, 2.0)),     # L1
    }

base_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": SEED,
}

best = {"rmse": float("inf"), "booster": None, "params": None}
n_trials = 48  # bump to 48 later if you want a deeper sweep

for _ in range(n_trials):
    params = {**base_params, **sample_params()}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    # rmse on validation set
    rmse_val = float(booster.eval(dval).split(":")[-1])
    if rmse_val < best["rmse"]:
        best = {"rmse": rmse_val, "booster": booster, "params": params}

bst = best["booster"]
best_rounds = get_best_rounds(bst)
print(f"[XGB] Best val RMSE: {best['rmse']:.3f} at rounds={best_rounds}")
print("[XGB] Best params:", best["params"])

def _pred(dm):
    # For new xgboost, prefer iteration_range; if best_rounds is None, predict full
    if best_rounds is not None:
        return np.maximum(0.0, bst.predict(dm, iteration_range=(0, best_rounds)))
    return np.maximum(0.0, bst.predict(dm))

yhat_train_xgb = _pred(dtrain)
yhat_val_xgb   = _pred(dval)
yhat_test_xgb  = _pred(dtest)

xgb_metrics = {
    "train": metrics_dict(y_train, yhat_train_xgb),
    "val":   metrics_dict(y_val,   yhat_val_xgb),
    "test":  metrics_dict(y_test,  yhat_test_xgb),
    "hyperparams": best["params"],
    "best_rounds": best_rounds if best_rounds is not None else "all",
}

print(f"[XGB] Val RMSE: {xgb_metrics['val']['rmse']:.3f} | "
      f"Test RMSE: {xgb_metrics['test']['rmse']:.3f} | "
      f"Val MAPE: {xgb_metrics['val']['mape']:.2f}%")

# Success checks vs baseline
val_vs_test_ok = xgb_metrics["val"]["rmse"] <= 1.10 * xgb_metrics["test"]["rmse"]
improves_20pct = xgb_metrics["val"]["rmse"] <= 0.80 * baseline_metrics["val"]["rmse"]
print(f"[CHECK] No strong overfit (val ≤ 1.10×test): {val_vs_test_ok}")
print(f"[CHECK] XGB improves ≥20% vs linear:       {improves_20pct}")

# Save model + metrics
bst.save_model(os.path.join(MODEL_DIR, "xgb.json"))

# Top importances (gain)
imp_gain = []
try:
    scores = bst.get_score(importance_type="gain")
    imp_gain = sorted(
        [{"feature": k, "gain": float(v)} for k, v in scores.items()],
        key=lambda d: -d["gain"]
    )[:50]
except Exception:
    pass

metrics_bundle = {
    "dataset_sizes": {"train": int(len(X_train_tree)), "val": int(len(X_val_tree)), "test": int(len(X_test_tree))},
    "feature_count": len(feature_order),
    "feature_order": feature_order,
    "baseline_linear": baseline_metrics,
    "xgb": xgb_metrics,
    "success_checks": {
        "val_vs_test_rmse_ok": val_vs_test_ok,
        "xgb_beats_linear_by_20pct": improves_20pct,
    },
    "top_importances_gain": imp_gain,
    "seed": SEED,
}

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_bundle, f, indent=2, default=float)

print("[SAVE] Model → ./model/xgb.json")
print("[SAVE] Metrics → ./model/metrics.json")

# ===== OPTIONAL: LightGBM / CatBoost comparison =====
try:
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X_train_tree, label=y_train)
    lgb_val   = lgb.Dataset(X_val_tree,   label=y_val)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": SEED,
    }
    lgb_model = lgb.train(
        lgb_params, lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train","val"],
        early_stopping_rounds=200,
        verbose_eval=False
    )
    yhat_test_lgb = np.maximum(0.0, lgb_model.predict(X_test_tree, num_iteration=lgb_model.best_iteration))
    lgb_rmse = float(np.sqrt(((yhat_test_lgb - y_test.values)**2).mean()))
    print(f"[LGBM] Test RMSE: {lgb_rmse:.3f}")
except Exception as e:
    print("[LGBM] Skipped (not installed or failed):", e)

try:
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor(
        loss_function="RMSE", depth=8, learning_rate=0.05,
        l2_leaf_reg=3.0, iterations=5000, random_seed=SEED,
        od_type="Iter", od_wait=200, verbose=False
    )
    cb.fit(X_train_tree, y_train, eval_set=(X_val_tree, y_val), use_best_model=True)
    yhat_test_cb = np.maximum(0.0, cb.predict(X_test_tree))
    cb_rmse = float(np.sqrt(((yhat_test_cb - y_test.values)**2).mean()))
    print(f"[CATBOOST] Test RMSE: {cb_rmse:.3f}")
except Exception as e:
    print("[CATBOOST] Skipped (not installed or failed):", e)


# ===== STEP 6: DIAGNOSTICS =====
import pandas as pd

# 6.1 helper for metrics (reuse the one you already defined)
def soc_band_errors(X_df, y_true, y_pred, soc_col="soc"):
    out = {}
    if soc_col not in X_df.columns:
        print(f"[WARN] SOC column '{soc_col}' not in features.")
        return out
    df = X_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["soc_band"] = pd.cut(df[soc_col],
                            bins=[0,20,40,60,80,100],
                            labels=["0-20","20-40","40-60","60-80","80-100"],
                            include_lowest=True)
    for band, g in df.groupby("soc_band"):
        if len(g) == 0: 
            continue
        m = metrics_dict(g["y_true"].values, g["y_pred"].values)
        out[str(band)] = m
    return out

# 6.2 Errors by SOC band
val_soc_errs  = soc_band_errors(X_val,  y_val,  yhat_val_xgb)
test_soc_errs = soc_band_errors(X_test, y_test, yhat_test_xgb)
print("\n[SOC-BAND ERRORS] Val:", val_soc_errs)
print("[SOC-BAND ERRORS] Test:", test_soc_errs)

# 6.3 Errors by current deciles
def current_decile_errors(X_df, y_true, y_pred, cur_col="current"):
    out = {}
    if cur_col not in X_df.columns:
        print(f"[WARN] current column '{cur_col}' not in features.")
        return out
    df = X_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    # absolute current, split into 10 bins
    df["cur_decile"] = pd.qcut(df[cur_col].abs(), q=10, duplicates="drop")
    for dec, g in df.groupby("cur_decile"):
        if len(g) == 0:
            continue
        m = metrics_dict(g["y_true"].values, g["y_pred"].values)
        out[str(dec)] = m
    return out

val_cur_errs  = current_decile_errors(X_val,  y_val,  yhat_val_xgb)
test_cur_errs = current_decile_errors(X_test, y_test, yhat_test_xgb)
print("\n[CURRENT-DECILE ERRORS] Val:", val_cur_errs)
print("[CURRENT-DECILE ERRORS] Test:", test_cur_errs)

# 6.4 Feature importance (gain)
try:
    gains = bst.get_score(importance_type="gain")
    top5 = sorted(gains.items(), key=lambda x: -x[1])[:5]
    print("\n[FEATURE IMPORTANCE - Top 5 by gain]:", top5)
except Exception as e:
    print("[WARN] Could not extract importance:", e)

# ===== STEP 7: SAVE EVERYTHING (artifacts + inference helper) =====
import os, json, joblib
from pathlib import Path

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# 7.1 Save models
# - XGBoost primary
bst.save_model(os.path.join(MODEL_DIR, "xgb.json"))

# - Linear baseline (we already wrapped preprocessor+ridge in linear.joblib at Step-4)
#   If you want to be explicit again, uncomment the following two lines:
# baseline_pipeline = Pipeline([("prep", linear_preprocessor), ("ridge", linear_model)])
# joblib.dump({"pipeline": baseline_pipeline, "features": feature_order}, os.path.join(MODEL_DIR, "linear.joblib"))

# 7.2 Save feature order the model expects
with open(os.path.join(MODEL_DIR, "feature_list.txt"), "w") as f:
    f.write("\n".join(feature_order) + "\n")

# 7.3 Save a consolidated metrics report (built at Step-5 + Step-6)
bundle = {
    "dataset_sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
    "feature_count": len(feature_order),
    "features_preview": feature_order[:10],
    "baseline_linear": baseline_metrics,       # train/val/test metrics for Ridge
    "xgb": xgb_metrics,                         # train/val/test metrics for XGB
    "success_checks": {
        "val_vs_test_rmse_ok": xgb_metrics["val"]["rmse"] <= 1.10 * xgb_metrics["test"]["rmse"],
        "xgb_beats_linear_by_20pct": xgb_metrics["val"]["rmse"] <= 0.80 * baseline_metrics["val"]["rmse"],
    },
    "diagnostics": {
        "soc_band_errors_val": val_soc_errs,
        "soc_band_errors_test": test_soc_errs,
        "current_decile_errors_val": val_cur_errs,
        "current_decile_errors_test": test_cur_errs,
    },
    "top_importances_gain": imp_gain if 'imp_gain' in globals() else [],
    "seed": SEED,
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(bundle, f, indent=2, default=float)

# 7.4 Write a tiny inference helper you can import anywhere
infer_code = f'''# Auto-generated inference helper for RCT model
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
'''
with open(os.path.join(MODEL_DIR, "inference.py"), "w") as f:
    f.write(infer_code)

print("[SAVE] Artifacts written to ./model/: xgb.json, linear.joblib, feature_list.txt, metrics.json, inference.py")


# 7.5 (Optional) Mirror to S3
# If your role has permissions, you can sync the folder with:
#   aws s3 sync ./model s3://bortana-sagemaker-analysis/bev2/models/rct/

# ===== STEP 8A: TEST & PLOTS =====
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Parity (y vs ŷ) on TEST
plt.figure()
plt.scatter(y_test, yhat_test_xgb, s=8, alpha=0.6)
min_y = float(np.min([y_test.min(), yhat_test_xgb.min()]))
max_y = float(np.max([y_test.max(), yhat_test_xgb.max()]))
plt.plot([min_y, max_y], [min_y, max_y])  # y=x reference
plt.xlabel("Actual RCT minutes")
plt.ylabel("Predicted RCT minutes")
plt.title("XGBoost parity plot (TEST)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "parity_test.png"), dpi=160)
plt.close()

# Residuals (ŷ - y) vs y
resid = yhat_test_xgb - y_test.values
plt.figure()
plt.scatter(y_test, resid, s=8, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Actual RCT minutes")
plt.ylabel("Residual (pred - actual)")
plt.title("Residuals vs Actual (TEST)")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "residuals_test.png"), dpi=160)
plt.close()

print("[PLOTS] Saved parity_test.png and residuals_test.png in ./model/")
