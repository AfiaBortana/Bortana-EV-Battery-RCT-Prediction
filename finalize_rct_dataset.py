# filepath: finalize_rct_dataset.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finalize the RCT dataset for model training (RCT-only, no efficiency).
- Drops efficiency/energy and other non-training columns.
- Keeps only informative numeric predictors.
- Ensures valid, clipped label (rct_minutes).
- Splits by session_id to avoid leakage; robust when groups are few.
- Writes train/val/test CSVs + feature_list.txt (local or s3://).
"""

import argparse
import os
import re
import json
from typing import List, Tuple, Set, Dict, Optional

import numpy as np
import pandas as pd

try:
    import s3fs  # enables s3:// IO via fsspec
except Exception:
    s3fs = None

LABEL = "rct_minutes"

# Columns that must never be used as predictors
DROP_ALWAYS: Set[str] = {
    "timestamp_local", "session_id", "t_start", "t_end",
    "vehicle", "fail_reason",
    "speed", "idle",  # operational flags; keep out of base model
    "slave_power_kw",  # implementation detail / not broadly available
}

# Efficiency/Energy columns to purge if present (exact + loose patterns)
EFFICIENCY_EXACT: Set[str] = {
    "estimated_energy_stored_kwh",
    "efficiency_pct",
    "charging_efficiency_pct",
    "charger_energy_kwh",
    "session_energy_in_kwh",
    "pack_energy_in_kwh",
    "dc_energy_in_kwh",
    "ac_energy_in_kwh",
    "energy_in_kwh",
    "energy_stored_kwh_estimate",
    "inferred_efficiency",
    "efficiency",
    "loss_kwh",
    "net_energy_kwh",
    "total_energy_kwh",  # sometimes carried through; not a per-minute feature
}
EFFICIENCY_REGEX = re.compile(
    r"(efficien|energy[_\-]stored|charger[_\-]?energy|energy[_\-]?loss|loss[_\-]?kwh)",
    re.IGNORECASE,
)


# ---------------- IO helpers ----------------
def _fs_storage() -> Dict:
    return {"storage_options": {}} if s3fs is not None else {}


def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, **_fs_storage())
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, **_fs_storage())


def write_text(text: str, path: str) -> None:
    # Why: feature_list may need to go to s3:// as well
    if str(path).startswith("s3://") and s3fs is not None:
        import fsspec
        with fsspec.open(path, "w") as f:
            f.write(text)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


# ---------------- Utilities ----------------
def as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object-like numeric columns to numeric; keep original floats/ints."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        # Try numeric coercion only for 'object' and 'string' dtypes
        if out[c].dtype == "object" or pd.api.types.is_string_dtype(out[c]):
            coerced = pd.to_numeric(out[c], errors="coerce")
            # adopt if conversion is meaningful (not all NaN)
            if coerced.notna().sum() > 0:
                out[c] = coerced
    return out


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def drop_efficiency_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    to_drop = {c for c in cols if c in EFFICIENCY_EXACT or EFFICIENCY_REGEX.search(c or "")}
    if not to_drop:
        return df
    return df.drop(columns=sorted(to_drop))


def drop_non_informative(df: pd.DataFrame, max_missing_frac: float) -> pd.DataFrame:
    """Drop all-null, constant, or too-missing numeric features. Keeps label."""
    keep = []
    for c in df.columns:
        if c == LABEL:
            keep.append(c); continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        miss = 1.0 - (s.notna().mean())
        if miss >= max_missing_frac:
            continue
        keep.append(c)
    # Always include label at end
    cols = [c for c in keep if c != LABEL] + [LABEL]
    return df[cols]


def group_split(df: pd.DataFrame, group_col: str,
                test_size=0.2, val_size=0.1, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-wise split to avoid leakage across the same session."""
    rng = np.random.default_rng(seed)
    if group_col in df.columns and df[group_col].notna().any():
        groups = pd.Series(df[group_col].dropna().unique()).tolist()
        rng.shuffle(groups)
        n = len(groups)
        if n <= 1:
            # Degenerate: single group → train only.
            test_g, val_g = set(), set()
        elif n == 2:
            # Two groups: 1 train, 1 test.
            test_g, val_g = {groups[0]}, set()
        else:
            n_test = max(1, int(round(test_size * n)))
            n_rem = max(0, n - n_test)
            n_val = max(1, int(round(val_size * n_rem))) if n_rem >= 2 else 0
            test_g = set(groups[:n_test])
            val_g = set(groups[n_test:n_test + n_val])
        def mask(gs):
            return df[group_col].isin(gs) if gs else pd.Series(False, index=df.index)
        test = df[mask(test_g)]
        val = df[mask(val_g)]
        train = df.loc[~df.index.isin(test.index) & ~df.index.isin(val.index)]
        return train, val, test
    # No group column → random row split (last resort).
    idx = df.index.to_numpy()
    rng.shuffle(idx)
    n = len(idx)
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * (n - n_test))))
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test:n_test + n_val])
    test = df.loc[list(test_idx)]
    val = df.loc[list(val_idx)]
    train = df.drop(test.index.union(val.index))
    return train, val, test


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to rct_all.(csv|parquet) — local or s3://")
    ap.add_argument("--outdir", required=True, help="Output folder — local or s3://")
    ap.add_argument("--clip-max-minutes", type=float, default=24 * 60, help="Upper clip for label")
    ap.add_argument("--min-voltage", type=float, default=200.0)
    ap.add_argument("--max-voltage", type=float, default=900.0)
    ap.add_argument("--min-current", type=float, default=-500.0)
    ap.add_argument("--max-current", type=float, default=500.0)
    ap.add_argument("--max-missing-frac", type=float, default=0.98, help="Drop columns with >= this missing fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # IO
    df = read_any(args.src)
    if df.empty:
        raise SystemExit("Empty RCT dataset")

    # Coerce potential numeric strings → numeric
    df = as_numeric(df)

    # Label checks
    if LABEL not in df.columns:
        raise SystemExit(f"Label '{LABEL}' missing")
    df = df[df[LABEL].notna() & (df[LABEL] > 0)]
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").clip(upper=args.clip_max_minutes)

    # Light sanity bounds (kept loose; guard against corrupt rows)
    if "voltage" in df.columns:
        df = df[(df["voltage"].between(args.min_voltage, args.max_voltage) | df["voltage"].isna())]
    if "current" in df.columns:
        df = df[(df["current"].between(args.min_current, args.max_current) | df["current"].isna())]

    # Deduplicate if both timestamp/session exist
    keys = [k for k in ["timestamp_local", "session_id"] if k in df.columns]
    if keys:
        df = df[~df.duplicated(subset=keys, keep="first")]

    # Drop non-training columns
    drop_cols = [c for c in DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop any efficiency/energy columns (both explicit and pattern-based)
    df = drop_efficiency_cols(df)

    # Keep numeric columns only
    num_only = df[numeric_cols(df) + [LABEL]] if LABEL in df else df
    num_only = num_only.loc[:, ~num_only.columns.duplicated()]
    # Remove non-informative features
    num_only = drop_non_informative(num_only, max_missing_frac=args.max_missing_frac)

    if LABEL not in num_only.columns:
        raise SystemExit("Label lost during column pruning — aborting")

    feats = [c for c in num_only.columns if c != LABEL]
    if not feats:
        raise SystemExit("No features left after pruning; check input schema")

    # Train/Val/Test split (grouped by session)
    # Keep original 'session_id' just for splitting, if it exists
    df_for_split = df.copy()
    if "session_id" in df_for_split.columns:
        num_only["session_id"] = df_for_split["session_id"]

    train, val, test = group_split(num_only, "session_id", seed=args.seed)

    # Drop helper column from outputs if present
    for part in (train, val, test):
        if "session_id" in part.columns:
            part.drop(columns=["session_id"], inplace=True)

    # Ensure output folder(s)
    # For local paths, create directory; for s3, fsspec will handle it.
    if not str(args.outdir).startswith("s3://"):
        os.makedirs(args.outdir, exist_ok=True)

    # Persist in consistent feature order
    for name, part in (("train", train), ("val", val), ("test", test)):
        out_path = os.path.join(args.outdir, f"{name}.csv")
        write_csv(part[feats + [LABEL]].copy(), out_path)

    # Feature list + minimal schema
    feat_list_path = os.path.join(args.outdir, "feature_list.txt")
    write_text("\n".join(feats) + "\n", feat_list_path)

    schema_path = os.path.join(args.outdir, "schema.json")
    schema = {"label": LABEL, "features": feats, "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test))}
    write_text(json.dumps(schema, indent=2), schema_path)

    # Summary
    print("Label    :", LABEL)
    print("Features :", feats)
    print("Shapes   :", {"train": train.shape, "val": val.shape, "test": test.shape})
    print("Saved to :", args.outdir)


if __name__ == "__main__":
    main()
