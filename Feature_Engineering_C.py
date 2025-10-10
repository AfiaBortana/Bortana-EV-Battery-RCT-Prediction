#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: Feature_Engineering_C.py

"""
EV Charging ML Feature Engineering — SageMaker Processing (RCT-only + SOC-interval analytics)

What changed (permanent):
- Removed ALL efficiency calculations/outputs (no estimated_energy_stored_kwh, no efficiency_pct, no features/efficiency/*).
- Kept alignment, sessionization, 1-min curation, RCT row building, gating; energy integration only for min-energy gating.
- Integrated time-weighted modal SOC-interval bucket picking as the standard post-processing step for interpretability.
- Outputs:
    curated/rows/YYYY-MM.parquet
    features/rct/rct_YYYY-MM.parquet
    features/rct/rct_YYYY-MM.csv
    features/rct/rct_all.parquet
    features/rct/rct_all.csv
    analytics/soc_intervals/soc_YYYY-MM.parquet
    analytics/soc_intervals/soc_intervals_all.parquet
"""

import os, io, re, math, argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import boto3, botocore
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# ---------- Constants ----------
PROC_OUT_BASE = os.environ.get("PROC_OUT_BASE", "/opt/ml/processing")
MEL_TZ = ZoneInfo("Australia/Melbourne")
UTC_TZ = ZoneInfo("UTC")
FIFTEEN_SEC = pd.Timedelta(seconds=15)

# ---------- Logging ----------
def log(msg: str, level: str="INFO") -> None:
    print(f"[{level}] {msg}", flush=True)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ---------- CLI ----------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EV Charging Feature Engineering (RCT-only + SOC-interval analytics)")
    # IO
    p.add_argument("--in-bucket", required=True)
    p.add_argument("--in-prefix", required=True)      # root; script detects subfolders
    p.add_argument("--out-bucket", required=True)
    p.add_argument("--out-prefix", required=True)

    # Battery (kept for context; not used for efficiency anymore)
    p.add_argument("--battery-capacity-kwh", type=float, default=53.280)

    # Sessionization / quality
    p.add_argument("--session-gap-min", type=int, default=10)
    p.add_argument("--idle-speed-thresh", type=float, default=0.5)
    p.add_argument("--min-charge-a", type=float, default=6.0)         # looser to catch AC
    p.add_argument("--max-charge-a", type=float, default=350.0)
    p.add_argument("--min-session-min", type=float, default=4.0)      # small, keep short top-ups
    p.add_argument("--min-delta-soc-pct", type=float, default=0.2)    # very small ΔSOC
    p.add_argument("--min-e-in-kwh", type=float, default=0.1)         # used only for gating, not for outputs
    p.add_argument("--min-idle-frac", type=float, default=0.8)
    p.add_argument("--min-median-charge-a", type=float, default=5.0)
    p.add_argument("--max-dt-sec", type=int, default=180)

    # SOC-interval modal bucket analytics (authoritative for current ranges)
    p.add_argument("--soc-step-pct", type=float, default=5.0)         # interval width in % SOC
    p.add_argument(
        "--current-buckets",
        type=str,
        default="0,1,5,10,20,40,80,200,1000",                         # inclusive left, exclusive right
        help="Comma-separated abs(A) edges, e.g. '0,1,5,10,20,40,80,200,1000'"
    )

    # Months filter (comma-separated YYYY-MM)
    p.add_argument("--months", type=str, default="")
    return p.parse_args()

# ---------- S3 helpers ----------
MASTER_CANDIDATES = ["raw_data_master/", "raw-data_master/"]
SLAVE_CANDIDATES  = ["raw_data_slave/",  "raw-data_slave/" ]

def s3_prefix_exists(s3, bucket: str, prefix: str) -> bool:
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return any(obj.get("Key", "").startswith(prefix) for obj in resp.get("Contents", []))
    except botocore.exceptions.ClientError:
        return False

def choose_subprefix(s3, bucket: str, root: str, candidates: List[str]) -> Optional[str]:
    root = root.rstrip("/") + "/"
    for c in candidates:
        p = root + c
        if s3_prefix_exists(s3, bucket, p):
            return p
    return None

MONTH_RE = re.compile(r"(\d{4}-\d{2})\.csv$", re.IGNORECASE)

def list_month_csvs(s3, bucket: str, prefix: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for it in page.get("Contents", []):
            key = it["Key"]
            if key.endswith("/"):
                continue
            bn = os.path.basename(key)
            m = MONTH_RE.search(bn)
            if not m:
                if not bn.lower().endswith(".csv"):
                    log(f"Skipping non-CSV: s3://{bucket}/{key}", "WARN")
                continue
            out[m.group(1)] = key
    return dict(sorted(out.items()))

def read_s3_csv(s3, bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except botocore.exceptions.ClientError as e:
        log(f"Cannot fetch {key}: {e}", "WARN"); return None
    if not raw: return None

    last_err = None
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python",
                             comment="#", on_bad_lines="skip", skip_blank_lines=True)
            if df.empty: return None
            df = df.dropna(axis=1, how="all")
            df.rename(columns=lambda c: str(c).replace("\ufeff","").strip(), inplace=True)
            alias = {
                # core
                "avg_soc":"soc", "battery_soc":"soc", "M_battery_soc":"soc",
                "avg_voltage":"voltage", "M_battery_voltage":"voltage", "pack_voltage":"voltage",
                "avg_current":"current", "avg_curren":"current", "M_battery_current":"current", "pack_current":"current",
                "veh_speed":"speed","vehicle_speed":"speed","M_vehicle_speed":"speed",
                "km_driven":"km","odometer":"km","odo":"km","odo_km":"km","distance_km":"km","total_km":"km",
                # optional thermal/limits/pack
                "cell_temp":"cell_temp", "cell_tem":"cell_temp", "bms_temp":"bms_temp",
                "pack_delta":"pack_delta", "pack_dlt":"pack_delta",
                "charge_limit":"charge_limit","charge_lim":"charge_limit",
                "discharge_limit":"discharge_limit","discharge_lim":"discharge_limit",
                # slave
                "slave_soc":"soc_slave","slave_voltage":"voltage_slave","slave_current":"current_slave",
            }
            df.rename(columns=alias, inplace=True)
            return df
        except Exception as e:
            last_err = e
    log(f"Failed to parse CSV {key}: {last_err}", "WARN")
    return None

# ---------- Timestamp parsing (local, tolerant) ----------
TS_COMPACT = re.compile(r"^\s*((?:[01]?\d|2[0-3])):([0-5]\d)(?::([0-5]\d))?\s*$")
TS_MMSS    = re.compile(r"^\s*(\d{1,3}):([0-5]\d)\s*$")
TS_NUM     = re.compile(r"^\s*\d+(\.\d+)?\s*$")
LOCALE_RE  = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s{1,2}(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\s*$")

def parse_timestamps_local(series: pd.Series, yyyy_mm: str) -> pd.Series:
    base_local = datetime.strptime(yyyy_mm + "-01 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=MEL_TZ)
    out_local: List[pd.Timestamp] = []
    last_local = base_local
    last_num = None

    for raw in series.astype(str).fillna(""):
        s = raw.strip()

        m = LOCALE_RE.match(s)
        if m:
            d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hh, mm = int(m.group(4)), int(m.group(5))
            ss = int(m.group(6)) if m.group(6) else 0
            ampm = (m.group(7) or "").lower()
            if ampm == "pm" and hh != 12: hh += 12
            if ampm == "am" and hh == 12: hh = 0
            loc = datetime(y, mth, d, hh, mm, ss, tzinfo=MEL_TZ)
            out_local.append(pd.Timestamp(loc)); last_local = loc; last_num=None; continue

        m = TS_COMPACT.match(s)
        if m:
            hh, mm = int(m.group(1)), int(m.group(2))
            ss = int(m.group(3)) if m.group(3) else 0
            loc = datetime(last_local.year, last_local.month, last_local.day, hh, mm, ss, tzinfo=MEL_TZ)
            if loc < last_local: loc = loc + timedelta(days=1)
            out_local.append(pd.Timestamp(loc)); last_local = loc; last_num=None; continue

        m = TS_MMSS.match(s)
        if m and not TS_COMPACT.match(s):
            minutes, sec = int(m.group(1)), int(m.group(2))
            loc = base_local + timedelta(minutes=minutes, seconds=sec)
            if loc < last_local: loc = last_local + timedelta(seconds=1)
            out_local.append(pd.Timestamp(loc)); last_local = loc; last_num=None; continue

        if TS_NUM.match(s):
            secs = float(s)
            if last_num is not None and secs < last_num: secs = last_num
            loc = base_local + timedelta(seconds=secs)
            out_local.append(pd.Timestamp(loc)); last_local = loc; last_num=secs; continue

        try:
            loc = pd.to_datetime(s, errors="raise")
            loc = (loc.tz_localize(MEL_TZ) if getattr(loc,"tzinfo",None) is None else loc.tz_convert(MEL_TZ))
            out_local.append(pd.Timestamp(loc)); last_local = loc; last_num=None
        except Exception:
            out_local.append(pd.NaT)

    return pd.to_datetime(pd.Series(out_local, dtype="datetime64[ns, Australia/Melbourne]"))

# ---------- Column picks ----------
MASTER_TS_CAND = ["timestamp","time","date","datetime","dt","Timestamp","DateTime"]
MASTER_SOC_CAND = ["soc","SOC","SoC","m_battery_soc"]
MASTER_VOLT_CAND = ["voltage","Voltage","m_battery_voltage","pack_voltage"]
MASTER_CURR_CAND = ["current","Current","m_battery_current","pack_current","avg_curren"]
MASTER_SPEED_CAND = ["speed","veh_speed","vehicle_speed","Speed","M_vehicle_speed"]
MASTER_CELL_TEMP = ["cell_temp"]
MASTER_BMS_TEMP  = ["bms_temp"]
MASTER_PACK_DELTA = ["pack_delta"]
MASTER_CHG_LIM   = ["charge_limit"]
MASTER_DCHG_LIM  = ["discharge_limit"]
MASTER_KM        = ["km","km_driven","odometer","odo_km","odo","distance_km","total_km"]

SLAVE_TS_CAND = MASTER_TS_CAND
SLAVE_SOC_CAND = ["soc_slave"]
SLAVE_VOLT_CAND = ["voltage_slave"]
SLAVE_CURR_CAND = ["current_slave"]

def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns: return c
    lower = {c.lower(): c for c in df.columns}
    for c in options:
        if c.lower() in lower: return lower[c.lower()]
    return None

# ---------- Prepare master / slave ----------
def prepare_master(df: pd.DataFrame, yyyy_mm: str) -> pd.DataFrame:
    ts_col = pick_col(df, MASTER_TS_CAND)
    if not ts_col: raise ValueError("MASTER: no timestamp")
    ts_local = parse_timestamps_local(df[ts_col], yyyy_mm)

    out = pd.DataFrame({"ts_local": ts_local})
    out["soc_master"]     = pd.to_numeric(df.get(pick_col(df, MASTER_SOC_CAND)), errors="coerce")
    out["voltage_master"] = pd.to_numeric(df.get(pick_col(df, MASTER_VOLT_CAND)), errors="coerce")
    out["current_master"] = pd.to_numeric(df.get(pick_col(df, MASTER_CURR_CAND)), errors="coerce")
    out["speed"]          = pd.to_numeric(df.get(pick_col(df, MASTER_SPEED_CAND)), errors="coerce")
    out["cell_temp"]      = pd.to_numeric(df.get(pick_col(df, MASTER_CELL_TEMP)), errors="coerce")
    out["bms_temp"]       = pd.to_numeric(df.get(pick_col(df, MASTER_BMS_TEMP)), errors="coerce")
    out["pack_delta"]     = pd.to_numeric(df.get(pick_col(df, MASTER_PACK_DELTA)), errors="coerce")
    out["charge_limit"]   = pd.to_numeric(df.get(pick_col(df, MASTER_CHG_LIM)), errors="coerce")
    out["discharge_limit"]= pd.to_numeric(df.get(pick_col(df, MASTER_DCHG_LIM)), errors="coerce")
    out["km"]             = pd.to_numeric(df.get(pick_col(df, MASTER_KM)), errors="coerce")

    return out.dropna(subset=["ts_local"]).sort_values("ts_local").reset_index(drop=True)

def prepare_slave(df: pd.DataFrame, yyyy_mm: str) -> pd.DataFrame:
    ts_col = pick_col(df, SLAVE_TS_CAND)
    if not ts_col: raise ValueError("SLAVE: no timestamp")
    ts_local = parse_timestamps_local(df[ts_col], yyyy_mm)

    out = pd.DataFrame({"ts_local": ts_local})
    out["soc_slave"]     = pd.to_numeric(df.get(pick_col(df, SLAVE_SOC_CAND)), errors="coerce")
    out["voltage_slave"] = pd.to_numeric(df.get(pick_col(df, SLAVE_VOLT_CAND)), errors="coerce")
    out["current_slave"] = pd.to_numeric(df.get(pick_col(df, SLAVE_CURR_CAND)), errors="coerce")
    return out.dropna(subset=["ts_local"]).sort_values("ts_local").reset_index(drop=True)

# ---------- Align master + slave ----------
def align_master_slave(df_m: Optional[pd.DataFrame], df_s: Optional[pd.DataFrame]) -> pd.DataFrame:
    def tscol(df):
        if df is None or df.empty: return df
        if "ts_local" in df.columns: return df
        if isinstance(df.index, pd.DatetimeIndex): return df.reset_index().rename(columns={"index":"ts_local"})
        raise ValueError("expected ts_local column")
    df_m, df_s = tscol(df_m), tscol(df_s)

    pieces = []
    if df_m is not None and not df_m.empty: pieces.append(df_m[["ts_local"]])
    if df_s is not None and not df_s.empty: pieces.append(df_s[["ts_local"]])
    if not pieces:
        return pd.DataFrame(columns=[
            "ts_local","soc","soc_master","soc_slave","voltage","voltage_master","voltage_slave",
            "current","current_master","current_slave","speed","cell_temp","bms_temp","pack_delta",
            "charge_limit","discharge_limit","km"
        ])

    base = (pd.concat(pieces, ignore_index=True)
                .dropna().drop_duplicates()
                .sort_values("ts_local").reset_index(drop=True))

    if df_m is not None and not df_m.empty:
        base = pd.merge_asof(base.sort_values("ts_local"), df_m.sort_values("ts_local"),
                             on="ts_local", direction="nearest", tolerance=FIFTEEN_SEC)
    if df_s is not None and not df_s.empty:
        base = pd.merge_asof(base.sort_values("ts_local"), df_s.sort_values("ts_local"),
                             on="ts_local", direction="nearest", tolerance=FIFTEEN_SEC, suffixes=("","_s"))

    base["soc"]     = base["soc_master"].combine_first(base["soc_slave"])
    base["voltage"] = base[["voltage_master","voltage_slave"]].mean(axis=1, skipna=True)
    base.loc[base[["voltage_master","voltage_slave"]].isna().all(axis=1),"voltage"] = np.nan
    i = base[["current_master","current_slave"]].fillna(0.0).sum(axis=1)
    base["current"] = i.mask(base[["current_master","current_slave"]].isna().all(axis=1))
    return base[[
        "ts_local",
        "soc","soc_master","soc_slave",
        "voltage","voltage_master","voltage_slave",
        "current","current_master","current_slave",
        "speed","cell_temp","bms_temp","pack_delta","charge_limit","discharge_limit","km"
    ]]

# ---------- 1-minute curation ----------
def sanitize_km(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s < 0, np.nan)
    s = s.mask(s.diff() < 0, np.nan)
    return s.ffill()

def curate_to_minute(aligned: pd.DataFrame, idle_speed_thresh: float) -> pd.DataFrame:
    if aligned.empty:
        idx = pd.DatetimeIndex([], tz=MEL_TZ, name="ts_local")
        return pd.DataFrame(index=idx)

    df = aligned.dropna(subset=["ts_local"]).sort_values("ts_local").copy()
    df["minute"] = df["ts_local"].dt.floor("min")
    g = df.groupby("minute", sort=True)

    out = pd.DataFrame({
        
        "soc": g["soc"].last(),
        "soc_master": g["soc_master"].last(),
        "soc_slave": g["soc_slave"].last(),
        "voltage": g["voltage"].mean(),
        "voltage_master": g["voltage_master"].mean(),
        "voltage_slave": g["voltage_slave"].mean(),
        "current": g["current"].mean(),
        "current_master": g["current_master"].mean(),
        "current_slave": g["current_slave"].mean(),
        "speed": g["speed"].mean(),
        "cell_temp": g["cell_temp"].mean(),
        "bms_temp": g["bms_temp"].mean(),
        "pack_delta": g["pack_delta"].mean(),
        "charge_limit": g["charge_limit"].mean(),
        "discharge_limit": g["discharge_limit"].mean(),
        "km": g["km"].last(),
    })
    out.index.name = "ts_local"
    out["power_kw"] = (out["voltage"] * out["current"]) / 1000.0
    out["idle"] = (out["speed"].fillna(0.0).abs() <= idle_speed_thresh)
    out["km"] = sanitize_km(out["km"]) if "km" in out.columns else np.nan
    return out

# ---------- charge sign & sessions ----------
def detect_charge_sign(per_min: pd.DataFrame) -> int:
    s = per_min["soc"].astype(float)
    ds = s.diff()
    med_i = per_min.loc[ds > 0, "current"].median()
    if pd.isna(med_i):
        corr = np.corrcoef(per_min["current"].fillna(0), ds.fillna(0))[0,1]
        return -1 if corr > 0 else 1
    return -1 if med_i > 0 else 1   # want negative while charging

def build_charge_mask(per_min: pd.DataFrame, sign: int, min_a: float, max_a: float) -> pd.Series:
    cur = sign * per_min["current"]      # charging → cur<0
    v   = per_min["voltage"].astype(float)
    dsoc = per_min["soc"].astype(float).diff()
    # permissive; tiny positive ΔSOC allowed to reduce noise
    return (cur < 0) & (cur.abs().between(min_a, max_a)) & (v > 300.0) & (dsoc.fillna(0.0) >= -0.05)

def split_sessions(mask: pd.Series, session_gap_min: int) -> pd.Series:
    idx = mask.index
    on = mask.values
    sid = np.full(len(idx), np.nan)
    last_t = None
    ctr = 0
    for k, t in enumerate(idx):
        if not on[k]: continue
        if last_t is None: ctr += 1
        else:
            gap = (t - last_t).total_seconds()/60.0
            if gap > session_gap_min: ctr += 1
        sid[k] = ctr; last_t = t
    return pd.Series(sid, index=idx, name="session_id")

# ---------- energy integration (kept ONLY for gating) ----------
def _energy_from_power(df: pd.DataFrame, power_col: str, max_dt_sec: int) -> float:
    """Integrate pack power → kWh using trapezoidal rule.
    Why: data is already 1-min averaged; no extra smoothing to avoid undercount.
    NaNs mapped to 0 to keep gating conservative/stable.
    """
    if df.empty or power_col not in df:
        return 0.0
    p = pd.to_numeric(df[power_col], errors="coerce").to_numpy(dtype=float)
    p = np.nan_to_num(p, nan=0.0)  # avoid NaN → NaN area
    ts = (df.index.view("int64") // 10**9).astype(np.int64)
    dt = np.diff(ts, prepend=ts[0])
    dt = np.clip(dt, 0, max_dt_sec)  # cap irregular gaps
    area = (p + np.roll(p, 1)) * 0.5 * dt  # trapezoid (kW * s)
    return abs(float(area.sum() / 3600.0))  # → kWh


def compute_session_energies(df: pd.DataFrame, max_dt_sec: int) -> Tuple[float,float,float,float]:
    if df.empty: return 0.0, 0.0, 0.0, 0.0
    tmp = df.copy()
    tmp["p_master"] = (df["voltage_master"] * df["current_master"]) / 1000.0
    tmp["p_slave"]  = (df["voltage_slave"]  * df["current_slave"])  / 1000.0
    tmp["p_pack"]   = tmp["p_master"].fillna(0.0) + tmp["p_slave"].fillna(0.0)
    e_m = _energy_from_power(tmp, "p_master", max_dt_sec)
    e_s = _energy_from_power(tmp, "p_slave",  max_dt_sec)
    e_p = _energy_from_power(tmp, "p_pack",   max_dt_sec)
    p_avg = float(tmp["p_pack"].mean()) if "p_pack" in tmp else float("nan")
    return float(e_m), float(e_s), float(e_p), p_avg

# ---------- per-session quality ----------
def session_quality(df: pd.DataFrame,
                    min_session_min: float,
                    min_delta_soc_pct: float,
                    min_e_in_kwh: float,
                    min_idle_frac: float,
                    min_median_charge_a: float,
                    max_dt_sec: int) -> Tuple[bool, str, Dict[str,float]]:
    if df.empty: return False, "empty", {}
    dur = (df.index[-1] - df.index[0]).total_seconds()/60.0
    dsoc = float(df["soc"].iloc[-1] - df["soc"].iloc[0])
    idle_frac = float(df["idle"].mean()) if "idle" in df else 0.0
    med_i_abs = float(df["current"].abs().median())
    reasons = []
    if dur < min_session_min: reasons.append("duration")
    if dsoc < min_delta_soc_pct: reasons.append("delta_soc")
    if idle_frac < min_idle_frac: reasons.append("idle_frac")
    if med_i_abs < min_median_charge_a: reasons.append("median_current")

    # energy gating (kept; avoids bogus sessions). Not emitted later.
    _, _, e_p, _ = compute_session_energies(df, max_dt_sec)
    if e_p < min_e_in_kwh: reasons.append("min_energy")

    return len(reasons)==0, ",".join(reasons), {
        "duration_min": dur,
        "delta_soc": dsoc,
        "idle_frac": idle_frac,
        "median_abs_current": med_i_abs,
        "total_energy_kwh": e_p
    }

# ---------- build RCT rows ----------
def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    ts = (df.index.view("int64") // 10**9).astype(np.int64)
    dt_s = np.diff(ts, prepend=ts[0]); dt_s[dt_s==0] = 60
    dt_min = dt_s / 60.0
    df = df.copy()
    df["soc_rate_per_min"] = df["soc"].diff() / dt_min
    temp = df["cell_temp"].combine_first(df["bms_temp"])
    df["temp_rise_rate"] = temp.diff() / dt_min
    return df

def build_rct_rows(session_id: int, sdf: pd.DataFrame, t_end: pd.Timestamp) -> pd.DataFrame:
    if sdf.empty: return sdf
    sdf = add_rate_features(sdf)
    elapsed = (sdf.index - sdf.index[0]).total_seconds()/60.0
    rct = (t_end - sdf.index).total_seconds()/60.0
    out = sdf.copy()
    out["session_id"] = session_id
    out["elapsed_min"] = elapsed
    out["rct_minutes"] = rct
    out = out[out["rct_minutes"] > 0.0]   # strict >0 to prevent label leak
    out["slave_power_kw"] = (out["voltage_slave"] * out["current_slave"]) / 1000.0
    out["soc_delta_to_full"] = 100.0 - out["soc"]
    return out.reset_index().rename(columns={"ts_local":"timestamp_local"})

# ---------- SOC-interval modal bucket analytics (authoritative) ----------
def _format_bucket_labels(edges: List[float]) -> List[str]:
    labels = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        labels.append(f"{int(lo)}–{int(hi)} A" if hi < 1000 else f"{int(lo)}+ A")
    return labels

def soc_modal_buckets_for_session(
    session_id: int,
    sdf: pd.DataFrame,
    sign: int,
    soc_step_pct: float,
    bucket_edges: List[float],
    max_dt_sec: int
) -> pd.DataFrame:
    if sdf.empty: 
        return pd.DataFrame(columns=[
            "session_id","soc_lo","soc_hi","minutes_total",
            "mode_bucket","mode_minutes","confidence",
            "median_abs_current","mean_abs_current"
        ])

    # dt weighting
    ts = (sdf.index.view("int64") // 10**9).astype(np.int64)
    dt_s = np.diff(ts, prepend=ts[0]); dt_s = np.clip(dt_s, 1, max_dt_sec)
    dt_min = dt_s / 60.0

    df = sdf.copy()
    df["dt_min"] = dt_min

    # charging rows only (authoritative current ranges must be during charge)
    cur_eff = sign * df["current"]
    df = df[(cur_eff < 0) & (cur_eff.abs().between(bucket_edges[0], bucket_edges[-1]))].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "session_id","soc_lo","soc_hi","minutes_total",
            "mode_bucket","mode_minutes","confidence",
            "median_abs_current","mean_abs_current"
        ])

    # absolute current + bucket labels
    abs_i = df["current"].abs()
    labels = _format_bucket_labels(bucket_edges)
    df["i_bucket"] = pd.cut(abs_i, bucket_edges, right=False, labels=labels, include_lowest=True)

    # assign SOC interval
    # note: 1-min resolution → assign by trailing SOC value (time-weighted via dt_min)
    soc0, soc1 = float(df["soc"].min()), float(df["soc"].max())
    step = max(0.1, float(soc_step_pct))
    lo = math.floor(soc0 / step) * step
    hi = math.ceil(soc1 / step) * step
    df["soc_bin_lo"] = (np.floor(df["soc"] / step) * step).astype(float)
    df["soc_bin_hi"] = df["soc_bin_lo"] + step

    # aggregate minutes by (interval, bucket)
    grp = df.groupby(["soc_bin_lo","soc_bin_hi","i_bucket"], dropna=True, sort=True)["dt_min"].sum().reset_index()
    # pick modal bucket per interval
    rows = []
    for (lo_i, hi_i), g in grp.groupby(["soc_bin_lo","soc_bin_hi"]):
        total_min = float(g["dt_min"].sum())
        mode_idx = g["dt_min"].idxmax()
        mode_bucket = str(g.loc[mode_idx, "i_bucket"])
        mode_min = float(g.loc[mode_idx, "dt_min"])
        conf = (mode_min / total_min) if total_min > 0 else float("nan")
        # stat summaries within interval (time-weighted median/mean approximated by simple stats over per-minute; robust enough here)
        mask = (df["soc_bin_lo"] == lo_i) & (df["soc_bin_hi"] == hi_i)
        med_abs_i = float(df.loc[mask, "current"].abs().median())
        mean_abs_i = float(df.loc[mask, "current"].abs().mean())
        rows.append({
            "session_id": session_id,
            "soc_lo": float(lo_i),
            "soc_hi": float(hi_i),
            "minutes_total": total_min,
            "mode_bucket": mode_bucket,
            "mode_minutes": mode_min,
            "confidence": conf,
            "median_abs_current": med_abs_i,
            "mean_abs_current": mean_abs_i,
        })
    return pd.DataFrame(rows).sort_values(["session_id","soc_lo"]).reset_index(drop=True)

# ---------- write outputs ----------
def write_parquet_csv(df: pd.DataFrame, lpq: str, lcsv: Optional[str]=None) -> None:
    ensure_dir(os.path.dirname(lpq)); df.to_parquet(lpq, index=False)
    if lcsv: df.to_csv(lcsv, index=False)

def mirror_to_s3(local_path: str, bucket: str, key: str) -> None:
    boto3.client("s3").upload_file(local_path, bucket, key)
    log(f"OK | {os.path.basename(local_path)} → s3://{bucket}/{key}")

# ---------- per-month processing ----------
def process_month(
    s3,
    bucket: str,
    yyyymm: str,
    key_master: Optional[str],
    key_slave: Optional[str],
    args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_m = read_s3_csv(s3, bucket, key_master) if key_master else None
    df_s = read_s3_csv(s3, bucket, key_slave) if key_slave else None

    if df_m is not None and not df_m.empty:
        try: df_m = prepare_master(df_m, yyyymm)
        except Exception as e: log(f"MASTER parse error {key_master}: {e}", "WARN"); df_m=None
    if df_s is not None and not df_s.empty:
        try: df_s = prepare_slave(df_s, yyyymm)
        except Exception as e: log(f"SLAVE parse error {key_slave}: {e}", "WARN"); df_s=None

    aligned = align_master_slave(df_m, df_s)
    curated = curate_to_minute(aligned, args.idle_speed_thresh)

    # charge sessions
    sign = detect_charge_sign(curated)
    chg_mask = build_charge_mask(curated, sign, args.min_charge_a, args.max_charge_a)
    sid = split_sessions(chg_mask, args.session_gap_min)
    curated = curated.copy()
    curated["session_id"] = sid

    # aggregates
    rct_rows: List[pd.DataFrame] = []
    soc_rows: List[pd.DataFrame] = []

    # buckets parse
    bucket_edges = [float(x) for x in str(args.current_buckets).split(",") if str(x).strip() != ""]
    if len(bucket_edges) < 2: 
        bucket_edges = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 200.0, 1000.0]  # safety
    bucket_edges = sorted(bucket_edges)

    for sess_id, sdf in curated.groupby("session_id", dropna=True):
        if sdf.empty: continue
        # quality & gating (no efficiency outputs)
        ok, reasons, _ = session_quality(
            sdf, args.min_session_min, args.min_delta_soc_pct,
            args.min_e_in_kwh, args.min_idle_frac, args.min_median_charge_a, args.max_dt_sec
        )
        if not ok:
            # duration and energy are hard fails; others keep us conservative
            if ("duration" in reasons) or ("min_energy" in reasons):
                continue

        # RCT per-minute rows
        rct_rows.append(build_rct_rows(int(sess_id), sdf, sdf.index[-1]))

        # SOC-interval modal bucket analytics (authoritative)
        soc_tbl = soc_modal_buckets_for_session(
            int(sess_id), sdf, sign, args.soc_step_pct, bucket_edges, args.max_dt_sec
        )
        if not soc_tbl.empty:
            soc_rows.append(soc_tbl)

    # month outputs
    rct_df = (pd.concat(rct_rows, ignore_index=True) if rct_rows else
              pd.DataFrame(columns=[ "vehicle",
                  "timestamp_local","session_id","elapsed_min","rct_minutes",
                  "soc","soc_master","soc_slave","soc_rate_per_min","soc_delta_to_full",
                  "voltage","voltage_master","voltage_slave",
                  "current","current_master","current_slave",
                  "power_kw","slave_power_kw",
                  "cell_temp","bms_temp","temp_rise_rate",
                  "pack_delta","charge_limit","discharge_limit","speed","idle","km"
              ]))
    soc_df = (pd.concat(soc_rows, ignore_index=True) if soc_rows else
              pd.DataFrame(columns=[
                  "session_id","soc_lo","soc_hi","minutes_total",
                  "mode_bucket","mode_minutes","confidence",
                  "median_abs_current","mean_abs_current"
              ]))

    return curated, rct_df, soc_df

# ---------- main ----------
def main():
    args = build_args()
    ensure_dir(os.path.join(PROC_OUT_BASE, "curated", "rows"))
    ensure_dir(os.path.join(PROC_OUT_BASE, "features", "rct"))
    ensure_dir(os.path.join(PROC_OUT_BASE, "analytics", "soc_intervals"))

    s3 = boto3.client("s3")
    root = args.in_prefix.rstrip("/") + "/"
    master_prefix = choose_subprefix(s3, args.in_bucket, root, MASTER_CANDIDATES)
    slave_prefix  = choose_subprefix(s3, args.in_bucket, root, SLAVE_CANDIDATES)
    if not master_prefix and not slave_prefix:
        log("No master or slave subfolders found.", "ERROR"); raise SystemExit(2)

    months_filter = [m.strip() for m in args.months.split(",") if m.strip()] if args.months else []

    m_map = list_month_csvs(s3, args.in_bucket, master_prefix) if master_prefix else {}
    s_map = list_month_csvs(s3, args.in_bucket, slave_prefix)  if slave_prefix  else {}
    all_months = sorted(set(m_map.keys()) | set(s_map.keys()))
    if months_filter: all_months = [m for m in all_months if m in months_filter]
    if not all_months: log("No monthly CSVs discovered for master/slave.", "WARN")

    curated_frames: List[Tuple[str,pd.DataFrame]] = []
    rct_tables: List[pd.DataFrame] = []
    soc_tables: List[pd.DataFrame] = []

    for yyyymm in all_months:
        key_m, key_s = m_map.get(yyyymm), s_map.get(yyyymm)
        log(f"Month {yyyymm} (MASTER:{bool(key_m)} SLAVE:{bool(key_s)})")
        curated, rct_df, soc_df = process_month(s3, args.in_bucket, yyyymm, key_m, key_s, args)
        curated_frames.append((yyyymm, curated))
        if not rct_df.empty: rct_tables.append(rct_df)
        if not soc_df.empty: soc_tables.append(soc_df)

        # 1) curated rows (per-minute aligned)
        lp = os.path.join(PROC_OUT_BASE, "curated", "rows", f"{yyyymm}.parquet")
        ensure_dir(os.path.dirname(lp))
        cols = ["soc","soc_master","soc_slave","voltage","voltage_master","voltage_slave",
                "current","current_master","current_slave","power_kw","speed","idle",
                "cell_temp","bms_temp","pack_delta","charge_limit","discharge_limit","km","session_id"]
        for c in cols:
            if c not in curated.columns: curated[c] = np.nan
        curated_out = curated[cols].copy()
        curated_out.index.name = "ts_local"
        curated_out.to_parquet(lp, index=True)
        s3_key = f"{args.out_prefix.rstrip('/')}/curated/rows/{yyyymm}.parquet"
        mirror_to_s3(lp, args.out_bucket, s3_key)

        # 2) per-minute RCT for the month
        rct_lp = os.path.join(PROC_OUT_BASE, "features", "rct", f"rct_{yyyymm}.parquet")
        rct_lc = os.path.join(PROC_OUT_BASE, "features", "rct", f"rct_{yyyymm}.csv")
        write_parquet_csv(rct_df, rct_lp, rct_lc)
        mirror_to_s3(rct_lp, args.out_bucket, f"{args.out_prefix.rstrip('/')}/features/rct/rct_{yyyymm}.parquet")
        mirror_to_s3(rct_lc, args.out_bucket, f"{args.out_prefix.rstrip('/')}/features/rct/rct_{yyyymm}.csv")

        # 3) SOC-interval modal analytics (interpretability only)
        soc_lp = os.path.join(PROC_OUT_BASE, "analytics", "soc_intervals", f"soc_{yyyymm}.parquet")
        soc_df.to_parquet(soc_lp, index=False)
        mirror_to_s3(soc_lp, args.out_bucket, f"{args.out_prefix.rstrip('/')}/analytics/soc_intervals/soc_{yyyymm}.parquet")

    # Consolidate
    rct_all = (pd.concat(rct_tables, ignore_index=True) if rct_tables
               else pd.DataFrame(columns=[ "vehicle",
                    "timestamp_local","session_id","elapsed_min","rct_minutes",
                    "soc","soc_master","soc_slave","soc_rate_per_min","soc_delta_to_full",
                    "voltage","voltage_master","voltage_slave","current","current_master","current_slave",
                    "power_kw","slave_power_kw","cell_temp","bms_temp","temp_rise_rate",
                    "pack_delta","charge_limit","discharge_limit","speed","idle","km"
               ]))
    soc_all = (pd.concat(soc_tables, ignore_index=True) if soc_tables
               else pd.DataFrame(columns=[
                   "session_id","soc_lo","soc_hi","minutes_total",
                   "mode_bucket","mode_minutes","confidence",
                   "median_abs_current","mean_abs_current"
               ]))

    # Persist consolidated RCT (single training import)
    rct_pq = os.path.join(PROC_OUT_BASE, "features", "rct", "rct_all.parquet")
    rct_cs = os.path.join(PROC_OUT_BASE, "features", "rct", "rct_all.csv")
    write_parquet_csv(rct_all, rct_pq, rct_cs)
    mirror_to_s3(rct_pq, args.out_bucket, f"{args.out_prefix.rstrip('/')}/features/rct/rct_all.parquet")
    mirror_to_s3(rct_cs, args.out_bucket, f"{args.out_prefix.rstrip('/')}/features/rct/rct_all.csv")

    # Persist consolidated SOC-interval analytics
    soc_pq = os.path.join(PROC_OUT_BASE, "analytics", "soc_intervals", "soc_intervals_all.parquet")
    ensure_dir(os.path.dirname(soc_pq))
    soc_all.to_parquet(soc_pq, index=False)
    mirror_to_s3(soc_pq, args.out_bucket, f"{args.out_prefix.rstrip('/')}/analytics/soc_intervals/soc_intervals_all.parquet")

    log("DONE | Outputs: curated rows + RCT dataset (train) + SOC-interval modal analytics (interpretability).", "OK")




if __name__ == "__main__":
    main()

