# filepath: soc_interval_strategy.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SOC-interval analytics with authoritative time-weighted modal amp buckets.

Guarantees:
- No efficiency features; interpretability-only artifacts.
- Modal bucket picking is authoritative; mean current is NOT used for bin selection.
- Robust to irregular sampling via time weighting and SOC-edge interpolation.
"""

from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
from typing import Tuple, List, Optional

try:
    import s3fs  # enables s3:// paths
except Exception:
    s3fs = None


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser("SOC-interval tables (modal amp buckets)")
    p.add_argument("--src", required=True, help="rct_all.(parquet|csv), local or s3://")
    p.add_argument("--outdir", default="./soc_intervals")
    p.add_argument("--band-size", type=int, default=5, help="SOC band width in percentage points")
    p.add_argument(
        "--current-bins",
        type=str,
        default="0,1,5,10,20,40,80,200,1000",
        help="abs(A) edges, comma-separated; left-closed, right-open bins",
    )
    p.add_argument("--min-band-mins", type=float, default=0.5, help="min minutes required to accept a band")
    return p.parse_args()


# ----------------------------- IO ------------------------------
def read_any(path: str) -> pd.DataFrame:
    is_s3 = str(path).startswith("s3://")
    fs = {"filesystem": s3fs.S3FileSystem()} if (is_s3 and s3fs is not None) else {}
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, **fs)
    return pd.read_csv(path)


def write_text(text: str, path: str) -> None:
    if str(path).startswith("s3://") and s3fs is not None:
        import fsspec
        with fsspec.open(path, "w") as f:
            f.write(text)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


# ------------------------ Helpers ------------------------------
def label_bins_from_edges(edges: np.ndarray) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"{int(lo)}–{int(hi)} A" if hi < 1000 else f"{int(lo)}+ A")
    return labels


def bucket_index(val_abs: float, edges: np.ndarray) -> int:
    if np.isnan(val_abs):  # map NaN to lowest bin
        return 0
    idx = np.digitize([val_abs], edges, right=False)[0] - 1  # left-closed
    return int(np.clip(idx, 0, len(edges) - 2))


def detect_charge_sign(df: pd.DataFrame) -> int:
    """Infer sign convention of charging current. Returns -1 when charging current is negative."""
    s = pd.to_numeric(df["soc"], errors="coerce")
    ds = s.diff()
    cur = pd.to_numeric(df.get("current"), errors="coerce")
    med_i_when_chg = cur.loc[ds > 0].median()
    if np.isfinite(med_i_when_chg):
        return -1 if med_i_when_chg > 0 else 1
    corr = np.corrcoef(cur.fillna(0), ds.fillna(0))[0, 1]
    return -1 if corr > 0 else 1


def interpolate_time_at_soc(t_minutes: np.ndarray, soc: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Monotone SOC and de-duplicate for stable interpolation."""
    soc_mono = np.maximum.accumulate(soc.astype(float))
    keep = np.r_[True, np.diff(soc_mono) > 1e-9]
    t_k = t_minutes[keep]
    s_k = soc_mono[keep]
    s_k, uniq_idx = np.unique(s_k, return_index=True)
    t_k = t_k[uniq_idx]
    lo, hi = np.nanmin(s_k), np.nanmax(s_k)
    out = np.full_like(targets, np.nan, dtype=float)
    ok = (targets >= lo) & (targets <= hi)
    out[ok] = np.interp(targets[ok], s_k, t_k)
    return out


def time_weighted_stats_in_slice(
    df: pd.DataFrame,
    tcol: str,
    icol: str,
    t0: float,
    t1: float,
    edges: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """
    Returns (time_weighted_mean_abs_current, bucket_time_shares, approx_median_abs_current).
    bucket_time_shares has length len(edges)-1 and sums to 1.0 (if duration>0).
    """
    d = df[[tcol, icol]].dropna().copy()
    if d.empty:
        return np.nan, np.zeros(len(edges) - 1, dtype=float), np.nan

    # Ensure coverage and clamp to [t0,t1]
    if t0 < d[tcol].iloc[0]:
        d.loc[len(d)] = [t0, d[icol].iloc[0]]
    if t1 > d[tcol].iloc[-1]:
        d.loc[len(d)] = [t1, d[icol].iloc[-1]]
    d = d[(d[tcol] >= t0) & (d[tcol] <= t1)].sort_values(tcol).reset_index(drop=True)
    if d.empty:
        return np.nan, np.zeros(len(edges) - 1, dtype=float), np.nan

    t = d[tcol].to_numpy(float)
    iabs = np.abs(d[icol].to_numpy(float))
    if len(t) == 1:
        b = bucket_index(iabs[0], edges)
        shares = np.eye(1, len(edges) - 1, b).ravel()
        return float(iabs[0]), shares, float(iabs[0])

    dt = np.diff(t)
    if np.any(dt < 0):
        o = np.argsort(t)
        t, iabs = t[o], iabs[o]
        dt = np.diff(t)

    mid_i = (iabs[:-1] + iabs[1:]) / 2.0
    dur = float(dt.sum())
    if dur <= 0:
        return float(np.nanmean(iabs)), np.zeros(len(edges) - 1, dtype=float), float(np.nanmedian(iabs))

    # Time-weighted mean
    mean_abs = float(np.sum(mid_i * dt) / dur)

    # Bucket shares (time in each mid-point bucket)
    shares = np.zeros(len(edges) - 1, dtype=float)
    for k in range(len(dt)):
        shares[bucket_index(mid_i[k], edges)] += dt[k]
    shares /= dur

    # Approx median by weighted midpoints
    # (sufficient for interpretability; exact weighted median not needed here)
    order = np.argsort(mid_i)
    cum = np.cumsum(dt[order]) / dur
    approx_median = float(mid_i[order][np.searchsorted(cum, 0.5, side="left")])

    return mean_abs, shares, approx_median


# --------------------------- Builder ---------------------------
def build_interval_tables(
    rct: pd.DataFrame,
    band: int,
    edges: np.ndarray,
    min_band_mins: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    need = {"timestamp_local", "soc", "session_id"}
    miss = [c for c in need if c not in rct.columns]
    if miss:
        raise SystemExit(f"Missing columns: {miss}")

    # Prioritize pack current for analytics; fallback cleanly
    has_pack = "current" in rct.columns
    has_m = "current_master" in rct.columns
    has_s = "current_slave" in rct.columns
    if not (has_pack or has_m or has_s):
        raise SystemExit("No current columns found (need 'current' and/or 'current_master'/'current_slave').")

    rct = rct.copy()
    rct["timestamp_local"] = pd.to_datetime(rct["timestamp_local"], errors="coerce")
    rct.sort_values(["session_id", "timestamp_local"], inplace=True, ignore_index=True)

    # Charge sign + mask (authoritative filter)
    sign = detect_charge_sign(rct)
    def is_charging(series: pd.Series) -> pd.Series:
        return (sign * pd.to_numeric(series, errors="coerce")) < 0

    chg_mask = pd.Series(False, index=rct.index)
    if has_pack: chg_mask |= is_charging(rct["current"])
    if has_m:    chg_mask |= is_charging(rct["current_master"])
    if has_s:    chg_mask |= is_charging(rct["current_slave"])
    rct = rct.loc[chg_mask].copy()

    # Effective current column (pack > master > slave)
    if has_pack:
        rct["_i_eff"] = rct["current"]
    elif has_m:
        rct["_i_eff"] = rct["current_master"]
    else:
        rct["_i_eff"] = rct["current_slave"]

    # Minutes since session start
    rct["_t0"] = rct.groupby("session_id")["timestamp_local"].transform("min")
    rct["_tmin"] = (rct["timestamp_local"] - rct["_t0"]).dt.total_seconds() / 60.0

    labels = label_bins_from_edges(edges)
    rows: List[dict] = []

    for sid, g in rct.groupby("session_id", dropna=True):
        g = g[["_tmin", "soc", "_i_eff", "current_master", "current_slave"]].copy()
        g = g.dropna(subset=["soc"]).sort_values("_tmin")
        if len(g) < 3:
            continue

        s_lo, s_hi = float(np.nanmin(g["soc"])), float(np.nanmax(g["soc"]))
        # integer grid (monotone) for boundary interpolation
        targets = np.arange(np.floor(s_lo), np.ceil(s_hi) + 1, 1.0)
        if len(targets) < band + 1:
            continue
        t_at = interpolate_time_at_soc(g["_tmin"].to_numpy(), g["soc"].to_numpy(), targets)

        for s0 in range(int(np.floor(s_lo)), int(np.floor(s_hi)) - band + 1, 1):
            s1 = s0 + band
            i0 = np.where(np.isclose(targets, s0))[0]
            i1 = np.where(np.isclose(targets, s1))[0]
            if i0.size == 0 or i1.size == 0:
                continue
            t0, t1 = t_at[i0[0]], t_at[i1[0]]
            if not np.isfinite(t0) or not np.isfinite(t1):
                continue
            dt = t1 - t0
            if dt < min_band_mins:
                continue

            # Pack/effective current stats (authoritative modal selection)
            mean_eff, shares_eff, med_eff = time_weighted_stats_in_slice(g, "_tmin", "_i_eff", t0, t1, edges)
            b_eff = int(np.argmax(shares_eff)) if shares_eff.sum() > 0 else 0
            conf_eff = float(shares_eff[b_eff]) if shares_eff.size else np.nan

            # Optional master/slave analytics for diagnostics
            if has_m:
                mean_m, shares_m, med_m = time_weighted_stats_in_slice(g, "_tmin", "current_master", t0, t1, edges)
                b_m = int(np.argmax(shares_m)) if shares_m.sum() > 0 else 0
                conf_m = float(shares_m[b_m]) if shares_m.size else np.nan
            else:
                mean_m = med_m = np.nan
                b_m, conf_m = 0, np.nan

            if has_s:
                mean_s, shares_s, med_s = time_weighted_stats_in_slice(g, "_tmin", "current_slave", t0, t1, edges)
                b_s = int(np.argmax(shares_s)) if shares_s.sum() > 0 else 0
                conf_s = float(shares_s[b_s]) if shares_s.size else np.nan
            else:
                mean_s = med_s = np.nan
                b_s, conf_s = 0, np.nan

            rows.append({
                "session_id": sid,
                "soc_start": s0,
                "soc_end": s1,
                "soc_band": f"{s0}-{s1}",
                "minutes_total": float(dt),
                # Authoritative modal bucket (effective current)
                "mode_bucket_idx": b_eff,
                "mode_bucket_label": labels[b_eff],
                "mode_minutes": float(conf_eff * dt) if np.isfinite(conf_eff) else np.nan,
                "confidence": conf_eff,
                "mean_abs_current": mean_eff,
                "median_abs_current": med_eff,
                # Diagnostics (optional)
                "m_bucket_idx": b_m,
                "m_bucket_label": labels[b_m],
                "m_conf": conf_m,
                "m_mean_abs_current": mean_m,
                "m_median_abs_current": med_m,
                "s_bucket_idx": b_s,
                "s_bucket_label": labels[b_s],
                "s_conf": conf_s,
                "s_mean_abs_current": mean_s,
                "s_median_abs_current": med_s,
            })

    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise SystemExit("No SOC-interval rows produced. Check inputs and band size.")

    # Pivots (minutes by modal bucket per band) — effective/pack is authoritative
    pivot_pack = intervals.pivot_table(index="soc_band", columns="mode_bucket_label", values="minutes_total", aggfunc="mean").sort_index()
    pivot_master = None
    pivot_slave = None
    if "m_bucket_label" in intervals.columns and intervals["m_bucket_label"].notna().any():
        pivot_master = intervals.pivot_table(index="soc_band", columns="m_bucket_label", values="minutes_total", aggfunc="mean").sort_index()
    if "s_bucket_label" in intervals.columns and intervals["s_bucket_label"].notna().any():
        pivot_slave = intervals.pivot_table(index="soc_band", columns="s_bucket_label", values="minutes_total", aggfunc="mean").sort_index()

    return intervals, pivot_pack, pivot_master, pivot_slave, labels


# --------------------- Estimator (modal pivots) ----------------
def build_estimator(pivot_pack: pd.DataFrame, band: int):
    def parse_mid(label: str) -> float:
        if label.endswith("+ A"):
            base = float(label.split("+")[0])
            return base + 1.0
        lo, hi = label.replace(" A", "").split("–")
        return (float(lo) + float(hi)) / 2.0

    band_means = pivot_pack.mean(axis=1, skipna=True)
    global_min_per_pct = (band_means.mean() / band) if band_means.notna().any() else 1.0

    def pick_bucket_label(amps_abs: float) -> Optional[str]:
        if pivot_pack.empty or not np.isfinite(amps_abs):
            return None
        mids = {c: parse_mid(str(c)) for c in pivot_pack.columns}
        return min(mids, key=lambda c: abs(mids[c] - amps_abs))

    def estimate_remaining_time(cur_soc: float, tgt_soc: float, cur_pack_a: Optional[float] = None) -> float:
        if tgt_soc <= cur_soc:
            return 0.0
        starts = list(range(int(np.floor(cur_soc / band) * band), int(np.floor(tgt_soc / band) * band), band))
        rem = 0.0
        for s0 in starts:
            s1 = s0 + band
            key = f"{s0}-{s1}"
            m = np.nan
            if cur_pack_a is not None and not np.isnan(cur_pack_a) and not pivot_pack.empty:
                col = pick_bucket_label(abs(cur_pack_a))
                if col in pivot_pack.columns and key in pivot_pack.index:
                    m = pivot_pack.loc[key, col]
            if np.isnan(m) or not np.isfinite(m):
                m = band_means.get(key, np.nan)
            if np.isnan(m) or not np.isfinite(m):
                m = global_min_per_pct * band
            rem += float(m)
        return rem

    return estimate_remaining_time


# ----------------------------- Main ---------------------------
def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    edges = np.array([float(x) for x in a.current_bins.split(",") if str(x).strip() != ""], float)
    edges = np.unique(np.sort(np.r_[edges]))
    if edges[0] > 0:
        edges = np.r_[0.0, edges]  # include 0
    if len(edges) < 2:
        raise SystemExit("Need at least two current bin edges")

    rct = read_any(a.src)

    intervals, pivot_pack, pivot_master, pivot_slave, labels = build_interval_tables(
        rct=rct, band=int(a.band_size), edges=edges, min_band_mins=float(a.min_band_mins)
    )

    # Save artifacts (authoritative pack + optional diagnostics)
    intervals_out = os.path.join(a.outdir, "interval_samples.csv")
    intervals.to_csv(intervals_out, index=False)

    pack_out = os.path.join(a.outdir, "pivot_pack.csv")
    pivot_pack.to_csv(pack_out)

    if pivot_master is not None:
        pivot_master.to_csv(os.path.join(a.outdir, "pivot_master.csv"))
    if pivot_slave is not None:
        pivot_slave.to_csv(os.path.join(a.outdir, "pivot_slave.csv"))

    write_text(",".join(labels), os.path.join(a.outdir, "bucket_labels.txt"))

    # Quick QA summary
    decisive = intervals["confidence"].fillna(0).mean()
    print(f"Average modal-bucket confidence (time share): {decisive:.2f}")

    # Example band print (50–60%)
    band_label = "50-60"
    if band_label in pivot_pack.index:
        print("\n[Pack] 50–60% minutes by current bucket (modal):")
        print(pivot_pack.loc[band_label].dropna().round(2).to_string())

    # Demo estimator
    estimator = build_estimator(pivot_pack, int(a.band_size))
    demo = estimator(20, 80, cur_pack_a=12.0)
    print(f"\nEstimated minutes 20→80 using modal pivots: {demo:.1f}")


if __name__ == "__main__":
    main()
