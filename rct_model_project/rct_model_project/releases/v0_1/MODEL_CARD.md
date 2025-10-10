# RCT Model v0.1
- Target: rct_minutes (per-minute charging rows)
- Train/Val/Test: group-split by session_id
- Features: see feature_list.txt (numeric only)
- Algorithms: Linear baseline + XGBoost (current default: XGBoost)
- Notes: No efficiency features; SOC-interval analytics are interpretability-only.
