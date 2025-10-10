# RCT (Remaining Charging Time) Model Project

This folder contains feature engineering, dataset finalization, and SageMaker
training scripts for predicting **remaining charging time (RCT)** for Bortana EV.
Components:
- `Feature_Engineering_C.py` – core 1-min feature builder
- `finalize_rct_dataset.py` – builds `train/val/test` splits
- `soc_interval_strategy.py` – interval analytics for interpretability
- `run_processing_job_C.py` – example Processing-Job launcher
- `rct_model_project/` – model training and inference code
- `output/` – (ignored) generated features/models
