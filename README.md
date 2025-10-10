# RCT (Remaining Charging Time) Model Project

Feature engineering, dataset finalization, and SageMaker scripts to predict EV charging **remaining time**.

## Key files
- `Feature_Engineering_C.py` – 1-min curated features
- `finalize_rct_dataset.py` – builds train/val/test
- `soc_interval_strategy.py` – SOC-interval analytics
- `run_processing_job_C.py` – SageMaker Processing launcher
- `rct_model_project/` – training & inference package

## Local quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python Feature_Engineering_C.py --help
