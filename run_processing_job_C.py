# filepath: run_processing_job_C.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# Session & role
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Image
script_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=sess.boto_region_name,
        version="1.2-1",
    ),
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    max_runtime_in_seconds=3600,
    sagemaker_session=sess,
)

# S3 paths
IN_BUCKET  = "bortana-sagemaker-analysis"
IN_PREFIX  = "bev2/raw-data"              # root; FE script reads from S3
OUT_BUCKET = "bortana-sagemaker-analysis"
OUT_PREFIX = "bev2/processing_c"          # destination prefix

# Inputs (optional local mount for reference)
processing_inputs = [
    ProcessingInput(
        source=f"s3://{IN_BUCKET}/{IN_PREFIX}/",
        destination="/opt/ml/processing/input/raw-data",
    )
]

# Outputs: curated rows + RCT features + SOC-interval analytics (no efficiency)
processing_outputs = [
    ProcessingOutput(
        source="/opt/ml/processing/curated/rows",
        destination=f"s3://{OUT_BUCKET}/{OUT_PREFIX}/curated/rows/",
    ),
    ProcessingOutput(
        source="/opt/ml/processing/features/rct",
        destination=f"s3://{OUT_BUCKET}/{OUT_PREFIX}/features/rct/",
    ),
    ProcessingOutput(
        source="/opt/ml/processing/analytics/soc_intervals",
        destination=f"s3://{OUT_BUCKET}/{OUT_PREFIX}/analytics/soc_intervals/",
    ),
]

# Run Feature Engineering
script_processor.run(
    code="Feature_Engineering_C.py",
    inputs=processing_inputs,
    outputs=processing_outputs,
    arguments=[
        "--in-bucket", IN_BUCKET,
        "--in-prefix", IN_PREFIX,
        "--out-bucket", OUT_BUCKET,
        "--out-prefix", OUT_PREFIX,


        # Modal SOC-interval analytics
        "--soc-step-pct", "5",
        "--current-buckets", "0,1,5,10,20,40,80,200,1000",

        # FE knobs (unchanged)
        "--battery-capacity-kwh", "53.280",
        "--session-gap-min", "10",
        "--idle-speed-thresh", "0.5",
        "--min-charge-a", "6",
        "--max-charge-a", "350",
        "--min-session-min", "4",
        "--min-delta-soc-pct", "0.2",
        "--min-e-in-kwh", "0.1",
        "--min-idle-frac", "0.8",
        "--min-median-charge-a", "5",
        "--max-dt-sec", "180",

    ],
    wait=True,
    logs=True,
)


print("\nSubmitted Processing job. Outputs:")
print(f"  • Curated rows              → s3://{OUT_BUCKET}/{OUT_PREFIX}/curated/rows/")
print(f"  • RCT features (train set)  → s3://{OUT_BUCKET}/{OUT_PREFIX}/features/rct/")
print(f"  • SOC-interval analytics    → s3://{OUT_BUCKET}/{OUT_PREFIX}/analytics/soc_intervals/")
