# pipeline/rct_pipeline.py
# SageMaker RCT Retraining Pipeline – CI/CD ready version

import boto3
from sagemaker import image_uris, Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.steps import TrainingStep, CreateModelStep

# ---------- Optional imports for endpoint deployment ----------
HAVE_ENDPOINT_STEPS = False
EndpointConfigStep = None
EndpointStep = None
try:
    from sagemaker.workflow.steps import CreateEndpointConfigStep, CreateEndpointStep  # type: ignore
    EndpointConfigStep = CreateEndpointConfigStep
    EndpointStep = CreateEndpointStep
    HAVE_ENDPOINT_STEPS = True
except Exception:
    try:
        from sagemaker.workflow.step_collections import EndpointConfigStep as _EPCfg, EndpointStep as _EPStep  # type: ignore
        EndpointConfigStep = _EPCfg
        EndpointStep = _EPStep
        HAVE_ENDPOINT_STEPS = True
    except Exception:
        HAVE_ENDPOINT_STEPS = False
# ----------------------------------------------------------------

# ===== USER CONFIG =====
REGION = "ap-southeast-2"
ROLE_ARN = "arn:aws:iam::383868855578:role/BortanaSageMakerExecutionRole"
BUCKET = "bortana-sagemaker-analysis"
PIPELINE_NAME = "RCT-Retrain-Pipeline"
TRAIN_S3 = "s3://bortana-sagemaker-analysis/bev2/processing_c/features/rct/train.csv"
ENDPOINT_NAME = "rct-realtime-endpoint"
# =======================

# Session setup
boto_sess = boto3.Session(region_name=REGION)
sm_sess = Session(boto_session=boto_sess)
pipeline_sess = PipelineSession(
    boto_session=boto_sess,
    sagemaker_client=sm_sess.sagemaker_client,
)

# Parameters (for override during CI/CD)
p_train_data = ParameterString(name="TrainDataS3Uri", default_value=TRAIN_S3)
p_endpoint_name = ParameterString(name="EndpointName", default_value=ENDPOINT_NAME)

# XGBoost built-in image
xgb_image = image_uris.retrieve(framework="xgboost", region=REGION, version="1.7-1")

# Estimator
estimator = Estimator(
    image_uri=xgb_image,
    role=ROLE_ARN,  # Explicit for CI/CD
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{BUCKET}/rct_model/artifacts/",
    sagemaker_session=pipeline_sess,
    disable_profiler=True,
    max_run=3600,
)
estimator.set_hyperparameters(objective="reg:squarederror", num_round=100)

# Channel input
train_input = TrainingInput(s3_data=p_train_data, content_type="text/csv")

# Pipeline steps
train_step = TrainingStep(name="TrainXGBModel", estimator=estimator, inputs={"train": train_input})

model = Model(
    image_uri=xgb_image,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=ROLE_ARN,
    sagemaker_session=pipeline_sess,
)
create_model_step = CreateModelStep(name="CreateModel", model=model)

steps = [train_step, create_model_step]
if HAVE_ENDPOINT_STEPS and EndpointConfigStep and EndpointStep:
    ep_config_step = EndpointConfigStep(
        name="CreateEndpointConfig",
        model_name=create_model_step.properties.ModelName,
        initial_instance_count=1,
        instance_type="ml.m5.large",
    )
    ep_step = EndpointStep(
        name="DeployOrUpdateEndpoint",
        endpoint_name=p_endpoint_name,
        endpoint_config_name=ep_config_step.properties.EndpointConfigName,
    )
    steps.extend([ep_config_step, ep_step])

# Assemble pipeline
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[p_train_data, p_endpoint_name],
    steps=steps,
    sagemaker_session=pipeline_sess,
)

def get_pipeline(session=None):
    """Return the SageMaker Pipeline object."""
    return pipeline


# ------------- MAIN (CI/CD entry point) -------------
def main():
    print(f" Upserting pipeline: {PIPELINE_NAME}")
    pipeline.upsert(role_arn=ROLE_ARN)
    print(" Upsert complete. Starting execution...")
    exe = pipeline.start(parameters={"TrainDataS3Uri": TRAIN_S3, "EndpointName": ENDPOINT_NAME})
    print(f" Execution started: {exe.arn}")


if __name__ == "__main__":
    main()
