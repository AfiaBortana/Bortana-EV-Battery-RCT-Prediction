# rct_pipeline.py
# Build a robust SageMaker Pipeline for RCT model retraining.
# - Steps: Train (XGBoost) -> CreateModel -> (optionally) CreateEndpointConfig & DeployOrUpdateEndpoint
# - Endpoint steps are included when the installed SageMaker SDK exposes the classes;
#   otherwise the pipeline is created without them (avoids import-time failures).

import boto3
from sagemaker import image_uris, Session, get_execution_role
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.steps import TrainingStep, CreateModelStep

# ---------- Try to import endpoint-related steps (SDKs differ) ----------
HAVE_ENDPOINT_STEPS = False
EndpointConfigStep = None
EndpointStep = None

# Newer SDKs sometimes expose CreateEndpoint* in steps; others use step_collections
try:
    # Path 1 (some SDKs): steps has CreateEndpoint* classes
    from sagemaker.workflow.steps import CreateEndpointConfigStep, CreateEndpointStep  # type: ignore
    EndpointConfigStep = CreateEndpointConfigStep
    EndpointStep = CreateEndpointStep
    HAVE_ENDPOINT_STEPS = True
except Exception:
    try:
        # Path 2 (common on many recent SDKs)
        from sagemaker.workflow.step_collections import EndpointConfigStep as _EPCfg, EndpointStep as _EPStep  # type: ignore
        EndpointConfigStep = _EPCfg
        EndpointStep = _EPStep
        HAVE_ENDPOINT_STEPS = True
    except Exception:
        try:
            # Path 3 (older SDKs)
            from sagemaker.workflow._step_collections import EndpointConfigStep as _EPCfgOld, EndpointStep as _EPStepOld  # type: ignore
            EndpointConfigStep = _EPCfgOld
            EndpointStep = _EPStepOld
            HAVE_ENDPOINT_STEPS = True
        except Exception:
            # Final fallback: endpoint steps not available; pipeline will stop at CreateModel
            HAVE_ENDPOINT_STEPS = False
# -----------------------------------------------------------------------


# ===================== EDIT THESE (names/values only) =====================
REGION = "ap-southeast-2"
ROLE_ARN = "arn:aws:iam::383868855578:role/BortanaSageMakerExecutionRole"  
BUCKET = "bortana-sagemaker-analysis"
TRAIN_S3 = "s3://bortana-sagemaker-analysis/bev2/processing_c/features/rct/train.csv"
PREFIX = "rct_model/artifacts"
MODEL_NAME = "rct-xgb-model"
ENDPOINT_NAME = "rct-realtime-endpoint"
# ==========================================================================


# Resolve execution role (works on Notebook Instance / Studio)
SAGEMAKER_EXEC_ROLE = get_execution_role()

# Sessions
_boto = boto3.Session(region_name=REGION)
_sm_sess = Session(boto_session=_boto)
pipeline_sess = PipelineSession(
    boto_session=_boto,
    sagemaker_client=_sm_sess.sagemaker_client,
)

# Parameters (allow overrides at start time / CI)
p_train_data = ParameterString(name="TrainDataS3Uri", default_value=TRAIN_S3_PATH)
p_endpoint_name = ParameterString(name="EndpointName", default_value=ENDPOINT_NAME_DEFAULT)

# Training image (built-in XGBoost)
xgb_image = image_uris.retrieve(framework="xgboost", region=REGION, version="1.7-1")

# Estimator (reliable defaults)
estimator = Estimator(
    image_uri=xgb_image,
    role=SAGEMAKER_EXEC_ROLE,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{ARTIFACTS_BUCKET}/rct_model/artifacts/",
    sagemaker_session=pipeline_sess,
    disable_profiler=True,
    max_run=3600,
)
# XGBoost expects CSV with label in the first column and NO header
estimator.set_hyperparameters(objective="reg:squarederror", num_round=100)

# Channels
train_input = TrainingInput(s3_data=p_train_data, content_type="text/csv")

# Step 1 — Train
train_step = TrainingStep(
    name="TrainXGBModel",
    estimator=estimator,
    inputs={"train": train_input},
)

# Step 2 — Create Model
model = Model(
    image_uri=xgb_image,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=SAGEMAKER_EXEC_ROLE,
    sagemaker_session=pipeline_sess,
)
create_model_step = CreateModelStep(name="CreateModel", model=model)

# Optional Step 3/4 — EndpointConfig + Endpoint (only if SDK exposes classes)
steps = [train_step, create_model_step]
if HAVE_ENDPOINT_STEPS and EndpointConfigStep is not None and EndpointStep is not None:
    endpoint_config_step = EndpointConfigStep(
        name="CreateEndpointConfig",
        model_name=create_model_step.properties.ModelName,
        initial_instance_count=1,
        instance_type="ml.m5.large",
    )
    endpoint_step = EndpointStep(
        name="DeployOrUpdateEndpoint",
        endpoint_name=p_endpoint_name,
        endpoint_config_name=endpoint_config_step.properties.EndpointConfigName,
    )
    steps.extend([endpoint_config_step, endpoint_step])
else:
    # No exception: we simply inform at construction time. Execution will still succeed for the first two steps.
    print(
        " Endpoint deployment steps are not available in this SageMaker SDK build. "
        "The pipeline will run up to CreateModel. You can deploy the model with a small "
        "boto3 script afterwards if needed."
    )

# Assemble Pipeline
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[p_train_data, p_endpoint_name],
    steps=steps,
    sagemaker_session=pipeline_sess,
)


def get_pipeline(session=None):
    """Return the SageMaker Pipeline object for external callers (Studio/CI)."""
    return pipeline
