# pipeline/rct_pipeline.py
import os, boto3
from sagemaker import image_uris, Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.steps import TrainingStep, CreateModelStep

# Endpoint step imports (SDK-flex)
HAVE_ENDPOINT_STEPS = False
EndpointConfigStep = EndpointStep = None
try:
    from sagemaker.workflow.steps import CreateEndpointConfigStep, CreateEndpointStep  # type: ignore
    EndpointConfigStep, EndpointStep = CreateEndpointConfigStep, CreateEndpointStep
    HAVE_ENDPOINT_STEPS = True
except Exception:
    try:
        from sagemaker.workflow.step_collections import EndpointConfigStep as _EPC, EndpointStep as _EPS  # type: ignore
        EndpointConfigStep, EndpointStep = _EPC, _EPS
        HAVE_ENDPOINT_STEPS = True
    except Exception:
        pass

# ==== EDIT THESE ====
REGION = "ap-southeast-2"
ROLE_ARN = "arn:aws:iam::383868855578:role/BortanaSageMakerExecutionRole"  # SageMaker execution role
BUCKET = "bortana-sagemaker-analysis"
TRAIN_S3 = "s3://bortana-sagemaker-analysis/bev2/processing_c/features/rct/train.csv"
PIPELINE_NAME = "RCT-Retrain-Pipeline"
ARTIFACT_PREFIX = "rct_model/artifacts"
ENDPOINT_NAME = "rct-realtime-endpoint"
# =====================

boto_sess = boto3.Session(region_name=REGION)
sm_sess = Session(boto_session=boto_sess)
pipe_sess = PipelineSession(boto_session=boto_sess, sagemaker_client=sm_sess.sagemaker_client)

p_train = ParameterString(name="TrainDataS3Uri", default_value=TRAIN_S3)
p_ep = ParameterString(name="EndpointName", default_value=ENDPOINT_NAME)

xgb_img = image_uris.retrieve(framework="xgboost", region=REGION, version="1.7-1")

est = Estimator(
    image_uri=xgb_img,
    role=ROLE_ARN,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{BUCKET}/{ARTIFACT_PREFIX}/",
    sagemaker_session=pipe_sess,
    disable_profiler=True,
    max_run=3600,
)
est.set_hyperparameters(objective="reg:squarederror", num_round=100)

train_input = TrainingInput(s3_data=p_train, content_type="text/csv")
train_step = TrainingStep(name="TrainXGBModel", estimator=est, inputs={"train": train_input})

model = Model(image_uri=xgb_img,
              model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
              role=ROLE_ARN,
              sagemaker_session=pipe_sess)
create_model = CreateModelStep(name="CreateModel", model=model)

steps = [train_step, create_model]
if HAVE_ENDPOINT_STEPS and EndpointConfigStep and EndpointStep:
    ep_cfg = EndpointConfigStep(name="CreateEndpointConfig",
                                model_name=create_model.properties.ModelName,
                                initial_instance_count=1,
                                instance_type="ml.m5.large")
    ep_step = EndpointStep(name="DeployOrUpdateEndpoint",
                           endpoint_name=p_ep,
                           endpoint_config_name=ep_cfg.properties.EndpointConfigName)
    steps += [ep_cfg, ep_step]
else:
    print(" Endpoint classes not present in this SDK; pipeline will stop at CreateModel.")

pipeline = Pipeline(name=PIPELINE_NAME,
                    parameters=[p_train, p_ep],
                    steps=steps,
                    sagemaker_session=pipe_sess)

def get_pipeline(session=None):
    return pipeline

def main():
    print(f" Upserting pipeline: {PIPELINE_NAME}")
    pipeline.upsert(role_arn=ROLE_ARN)
    print(" Upserted. Starting execution…")
    exe = pipeline.start(parameters={"TrainDataS3Uri": TRAIN_S3, "EndpointName": ENDPOINT_NAME})
    print(f" Execution ARN: {exe.arn}")

if __name__ == "__main__":
    main()
