import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.local import LocalSession

# --- SWITCH POINT for real AWS compute later ---
INSTANCE_TYPE = "local"
boto_session = boto3.Session(region_name="ap-south-1")
sm_session = LocalSession(boto_session=boto_session)
sm_session.config = {"local": {"local_code": True}}
# Real AWS: sm_session = sagemaker.Session(boto_session=boto_session); INSTANCE_TYPE = "ml.m5.large"
# (and at that point, wrap this in a real Pipeline object with TrainingStep + ModelStep(register))
# ------------------------------------------------

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
role = f"arn:aws:iam::{account_id}:role/attrition-mlops-sagemaker-role"


def train_and_register(name, train_entry, data_file, model_package_group):
    estimator = SKLearn(
        entry_point=train_entry,
        source_dir="app/models",
        role=role,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
        base_job_name=name,
    )
    sm_session.upload_data(f"app/data/{data_file}", bucket=bucket, key_prefix=f"data/{name}")
    estimator.fit({"train": f"s3://{bucket}/data/{name}"})

    # Upload local artifact to S3 (local mode doesn't auto-upload)
    s3_client = boto3.client("s3", region_name="ap-south-1")
    local_path = estimator.model_data.replace("file://", "")
    s3_key = f"models/{name}/model.tar.gz"
    s3_client.upload_file(local_path, bucket, s3_key)
    model_s3_uri = f"s3://{bucket}/{s3_key}"

    # Register — real AWS API call, works regardless of local mode
    real_session_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        entry_point=train_entry,
        source_dir="app/models",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=boto3.Session(region_name="ap-south-1"),  # not local, real registry call
    )
    import sagemaker
    real_session_model.sagemaker_session = sagemaker.Session(boto_session=boto_session)
    real_session_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="Approved",
    )
    print(f"{name}: trained + registered, model at {model_s3_uri}")


if __name__ == "__main__":
    train_and_register("attrition", "sm_train_attrition.py", "attrition_raw.csv", "attrition-models")
    train_and_register("promotion", "sm_train_promotion.py", "promotion_raw.csv", "promotion-models")

    
# import boto3
# import sagemaker
# from sagemaker.sklearn.estimator import SKLearn
# from sagemaker.sklearn.model import SKLearnModel
# from sagemaker.workflow.pipeline_context import LocalPipelineSession
# from sagemaker.workflow.steps import TrainingStep
# from sagemaker.workflow.model_step import ModelStep
# from sagemaker.workflow.pipeline import Pipeline

# # --- SWITCH POINT for real AWS compute later ---
# INSTANCE_TYPE = "local"
# pipeline_session = LocalPipelineSession()
# # Real AWS: pipeline_session = PipelineSession(); INSTANCE_TYPE = "ml.m5.large"
# # ------------------------------------------------

# boto_session = boto3.Session(region_name="ap-south-1")
# account_id = boto3.client("sts").get_caller_identity()["Account"]
# bucket = f"attrition-hybrid-mlops-{account_id}"
# role = f"arn:aws:iam::{account_id}:role/attrition-mlops-sagemaker-role"


# def build_pipeline(name, train_entry, model_package_group):
#     estimator = SKLearn(
#         entry_point=train_entry,
#         source_dir="app/models",
#         role=role,
#         instance_type=INSTANCE_TYPE,
#         instance_count=1,
#         framework_version="1.2-1",
#         py_version="py3",
#         sagemaker_session=pipeline_session,
#         base_job_name=name,
#     )

#     train_step = TrainingStep(
#         name=f"Train-{name}",
#         estimator=estimator,
#         inputs={"train": f"s3://{bucket}/data/{name}"},
#     )

#     model = SKLearnModel(
#         model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
#         role=role,
#         entry_point=train_entry,
#         source_dir="app/models",
#         framework_version="1.2-1",
#         py_version="py3",
#         sagemaker_session=pipeline_session,
#     )

#     register_step = ModelStep(
#         name=f"Register-{name}",
#         step_args=model.register(
#             content_types=["text/csv"],
#             response_types=["text/csv"],
#             inference_instances=["ml.m5.large"],
#             transform_instances=["ml.m5.large"],
#             model_package_group_name=model_package_group,
#             approval_status="Approved",
#         ),
#     )

#     return Pipeline(
#         name=f"{name}-train-register-pipeline",
#         steps=[train_step, register_step],
#         sagemaker_session=pipeline_session,
#     )


# if __name__ == "__main__":
#     for name, entry in [("attrition", "sm_train_attrition.py"), ("promotion", "sm_train_promotion.py")]:
#         p = build_pipeline(name, entry, f"{name}-models")
#         p.upsert(role_arn=role)
#         execution = p.start()
#         print(f"{name} pipeline started:", execution)