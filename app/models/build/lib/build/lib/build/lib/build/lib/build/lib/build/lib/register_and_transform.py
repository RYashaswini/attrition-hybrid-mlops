import json
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.local import LocalSession

boto_session = boto3.Session(region_name="ap-south-1")

# Real session — for Model Registry API calls (metadata only, no compute)
real_session = sagemaker.Session(boto_session=boto_session)

# Local session — for Batch Transform compute (avoids quota)
local_session = LocalSession(boto_session=boto_session)
local_session.config = {"local": {"local_code": True}}

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
role = f"arn:aws:iam::{account_id}:role/attrition-mlops-sagemaker-role"

with open("app/models/model_uris.json") as f:
    model_uris = json.load(f)

def register_model(name, model_data, entry_point):
    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point=entry_point,
        source_dir="app/models",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=real_session,  # real session for registry
    )
    model_package_group = f"{name}-models"
    model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="Approved",
    )
    print(f"Registered {name} model into group '{model_package_group}'")

def run_batch_transform(name, model_data, entry_point, score_file):
    # separate model object using the LOCAL session, for local Batch Transform
    local_model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point=entry_point,
        source_dir="app/models",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=local_session,
    )

    local_session.upload_data(f"app/data/{score_file}", bucket=bucket, key_prefix=f"scoring/{name}")

    transformer = local_model.transformer(
        instance_count=1,
        instance_type="local",
        output_path=f"s3://{bucket}/predictions/{name}/",
        accept="text/csv",
    )
    transformer.transform(
        data=f"s3://{bucket}/scoring/{name}/{score_file}",
        content_type="text/csv",
        split_type="Line",
    )
    transformer.wait()
    print(f"Batch Transform complete for {name}. Output: s3://{bucket}/predictions/{name}/")

if __name__ == "__main__":
    register_model("attrition", model_uris["attrition"], "sm_train_attrition.py")
    run_batch_transform("attrition", model_uris["attrition"], "sm_train_attrition.py", "score_attrition.csv")

    register_model("promotion", model_uris["promotion"], "sm_train_promotion.py")
    run_batch_transform("promotion", model_uris["promotion"], "sm_train_promotion.py", "score_promotion.csv")