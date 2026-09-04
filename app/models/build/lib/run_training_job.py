import json
import boto3
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn

boto_session = boto3.Session(region_name="ap-south-1")

# Use LocalSession when running local mode, regular Session otherwise
sm_session = LocalSession(boto_session=boto_session)
sm_session.config = {"local": {"local_code": True}}

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
role = f"arn:aws:iam::{account_id}:role/attrition-mlops-sagemaker-role"

def run_job(name, entry_point, data_file):
    sm_session.upload_data(f"app/data/{data_file}", bucket=bucket, key_prefix=f"data/{name}")
    estimator = SKLearn(
        entry_point=entry_point,
        source_dir="app/models",
        role=role,
        instance_type="local",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
        base_job_name=name,
    )
    estimator.fit({"train": f"s3://{bucket}/data/{name}"})
    return estimator

if __name__ == "__main__":
    attrition_estimator = run_job("attrition", "sm_train_attrition.py", "attrition_raw.csv")
    promotion_estimator = run_job("promotion", "sm_train_promotion.py", "promotion_raw.csv")

    s3_client = boto3.client("s3", region_name="ap-south-1")

    def upload_local_model(local_uri, name):
        local_path = local_uri.replace("file://", "")
        s3_key = f"models/{name}/model.tar.gz"
        s3_client.upload_file(local_path, bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"

    model_uris = {
        "attrition": upload_local_model(attrition_estimator.model_data, "attrition"),
        "promotion": upload_local_model(promotion_estimator.model_data, "promotion"),
    }
    with open("app/models/model_uris.json", "w") as f:
        json.dump(model_uris, f, indent=2)
    print("Saved model_uris.json:", model_uris)