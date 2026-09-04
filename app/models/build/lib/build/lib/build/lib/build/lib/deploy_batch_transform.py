import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.local import LocalSession

# --- SWITCH POINT ---
INSTANCE_TYPE = "local"
boto_session = boto3.Session(region_name="ap-south-1")
sm_session = LocalSession(boto_session=boto_session)
sm_session.config = {"local": {"local_code": True}}
# Real AWS: sm_session = sagemaker.Session(boto_session=boto_session); INSTANCE_TYPE = "ml.m5.large"
# ---------------------

sm_client = boto3.client("sagemaker", region_name="ap-south-1")
account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
role = f"arn:aws:iam::{account_id}:role/attrition-mlops-sagemaker-role"


def get_latest_approved_model_data(model_package_group):
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not response["ModelPackageSummaryList"]:
        raise ValueError(f"No approved model found in {model_package_group}")
    package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
    details = sm_client.describe_model_package(ModelPackageName=package_arn)
    return details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]


def deploy_and_transform(name, train_entry, score_file, model_package_group):
    model_data = get_latest_approved_model_data(model_package_group)
    print(f"Latest approved {name} model: {model_data}")

    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point=train_entry,
        source_dir="app/models",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
    )

    sm_session.upload_data(f"app/data/{score_file}", bucket=bucket, key_prefix=f"scoring/{name}")

    transformer = model.transformer(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        output_path=f"s3://{bucket}/predictions/{name}/",
        accept="text/csv",
    )
    transformer.transform(
        data=f"s3://{bucket}/scoring/{name}/{score_file}",
        content_type="text/csv",
        split_type="Line",
    )
    transformer.wait()
    print(f"Deployed + Batch Transform complete for {name}")


if __name__ == "__main__":
    deploy_and_transform("attrition", "sm_train_attrition.py", "score_attrition.csv", "attrition-models")
    deploy_and_transform("promotion", "sm_train_promotion.py", "score_promotion.csv", "promotion-models")