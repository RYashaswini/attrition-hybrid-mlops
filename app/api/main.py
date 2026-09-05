import os
import io
import tarfile
import tempfile
import joblib
import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi import Depends, Header

API_KEY = os.environ.get("API_KEY", "")

def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI(title="Attrition Hybrid MLOps Serving API")

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

MODEL_PACKAGE_GROUPS = {
    "attrition": "attrition-models",
    "promotion": "promotion-models",
}

_loaded_models: Dict[str, Any] = {}


def _get_latest_approved_model_data_url(package_group: str) -> str:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(f"No Approved model package found in group {package_group}")

    arn = packages[0]["ModelPackageArn"]
    detail = sm_client.describe_model_package(ModelPackageName=arn)
    model_data_url = detail["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    return model_data_url


def _download_and_load_model(model_data_url: str):
    assert model_data_url.startswith("s3://")
    _, _, rest = model_data_url.partition("s3://")
    bucket, _, key = rest.partition("/")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_tar_path = os.path.join(tmp_dir, "model.tar.gz")
        s3_client.download_file(bucket, key, local_tar_path)

        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)

        # Adjust this filename if your training scripts save the model under a different name
        model_path = os.path.join(tmp_dir, "model.joblib")
        if not os.path.exists(model_path):
            candidates = [f for f in os.listdir(tmp_dir) if f.endswith((".joblib", ".pkl"))]
            if not candidates:
                raise RuntimeError(f"No model file found in extracted artifact: {os.listdir(tmp_dir)}")
            model_path = os.path.join(tmp_dir, candidates[0])

        return joblib.load(model_path)


def load_all_models():
    for name, group in MODEL_PACKAGE_GROUPS.items():
        model_data_url = _get_latest_approved_model_data_url(group)
        _loaded_models[name] = _download_and_load_model(model_data_url)


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_loaded_models.keys())}


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/predict/attrition")
def predict_attrition(req: PredictRequest, _: None = Depends(verify_api_key)):
    if "attrition" not in _loaded_models:
        raise HTTPException(status_code=503, detail="Attrition model not loaded")
    df = pd.DataFrame([req.features])
    model = _loaded_models["attrition"]
    pred = model.predict_proba(df)[:, 1][0]
    return {"model": "attrition", "risk_score": float(pred)}


@app.post("/predict/promotion")
def predict_promotion(req: PredictRequest, _: None = Depends(verify_api_key)):
    if "promotion" not in _loaded_models:
        raise HTTPException(status_code=503, detail="Promotion model not loaded")
    df = pd.DataFrame([req.features])
    model = _loaded_models["promotion"]
    pred = model.predict_proba(df)[:, 1][0]
    return {"model": "promotion", "readiness_score": float(pred)}


@app.post("/reload-models")
def reload_models():
    load_all_models()
    return {"status": "reloaded", "models_loaded": list(_loaded_models.keys())}# ci/cd test
