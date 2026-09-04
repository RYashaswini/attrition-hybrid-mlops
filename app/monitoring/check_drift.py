import json
import boto3
import numpy as np
import pandas as pd
from compute_baseline import NUMERIC_COLUMNS

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
s3 = boto3.client("s3", region_name="ap-south-1")
cloudwatch = boto3.client("cloudwatch", region_name="ap-south-1")

def psi(expected_pct, actual_pct):
    expected_pct = np.clip(np.array(expected_pct, dtype=float), 1e-4, None)
    actual_pct = np.clip(np.array(actual_pct, dtype=float), 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def check_drift(name, score_file):
    obj = s3.get_object(Bucket=bucket, Key=f"monitoring/{name}/baseline.json")
    baseline = json.loads(obj["Body"].read())

    df = pd.read_csv(f"app/data/{score_file}")
    scores = {}
    for col in NUMERIC_COLUMNS[name]:
        edges = baseline[col]["edges"]
        expected_dist = baseline[col]["distribution"]
        actual_dist = pd.cut(df[col], bins=edges).value_counts(normalize=True).sort_index().tolist()
        scores[col] = psi(expected_dist, actual_dist)

    max_drift = max(scores.values())
    print(f"{name} drift scores: {scores}")
    print(f"{name} max drift: {max_drift:.3f}")

    cloudwatch.put_metric_data(
        Namespace=f"{name.capitalize()}MLOps",
        MetricData=[{"MetricName": "MaxDriftScore", "Value": max_drift, "Unit": "None"}],
    )
    return max_drift

# if __name__ == "__main__":
#     check_drift("attrition", "score_attrition.csv")
#     check_drift("promotion", "score_promotion.csv")

if __name__ == "__main__":
    check_drift("attrition", "score_attrition_shifted.csv")
    check_drift("promotion", "score_promotion_shifted.csv")