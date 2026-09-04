import json
import boto3
import numpy as np
import pandas as pd

account_id = boto3.client("sts").get_caller_identity()["Account"]
bucket = f"attrition-hybrid-mlops-{account_id}"
s3 = boto3.client("s3", region_name="ap-south-1")

NUMERIC_COLUMNS = {
    "attrition": [
        "age", "tenure_months", "months_since_last_hike", "current_salary",
        "last_hike_percentage", "total_leaves_taken", "productivity_score",
        "bench_days_last_6mo", "manager_change_count", "number_of_client_switches",
    ],
    "promotion": [
        "tenure_months", "months_since_last_promotion", "performance_rating",
        "avg_performance_rating_last_2yrs", "project_completion_rate",
        "training_hours_completed", "peer_review_score", "manager_rating",
        "certifications_count",
    ],
}

def compute_baseline(name, raw_file):
    df = pd.read_csv(f"app/data/{raw_file}")
    baseline = {}
    for col in NUMERIC_COLUMNS[name]:
        binned, edges = pd.qcut(df[col], q=10, duplicates="drop", retbins=True)
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        dist = pd.cut(df[col], bins=edges).value_counts(normalize=True).sort_index().tolist()
        baseline[col] = {
            "edges": edges.tolist(),
            "distribution": dist,
        }
    key = f"monitoring/{name}/baseline.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(baseline))
    print(f"Baseline saved: s3://{bucket}/{key}")

if __name__ == "__main__":
    compute_baseline("attrition", "attrition_raw.csv")
    compute_baseline("promotion", "promotion_raw.csv")