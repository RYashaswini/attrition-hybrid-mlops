import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    import io
    if request_content_type == "text/csv":
        return pd.read_csv(io.StringIO(request_body), header=None)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict_proba(input_data)[:, 1]

def output_fn(prediction, content_type):
    return "\n".join([str(p) for p in prediction])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "promotion_raw.csv"))
    y = df["PromotionReady"]
    X = df.drop(columns=["PromotionReady", "employee_id"])

    categorical = ["department", "current_designation"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f"roc_auc: {auc:.3f}")

    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))