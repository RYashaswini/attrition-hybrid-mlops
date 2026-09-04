import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("app/data/attrition_raw.csv")

y = df["Attrition"]
X = df.drop(columns=["Attrition", "employee_id"])

categorical = ["department", "designation", "employment_type", "location"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

mlflow.set_experiment("attrition-risk")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        print(f"{k}: {v:.3f}")

    mlflow.sklearn.log_model(pipeline, "model")

joblib.dump(pipeline, "app/models/attrition_model.joblib")
print("Saved app/models/attrition_model.joblib")