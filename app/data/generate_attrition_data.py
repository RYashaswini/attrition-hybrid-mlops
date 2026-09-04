import pandas as pd
import numpy as np

np.random.seed(42)
N = 3000

departments = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
designations = ["Associate", "Senior Associate", "Team Lead", "Manager"]
employment_types = ["Full-time", "Contract"]
locations = ["Onsite", "Remote", "Hybrid"]

df = pd.DataFrame({
    "employee_id": range(1, N + 1),
    "department": np.random.choice(departments, N),
    "designation": np.random.choice(designations, N),
    "employment_type": np.random.choice(employment_types, N, p=[0.85, 0.15]),
    "location": np.random.choice(locations, N),
    "age": np.random.randint(22, 58, N),
    "tenure_months": np.random.randint(1, 120, N),
    "months_since_last_hike": np.random.randint(0, 36, N),
    "current_salary": np.random.randint(30000, 150000, N),
    "last_hike_percentage": np.round(np.random.uniform(0, 25, N), 1),
    "total_leaves_taken": np.random.randint(0, 30, N),
    "leave_trend_flag": np.random.choice([0, 1], N, p=[0.7, 0.3]),
    "productivity_score": np.round(np.random.uniform(40, 100, N), 1),
    "productivity_trend_flag": np.random.choice([0, 1], N, p=[0.65, 0.35]),
    "bench_days_last_6mo": np.random.randint(0, 60, N),
    "manager_change_count": np.random.poisson(0.4, N),
    "number_of_client_switches": np.random.poisson(0.6, N),
})

# Build attrition risk as a weighted logistic-style signal so the model has real patterns to learn
risk_score = (
    0.035 * df["months_since_last_hike"]
    + 0.03 * df["bench_days_last_6mo"]
    + 0.8 * df["leave_trend_flag"]
    + 0.9 * df["productivity_trend_flag"]
    - 0.02 * df["productivity_score"]
    - 0.015 * df["last_hike_percentage"]
    + 0.5 * df["manager_change_count"]
    + 0.4 * df["number_of_client_switches"]
    - 0.01 * df["tenure_months"]
    + np.random.normal(0, 1.5, N)  # noise
)

prob = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / risk_score.std()))
df["Attrition"] = (prob > np.random.uniform(0.5, 0.75, N)).astype(int)

df.to_csv("app/data/attrition_raw.csv", index=False)
print(df.shape)
print(df["Attrition"].value_counts(normalize=True))