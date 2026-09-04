import pandas as pd
import numpy as np

np.random.seed(7)
N = 3000

departments = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
designations = ["Associate", "Senior Associate", "Team Lead", "Manager"]

df = pd.DataFrame({
    "employee_id": range(1, N + 1),
    "department": np.random.choice(departments, N),
    "current_designation": np.random.choice(designations, N),
    "tenure_months": np.random.randint(1, 120, N),
    "months_since_last_promotion": np.random.randint(0, 60, N),
    "performance_rating": np.random.randint(1, 6, N),  # 1-5 scale
    "avg_performance_rating_last_2yrs": np.round(np.random.uniform(1, 5, N), 2),
    "project_completion_rate": np.round(np.random.uniform(50, 100, N), 1),
    "training_hours_completed": np.random.randint(0, 80, N),
    "peer_review_score": np.round(np.random.uniform(1, 5, N), 2),
    "manager_rating": np.random.randint(1, 6, N),
    "leadership_flag": np.random.choice([0, 1], N, p=[0.8, 0.2]),
    "certifications_count": np.random.poisson(1.2, N),
})

signal = (
    0.6 * df["performance_rating"]
    + 0.5 * df["avg_performance_rating_last_2yrs"]
    + 0.02 * df["project_completion_rate"]
    + 0.03 * df["training_hours_completed"]
    + 0.4 * df["peer_review_score"]
    + 0.5 * df["manager_rating"]
    + 1.2 * df["leadership_flag"]
    + 0.3 * df["certifications_count"]
    - 0.02 * df["months_since_last_promotion"]
    + np.random.normal(0, 1.5, N)
)

prob = 1 / (1 + np.exp(-(signal - signal.mean()) / signal.std()))
df["PromotionReady"] = (prob > np.random.uniform(0.55, 0.8, N)).astype(int)

df.to_csv("app/data/promotion_raw.csv", index=False)
print(df.shape)
print(df["PromotionReady"].value_counts(normalize=True))