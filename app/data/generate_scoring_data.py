import pandas as pd
import numpy as np

np.random.seed(99)
N = 100

departments = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
designations = ["Associate", "Senior Associate", "Team Lead", "Manager"]
employment_types = ["Full-time", "Contract"]
locations = ["Onsite", "Remote", "Hybrid"]

attrition_score_df = pd.DataFrame({
    "employee_id": range(9001, 9001 + N),
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
attrition_score_df.to_csv("app/data/score_attrition.csv", index=False)

promotion_score_df = pd.DataFrame({
    "employee_id": range(9001, 9001 + N),
    "department": np.random.choice(departments, N),
    "current_designation": np.random.choice(designations, N),
    "tenure_months": np.random.randint(1, 120, N),
    "months_since_last_promotion": np.random.randint(0, 60, N),
    "performance_rating": np.random.randint(1, 6, N),
    "avg_performance_rating_last_2yrs": np.round(np.random.uniform(1, 5, N), 2),
    "project_completion_rate": np.round(np.random.uniform(50, 100, N), 1),
    "training_hours_completed": np.random.randint(0, 80, N),
    "peer_review_score": np.round(np.random.uniform(1, 5, N), 2),
    "manager_rating": np.random.randint(1, 6, N),
    "leadership_flag": np.random.choice([0, 1], N, p=[0.8, 0.2]),
    "certifications_count": np.random.poisson(1.2, N),
})
promotion_score_df.to_csv("app/data/score_promotion.csv", index=False)

print("Generated score_attrition.csv and score_promotion.csv")