import pandas as pd
import numpy as np

np.random.seed(123)
N = 100

departments = ["Engineering", "Sales", "HR", "Finance", "Operations", "Marketing"]
designations = ["Associate", "Senior Associate", "Team Lead", "Manager"]
employment_types = ["Full-time", "Contract"]
locations = ["Onsite", "Remote", "Hybrid"]

# Attrition: shift several numeric columns far outside the training distribution
attrition_shifted = pd.DataFrame({
    "employee_id": range(9101, 9101 + N),
    "department": np.random.choice(departments, N),
    "designation": np.random.choice(designations, N),
    "employment_type": np.random.choice(employment_types, N, p=[0.85, 0.15]),
    "location": np.random.choice(locations, N),
    "age": np.random.randint(22, 58, N),
    "tenure_months": np.random.randint(1, 120, N),
    "months_since_last_hike": np.random.randint(60, 100, N),       # shifted way up
    "current_salary": np.random.randint(200000, 400000, N),        # shifted way up
    "last_hike_percentage": np.round(np.random.uniform(0, 2, N), 1),  # shifted way down
    "total_leaves_taken": np.random.randint(40, 80, N),             # shifted way up
    "leave_trend_flag": np.random.choice([0, 1], N, p=[0.3, 0.7]),
    "productivity_score": np.round(np.random.uniform(0, 30, N), 1),  # shifted way down
    "productivity_trend_flag": np.random.choice([0, 1], N, p=[0.3, 0.7]),
    "bench_days_last_6mo": np.random.randint(80, 150, N),           # shifted way up
    "manager_change_count": np.random.poisson(3, N),                # shifted up
    "number_of_client_switches": np.random.poisson(3, N),           # shifted up
})
attrition_shifted.to_csv("app/data/score_attrition_shifted.csv", index=False)

promotion_shifted = pd.DataFrame({
    "employee_id": range(9101, 9101 + N),
    "department": np.random.choice(departments, N),
    "current_designation": np.random.choice(designations, N),
    "tenure_months": np.random.randint(1, 120, N),
    "months_since_last_promotion": np.random.randint(80, 120, N),   # shifted way up
    "performance_rating": np.random.randint(1, 3, N),                # shifted way down
    "avg_performance_rating_last_2yrs": np.round(np.random.uniform(1, 2, N), 2),  # shifted down
    "project_completion_rate": np.round(np.random.uniform(10, 40, N), 1),  # shifted down
    "training_hours_completed": np.random.randint(0, 5, N),          # shifted down
    "peer_review_score": np.round(np.random.uniform(1, 2, N), 2),    # shifted down
    "manager_rating": np.random.randint(1, 3, N),                     # shifted down
    "leadership_flag": np.random.choice([0, 1], N, p=[0.95, 0.05]),
    "certifications_count": np.random.poisson(0.1, N),
})
promotion_shifted.to_csv("app/data/score_promotion_shifted.csv", index=False)

print("Generated shifted scoring files")