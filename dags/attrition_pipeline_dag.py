from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "attrition-hybrid-mlops",
    "retries": 1,
}

with DAG(
    dag_id="attrition_pipeline_dag",
    default_args=default_args,
    description="Train+register then batch-transform for both models",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hybrid-mlops", "sagemaker"],
) as dag:

    train_register = BashOperator(
    task_id="train_register_both_models",
    bash_command="cd /opt/airflow/dags/repo/app/models && python pipeline_train_register.py",
)

    batch_transform = BashOperator(
        task_id="batch_transform_both_models",
        bash_command="cd /opt/airflow/dags/repo/app/models && python deploy_batch_transform.py",
    )

    train_register >> batch_transform