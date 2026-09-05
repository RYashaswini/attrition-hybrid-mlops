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
    description="Train+register then batch-transform for attrition and promotion models",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hybrid-mlops", "sagemaker"],
) as dag:

    train_register_attrition = BashOperator(
        task_id="train_register_attrition",
        bash_command="cd /opt/airflow/dags/repo && python pipeline_train_register.py --model attrition",
    )

    transform_attrition = BashOperator(
        task_id="transform_attrition",
        bash_command="cd /opt/airflow/dags/repo && python deploy_batch_transform.py --model attrition",
    )

    train_register_promotion = BashOperator(
        task_id="train_register_promotion",
        bash_command="cd /opt/airflow/dags/repo && python pipeline_train_register.py --model promotion",
    )

    transform_promotion = BashOperator(
        task_id="transform_promotion",
        bash_command="cd /opt/airflow/dags/repo && python deploy_batch_transform.py --model promotion",
    )

    train_register_attrition >> transform_attrition
    train_register_promotion >> transform_promotion