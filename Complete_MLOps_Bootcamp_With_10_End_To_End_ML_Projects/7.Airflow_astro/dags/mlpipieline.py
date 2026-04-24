from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def preprocess_data():
    print("Preprocessing data...")


def train_model():
    print("Training model...")


def evaluate_model():
    print("Evaluating model...")


with DAG(
    "ml_pipeline",
    start_date = datetime(2020, 1, 1),
    schedule="@weekly") as dag:

        preprocess = PythonOperator(task_id = "processing_data", python_callable = preprocess_data)
        train = PythonOperator(task_id = "training", python_callable = train_model)
        evaluate = PythonOperator(task_id = "evaluation", python_callable = evaluate_model)
        preprocess >> train >> evaluate


