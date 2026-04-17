import os
import mlflow

#mlflow ui

mlflow.set_tracking_uri("sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlflow.db")

#mlflow ui --backend-store-uri sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlflow.db


mlflow.set_experiment("clean_test")

with mlflow.start_run():
    mlflow.log_metric("test", 1)
    mlflow.log_metric("ben", 2)


with mlflow.start_run():
    mlflow.log_metric("test1", 1)
    mlflow.log_metric("ben1", 2)
