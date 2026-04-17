import mlflow
import os

#mlflow ui
#cd /Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow
#mlflow ui --backend-store-uri file:./mlruns

"""mlflow.set_tracking_uri(
    "file:/Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlruns"
)"""


mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("clean_test")

with mlflow.start_run():
    mlflow.log_metric("test", 1)
    mlflow.log_metric("ben", 2)


with mlflow.start_run():
    mlflow.log_metric("test1", 1)
    mlflow.log_metric("ben1", 2)
