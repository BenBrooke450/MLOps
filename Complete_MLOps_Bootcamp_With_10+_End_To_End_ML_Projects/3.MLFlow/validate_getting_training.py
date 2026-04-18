import mlflow
import pandas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature, validate_serving_input

mlflow.set_tracking_uri("sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlflow.db")

run_id = "4ecd70829aee469bb3d5dfe607b38d3e"
model_uri = f"runs:/{run_id}/iris_model"

data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])

loaded_model = mlflow.pyfunc.load_model(model_uri)

prediction = loaded_model.predict(data)

print(prediction)





