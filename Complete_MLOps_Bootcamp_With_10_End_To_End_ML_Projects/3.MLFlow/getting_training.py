import mlflow
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature



mlflow.set_tracking_uri("sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlflow.db")

#mlflow ui --backend-store-uri sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/3.MLFlow/mlflow.db

X,y = load_iris(return_X_y=True)

print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"penalty": "l2", "solver": "lbfgs", "random_state": 42,"max_iter":1000}

lr = LogisticRegression(**params)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

mlflow.set_experiment("Logistic_Regression_Testing")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_param("lr", lr)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.set_tag("Training info", "basic LR model")
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info=mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="iris_model_v3")


