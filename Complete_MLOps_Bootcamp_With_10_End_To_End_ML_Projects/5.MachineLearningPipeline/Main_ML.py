#%%
import mlflow
import yaml
import pandas as pd
import torch
from torch import nn
import numpy as np
from hyperopt import STATUS_OK,Trials,fmin,hp,tpe
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature


mlflow.set_tracking_uri("sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/5.MachineLearningPipeline/mlflow.db")

#mlflow ui --backend-store-uri sqlite:////Users/benjaminbrooke/PycharmProjects/MLOps/Complete_MLOps_Bootcamp_With_10+_End_To_End_ML_Projects/5.MachineLearningPipeline/mlflow.db

def hyperparameter_truning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

params = yaml.safe_load(open("params.yaml"))["train"]


def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    with mlflow.start_run():

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random_state)

        signature = infer_signature(X_test,y_test)

        para_grid = {
            "n_estimators": [100,200],
            "max_depth": [5,10,None],
            "min_samples_split": [2,5],
            "min_samples_leaf": [1,2]}

        grid_search = hyperparameter_truning(X_train, y_train, para_grid)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)

        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("Accuracy",accuracy)

        mlflow.log_param("n_estimators",best_model.n_estimators)
        mlflow.log_param("max_depth",best_model.max_depth)
        mlflow.log_param("best_sample_split",best_model.min_samples_split)
        mlflow.log_param("best_samples_leaf",best_model.min_samples_leaf)

        cm = confusion_matrix(y_test,y_pred)

        cr = classification_report(y_test,y_pred)

        mlflow.log_text(cr,"classification_report")

        mlflow.sklearn.log_model(best_model,"model",signature=signature)

if __name__ == "__main__":
    train(params["data"],params["model_path"],params["random_state"],params["n_estimators"],params["max_depth"])


print("CHECK MY CODE")

print("CHECK MY CODE")

