import pandas as pd
import os
import sys
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

if __name__ == "__main__":
    pass

    try:
        data = pd.read_csv(url)
    except:
        logger.exception("Error, unable to download the data")

    train , test = train_test_split(data)

    X_train = train.drop('quality',axis=1)
    X_test = test.drop('quality',axis=1)
    y_train = train[['quality']]
    y_test = test[['quality']]

    alpha  = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv)>2 else 0.5

    with mlflow.start_run():
        lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(X_train,y_train)

        predict_qualities = lr.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test,predict_qualities)

        mlflow.log_metric("alpha", alpha)
        mlflow.log_metric("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("alpha", alpha)
        mlflow.log_metric("mae", mae)
















