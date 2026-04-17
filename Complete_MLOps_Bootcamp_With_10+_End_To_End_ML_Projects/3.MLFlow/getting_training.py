import mlflow
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

X,y = load_iris(return_X_y=True)

print(X,y)

X_train, y_train, X_test, x_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "multi_class": 'auto'}

lr = LogisticRegression(**params)

lr.fit(X_train, y_train)

