import mlflow
import numpy as np


mlflow.set_tracking_uri("http://localhost:5001")


mlflow.set_experiment("My experiment for LLM")


with mlflow.start_run():
    mlflow.log_param("test", 1)




run = mlflow.start_run(run_name="new_test")


mlflow.log_param("learning_rate", 0.5)
mlflow.log_param("batch_size", 23)



for epochs in range(10):
    mlflow.log_metric("accuracy", np.random.random(),step=epochs)
    mlflow.log_metric("loss", np.random.random(), step=epochs)

mlflow.end_run()



mlflow.set_experiment("My other experiment for LLM")

with mlflow.start_run(run_name="sin"):
    for t in range(100):
        metric_value = np.sin(t * np.pi/50)
        mlflow.log_metric("matric_value", metric_value, step = t)





with mlflow.start_run(run_name="cos"):
    for t in range(100):
        metric_value = np.cos(t * np.pi / 50)
        mlflow.log_metric("matric_value", metric_value, step=t)



