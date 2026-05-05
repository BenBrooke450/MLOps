import mlflow
import numpy as np
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



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



with mlflow.start_run(run_name="numbers"):
    x = list(range(100))
    y = list(range(1,200,2))

    df = pd.DataFrame({"x": x, "y": y})


    with open("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/data.csv", "w") as f:
        df.to_csv(f, index=False)

    mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/data.csv")






save_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)



with mlflow.start_run(run_name="model"):
    mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small")



