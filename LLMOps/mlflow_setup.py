import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import pandas as pd

params = {
    "model_name":"distilbert-base-uncased",
    "learning_rate":0.001,
    "batch_size":32,
    "num_epochs":3,
    "dataset_name":"ag_news",
    "task_name":"sequence_classification",
    "max_seq_length":128,
    "output_dir":"models/distilbert_ag-news"
    }

mlflow.set_tracking_uri("http://localhost:5001")

mlflow.set_experiment("distilbert-base-uncased")

run = mlflow.start_run(run_name="distilbert-base-uncased")

mlflow.log_params(params)




dataset = load_dataset("imdb", data_dir="/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps")

with open(dataset, "w") as df:
    pd.read_csv("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/df")







save_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def tokenise(batch):
    return tokenizer(batch["text"],padding="max_length",
                     truncate=True,
                     max_length=params["max_seq_length"])

train_dataset = dataset["train"].shuffle()\
                 .select(range(200))\
                 .map(tokenise,batched=True)

test_dataset = dataset["test"].shuffle()\
                 .select(range(200))\
                 .map(tokenise,batched=True)


print(dataset)

