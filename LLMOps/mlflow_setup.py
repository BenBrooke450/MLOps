import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from load_imbd import load_imdb_split

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



"""save_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

"""








base_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/aclImdb"

train_dataset = load_imdb_split(f"{base_path}/train")
test_dataset = load_imdb_split(f"{base_path}/test")




train_short = train_dataset[:]
print(test_dataset[:100])


