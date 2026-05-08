import torch
import mlflow
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from load_imbd import load_imdb_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


params = {
    "model_name":"flan-t5-small",
    "learning_rate":0.001,
    "batch_size":32,
    "num_epochs":3,
    "dataset_name":"aclImdb",
    "task_name":"sequence_classification",
    "max_seq_length":128,
    "output_dir":"models/flan-t5-small"
    }

mlflow.set_tracking_uri("http://localhost:5001")

mlflow.set_experiment("flan-t5-small")

run = mlflow.start_run(run_name="flan-t5-small")

mlflow.log_params(params)



"""save_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

"""








base_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/aclImdb"

train_dataset = load_imdb_split(f"{base_path}/train")
test_dataset = load_imdb_split(f"{base_path}/test")

train_short = train_dataset.select(range(2000))
test_short = test_dataset.select(range(2000))

train_short.to_parquet("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/train.parquet")
test_short.to_parquet("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/text.parquet")

mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/text.parquet", artifact_path="datasets")
mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/train.parquet", artifact_path="datasets")

train_short.set_format("torch", columns=["text", "label"])
test_short.set_format("torch", columns=["text", "label"])

train_loder = DataLoader(train_short, batch_size=params["batch_size"], shuffle=False)
test_loader = DataLoader(test_short, batch_size=params["batch_size"], shuffle=False)

labels = train_short.features["label"]

print(labels)










model = AutoModelForSeq2SeqLM.from_pretrained(params["model_name"])


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)



optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

def evaluate_model(model, dataloader)
    model.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            inputs, label = batch[0].to(device), batch[1].to(device)

            outputs = model(inputs)

            predictions.extend(outputs.cpu().detach().numpy())
            true_labels.extend(label.cpu().detach().numpy())


    accuracy_score = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

    return accuracy_score, precision, recall, f1


for epoch in range(params["num_epochs"]):
    running_loss = 0.0

    for i, batch in enumerate(train_loder,0):
        inputs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i in i%params["log_steps"] == 0:
            avg_loss = running_loss / params["log_steps"]

            mlflow.log_metric("train_loss", avg_loss, step=epoch*len(train_loder)+i)

            running_loss = 0.0


    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)

    mlflow.log_metric({"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1)


