import torch
import mlflow
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from load_imbd import load_imdb_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

params = {
    "model_name":"flan-t5-small",
    "learning_rate":0.001,
    "batch_size":10,
    "num_epochs":1,
    "dataset_name":"aclImdb",
    "task_name":"sequence_classification",
    "max_seq_length":128,
    "output_dir":"models/flan-t5-small"
    }

mlflow.set_tracking_uri("http://localhost:5001")

mlflow.set_experiment("flan-t5-small_runpod")

run = mlflow.start_run(run_name="flan-t5-small")

mlflow.log_params(params)



"""save_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

"""




from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/flan-t5-small")



def tokenise(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )





base_path = "/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/aclImdb"

train_dataset = load_imdb_split(f"{base_path}/train")
test_dataset = load_imdb_split(f"{base_path}/test")

print(train_dataset)

train_short = train_dataset.select(range(2000))
test_short = test_dataset.select(range(2000))

"""
train_short.to_parquet("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/text.parquet")
test_short.to_parquet("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/train.parquet")
"""

print(train_short)



mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/text.parquet", artifact_path="datasets")
mlflow.log_artifact("/Users/benjaminbrooke/PycharmProjects/MLOps/LLMOps/Date/train.parquet", artifact_path="datasets")

train_short = train_short.map(tokenise, batched=True)
test_short = test_short.map(tokenise, batched=True)


print(test_short)






train_short.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_short.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)


train_loader = DataLoader(train_short, batch_size=params["batch_size"], shuffle=False)
test_loader = DataLoader(test_short, batch_size=params["batch_size"], shuffle=False)













model = AutoModelForSequenceClassification.from_pretrained(params["model_name"], num_labels=2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model.to(device)

optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

def evaluate_model(model, dataloader,device):
    model.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:

            input_ids, attention_mask, label = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label
            )

            logits = outputs.logits

            preds = logits.argmax(dim=-1)

            if len(preds.shape) > 1:
                preds = preds[:, 0]

            predictions.extend(preds.tolist())
            true_labels.extend(label.tolist())


    accuracy_score_ = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

    return accuracy_score_, precision, recall, f1



print("-----model begins here-----")


with tqdm(total=params["num_epochs"]*len(train_short), desc = f"Epoch [1 / {params['num_epochs']} ]") as pbar:
    for epoch in range(params["num_epochs"]):
        running_loss = 0.0

        for i, batch in enumerate(train_loader,0):

            print(i)

            input_ids, attention_mask, label = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 0 and i > 0:
                avg_loss = running_loss / 5

                mlflow.log_metric("train_loss", avg_loss, step=epoch*len(train_loader)+i)

                running_loss = 0.0


        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)

        mlflow.log_metrics({"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1})


mlflow.pytorch.log_model(model,"flan-t5-small")