import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

RUN_ID = "8167ebcdc4384d15b5603d89d8a9a4fb"
model_uri = f"runs/{RUN_ID}/flan_t5_classifier"

mlflow.register_model(model_uri=model_uri, name="Flan-T5-Ops-Model")