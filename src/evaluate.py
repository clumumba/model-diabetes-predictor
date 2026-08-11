import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import pandas as pd
import yaml, os, pickle
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

while not (_root / "params.yml").exists() and _root != _root.parent:
    _root = _root.parent

with open(_root / "params.yml", 'r')as f:
    params=yaml.safe_load(f)["train"]

def evaluate_model (data_file, model_file):
    #load the trained model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    #load the test data
    df = pd.read_csv(data_file)
    X = df.drop(columns=params["target_column"])
    y = df[params["target_column"]]

    #make predictions
    y_pred = model.predict(X)

    #evaluate the model
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    confusion = confusion_matrix(y, y_pred)

    #log the metrics to mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("diabetes_predictor_evaluate_experiment")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision_score(y, y_pred)))
        mlflow.log_metric("recall", float(recall_score(y, y_pred)))
        mlflow.log_metric("f1-score", float(f1_score(y, y_pred)))
        mlflow.log_metric("confusion_matrix", float(confusion.tolist()))

    return accuracy, report, confusion

if __name__ == "__main__":
    data_file = _root / params["data_path"]
    model_file = _root / params["models"] / "model.pkl"
    accuracy, report, confusion = evaluate_model(data_file, model_file)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")