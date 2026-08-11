import mlflow
from mlflow.sklearn import log_model
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
import yaml, os, pickle
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

TRUSTED_SKOPS_TYPES = [
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.metrics._classification.accuracy_score",
    "sklearn.metrics._scorer._Scorer",
    "sklearn.model_selection._split.StratifiedKFold",
]

_root = Path(__file__).resolve().parent.parent

while not (_root / "params.yml").exists() and _root != _root.parent:
    _root = _root.parent

with open(_root / "params.yml", 'r')as f:
    params=yaml.safe_load(f)["train"]

load_dotenv()


class ModelTrainer:
    """Train and evaluate a Random Forest model."""

    def __init__(self, params: dict):
        self.params = params
        self.target = params["target_column"]
        self.rf = RandomForestClassifier()
        self.model = GridSearchCV(
            estimator=self.rf,
            param_grid=params["model_params1"],
            cv=5,
            scoring=params.get("scoring", "accuracy"),
        )

    def load_data(self, data_path: str):
        """Load dataset and split features/target."""
        df = pd.read_csv(_root / data_path)

        X = df.drop(columns=self.target)
        y = df[self.target]
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("diabetes_predictor_experiment")

        return X, y

    def train(self, X, y):
        """Split data and train the model."""

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.params.get("test_size", 0.2),
            random_state=self.params.get("random_state", 42),
            stratify=y,
        )

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        return X_test, y_test, predictions

    def evaluate(self, y_test, predictions):
        """Evaluate model performance."""

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "classification_report": classification_report(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }

    def save_model(self, artifact_path="model", X_sample=None):
        """Log the trained model to MLflow."""
        model_to_log = self.model.best_estimator_ if hasattr(self.model, "best_estimator_") else self.model
        signature = None
        if X_sample is not None:
            signature = infer_signature(X_sample, model_to_log.predict(X_sample))

        log_model(
            sk_model=model_to_log,
            artifact_path=artifact_path,
            signature=signature, #type: ignore
            registered_model_name="insulin_model",
            skops_trusted_types=TRUSTED_SKOPS_TYPES,
        )
        os.makedirs(os.path.join(_root, params["models"]), exist_ok=True)
        with open(os.path.join(_root, params["models"]), "wb") as f:
            pickle.dump(model_to_log, f)

        return artifact_path


if __name__ == "__main__":
    trainer = ModelTrainer(params)
    X,y = trainer.load_data(params['data_path'])
    X_test, y_test, predictions = trainer.train(X, y)
    metrics = trainer.evaluate(y_test, predictions)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_text(metrics["classification_report"], "classification_report.txt")
        mlflow.log_text(str(metrics["confusion_matrix"]), "confusion_matrix.txt")
        trainer.save_model(artifact_path="diabetes-mbogi", X_sample=X_test)
