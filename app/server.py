from datetime import datetime
import json
from time import time
from typing import Dict, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import torch


GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "./data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "./data/logs.out"
    }
}


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model = SentenceTransformer(f"sentence-transformers/{model_name}")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.model.encode(
            sentences=X,
            normalize_embeddings=True,
            batch_size=32,
        )


class NewsCategoryClassifier:
    def __init__(self, transformer_model_name: str, serialized_classifier_path: str) -> None:
        featurizer = TransformerFeaturizer(transformer_model_name)
        classifier = joblib.load(serialized_classifier_path)
        self.class_names = classifier.classes_
        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', classifier)
        ])

    def predict(self, document: str) -> Tuple[str, Dict[str, float]]:
        pred_proba = self.pipeline.predict_proba([document])
        pred_label_scores = {
            self.class_names[i]: pred_proba[0][i]
            for i in range(len(self.class_names))
        }
        pred_label = self.class_names[pred_proba[0].argmax()]
        return pred_label, pred_label_scores


app = FastAPI()

@app.on_event("startup")
def startup_event():
    global classifier
    global log_file
    classifier = NewsCategoryClassifier(
        transformer_model_name=GLOBAL_CONFIG["model"]["featurizer"]["sentence_transformer_model"],
        serialized_classifier_path=GLOBAL_CONFIG["model"]["classifier"]["serialized_model_path"]
    )
    log_file = open(GLOBAL_CONFIG["service"]["log_destination"], "a")
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    log_file.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    start_time = datetime.now()
    pred_label, pred_label_scores = classifier.predict(request.description)
    end_time = datetime.now()
    inference_time_ms = (end_time - start_time).total_seconds() * 1000

    # construct the data to be logged
    log_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request": request.dict(),
        "prediction": pred_label,
        "scores": pred_label_scores,
        "inference_time_ms": int(inference_time_ms),
        "inference_device": torch.cuda.get_device_name(),
    }
    log_file.write(f"{json.dumps(log_info)}\n")
    log_file.flush()

    # construct response
    return PredictResponse(scores=pred_label_scores, label=pred_label)


@app.get("/")
def read_root():
    return {"message": "Please use the /predict endpoint to make predictions"}
