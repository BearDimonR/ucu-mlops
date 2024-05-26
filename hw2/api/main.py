from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
app = FastAPI()


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]


@app.get("/")
def read_root():
    return {"message": "Antiscam Model"}


@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features).tolist()
    return PredictionResponse(prediction=int(prediction[0]), probabilities=probabilities[0])
