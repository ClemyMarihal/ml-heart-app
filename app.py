from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class InputData(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)

    return {
        "prediction": int(prediction[0]),
        "result": "Disease Detected" if prediction[0] == 1 else "No Disease"
    }