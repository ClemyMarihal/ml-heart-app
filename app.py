from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI()

# --- Debug (optional, helps in deployment issues) ---
print("Working Directory:", os.getcwd())
print("Files:", os.listdir())

# --- Load Model ---
model_path = "model.pkl"

if not os.path.exists(model_path):
    raise Exception("❌ model.pkl NOT FOUND")

model = joblib.load(model_path)


# --- Input Schema ---
class InputData(BaseModel):
    features: List[float]


# --- Root Endpoint ---
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}


# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to numpy array
        arr = np.array(data.features).reshape(1, -1)

        # Prediction
        prediction = model.predict(arr)[0]

        # Probability (safe handling)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(arr)[0][1]
        else:
            probability = 0.0  # fallback if model doesn't support it

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "result": "Disease Detected" if prediction == 1 else "No Disease"
        }

    except Exception as e:
        return {
            "error": str(e)
        }