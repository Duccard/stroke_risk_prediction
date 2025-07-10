from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("stroke_risk_pipeline_ensemble_model.joblib")


class PatientData(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str


@app.get("/")
def root():
    return {"message": "Your Stroke Prediction API is running!"}


@app.post("/predict")
def predict_stroke(data: PatientData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return {"prediction": int(prediction), "probability": round(float(probability), 4)}
