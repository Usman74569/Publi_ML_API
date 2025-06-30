# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:19:45 2025

@author: DELL
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Add CORS middleware
origins = ["*"]  # allow all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and scaler
model = pickle.load(open("diabetes_model.sav", "rb"))
scaler = pickle.load(open("diabetes_scaler.sav", "rb"))

# Input schema
class DiabetesFeatures(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict_diabetes(data: DiabetesFeatures):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                            data.SkinThickness, data.Insulin, data.BMI,
                            data.DiabetesPedigreeFunction, data.Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return {"prediction": "Diabetic" if prediction[0] == 1 else "Non-Diabetic"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API. Go to /docs for Swagger UI."}

