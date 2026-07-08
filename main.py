from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import joblib
import os
from .schema import PatientData

app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI backend serving a scikit-learn pipeline for CDC 2020 Heart Disease risk estimation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'heart_model_2020.pkl')
model_pipeline = None


@app.on_event("startup")
def load_model():
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Serialized model file not found at '{MODEL_PATH}'. "
            f"Please execute your training pipeline script first to generate it."
        )
    print(f"Loading deployable pipeline framework from {MODEL_PATH}...")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Pipeline framework loaded successfully. Ready for inference requests.")







@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the Heart Disease Prediction API. Use the /predict endpoint for inference."
    }

@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    """
    Accepts raw patient data from the UI, routes it through the preprocessor 
    and model pipeline, and returns the classification result.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction model is not initialized.")
    
    try:
       
        input_dict = patient.dict()
        input_df = pd.DataFrame([input_dict])
        
        prediction = model_pipeline.predict(input_df)
        
        prediction_result = int(prediction[0])
        
        try:
            probabilities = model_pipeline.predict_proba(input_df)[0]
            confidence = float(probabilities[prediction_result])
        except AttributeError:
            confidence = 1.0  
            
    
        return {
            "heart_disease_detected": True if prediction_result == 1 else False,
            "prediction_code": prediction_result,
            "confidence_score": round(confidence, 4),
            "interpreted_result": "High Risk Detected" if prediction_result == 1 else "Normal / Low Risk"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Engine Error: {str(e)}")