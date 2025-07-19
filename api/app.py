"""
Production API for real-time fraud detection
"""

from fastapi import FastAPI, HTTPException
from scripts.pipeline import FraudDetectionPipeline
from pydantic import BaseModel
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for financial transactions",
    version="1.0.0"
)

# Load pipeline
try:
    pipeline = joblib.load('models/production_pipeline.pkl')
    logger.info("Loaded production fraud detection pipeline")
except Exception as e:
    logger.error(f"Failed to load pipeline: {str(e)}")
    pipeline = None

class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    purchase_value: float
    signup_time: str
    purchase_time: str
    device_id: str
    ip_address: str
    browser: str
    age: int
    sex: str

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """Predict fraud probability and return decision"""
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection system unavailable"
        )
    
    try:
        result = pipeline.predict(transaction.dict())
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error processing transaction"
        )

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}