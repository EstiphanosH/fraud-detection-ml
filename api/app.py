"""
Production API for real-time fraud detection
"""

from fastapi import FastAPI, HTTPException, Request, status
from scripts.pipeline import FraudDetectionPipeline
from scripts.metrics import BusinessMetrics
from scripts.utils import load_config, prepare_api_data, validate_transaction
import joblib
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for financial transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Initialize metrics and pipeline
metrics = BusinessMetrics()
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    try:
        config = load_config("config/pipeline_config.yaml")
        pipeline = FraudDetectionPipeline(config)
        logger.info("Fraud detection pipeline loaded successfully")
    except Exception as e:
        logger.critical(f"Pipeline loading failed: {str(e)}")
        metrics.log_error()
        pipeline = None

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_fraud(request: Request, transaction: dict):
    """Predict fraud probability and return decision"""
    start_time = time.time()
    
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection system unavailable"
        )
    
    # Validate input
    if not validate_transaction(transaction):
        metrics.log_error()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transaction data"
        )
    
    try:
        # Prepare data
        prepared_data = prepare_api_data(transaction)
        
        # Get prediction
        result = pipeline.predict(prepared_data)
        latency = time.time() - start_time
        
        # Update metrics
        metrics.update_prediction(transaction, result)
        metrics.log_latency(latency)
        
        return {
            "status": "success",
            "latency_ms": round(latency * 1000, 2),
            "result": result
        }
    except Exception as e:
        metrics.log_error()
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing transaction"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "metrics": {
            "predictions": metrics.prediction_count._value.get(),
            "errors": metrics.error_count._value.get()
        }
    }

@app.get("/metrics", status_code=status.HTTP_200_OK)
async def get_metrics():
    return metrics.generate_metrics_report()