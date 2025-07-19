"""
End-to-end fraud detection pipeline
Integrates preprocessing, feature engineering, and modeling
"""

import pandas as pd
import numpy as np
import logging
from .data_processing import FraudDataPreprocessor
from .feature_engineering import FraudFeatureGenerator
from .model import FraudModel
from .business_optimizer import apply_business_rules

logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """
    Production-grade fraud detection pipeline
    
    Features:
    - Complete E2E processing from raw data to fraud decision
    - Business rule integration
    - Real-time monitoring
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = FraudDataPreprocessor(config['preprocessing'])
        self.feature_engineer = FraudFeatureGenerator(config['feature_engineering'])
        self.model = None
        
    def train(self, data_path: str):
        """Train full pipeline"""
        logger.info("Starting pipeline training")
        
        # Load and process data
        raw_data = pd.read_csv(data_path)
        processed_data = self.preprocessor.transform(raw_data)
        
        # Generate features
        X = self.feature_engineer.fit_transform(
            processed_data.drop('class', axis=1),
            processed_data['class']
        )
        y = processed_data['class']
        
        # Train model
        self.model = FraudModel(self.config['modeling'])
        self.model.train(X, y)
        
        logger.info("Pipeline training complete")
        return self
    
    def predict(self, transaction: Dict) -> Dict:
        """Process transaction and return fraud decision"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame([transaction])
            
            # Preprocess
            processed = self.preprocessor.transform(df)
            if processed.empty:
                return self._fallback_decision(transaction)
                
            # Feature engineering
            features = self.feature_engineer.transform(processed)
            
            # Model prediction
            fraud_prob = self.model.predict_proba(features)[0]
            fraud_flag = fraud_prob >= self.model.threshold
            
            # Apply business rules
            decision = apply_business_rules(
                transaction, 
                fraud_prob, 
                self.config['business_rules']
            )
            
            # Generate explanations
            explanation = self.model.explain(features) if fraud_flag else {}
            
            return {
                "transaction_id": transaction.get('transaction_id'),
                "fraud_probability": float(fraud_prob),
                "decision": decision,
                "explanations": explanation,
                "status": "processed"
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_decision(transaction)
    
    def _fallback_decision(self, transaction: Dict) -> Dict:
        """Fallback mechanism when processing fails"""
        # Business rule: Flag high-risk transactions for manual review
        if transaction['purchase_value'] > self.config['business_rules']['high_value_threshold']:
            decision = "review"
        else:
            decision = "pass"
            
        return {
            "transaction_id": transaction.get('transaction_id'),
            "fraud_probability": None,
            "decision": decision,
            "explanations": {"error": "Processing failed - using fallback rules"},
            "status": "fallback"
        }
    
    def save(self, path: str):
        """Save entire pipeline"""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load trained pipeline"""
        return joblib.load(path)