"""
End-to-end model training pipeline with business optimization
"""

import argparse
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from .data_processing import FraudDataPreprocessor
from .feature_engineering import FraudFeatureEngineer
from .model import FraudModel
from .metrics import BusinessMetrics
from .utils import configure_logging, load_config

def main():
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--ecom-data', type=str, required=True,
                        help='Path to e-commerce raw data')
    parser.add_argument('--bank-data', type=str, required=True,
                        help='Path to banking transaction data')
    parser.add_argument('--ip-country', type=str, required=True,
                        help='Path to IP-country mapping')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                        help='Path to pipeline configuration')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        logger.info("Starting model training pipeline")
        
        # Process e-commerce data
        logger.info("Processing e-commerce data")
        ecom_processor = FraudDataPreprocessor(config['preprocessing'], 'ecommerce')
        ecom_df = pd.read_csv(args.ecom_data)
        ecom_processed = ecom_processor.process(ecom_df)
        
        ecom_feature_engineer = FraudFeatureEngineer('ecommerce', config['feature_engineering'])
        X_ecom = ecom_feature_engineer.transform(ecom_processed)
        y_ecom = ecom_processed['class']
        
        # Process banking data
        logger.info("Processing banking data")
        bank_processor = FraudDataPreprocessor(config['preprocessing'], 'bank')
        bank_df = pd.read_csv(args.bank_data)
        bank_processed = bank_processor.process(bank_df)
        
        bank_feature_engineer = FraudFeatureEngineer('bank', config['feature_engineering'])
        X_bank = bank_feature_engineer.transform(bank_processed)
        y_bank = bank_processed['Class']
        
        # Train e-commerce model
        logger.info("Training e-commerce model")
        ecom_model = FraudModel(config['modeling'])
        ecom_model.train(X_ecom, y_ecom)
        ecom_model_path = f"{args.output_dir}/ecommerce_model_v1.pkl"
        ecom_model.save(ecom_model_path)
        logger.info(f"Saved e-commerce model to {ecom_model_path}")
        
        # Train banking model
        logger.info("Training banking model")
        bank_model = FraudModel(config['modeling'])
        bank_model.train(X_bank, y_bank)
        bank_model_path = f"{args.output_dir}/bank_model_v1.pkl"
        bank_model.save(bank_model_path)
        logger.info(f"Saved banking model to {bank_model_path}")
        
        # Evaluate models
        ecom_metrics = ecom_model.evaluate(X_ecom, y_ecom)
        bank_metrics = bank_model.evaluate(X_bank, y_bank)
        
        logger.info(f"E-commerce model performance: {ecom_metrics}")
        logger.info(f"Banking model performance: {bank_metrics}")
        
    except Exception as e:
        logger.exception("Model training failed")
        raise

if __name__ == "__main__":
    main()