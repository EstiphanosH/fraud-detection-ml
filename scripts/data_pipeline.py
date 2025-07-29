"""
Data Pipeline Orchestration Module

Implements end-to-end data processing workflows:
1. E-commerce fraud data pipeline
2. Bank transaction data pipeline

Key Stages:
- Data ingestion
- Cleaning and validation
- Feature engineering
- Preprocessing
- Class imbalance handling

Author: ML Engineering Team
Date: 2023-10-15
Version: 4.0
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, ADASYN

# Import custom modules
from scripts.data_ingestion import DataIngestion
from scripts.data_cleaning import DataCleaner
from scripts.feature_engineering import FeatureEngineer
from scripts.ip_to_country import IPtoCountryMapper

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataPipeline')

# Project paths
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
MODELS_PATH = 'models'
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

class DataPipeline:
    """
    Orchestrates end-to-end data processing for fraud detection.
    
    Implements two parallel pipelines:
    1. E-commerce fraud data
    2. Bank transaction data
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.ingestor = DataIngestion(base_path=RAW_DATA_PATH)
        self.cleaner = DataCleaner()
        ip_lookup_path = os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv')
        self.ip_mapper = IPtoCountryMapper(ip_lookup_path)
        self.feature_engineer = FeatureEngineer()
        logger.info("DataPipeline initialized")

    def run_ecommerce_pipeline(self) -> None:
        """Full processing workflow for e-commerce data."""
        logger.info("🚀 Starting E-commerce Data Pipeline")
        
        # 1. Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        fraud_df = self.ingestor.load_fraud_data()
        if fraud_df is None or fraud_df.empty:
            logger.error("Failed to load e-commerce data")
            return
        logger.info(f"Loaded raw data: {fraud_df.shape}")
        
        # 2. Data Cleaning
        logger.info("Stage 2: Data Cleaning")
        cleaned_df = self.cleaner.clean_ecommerce_data(fraud_df)
        logger.info(f"Cleaned data: {cleaned_df.shape}")
        
        # 3. Feature Engineering
        logger.info("Stage 3: Feature Engineering")
        engineered_df = self.ip_mapper.map_ips_to_countries(cleaned_df)
        engineered_df = self.feature_engineer.create_time_based_features(engineered_df)
        engineered_df = self.feature_engineer.create_user_transaction_frequency(engineered_df)
        engineered_df = self.feature_engineer.create_transaction_amount_features(engineered_df)
        
        # Time-windowed features
        entities = ['user_id', 'device_id', 'ip_address']
        for entity in entities:
            if entity in engineered_df.columns:
                engineered_df = self.feature_engineer.create_time_window_features(
                    engineered_df,
                    time_col='purchase_time',
                    id_col=entity,
                    amount_col='purchase_value',
                    time_windows=[1, 7, 30]
                )
        logger.info(f"Engineered features: {engineered_df.shape}")
        
        # 4. Train-Test Split
        logger.info("Stage 4: Data Splitting")
        TARGET_COL = 'class'
        if TARGET_COL not in engineered_df.columns:
            logger.error(f"Missing target column: {TARGET_COL}")
            return
            
        # Drop non-feature columns
        drop_cols = [TARGET_COL, 'user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
        X = engineered_df.drop(columns=[c for c in drop_cols if c in engineered_df.columns])
        y = engineered_df[TARGET_COL]
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        logger.info(f"Split: Train {X_train.shape}, Test {X_test.shape}")
        
        # 5. Preprocessing
        logger.info("Stage 5: Data Preprocessing")
        scaler = StandardScaler()
        
        # Identify numerical and categorical columns
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()  # Now uses imported np
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), num_cols),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ], remainder='drop')
        
        # Fit and transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        # Save preprocessing artifacts
        joblib.dump(preprocessor, os.path.join(MODELS_PATH, 'ecommerce_preprocessor.pkl'))
        joblib.dump(feature_names, os.path.join(MODELS_PATH, 'ecommerce_features.joblib'))
        logger.info("Saved preprocessing artifacts")
        
        # 6. Handle Class Imbalance
        logger.info("Stage 6: Class Imbalance Handling")
        logger.info(f"Original class distribution:\n{y_train.value_counts(normalize=True)}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_df, y_train)
        logger.info(f"Resampled: {X_resampled.shape}")
        
        # 7. Save Processed Data
        logger.info("Stage 7: Data Persistence")
        X_resampled.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_ecommerce_train.csv'), index=False)
        y_resampled.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_ecommerce_train.csv'), index=False)
        X_test_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_ecommerce_test.csv'), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_ecommerce_test.csv'), index=False)
        
        logger.info("✅ E-commerce Pipeline Complete")

    def run_bank_pipeline(self) -> None:
        """Full processing workflow for bank transaction data."""
        logger.info("🚀 Starting Bank Transaction Pipeline")
        
        # 1. Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        bank_df = self.ingestor.load_creditcard_data()
        if bank_df is None or bank_df.empty:
            logger.error("Failed to load bank data")
            return
        logger.info(f"Loaded raw data: {bank_df.shape}")
        
        # 2. Data Cleaning
        logger.info("Stage 2: Data Cleaning")
        cleaned_df = self.cleaner.clean_bank_data(bank_df)
        logger.info(f"Cleaned data: {cleaned_df.shape}")
        
        # 3. Train-Test Split
        logger.info("Stage 3: Data Splitting")
        TARGET_COL = 'Class'
        if TARGET_COL not in cleaned_df.columns:
            logger.error(f"Missing target column: {TARGET_COL}")
            return
            
        X = cleaned_df.drop(columns=[TARGET_COL])
        y = cleaned_df[TARGET_COL]
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        logger.info(f"Split: Train {X_train.shape}, Test {X_test.shape}")
        
        # 4. Preprocessing
        logger.info("Stage 4: Data Preprocessing")
        # Only numerical features in this dataset
        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), X_train.columns)
        ])
        
        # Fit and transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        # Save artifacts
        joblib.dump(preprocessor, os.path.join(MODELS_PATH, 'bank_preprocessor.pkl'))
        joblib.dump(feature_names, os.path.join(MODELS_PATH, 'bank_features.joblib'))
        logger.info("Saved preprocessing artifacts")
        
        # 5. Handle Class Imbalance
        logger.info("Stage 5: Class Imbalance Handling")
        logger.info(f"Original class distribution:\n{y_train.value_counts(normalize=True)}")
        
        # Apply ADASYN
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X_train_df, y_train)
        logger.info(f"Resampled: {X_resampled.shape}")
        
        # 6. Save Processed Data
        logger.info("Stage 6: Data Persistence")
        X_resampled.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_bank_train.csv'), index=False)
        y_resampled.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_bank_train.csv'), index=False)
        X_test_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_bank_test.csv'), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_bank_test.csv'), index=False)
        
        logger.info("✅ Bank Transaction Pipeline Complete")

def main():
    """Execute data pipelines."""
    pipeline = DataPipeline()
    pipeline.run_ecommerce_pipeline()
    pipeline.run_bank_pipeline()

if __name__ == "__main__":
    main()
