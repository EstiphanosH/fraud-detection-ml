"""
Data processing pipeline with business-optimized cleaning rules
Handles missing values, duplicates, and geolocation merging
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logger = logging.getLogger(__name__)

class FraudDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade data preprocessor with business rule enforcement
    
    Features:
    - Automated missing value handling based on business impact
    - Duplicate detection with transaction fingerprinting
    - Geolocation enrichment with IP-to-country mapping
    - Data quality monitoring
    
    Business Rules Implemented:
    1. Minimum purchase value: $0.01 (fraudsters test with micro-transactions)
    2. Maximum purchase value: $10,000 (requires manual review)
    3. Age validation: 18-100 years
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with business configuration
        
        :param config: Dictionary containing:
            - ip_country_path: Path to IP-country mapping
            - min_purchase_value: Minimum valid transaction amount
            - max_purchase_value: Maximum auto-processed amount
            - valid_age_range: Tuple of (min_age, max_age)
        """
        self.config = config
        self.ip_mapping = self._load_ip_mapping()
        self.stats = {"missing_handled": 0, "duplicates_removed": 0}
        
    def _load_ip_mapping(self) -> pd.DataFrame:
        """Load and cache IP-country mapping data"""
        try:
            ip_df = pd.read_csv(self.config['ip_country_path'])
            logger.info(f"Loaded IP mapping with {len(ip_df)} records")
            return ip_df
        except FileNotFoundError:
            logger.error("IP mapping file not found")
            return pd.DataFrame()
        
    def _validate_transaction(self, row: pd.Series) -> bool:
        """Apply business validation rules"""
        valid = True
        # Purchase value validation
        if not (self.config['min_purchase_value'] <= row['purchase_value'] <= self.config['max_purchase_value']):
            valid = False
            logger.warning(f"Invalid purchase value: {row['purchase_value']}")
            
        # Age validation
        if not (self.config['valid_age_range'][0] <= row['age'] <= self.config['valid_age_range'][1]):
            valid = False
            logger.warning(f"Invalid age: {row['age']}")
            
        return valid
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Execute full cleaning pipeline with business rules"""
        logger.info(f"Processing {len(X)} transactions")
        
        # Convert timestamps
        X['signup_time'] = pd.to_datetime(X['signup_time'], errors='coerce')
        X['purchase_time'] = pd.to_datetime(X['purchase_time'], errors='coerce')
        
        # Handle missing values - business impact assessment
        initial_count = len(X)
        X['browser'].fillna('Unknown', inplace=True)
        X = X.dropna(subset=['purchase_value', 'age', 'signup_time', 'purchase_time'])
        self.stats['missing_handled'] = initial_count - len(X)
        
        # Remove duplicates - transaction fingerprinting
        initial_count = len(X)
        fingerprint = X[['user_id', 'device_id', 'purchase_time', 'purchase_value']].astype(str).sum(axis=1)
        X = X[~fingerprint.duplicated()]
        self.stats['duplicates_removed'] = initial_count - len(X)
        
        # Apply business validation rules
        validation_mask = X.apply(self._validate_transaction, axis=1)
        X = X[validation_mask]
        
        # Geolocation enrichment
        if not self.ip_mapping.empty:
            X['ip_int'] = X['ip_address'].apply(lambda x: int(x.replace('.', '')) if isinstance(x, str) else np.nan)
            X = pd.merge(X, self.ip_mapping, 
                         left_on='ip_int',
                         right_on='lower_bound_ip_address',
                         how='left')
        
        logger.info(f"Processed {len(X)} valid transactions")
        return X
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Return data quality metrics for monitoring"""
        return self.stats