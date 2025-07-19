"""
Feature engineering optimized for fraud pattern detection
Creates behavioral, temporal, and geospatial features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class FraudFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates predictive features with business logic:
    
    Key Features:
    1. Temporal Patterns:
       - time_since_signup (hours)
       - transaction_hour (categorized)
       
    2. Behavioral Patterns:
       - transaction_velocity (txns/hour)
       - purchase_value_deviation
       
    3. Geospatial Patterns:
       - country_risk_score
       - distance_from_home (if location data available)
    
    Business Hypothesis:
    - Fraudsters act quickly after signup (time_since_signup)
    - High-value deviations from normal behavior are suspicious
    - Transactions from high-risk countries need scrutiny
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_profiles = None
        self.country_risk_scores = self._load_country_risk_data()
        
    def _load_country_risk_data(self) -> Dict[str, float]:
        """Load business-defined country risk scores"""
        # Default medium risk for unknown countries
        return {
            'US': 0.2, 'GB': 0.3, 'DE': 0.25,
            'NG': 0.85, 'RU': 0.8, 'CN': 0.7
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Build user behavioral profiles from historical data"""
        logger.info("Building user behavioral profiles")
        
        # Calculate user-specific baselines
        self.user_profiles = X.groupby('user_id').agg(
            avg_purchase_value=('purchase_value', 'mean'),
            std_purchase_value=('purchase_value', 'std'),
            common_country=('country', lambda x: x.mode()[0] if not x.mode().empty else None)
        ).reset_index()
        
        # Handle new users
        self.user_profiles['std_purchase_value'].fillna(
            self.user_profiles['avg_purchase_value'] * 0.5, inplace=True)
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate fraud prediction features"""
        logger.info("Generating fraud detection features")
        
        # Temporal features
        X['time_since_signup'] = (X['purchase_time'] - X['signup_time']).dt.total_seconds() / 3600
        X['transaction_hour'] = X['purchase_time'].dt.hour
        X['is_night'] = (X['transaction_hour'] < 6) | (X['transaction_hour'] > 22)
        
        # Behavioral features
        X = pd.merge(X, self.user_profiles, on='user_id', how='left')
        X['value_deviation'] = (X['purchase_value'] - X['avg_purchase_value']) / X['std_purchase_value']
        X['value_deviation'].fillna(0, inplace=True)
        
        # Velocity features
        X.sort_values(['user_id', 'purchase_time'], inplace=True)
        X['prev_purchase_time'] = X.groupby('user_id')['purchase_time'].shift(1)
        X['time_since_last'] = (X['purchase_time'] - X['prev_purchase_time']).dt.total_seconds() / 3600
        X['txn_velocity'] = 1 / X['time_since_last'].replace(0, 0.1)  # Avoid division by zero
        
        # Geospatial features
        X['country_risk'] = X['country'].map(self.country_risk_scores).fillna(0.5)
        X['country_mismatch'] = (X['country'] != X['common_country']).astype(int)
        
        # High-risk feature combinations (business rules)
        X['new_account_high_value'] = (
            (X['time_since_signup'] < self.config['new_account_threshold']) & 
            (X['purchase_value'] > self.config['high_value_threshold'])
        ).astype(int)
        
        return X