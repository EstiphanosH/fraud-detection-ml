"""
Robust feature engineering with dynamic handling
for e-commerce and banking datasets
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FraudFeatureEngineer:
    """
    Creates domain-specific features with:
    - Automatic feature selection
    - Business-driven transformations
    - Leakage prevention
    """
    
    def __init__(self, dataset_type: str, config: Dict[str, Any]):
        self.dataset_type = dataset_type
        self.config = config
        self.feature_mapping = {
            'ecommerce': self._ecom_features,
            'bank': self._bank_features
        }
        
    def _ecom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """E-commerce feature engineering"""
        # Time-based features
        if 'signup_time' in df.columns:
            df['time_since_signup'] = (
                (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
            )
            df['new_account_flag'] = (df['time_since_signup'] < 2).astype(int)
        
        # Behavioral features
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id').agg(
                avg_purchase=('purchase_value', 'mean'),
                tx_count=('purchase_time', 'count')
            ).reset_index()
            df = df.merge(user_stats, on='user_id', how='left')
            df['value_deviation'] = (
                (df['purchase_value'] - df['avg_purchase']) / 
                df['avg_purchase'].replace(0, 1)
            )
            
        # Geolocation features
        if 'country' in df.columns:
            df['high_risk_country'] = df['country'].isin(
                self.config['high_risk_countries']
            ).astype(int)
            
        return df
    
    def _bank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Banking feature engineering"""
        # Time-based features
        if 'Time' in df.columns:
            df['hour_of_day'] = (df['Time'] // 3600) % 24
            df['is_night'] = ((df['hour_of_day'] >= 1) & (df['hour_of_day'] <= 5)).astype(int)
        
        # Transaction features
        if 'Amount' in df.columns:
            df['log_amount'] = np.log1p(df['Amount'])
            df['high_value'] = (df['Amount'] > self.config['high_value_threshold']).astype(int)
            
        # PCA feature interactions
        for i in [4, 14, 17]:
            if f'V{i}' in df.columns:
                df[f'V{i}_amount'] = df[f'V{i}'] * df['Amount']
                
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-specific feature engineering"""
        if self.dataset_type not in self.feature_mapping:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        return self.feature_mapping[self.dataset_type](df)