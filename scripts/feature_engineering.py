"""
Feature Engineering Module

Creates derived features for fraud detection models:
1. Time-based features from datetime columns
2. User transaction frequencies
3. Transaction amount transformations
4. Time-windowed aggregations

Author: Data Science Team
Date: 2023-10-15
Version: 3.0
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection datasets.
    
    Methods implement feature creation with validation checks and
    efficient computation patterns.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline."""
        logging.info("FeatureEngineer initialized")
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.
        
        Features created:
        - purchase_hour: Hour of transaction (0-23)
        - purchase_dayofweek: Day of week (0=Monday)
        - purchase_month: Month of year (1-12)
        - time_since_signup_hours: Hours between signup and purchase
        
        Args:
            df: DataFrame with 'signup_time' and 'purchase_time'
            
        Returns:
            DataFrame with added temporal features
        """
        logging.info("Creating time-based features")
        df_copy = df.copy()
        
        # Validate datetime columns
        datetime_cols = ['signup_time', 'purchase_time']
        for col in datetime_cols:
            if col not in df_copy.columns:
                logging.warning(f"Missing datetime column: {col}")
                continue
                
            # Ensure proper datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col], utc=True, errors='coerce')
                except Exception as e:
                    logging.error(f"Datetime conversion failed for {col}: {str(e)}")
        
        # Extract purchase time features
        if 'purchase_time' in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy['purchase_time']):
                df_copy['purchase_hour'] = df_copy['purchase_time'].dt.hour
                df_copy['purchase_dayofweek'] = df_copy['purchase_time'].dt.dayofweek
                df_copy['purchase_month'] = df_copy['purchase_time'].dt.month
                logging.info("Added purchase time features")
            else:
                logging.warning("Invalid purchase_time dtype")
        
        # Calculate time since signup
        if all(col in df_copy.columns for col in ['signup_time', 'purchase_time']):
            if pd.api.types.is_datetime64_any_dtype(df_copy['signup_time']) and \
               pd.api.types.is_datetime64_any_dtype(df_copy['purchase_time']):
                
                # Calculate time difference in hours
                time_diff = (df_copy['purchase_time'] - df_copy['signup_time']).dt.total_seconds() / 3600
                df_copy['time_since_signup_hours'] = time_diff
                
                # Handle negative values (data errors)
                neg_mask = df_copy['time_since_signup_hours'] < 0
                if neg_mask.any():
                    logging.warning(f"Correcting {neg_mask.sum()} negative time differences")
                    df_copy.loc[neg_mask, 'time_since_signup_hours'] = np.nan
                
                # Impute missing values
                null_count = df_copy['time_since_signup_hours'].isna().sum()
                if null_count > 0:
                    median_val = df_copy['time_since_signup_hours'].median()
                    df_copy['time_since_signup_hours'].fillna(median_val, inplace=True)
                    logging.info(f"Imputed {null_count} missing time differences")
            else:
                logging.warning("Invalid dtypes for time difference calculation")
        
        logging.info("Completed time-based features")
        return df_copy

    def create_user_transaction_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transaction frequency per user.
        
        Args:
            df: DataFrame containing 'user_id'
            
        Returns:
            DataFrame with added 'user_transaction_count'
        """
        logging.info("Creating user transaction frequency")
        df_copy = df.copy()
        
        if 'user_id' not in df_copy.columns:
            logging.warning("Missing user_id column")
            df_copy['user_transaction_count'] = 1
            return df_copy
            
        # Calculate transaction counts
        user_counts = df_copy.groupby('user_id').size().reset_index(name='user_transaction_count')
        df_copy = df_copy.merge(user_counts, on='user_id', how='left')
        
        # Fill missing values (new users)
        df_copy['user_transaction_count'].fillna(1, inplace=True)
        
        logging.info(f"User frequency range: {df_copy['user_transaction_count'].min()}-{df_copy['user_transaction_count'].max()}")
        return df_copy

    def create_transaction_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform transaction amount values.
        
        Features created:
        - transaction_amount_log: Log(1 + amount)
        - transaction_amount_per_user_avg: Amount relative to user's average
        
        Args:
            df: DataFrame with 'purchase_value' and 'user_id'
            
        Returns:
            DataFrame with added amount features
        """
        logging.info("Creating transaction amount features")
        df_copy = df.copy()
        
        # Log transform for purchase value
        if 'purchase_value' in df_copy.columns:
            df_copy['transaction_amount_log'] = np.log1p(df_copy['purchase_value'])
        else:
            logging.warning("Missing purchase_value column")
            
        # Relative amount calculation
        if all(col in df_copy.columns for col in ['purchase_value', 'user_id']):
            user_avg = df_copy.groupby('user_id')['purchase_value'].transform('mean')
            df_copy['transaction_amount_per_user_avg'] = df_copy['purchase_value'] / (user_avg + 1e-6)
            
            # Handle infinite values
            inf_mask = np.isinf(df_copy['transaction_amount_per_user_avg'])
            if inf_mask.any():
                logging.warning(f"Correcting {inf_mask.sum()} infinite values")
                df_copy.loc[inf_mask, 'transaction_amount_per_user_avg'] = np.nan
            
            # Impute extreme values
            p99 = df_copy['transaction_amount_per_user_avg'].quantile(0.99)
            df_copy['transaction_amount_per_user_avg'] = df_copy['transaction_amount_per_user_avg'].clip(upper=p99)
            df_copy['transaction_amount_per_user_avg'].fillna(1, inplace=True)
        else:
            logging.warning("Missing columns for relative amount calculation")
            
        return df_copy

    def create_time_window_features(self, df: pd.DataFrame, time_col: str, 
                                   id_col: str, amount_col: str, 
                                   time_windows: list = [1, 7, 30]) -> pd.DataFrame:
        """
        Create time-window aggregated features.
        
        For each time window (in days), calculates:
        - {id_col}_count_last_{N}d: Transaction count
        - {id_col}_sum_amount_last_{N}d: Total amount
        - {id_col}_mean_amount_last_{N}d: Average amount
        
        Args:
            df: Input DataFrame
            time_col: Timestamp column name
            id_col: Entity ID column (user, device, IP)
            amount_col: Transaction amount column
            time_windows: List of day windows for aggregation
            
        Returns:
            DataFrame with added time-window features
        """
        logging.info(f"Creating time-window features for {id_col}")
        df_copy = df.copy()
        
        # Validate required columns
        required_cols = [time_col, id_col, amount_col]
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        if missing_cols:
            logging.error(f"Missing columns: {missing_cols}")
            return df_copy
            
        # Ensure proper datetime type
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
            try:
                df_copy[time_col] = pd.to_datetime(df_copy[time_col], utc=True, errors='coerce')
            except Exception as e:
                logging.error(f"Datetime conversion failed: {str(e)}")
                return df_copy
        
        # Create reference time (most recent transaction)
        max_time = df_copy[time_col].max()
        logging.info(f"Using reference time: {max_time}")
        
        # Create features for each time window
        for window in time_windows:
            logging.info(f"Processing {window}-day window")
            
            # Calculate time threshold
            threshold = max_time - timedelta(days=window)
            
            # Filter recent transactions
            recent_mask = df_copy[time_col] >= threshold
            recent_df = df_copy[recent_mask].copy()
            
            # Create aggregations
            aggregations = {
                f'{id_col}_count_last_{window}d': pd.NamedAgg(column=id_col, aggfunc='count'),
                f'{id_col}_sum_amount_last_{window}d': pd.NamedAgg(column=amount_col, aggfunc='sum'),
                f'{id_col}_mean_amount_last_{window}d': pd.NamedAgg(column=amount_col, aggfunc='mean')
            }
            
            # Group and aggregate
            window_features = recent_df.groupby(id_col).agg(**aggregations).reset_index()
            
            # Merge back to original data
            df_copy = df_copy.merge(window_features, on=id_col, how='left')
            
            # Fill missing values (no recent transactions)
            for col in aggregations.keys():
                df_copy[col].fillna(0, inplace=True)
                
        logging.info(f"Added {len(time_windows)*3} time-window features")
        return df_copy