"""
Data Cleaning Module

Implements comprehensive data cleaning pipeline:
1. Missing value handling
2. Duplicate removal
3. Data type correction
4. Dataset-specific cleaning workflows

Follows CRISP-DM methodology with audit logging.

Author: Data Quality Team
Date: 2023-10-15
Version: 2.2
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """
    Data cleaning pipeline for structured datasets.
    
    Implements:
    - Intelligent missing value treatment
    - Deduplication strategies
    - Type conversion with validation
    - Domain-specific cleaning workflows
    """
    
    def __init__(self):
        """Initialize data cleaner with default parameters."""
        logging.info("DataCleaner initialized")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive missing value treatment.
        
        Strategy:
        - Datetime columns: Drop rows if >5% missing, else median impute
        - Numerical columns: Median for skewed, mean otherwise
        - Categorical: 'Unknown' imputation
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values treated
        """
        logging.info("Handling missing values")
        df_copy = df.copy()
        initial_rows = len(df_copy)
        
        # Critical datetime columns
        datetime_cols = ['signup_time', 'purchase_time']
        for col in datetime_cols:
            if col in df_copy.columns:
                # Convert to UTC-aware datetime
                df_copy[col] = pd.to_datetime(df_copy[col], utc=True, errors='coerce')
                
                # Calculate missing rate
                null_count = df_copy[col].isna().sum()
                null_pct = null_count / len(df_copy)
                
                if null_count > 0:
                    if null_pct > 0.05:  # Significant missingness
                        df_copy.dropna(subset=[col], inplace=True)
                        logging.info(f"Dropped {null_count} rows ({null_pct:.1%}) for {col}")
                    else:
                        median_val = df_copy[col].median()
                        df_copy[col].fillna(median_val, inplace=True)
                        logging.info(f"Imputed {null_count} missing values in {col}")
        
        # Numerical columns
        num_cols = df_copy.select_dtypes(include=np.number).columns
        for col in num_cols:
            null_count = df_copy[col].isna().sum()
            if null_count > 0:
                # Use median for skewed distributions
                skew = df_copy[col].skew()
                if abs(skew) > 1:  # Highly skewed
                    impute_val = df_copy[col].median()
                    method = 'median'
                else:
                    impute_val = df_copy[col].mean()
                    method = 'mean'
                    
                df_copy[col].fillna(impute_val, inplace=True)
                logging.info(f"Imputed {null_count} nulls in {col} using {method} (skew={skew:.2f})")
        
        # Categorical columns
        cat_cols = df_copy.select_dtypes(include='object').columns
        for col in cat_cols:
            null_count = df_copy[col].isna().sum()
            if null_count > 0:
                df_copy[col].fillna('Unknown', inplace=True)
                logging.info(f"Imputed {null_count} nulls in {col} with 'Unknown'")
        
        # Report cleaning impact
        final_rows = len(df_copy)
        if initial_rows != final_rows:
            logging.info(f"Row count changed: {initial_rows} → {final_rows} ({initial_rows-final_rows} removed)")
            
        return df_copy

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows with auditing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        initial_rows = len(df)
        initial_dupes = df.duplicated().sum()
        
        if initial_dupes == 0:
            logging.info("No duplicates found")
            return df
            
        # Remove duplicates
        df_deduped = df.drop_duplicates()
        final_rows = len(df_deduped)
        
        logging.info(f"Removed {initial_rows - final_rows} duplicates "
                     f"({initial_dupes} identified)")
        
        return df_deduped

    def correct_data_types(self, df: pd.DataFrame, dataset_type: str = None) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            df: Input DataFrame
            dataset_type: 'ecommerce' or 'bank' for specific rules
            
        Returns:
            DataFrame with corrected data types
        """
        logging.info(f"Correcting data types for {dataset_type or 'generic'} dataset")
        df_copy = df.copy()
        
        # E-commerce specific conversions
        if dataset_type == 'ecommerce':
            # Datetime conversions
            for col in ['signup_time', 'purchase_time']:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col], utc=True, errors='coerce')
            
            # Numeric conversions
            num_cols = ['purchase_value', 'age', 'ip_address']
            for col in num_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Float64')
        
        # Bank transaction conversions
        elif dataset_type == 'bank':
            num_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            for col in num_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Float64')
        
        # Generic conversions
        else:
            for col in df_copy.columns:
                # Auto-detect datetime columns
                if 'date' in col.lower() or 'time' in col.lower():
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                # Convert numeric columns
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Report type changes
        type_counts = df_copy.dtypes.value_counts()
        logging.info(f"Final dtypes:\n{type_counts}")
        
        return df_copy

    def clean_ecommerce_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        End-to-end cleaning for e-commerce data.
        
        Steps:
        1. Data type correction
        2. Missing value treatment
        3. Deduplication
        
        Args:
            df: Raw e-commerce data
            
        Returns:
            Cleaned e-commerce DataFrame
        """
        logging.info("Starting e-commerce cleaning pipeline")
        df = self.correct_data_types(df, 'ecommerce')
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        logging.info("E-commerce cleaning complete")
        return df
    
    def clean_bank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        End-to-end cleaning for bank transaction data.
        
        Steps:
        1. Data type correction
        2. Missing value treatment
        3. Deduplication
        
        Args:
            df: Raw bank transaction data
            
        Returns:
            Cleaned bank transaction DataFrame
        """
        logging.info("Starting bank transaction cleaning pipeline")
        df = self.correct_data_types(df, 'bank')
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        logging.info("Bank transaction cleaning complete")
        return df