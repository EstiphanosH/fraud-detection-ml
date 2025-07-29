"""
IP-to-Country Mapping Module

Implements efficient IP address to country mapping using a precomputed lookup table.
Uses binary search on sorted IP ranges for O(log n) lookup performance.

Key Features:
- Supports both dotted-quad and integer IP formats
- Handles invalid IPs gracefully with 'Unknown' designation
- Vectorized operations for high performance
- Comprehensive error handling

Author: Data Engineering Team
Date: 2023-10-15
Version: 2.1
"""

import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IPtoCountryMapper:
    """
    Maps IP addresses to countries using a lookup file with IP ranges.
    
    Attributes:
        ip_country_filepath (str): Path to IP-country mapping CSV
        ip_country_df (DataFrame): Processed mapping data
    """
    
    def __init__(self, ip_country_filepath: str):
        """
        Initialize mapper and load IP-country data.
        
        Args:
            ip_country_filepath: Path to CSV with columns:
                lower_bound_ip_address, upper_bound_ip_address, country
        """
        self.ip_country_filepath = ip_country_filepath
        self.ip_country_df = self._load_ip_country_data()
        logging.info(f"IPtoCountryMapper initialized with {len(self.ip_country_df)} valid ranges")
    
    def _ip_to_int(self, ip_str: str) -> int:
        """
        Convert IP address to integer representation.
        
        Supports:
        - Dotted-quad format (e.g., '192.168.1.1')
        - Integer strings (e.g., '3232235777')
        - Actual numeric types
        
        Args:
            ip_str: IP address in string or numeric format
            
        Returns:
            Integer representation of IP or NaN for invalid inputs
        """
        try:
            # Handle numeric inputs directly
            if isinstance(ip_str, (int, float)):
                return int(ip_str)
            
            if isinstance(ip_str, str):
                # Process dotted-quad format
                if '.' in ip_str:
                    parts = ip_str.split('.')
                    if len(parts) == 4:
                        return (int(parts[0]) << 24) | (int(parts[1]) << 16) | (int(parts[2]) << 8) | int(parts[3])
                # Process integer strings
                else:
                    return int(ip_str)
        except (ValueError, TypeError):
            pass
        return np.nan

    def _load_ip_country_data(self) -> pd.DataFrame:
        """
        Load and validate IP-country mapping data.
        
        Returns:
            DataFrame with cleaned and sorted IP ranges
        """
        if not os.path.exists(self.ip_country_filepath):
            logging.error(f"IP lookup file not found at {self.ip_country_filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.ip_country_filepath)
            
            # Validate required columns
            required_cols = {'lower_bound_ip_address', 'upper_bound_ip_address', 'country'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                logging.error(f"Missing required columns: {missing}")
                return pd.DataFrame()
            
            # Convert to efficient integer types
            df['lower_bound_ip_address'] = pd.to_numeric(
                df['lower_bound_ip_address'], errors='coerce'
            ).astype('Int64')
            df['upper_bound_ip_address'] = pd.to_numeric(
                df['upper_bound_ip_address'], errors='coerce'
            ).astype('Int64')
            
            # Clean and sort
            df.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
            df.sort_values('lower_bound_ip_address', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            logging.info(f"Loaded {len(df)} valid IP ranges")
            return df
            
        except Exception as e:
            logging.error(f"Error loading IP data: {str(e)}")
            return pd.DataFrame()

    def map_ips_to_countries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map IP addresses to countries in the input DataFrame.
        
        Args:
            df: DataFrame containing 'ip_address' column
            
        Returns:
            Original DataFrame with added 'country' column
        """
        if self.ip_country_df.empty:
            logging.warning("No IP mapping data available")
            df['country'] = 'Unknown'
            return df

        if 'ip_address' not in df.columns:
            logging.error("Input DataFrame missing 'ip_address' column")
            df['country'] = 'Unknown'
            return df
            
        # Create working copy to avoid modifying original
        result_df = df.copy()
        result_df['country'] = 'Unknown'  # Default value
        
        # Convert IPs to numeric representation
        result_df['ip_address_numeric'] = result_df['ip_address'].apply(self._ip_to_int)
        
        # Identify invalid IPs
        invalid_mask = result_df['ip_address_numeric'].isna()
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logging.warning(f"{invalid_count} invalid IP addresses found")
        
        # Process valid IPs
        valid_df = result_df[~invalid_mask].copy()
        if valid_df.empty:
            return result_df.drop(columns=['ip_address_numeric'])
        
        # Prepare lookup references
        lower_bounds = self.ip_country_df['lower_bound_ip_address'].values
        upper_bounds = self.ip_country_df['upper_bound_ip_address'].values
        countries = self.ip_country_df['country'].values
        
        # Vectorized binary search
        ips = valid_df['ip_address_numeric'].values
        idx = np.searchsorted(lower_bounds, ips, side='right') - 1
        
        # Validate indices
        valid_idx_mask = (idx >= 0) & (idx < len(upper_bounds))
        
        # Check range containment
        within_range_mask = np.zeros(len(ips), dtype=bool)
        within_range_mask[valid_idx_mask] = ips[valid_idx_mask] <= upper_bounds[idx[valid_idx_mask]]
        
        # Apply mappings
        valid_df.loc[within_range_mask, 'country'] = countries[idx[within_range_mask]]
        
        # Merge results
        result_df.update(valid_df[['country']])
        return result_df.drop(columns=['ip_address_numeric'])