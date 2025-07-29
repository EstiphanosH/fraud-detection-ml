# scripts/data_ingestion.py

import os
import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self, base_path='data/raw'):
        self.base_path = base_path

    def load_fraud_data(self):
        # --- IMPORTANT: Ensure this matches your actual file name ---
        filepath = os.path.join(self.base_path, 'Fraud_Data.csv') # Changed from Ecommerce_Fraud.csv
        if not os.path.exists(filepath):
            print(f"Error: 'Fraud_Data.csv' not found at '{filepath}'")
            return None
        print(f"Loading e-commerce fraud data from {filepath}")
        return pd.read_csv(filepath)

    def load_ip_to_country_data(self):
        filepath = os.path.join(self.base_path, 'IpAddress_to_Country.csv')
        if not os.path.exists(filepath):
            print(f"Error: 'IpAddress_to_Country.csv' not found at '{filepath}'")
            return None
        print(f"Loading IP to Country data from {filepath}")
        return pd.read_csv(filepath, dtype={
            'lower_bound_ip_address': np.float64,
            'upper_bound_ip_address': np.float64
        })

    def load_creditcard_data(self):
        filepath = os.path.join(self.base_path, 'creditcard.csv')
        if not os.path.exists(filepath):
            print(f"Error: 'creditcard.csv' not found at '{filepath}'")
            return None
        print(f"Loading credit card fraud data from {filepath}")
        return pd.read_csv(filepath)

# Example Usage (for testing the script directly)
if __name__ == '__main__':
    data_ingestion = DataIngestion()

    ecommerce_df = data_ingestion.load_fraud_data()
    if ecommerce_df is not None:
        print("\nEcommerce Fraud Data Loaded (first 5 rows):")
        print(ecommerce_df.head())
        print(f"Shape: {ecommerce_df.shape}")

    ip_country_df = data_ingestion.load_ip_to_country_data()
    if ip_country_df is not None:
        print("\nIP to Country Data Loaded (first 5 rows):")
        print(ip_country_df.head())
        print(f"Shape: {ip_country_df.shape}")
        print("\nIP to Country Data Dtypes:")
        print(ip_country_df[['lower_bound_ip_address', 'upper_bound_ip_address']].dtypes)

    creditcard_df = data_ingestion.load_creditcard_data()
    if creditcard_df is not None:
        print("\nCredit Card Fraud Data Loaded (first 5 rows):")
        print(creditcard_df.head())
        print(f"Shape: {creditcard_df.shape}")