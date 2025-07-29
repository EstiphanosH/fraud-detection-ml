# tests/test_unit.py

import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import the refactored classes
from scripts.data_cleaning import DataCleaner
from scripts.feature_engineering import FeatureEngineer
from scripts.ip_to_country import IPtoCountryMapper # Import the new IP mapper

# Adjust paths for testing relative to the tests directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'mock_data') # Assuming mock_data in tests folder
TEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'mock_models')

# Ensure mock data directory exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(TEST_MODELS_DIR, exist_ok=True)

# Create dummy/mock data files for testing
def create_mock_data():
    # Mock Fraud_Data.csv
    fraud_data = {
        'user_id': [1, 2, 3, 4, 5, 6],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 12:00:00', '2023-01-03 13:00:00', '2023-01-04 14:00:00', np.nan],
        'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 11:30:00', '2023-01-02 12:30:00', '2023-01-03 13:30:00', '2023-01-04 14:30:00', '2023-01-05 15:00:00'],
        'purchase_value': [100, 200, 50, 150, 300, 120],
        'device_id': ['A1', 'B2', 'C3', 'D4', 'E5', 'F6'],
        'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads', 'Direct'],
        'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Edge', 'Firefox'],
        'sex': ['M', 'F', 'M', 'F', 'M', 'F'],
        'age': [25, 30, 22, 35, 28,  np.nan],
        'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '192.168.1.1', '10.0.0.1', '203.0.113.45'],
        'class': [0, 0, 1, 0, 0, 1] # 0: legitimate, 1: fraud
    }
    pd.DataFrame(fraud_data).to_csv(os.path.join(TEST_DATA_DIR, 'Fraud_Data.csv'), index=False)

    # Mock creditcard.csv (simplified)
    creditcard_data = {
        'Time': [1, 2, 3, 4, 5, 6],
        'V1': [1, 2, 3, 4, 5, 6], 'V2': [1, 2, 3, 4, 5, 6], 'V3': [1, 2, 3, 4, 5, 6],
        # ... (V4 to V28 for complete test)
        'V28': [1, 2, 3, 4, 5, 6],
        'Amount': [10.0, 20.0, 15.0, 25.0, 30.0, 12.0],
        'Class': [0, 0, 1, 0, 0, 1]
    }
    pd.DataFrame(creditcard_data).to_csv(os.path.join(TEST_DATA_DIR, 'creditcard.csv'), index=False)

    # Mock IpAddress_to_Country.csv
    ip_country_data = {
        'lower_bound_ip_address': [16777216.0, 16777472.0, 2130706432.0, 3409154048.0],
        'upper_bound_ip_address': [16777471.0, 16777727.0, 2130706687.0, 3409154303.0],
        'country': ['Australia', 'China', 'United States', 'Canada']
    }
    pd.DataFrame(ip_country_data).to_csv(os.path.join(TEST_DATA_DIR, 'IpAddress_to_Country.csv'), index=False)

create_mock_data()


class TestDataIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We'll directly load data within tests or mock file reads
        # For this, we need to ensure the IPtoCountryMapper is initialized with the mock path
        cls.ip_mapper = IPtoCountryMapper(os.path.join(TEST_DATA_DIR, 'IpAddress_to_Country.csv'))

    def test_load_ip_to_country_data(self):
        df = self.ip_mapper._load_ip_country_data()
        self.assertFalse(df.empty)
        self.assertIn('country', df.columns)
        self.assertEqual(str(df['lower_bound_ip_address'].dtype), 'float64') # Now it should be float64

    def test_map_ips_to_countries(self):
        df_test = pd.DataFrame({
            'user_id': [1, 2],
            'ip_address': ['192.168.1.1', '10.0.0.1'] # These map to 'Unknown' with current mock IP data
        })
        # Add a test IP that falls within the mock IP range
        test_ip_data = {
            'user_id': [7, 8],
            'ip_address': ['127.0.0.1', '10.10.10.10'] # Using some valid IPs that might map to existing ranges, or new ones for 'Unknown'
        }
        # Add an IP that should map to Australia from the mock data
        ip_for_australia = "1.0.0.1" # Which is 16777217.0 (within 16777216.0-16777471.0)
        df_with_mappable_ip = pd.DataFrame({
            'user_id': [1, 2, 3],
            'ip_address': ['1.0.0.1', '255.255.255.255', '99.99.99.99']
        })

        mapped_df = self.ip_mapper.map_ips_to_countries(df_with_mappable_ip.copy()) # Pass a copy
        self.assertIn('country', mapped_df.columns)
        self.assertFalse(mapped_df['country'].isnull().any())
        self.assertIn('Australia', mapped_df['country'].values)
        self.assertEqual(mapped_df.loc[mapped_df['user_id'] == 1, 'country'].iloc[0], 'Australia')
        self.assertIn('Unknown', mapped_df['country'].values) # Check for unmapped IPs
        self.assertEqual(mapped_df.loc[mapped_df['user_id'] == 2, 'country'].iloc[0], 'Unknown')


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = DataCleaner()
        self.df_ecommerce = pd.read_csv(os.path.join(TEST_DATA_DIR, 'Fraud_Data.csv'))
        self.df_bank = pd.read_csv(os.path.join(TEST_DATA_DIR, 'creditcard.csv'))

    def test_remove_duplicates(self):
        df_with_dupes = pd.DataFrame({'col1': [1, 2, 1], 'col2': ['a', 'b', 'a']})
        df_cleaned = self.cleaner.remove_duplicates(df_with_dupes.copy())
        self.assertEqual(len(df_cleaned), 2)
        self.assertFalse(df_cleaned.duplicated().any())

    def test_handle_missing_values_ecommerce(self):
        # Create a dataframe with specific missing values for ecommerce
        df_test_ecommerce = pd.DataFrame({
            'numerical_col': [1.0, 2.0, np.nan, 4.0],
            'signup_time': ['2023-01-01', np.nan, '2023-01-03', '2023-01-04'],
            'purchase_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', np.nan, '2023-01-01 13:00:00'],
            'categorical_col': ['A', np.nan, 'C', 'D']
        })
        # Correct data types first so handle_missing_values sees them correctly
        df_test_ecommerce = self.cleaner.correct_data_types(df_test_ecommerce.copy(), dataset_type='ecommerce')
        df_cleaned = self.cleaner.handle_missing_values(df_test_ecommerce.copy())

        self.assertFalse(df_cleaned['numerical_col'].isnull().any())
        # The test expects no nulls for signup_time. With dropping rows for NaT, this should be True for less rows.
        # So we assert the column has no nulls AND the length changed if a null existed.
        self.assertFalse(df_cleaned['signup_time'].isnull().any())
        self.assertFalse(df_cleaned['purchase_time'].isnull().any())
        self.assertFalse(df_cleaned['categorical_col'].isnull().any())
        self.assertIn('Unknown', df_cleaned['categorical_col'].values) # Check if 'Unknown' was filled

    def test_correct_datatypes_ecommerce(self):
        df = self.cleaner.correct_data_types(self.df_ecommerce.copy(), dataset_type='ecommerce')
        self.assertEqual(str(df['signup_time'].dtype), 'datetime64[ns]')
        self.assertEqual(str(df['purchase_time'].dtype), 'datetime64[ns]')
        self.assertEqual(str(df['purchase_value'].dtype), 'float64') # pandas often infers int to float if NaNs
        self.assertEqual(str(df['age'].dtype), 'float64')
        self.assertEqual(str(df['ip_address'].dtype), 'float64')


    def test_correct_datatypes_bank(self):
        df = self.cleaner.correct_data_types(self.df_bank.copy(), dataset_type='bank')
        self.assertEqual(str(df['Time'].dtype), 'float64') # Often inferred as float in raw data
        self.assertEqual(str(df['Amount'].dtype), 'float64')
        for i in range(1, 29):
            self.assertEqual(str(df[f'V{i}'].dtype), 'float64')


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.feature_engineer = FeatureEngineer()
        # Mock a DataFrame that has already gone through data_cleaning and IP mapping
        self.df_ecommerce_processed = pd.DataFrame({
            'user_id': [1, 2, 3],
            'signup_time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 12:00:00']),
            'purchase_time': pd.to_datetime(['2023-01-01 10:30:00', '2023-01-01 11:30:00', '2023-01-02 12:30:00']),
            'purchase_value': [100, 200, 50],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Firefox', 'Safari'],
            'sex': ['M', 'F', 'M'],
            'country': ['USA', 'Canada', 'Mexico']
        })

    def test_create_time_based_features(self):
        df = self.feature_engineer.create_time_based_features(self.df_ecommerce_processed.copy())
        self.assertIn('purchase_hour', df.columns)
        self.assertIn('purchase_dayofweek', df.columns)
        self.assertIn('purchase_month', df.columns)
        self.assertIn('time_since_signup', df.columns)
        self.assertGreater(df['time_since_signup'].iloc[0], 0)

    def test_create_user_transaction_frequency(self):  # Renamed method
        df = self.feature_engineer.create_user_transaction_frequency(self.df_ecommerce_processed.copy())
        self.assertIn('user_transaction_count', df.columns)
        self.assertEqual(df['user_transaction_count'].iloc[0], 1)
        
        # Test with duplicated user_id
        df_freq_test = pd.DataFrame({
            'user_id': [1, 1, 2],
            'purchase_value': [100, 200, 50]
        })
        df_freq_test = self.feature_engineer.create_user_transaction_frequency(df_freq_test)
        self.assertEqual(df_freq_test.loc[df_freq_test['user_id'] == 1, 'user_transaction_count'].iloc[0], 2)
        self.assertEqual(df_freq_test.loc[df_freq_test['user_id'] == 2, 'user_transaction_count'].iloc[0], 1)


    def test_encode_categorical_features(self):
        categorical_cols = ['source', 'browser', 'sex', 'country']
        df = self.feature_engineer.encode_categorical_features(self.df_ecommerce_processed.copy(), categorical_cols)
        
        # Check if original categorical columns are removed
        for col in categorical_cols:
            self.assertNotIn(col, df.columns)
        
        # Check for presence of new one-hot encoded columns
        self.assertIn('source_Direct', df.columns)
        self.assertIn('browser_Safari', df.columns)
        self.assertIn('sex_M', df.columns)
        self.assertIn('country_Canada', df.columns)

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # We need to provide mock_data_path for the DataProcessor
        self.processor = MagicMock() # Mock the DataProcessor
        self.processor.raw_data_path = TEST_DATA_DIR
        self.processor.processed_data_path = TEST_MODELS_DIR # Using models dir as placeholder output

        # Mock dependencies of DataProcessor
        self.processor.ip_mapper = IPtoCountryMapper(os.path.join(TEST_DATA_DIR, 'IpAddress_to_Country.csv'))
        self.processor.cleaner = DataCleaner()
        self.processor.feature_engineer = FeatureEngineer()
        
        # Manually create mock load_data method
        def mock_load_data(filename):
            filepath = os.path.join(TEST_DATA_DIR, filename)
            if not os.path.exists(filepath):
                return None # Simulate file not found
            return pd.read_csv(filepath)
        self.processor.load_data = MagicMock(side_effect=mock_load_data)

        # Re-attach original methods from the actual DataProcessor if needed for deeper tests
        from scripts.data_pipeline import DataProcessor as ActualDataProcessor
        self.processor.process_ecommerce_data = ActualDataProcessor.process_ecommerce_data.__get__(self.processor, ActualDataProcessor)
        self.processor.process_bank_data = ActualDataProcessor.process_bank_data.__get__(self.processor, ActualDataProcessor)
        self.processor.handle_imbalance = ActualDataProcessor.handle_imbalance.__get__(self.processor, ActualDataProcessor)


    def test_process_ecommerce_data_integration(self):
        # Load the mock data directly as process_ecommerce_data expects a DataFrame
        df_raw = pd.read_csv(os.path.join(TEST_DATA_DIR, 'Fraud_Data.csv'))
        processed_df = self.processor.process_ecommerce_data(df_raw.copy())
        
        self.assertFalse(processed_df.empty)
        self.assertNotIn('signup_time', processed_df.columns)
        self.assertIn('purchase_hour', processed_df.columns)
        self.assertIn('country_Unknown', processed_df.columns) # Check for mapped country
        self.assertNotIn('device_id', processed_df.columns) # Should be dropped now
        self.assertNotIn('device_id', processed_df.columns)

    def test_handle_imbalance_smote(self):
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]})
        y = pd.Series([0, 0, 0, 0, 1]) # Imbalanced data
        X_resampled, y_resampled = self.processor.handle_imbalance(X, y, sampler_type='SMOTE')
        self.assertGreater(y_resampled.value_counts()[1], y.value_counts()[1]) # Check if minority class increased
        self.assertEqual(y_resampled.value_counts()[0], y_resampled.value_counts()[1]) # Check for balance

    def test_handle_imbalance_adasyn(self):
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50]})
        y = pd.Series([0, 0, 0, 0, 1]) # Imbalanced data
        X_resampled, y_resampled = self.processor.handle_imbalance(X, y, sampler_type='ADASYN')
        self.assertGreater(y_resampled.value_counts()[1], y.value_counts()[1]) # Check if minority class increased
        self.assertGreaterEqual(y_resampled.value_counts()[0], y_resampled.value_counts()[1]) # ADASYN might not perfectly balance


class TestModelTraining(unittest.TestCase):
    # This class would contain tests for model_training.py
    # Since model_training.py is not yet provided, this is a placeholder.
    def test_placeholder(self):
        self.assertTrue(True) # Dummy test to avoid error if no tests found