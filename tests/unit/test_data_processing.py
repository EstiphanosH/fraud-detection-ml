import pytest
import pandas as pd
from scripts.data_processing import FraudDataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'user_id': ['U1', 'U1', 'U2', 'U3'],
        'purchase_value': [100, 100, 200, 50000],
        'age': [25, 25, 130, 40],
        'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'signup_time': ['2023-01-01 10:00']*4,
        'purchase_time': ['2023-01-01 10:05']*4,
        'browser': ['Chrome', 'Chrome', 'Firefox', None],
        'class': [0, 0, 1, 0]
    })

@pytest.fixture
def config():
    return {
        'ip_country_path': 'data/raw/IpAddress_to_Country.csv',
        'min_purchase_value': 0.01,
        'max_purchase_value': 10000,
        'valid_age_range': (18, 100)
    }

def test_duplicate_removal(sample_data, config):
    processor = FraudDataPreprocessor(config)
    cleaned = processor.transform(sample_data)
    assert cleaned.shape[0] == 3  # Removed 1 duplicate

def test_missing_value_handling(sample_data, config):
    processor = FraudDataPreprocessor(config)
    cleaned = processor.transform(sample_data)
    assert cleaned['browser'].isna().sum() == 0
    assert cleaned.shape[0] == 2  # Removed invalid age and high value

def test_business_rule_enforcement(sample_data, config):
    processor = FraudDataPreprocessor(config)
    cleaned = processor.transform(sample_data)
    # Should remove transaction with value 50,000 (exceeds max)
    assert cleaned[cleaned['purchase_value'] == 50000].empty
    # Should remove transaction with age 130 (exceeds max)
    assert cleaned[cleaned['age'] == 130].empty