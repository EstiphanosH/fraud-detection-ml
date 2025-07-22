import pytest
import pandas as pd
import numpy as np
from scripts.data_processing import FraudDataPreprocessor, DataSchema

@pytest.fixture
def sample_ecom_data():
    return pd.DataFrame({
        'user_id': ['U1', 'U1', 'U2', 'U3'],
        'purchase_value': [100, 100, 200, 50000],
        'signup_time': ['2023-01-01 10:00']*4,
        'purchase_time': ['2023-01-01 10:05']*4,
        'device_id': ['D1', 'D1', 'D2', 'D3'],
        'browser': ['Chrome', 'Chrome', 'Firefox', 'Safari'],
        'age': [25, 25, 130, 40],
        'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'class': [0, 0, 1, 0]
    })

@pytest.fixture
def sample_bank_data():
    return pd.DataFrame({
        'Time': [0, 86400, 172800],
        'V1': [1.2, -0.5, 3.1],
        'V2': [0.5, -1.2, 0.8],
        'Amount': [10.5, 5000, 150],
        'Class': [0, 1, 0]
    })

@pytest.fixture
def config():
    return {
        'preprocessing': {
            'min_purchase_value': 0.01,
            'max_purchase_value': 10000,
            'valid_age_range': (18, 100)
        }
    }

def test_ecom_processing(sample_ecom_data, config):
    processor = FraudDataPreprocessor(config['preprocessing'], 'ecommerce')
    cleaned = processor.process(sample_ecom_data)
    
    # Should remove duplicate and invalid records
    assert cleaned.shape[0] == 2
    assert cleaned['age'].max() <= 100
    assert cleaned['purchase_value'].max() <= 10000

def test_bank_processing(sample_bank_data, config):
    processor = FraudDataPreprocessor(config['preprocessing'], 'bank')
    cleaned = processor.process(sample_bank_data)
    
    # Should remove transaction with amount > 10000
    assert cleaned.shape[0] == 2
    assert cleaned['Amount'].max() <= 10000

def test_schema_validation():
    valid_transaction = {
        'purchase_value': 100,
        'purchase_time': '2023-01-01 10:05:00',
        'device_id': 'D1',
        'browser': 'Chrome',
        'age': 30,
        'sex': 'M',
        'class_': 0
    }
    assert DataSchema(**valid_transaction)
    
    invalid_transaction = valid_transaction.copy()
    invalid_transaction['age'] = 'thirty'
    with pytest.raises(ValidationError):
        DataSchema(**invalid_transaction)