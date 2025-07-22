import pytest
import pandas as pd
from scripts.pipeline import FraudDetectionPipeline

@pytest.fixture
def pipeline_config():
    return {
        'preprocessing': {
            'ip_country_path': 'data/raw/IpAddress_to_Country.csv',
            'min_purchase_value': 0.01,
            'max_purchase_value': 10000,
            'valid_age_range': (18, 100)
        },
        'feature_engineering': {
            'new_account_threshold': 2,
            'high_value_threshold': 500
        },
        'modeling': {
            'random_state': 42,
            'fp_cost': 5,
            'fn_cost': 250
        },
        'business_rules': {
            'decline_threshold': 0.9,
            'review_threshold': 0.7,
            'high_risk_countries': ['NG', 'RU'],
            'velocity_threshold': 5
        }
    }

def test_end_to_end_pipeline(pipeline_config):
    # Create pipeline
    pipeline = FraudDetectionPipeline(pipeline_config)
    
    # Train (using sample data in real implementation)
    # pipeline.train('data/raw/Fraud_Data.csv')
    
    # Test prediction
    transaction = {
        'transaction_id': 'T1',
        'user_id': 'U100',
        'purchase_value': 300,
        'signup_time': '2023-01-01 10:00:00',
        'purchase_time': '2023-01-01 10:05:00',
        'device_id': 'D1',
        'ip_address': '192.168.1.1',
        'browser': 'Chrome',
        'age': 30,
        'sex': 'M'
    }
    
    result = pipeline.predict(transaction)import pytest
import pandas as pd
from scripts.pipeline import FraudDetectionPipeline
from scripts.utils import load_config

@pytest.fixture
def pipeline_config():
    return load_config('config/pipeline_config.yaml')

@pytest.fixture
def sample_transaction():
    return {
        'transaction_id': 'T123',
        'user_id': 'U100',
        'purchase_value': 300,
        'signup_time': '2023-01-01 10:00:00',
        'purchase_time': '2023-01-01 10:05:00',
        'device_id': 'D1',
        'ip_address': '192.168.1.1',
        'browser': 'Chrome',
        'age': 30,
        'sex': 'M'
    }

def test_end_to_end_pipeline(pipeline_config, sample_transaction):
    # Create pipeline
    pipeline = FraudDetectionPipeline(pipeline_config)
    
    # Test prediction
    result = pipeline.predict(sample_transaction)
    
    assert 'decision' in result
    assert result['fraud_probability'] >= 0
    assert result['status'] == 'processed'
    
    # Test valid decisions
    assert result['decision'] in ['decline', 'review', 'pass']
    
    assert 'decision' in result
    assert result['status'] == 'processed'
    assert result['fraud_probability'] >= 0