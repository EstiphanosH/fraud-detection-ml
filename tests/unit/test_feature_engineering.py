import pytest
import pandas as pd
from datetime import datetime, timedelta
from scripts.feature_engineering import FraudFeatureEngineer

@pytest.fixture
def sample_ecom_data():
    return pd.DataFrame({
        'user_id': ['U1', 'U1', 'U2', 'U3'],
        'purchase_value': [100, 150, 200, 300],
        'signup_time': [
            datetime(2023, 1, 1, 10),
            datetime(2023, 1, 1, 10),
            datetime(2023, 1, 1, 11),
            datetime(2023, 1, 1, 9)
        ],
        'purchase_time': [
            datetime(2023, 1, 1, 10, 5),
            datetime(2023, 1, 1, 10, 30),
            datetime(2023, 1, 1, 11, 15),
            datetime(2023, 1, 1, 9, 30)
        ],
        'country': ['US', 'US', 'NG', 'GB'],
        'class': [0, 0, 1, 0]
    })

@pytest.fixture
def sample_bank_data():
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 10800],
        'V1': [1.2, -0.5, 3.1, -2.0],
        'V2': [0.5, -1.2, 0.8, 1.5],
        'Amount': [100, 50, 500, 1000],
        'Class': [0, 1, 0, 1]
    })

@pytest.fixture
def config():
    return {
        'feature_engineering': {
            'high_risk_countries': ['NG', 'RU'],
            'high_value_threshold': 500,
            'new_account_threshold': 2
        }
    }

def test_ecom_feature_engineering(sample_ecom_data, config):
    engineer = FraudFeatureEngineer('ecommerce', config['feature_engineering'])
    features = engineer.transform(sample_ecom_data)
    
    assert 'time_since_signup' in features.columns
    assert 'high_risk_country' in features.columns
    assert features['high_risk_country'].sum() == 1
    assert features['time_since_signup'].min() > 0

def test_bank_feature_engineering(sample_bank_data, config):
    engineer = FraudFeatureEngineer('bank', config['feature_engineering'])
    features = engineer.transform(sample_bank_data)
    
    assert 'hour_of_day' in features.columns
    assert 'is_night' in features.columns
    assert 'log_amount' in features.columns
    assert features['hour_of_day'].max() < 24