import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scripts.model import FraudModel

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=2, 
        weights=[0.95, 0.05],
        random_state=42
    )
    return X, y

@pytest.fixture
def model_config():
    return {
        'fp_cost': 5.0,
        'fn_cost': 250.0,
        'review_cost': 3.5
    }

def test_model_training(sample_data, model_config):
    """Test model training and threshold optimization"""
    X, y = sample_data
    model = FraudModel(model_config)
    model.train(X, y)
    
    assert model.threshold > 0.3
    assert model.threshold < 0.9
    assert hasattr(model, 'feature_importances_')

def test_prediction(sample_data, model_config):
    """Test prediction functionality"""
    X, y = sample_data
    model = FraudModel(model_config)
    model.train(X, y)
    
    probas = model.predict_proba(X[:10])
    predictions = model.predict(X[:10])
    
    assert probas.shape == (10,)
    assert predictions.shape == (10,)
    assert np.all((predictions == 0) | (predictions == 1))

def test_evaluation(sample_data, model_config):
    """Test business-aligned evaluation"""
    X, y = sample_data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    model = FraudModel(model_config)
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    assert 'precision' in results
    assert 'recall' in results
    assert 'business_savings' in results
    assert results['business_savings'] > -10000  # Shouldn't be catastrophic loss