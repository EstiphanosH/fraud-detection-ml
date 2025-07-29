# tests/test_api.py

import unittest
import json
import os
import joblib
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the Flask app (assuming it's in api/app.py relative to project root)
# Adjust import based on how you run tests. If running from project root:
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.app import app, load_ml_assets, MODELS_DIR

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a test client for the Flask app
        cls.app = app.test_client()
        cls.app.testing = True

        # Create mock models directory
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Create dummy/mock models for testing
        # These are simple mocks and don't represent real trained models
        # For actual model tests, you'd load proper dummy models or mock joblib.load
        
        # Create mock models
        mock_ecommerce_model = MagicMock()
        mock_ecommerce_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        joblib.dump(mock_ecommerce_model, os.path.join(MODELS_DIR, 'ecommerce_model.joblib'))  # Updated name

        mock_bank_model = MagicMock()
        mock_bank_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        joblib.dump(mock_bank_model, os.path.join(MODELS_DIR, 'bank_model.joblib'))  # Updated name

        # Reload ML assets to pick up mock models
        with app.app_context(): # Ensure app context for loading
            load_ml_assets()

    @classmethod
    def tearDownClass(cls):
        # Clean up mock models
        ecommerce_model_path = os.path.join(MODELS_DIR, 'ecommerce_model.joblib')
        bank_model_path = os.path.join(MODELS_DIR, 'bank_model.joblib')
        if os.path.exists(ecommerce_model_path):
            os.remove(ecommerce_model_path)
        if os.path.exists(bank_model_path):
            os.remove(bank_model_path)
        # Clean up models directory if empty
        if not os.listdir(MODELS_DIR):
            os.rmdir(MODELS_DIR)


    def test_ecommerce_fraud_prediction_success(self):
        # Example valid e-commerce data (should match the features expected by your preprocess_ecommerce_data)
        valid_ecommerce_data = {
            "user_id": 1001,
            "signup_time": "2023-01-01 10:00:00",
            "purchase_time": "2023-01-01 10:30:00",
            "purchase_value": 500,
            "device_id": "device_XYZ",
            "source": "SEO",
            "browser": "Chrome",
            "sex": "M",
            "age": 30,
            "ip_address": "192.168.1.1"
        }
        response = self.app.post('/predict_ecommerce_fraud', json=valid_ecommerce_data)
        response_data = json.loads(response.data)

        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}. Response: {response.data}")
        self.assertIn('is_fraud_probability', response_data)
        self.assertIn('is_fraud', response_data)
        self.assertEqual(response_data['user_id'], 1001)
        self.assertGreater(response_data['is_fraud_probability'], 0.5) # Based on mock model output
        self.assertEqual(response_data['is_fraud'], 1)
        valid_ecommerce_data["country"] = "US"  
        response = self.app.post('/predict_ecommerce_fraud', json=valid_ecommerce_data)

    def test_bank_fraud_prediction_success(self):
        # Example valid bank data (V1-V28, Time, Amount)
        valid_bank_data = {
            "Time": 1.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
            "V10": 0.090794, "V11": -0.551600, "V12": -0.617801, "V13": -0.991390, "V14": -0.311169,
            "V15": 1.468177, "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
            "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62,
            "transaction_id": "bank_tx_12345" # Added for context
        }
        response = self.app.post('/predict_bank_fraud', json=valid_bank_data)
        response_data = json.loads(response.data)

        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}. Response: {response.data}")
        self.assertIn('is_fraud_probability', response_data)
        self.assertIn('is_fraud', response_data)
        self.assertEqual(response_data['transaction_id'], "bank_tx_12345")
        self.assertLess(response_data['is_fraud_probability'], 0.5) # Based on mock model output
        self.assertEqual(response_data['is_fraud'], 0)
        # Ensure all 28 V-features are present
        for i in range(1, 29):
            valid_bank_data[f"V{i}"] = 0.5  # Add mock values
        response = self.app.post('/predict_bank_fraud', json=valid_bank_data)
    def test_invalid_json_input(self):
        response = self.app.post('/predict_ecommerce_fraud', data="this is not json", content_type='text/plain')
        response_data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response_data)
        self.assertIn('Invalid JSON input.', response_data['error'])

    def test_missing_ecommerce_model(self):
        # Temporarily remove the model to test 503
        ecommerce_model_path = os.path.join(MODELS_DIR, 'ecommerce_fraud_model.joblib')
        temp_ecommerce_model = joblib.load(ecommerce_model_path) # Load to restore later
        os.remove(ecommerce_model_path)
        
        with app.app_context():
            load_ml_assets() # Reload to make it aware the model is gone

        response = self.app.post('/predict_ecommerce_fraud', json={})
        response_data = json.loads(response.data)
        self.assertEqual(response.status_code, 503)
        self.assertIn('error', response_data)
        self.assertIn('E-commerce model not loaded', response_data['error'])

        # Restore the model
        joblib.dump(temp_ecommerce_model, ecommerce_model_path)
        with app.app_context():
            load_ml_assets() # Restore model in app context


    def test_missing_bank_model(self):
        # Temporarily remove the model to test 503
        bank_model_path = os.path.join(MODELS_DIR, 'bank_fraud_model.joblib')
        temp_bank_model = joblib.load(bank_model_path) # Load to restore later
        os.remove(bank_model_path)
        
        with app.app_context():
            load_ml_assets() # Reload to make it aware the model is gone

        response = self.app.post('/predict_bank_fraud', json={})
        response_data = json.loads(response.data)
        self.assertEqual(response.status_code, 503)
        self.assertIn('error', response_data)
        self.assertIn('Bank model not loaded', response_data['error'])

        # Restore the model
        joblib.dump(temp_bank_model, bank_model_path)
        with app.app_context():
            load_ml_assets() # Restore model in app context