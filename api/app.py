# api/app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import logging
import sys
import numpy as np
import shap # Required for SHAP explainability

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
RAW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

# --- Global ML Assets ---
ecommerce_model = None
ecommerce_preprocessor = None
ecommerce_feature_names = None # To store the ordered list of features expected by the model
ecommerce_optimal_threshold = 0.5 # Default, will be loaded/overridden

bank_model = None
bank_preprocessor = None
bank_feature_names = None # To store the ordered list of features expected by the model
bank_optimal_threshold = 0.5 # Default, will be loaded/overridden

# --- Initialize Helper Classes for Preprocessing ---
sys.path.insert(0, os.path.join(BASE_DIR, '..')) # Add project root to path for imports

from scripts.ip_to_country import IPtoCountryMapper
from scripts.data_cleaning import DataCleaner
from scripts.feature_engineering import FeatureEngineer
from models.model_explainability import ModelExplainer # For SHAP

data_cleaner = DataCleaner()
feature_engineer = FeatureEngineer()
# IPtoCountryMapper needs the direct path to IpAddress_to_Country.csv
ip_mapper = IPtoCountryMapper(os.path.join(RAW_DATA_DIR, 'IpAddress_to_Country.csv'))
model_explainer = ModelExplainer() # Initialize SHAP explainer utility

def load_ml_assets():
    """
    Loads the trained models, preprocessors, feature names, and optimal thresholds from disk.
    This function is called once on application startup.
    """
    global ecommerce_model, ecommerce_preprocessor, ecommerce_feature_names, ecommerce_optimal_threshold
    global bank_model, bank_preprocessor, bank_feature_names, bank_optimal_threshold

    logging.info("Attempting to load ML assets...")

    # --- Load E-commerce Assets ---
    # Assuming the best model is saved with a generic name like 'ecommerce_best_model.pkl'
    # Or you can load the specific tuned model you want to deploy.
    ecommerce_model_path = os.path.join(MODELS_DIR, 'ecommerce_best_model.pkl') # Or 'ecommerce_lgbm_model_tuned_{timestamp}.pkl'
    ecommerce_preprocessor_path = os.path.join(MODELS_DIR, 'ecommerce_preprocessor.pkl')
    ecommerce_features_path = os.path.join(MODELS_DIR, 'ecommerce_features.joblib')
    ecommerce_threshold_path = os.path.join(MODELS_DIR, 'ecommerce_optimal_threshold.joblib') # New path for threshold

    try:
        if os.path.exists(ecommerce_model_path) and os.path.exists(ecommerce_preprocessor_path) and \
           os.path.exists(ecommerce_features_path) and os.path.exists(ecommerce_threshold_path):
            ecommerce_model = joblib.load(ecommerce_model_path)
            ecommerce_preprocessor = joblib.load(ecommerce_preprocessor_path)
            ecommerce_feature_names = joblib.load(ecommerce_features_path)
            ecommerce_optimal_threshold = joblib.load(ecommerce_threshold_path)
            logging.info(f"✅ E-commerce assets loaded successfully. Optimal Threshold: {ecommerce_optimal_threshold:.4f}")
        else:
            logging.warning(f"⚠️ E-commerce assets not fully found. "
                            f"Model: {os.path.exists(ecommerce_model_path)}, "
                            f"Preprocessor: {os.path.exists(ecommerce_preprocessor_path)}, "
                            f"Features: {os.path.exists(ecommerce_features_path)}, "
                            f"Threshold: {os.path.exists(ecommerce_threshold_path)}. "
                            f"E-commerce API endpoints will be unavailable or use default threshold.")
            ecommerce_model = None
            ecommerce_preprocessor = None
            ecommerce_feature_names = None
            ecommerce_optimal_threshold = 0.5 # Fallback to default
    except Exception as e:
        logging.error(f"❌ Error loading E-commerce ML assets: {e}", exc_info=True)
        ecommerce_model = None
        ecommerce_preprocessor = None
        ecommerce_feature_names = None
        ecommerce_optimal_threshold = 0.5 # Fallback to default

    # --- Load Bank Assets ---
    bank_model_path = os.path.join(MODELS_DIR, 'bank_best_model.pkl') # Or 'bank_rf_model_tuned_{timestamp}.pkl'
    bank_preprocessor_path = os.path.join(MODELS_DIR, 'bank_preprocessor.pkl')
    bank_features_path = os.path.join(MODELS_DIR, 'bank_features.joblib')
    bank_threshold_path = os.path.join(MODELS_DIR, 'bank_optimal_threshold.joblib') # New path for threshold

    try:
        if os.path.exists(bank_model_path) and os.path.exists(bank_preprocessor_path) and \
           os.path.exists(bank_features_path) and os.path.exists(bank_threshold_path):
            bank_model = joblib.load(bank_model_path)
            bank_preprocessor = joblib.load(bank_preprocessor_path)
            bank_feature_names = joblib.load(bank_features_path)
            bank_optimal_threshold = joblib.load(bank_threshold_path)
            logging.info(f"✅ Bank assets loaded successfully. Optimal Threshold: {bank_optimal_threshold:.4f}")
        else:
            logging.warning(f"⚠️ Bank assets not fully found. "
                            f"Model: {os.path.exists(bank_model_path)}, "
                            f"Preprocessor: {os.path.exists(bank_preprocessor_path)}, "
                            f"Features: {os.path.exists(bank_features_path)}, "
                            f"Threshold: {os.path.exists(bank_threshold_path)}. "
                            f"Bank API endpoints will be unavailable or use default threshold.")
            bank_model = None
            bank_preprocessor = None
            bank_feature_names = None
            bank_optimal_threshold = 0.5 # Fallback to default
    except Exception as e:
        logging.error(f"❌ Error loading Bank ML assets: {e}", exc_info=True)
        bank_model = None
        bank_preprocessor = None
        bank_feature_names = None
        bank_optimal_threshold = 0.5 # Fallback to default

# Load models on app startup (for Gunicorn/WSGI, this runs once)
load_ml_assets()

# --- Preprocessing Functions for API Input ---

def preprocess_ecommerce_data_for_api(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw e-commerce transaction data received via API for model inference.
    Applies cleaning, feature engineering.
    The final ColumnTransformer (ecommerce_preprocessor) handles scaling and OHE.
    """
    df = pd.DataFrame([data])

    # Apply cleaning steps
    df = data_cleaner.correct_data_types(df.copy(), dataset_type='ecommerce')
    df = data_cleaner.handle_missing_values(df.copy())
    df = data_cleaner.remove_duplicates(df.copy())

    # Apply IP mapping
    df = ip_mapper.map_ips_to_countries(df.copy())

    # Apply feature engineering steps
    df = feature_engineer.create_time_based_features(df.copy())
    df = feature_engineer.create_user_transaction_frequency(df.copy())
    df = feature_engineer.create_transaction_amount_features(df.copy())
    
    # Add time-windowed features for user_id, device_id, ip_address
    df = feature_engineer.create_time_window_features(df.copy(), 
                                                      time_col='purchase_time', 
                                                      id_col='user_id', 
                                                      amount_col='purchase_value', 
                                                      time_windows=[1, 7, 30])
    df = feature_engineer.create_time_window_features(df.copy(), 
                                                      time_col='purchase_time', 
                                                      id_col='device_id', 
                                                      amount_col='purchase_value', 
                                                      time_windows=[1, 7, 30])
    df = feature_engineer.create_time_window_features(df.copy(), 
                                                      time_col='purchase_time', 
                                                      id_col='ip_address', 
                                                      amount_col='purchase_value', 
                                                      time_windows=[1, 7, 30])

    # Drop original raw columns and identifiers that are not features
    # This list must match the columns dropped *before* ColumnTransformer in data_pipeline.py
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'device_id']
    df_processed_initial = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Reindex to ensure all expected raw features are present before ColumnTransformer
    # This is crucial for the ColumnTransformer to work correctly.
    # We need the original feature names that went into the ColumnTransformer's `fit`.
    # This implies saving the *raw* feature names from X_train before transformation in data_pipeline.py
    # For now, let's assume the ColumnTransformer can handle missing columns by its `handle_unknown='ignore'`
    # in OneHotEncoder and `remainder='passthrough'`.
    # The `ecommerce_feature_names` loaded will be the *final* transformed feature names.

    # The actual transformation with ecommerce_preprocessor (ColumnTransformer) happens in the endpoint.
    return df_processed_initial

def preprocess_bank_data_for_api(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw bank transaction data received via API for model inference.
    The final ColumnTransformer (bank_preprocessor) handles scaling.
    """
    df = pd.DataFrame([data])

    # Apply cleaning steps
    df = data_cleaner.correct_data_types(df.copy(), dataset_type='bank')
    df = data_cleaner.handle_missing_values(df.copy())
    df = data_cleaner.remove_duplicates(df.copy())

    # Drop 'Class' if it somehow exists in the input (it shouldn't for prediction)
    df_processed_initial = df.drop(columns=['Class'], errors='ignore')

    # The actual transformation with bank_preprocessor (ColumnTransformer) happens in the endpoint.
    return df_processed_initial

# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Confirms the service and its ML assets are available."""
    status = {
        "status": "healthy",
        "ecommerce_model_loaded": ecommerce_model is not None and ecommerce_preprocessor is not None and ecommerce_feature_names is not None,
        "bank_model_loaded": bank_model is not None and bank_preprocessor is not None and bank_feature_names is not None
    }
    return jsonify(status), 200

@app.route('/predict_ecommerce_fraud', methods=['POST'])
def predict_ecommerce_fraud():
    """Predicts fraud for a single e-commerce transaction and provides SHAP explanations."""
    if not ecommerce_model or not ecommerce_preprocessor or not ecommerce_feature_names:
        logging.error("E-commerce model or preprocessor/feature names not loaded. Service unavailable.")
        return jsonify({"error": "Service unavailable: E-commerce model not loaded."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON input."}), 400

        # Preprocess the input data up to the point before ColumnTransformer
        df_for_transformer = preprocess_ecommerce_data_for_api(data)
        
        # Ensure the DataFrame has the correct columns for the preprocessor's input
        # This requires knowing the *raw* feature names that the ColumnTransformer expects.
        # This is a common challenge in API deployment.
        # A robust solution saves the list of raw features that went into ColumnTransformer.
        # For now, we rely on ColumnTransformer's `handle_unknown='ignore'` and `remainder='passthrough'`.
        
        if df_for_transformer.empty:
            raise ValueError("Processed data DataFrame is empty after initial steps.")

        # Apply the fitted preprocessor (ColumnTransformer)
        processed_data_array = ecommerce_preprocessor.transform(df_for_transformer)
        
        # Ensure processed_data_array is 2D, even for single instance
        if processed_data_array.ndim == 1:
            processed_data_array = processed_data_array.reshape(1, -1)

        prediction_proba = ecommerce_model.predict_proba(processed_data_array)[:, 1][0]
        
        # Use the loaded optimal threshold
        is_fraud = int(prediction_proba >= ecommerce_optimal_threshold)

        # --- SHAP Explanation ---
        top_shap_features = []
        try:
            # Convert processed_data_array back to DataFrame for SHAP, using the final feature names
            X_processed_df_for_shap = pd.DataFrame(processed_data_array, columns=ecommerce_feature_names)
            
            # Determine model type for SHAP
            model_type_shap = 'tree_based' if 'LGBMClassifier' in str(type(ecommerce_model)) or 'XGBClassifier' in str(type(ecommerce_model)) else 'linear'
            
            # Calculate SHAP values for this single instance
            explainer = shap.TreeExplainer(ecommerce_model) if model_type_shap == 'tree_based' else shap.LinearExplainer(ecommerce_model, X_processed_df_for_shap)
            
            instance_shap_values = explainer.shap_values(X_processed_df_for_shap.iloc[0])
            if isinstance(instance_shap_values, list) and len(instance_shap_values) == 2:
                instance_shap_values = instance_shap_values[1] # SHAP values for the positive class (fraud)

            abs_shap_values = np.abs(instance_shap_values)
            sorted_indices = np.argsort(abs_shap_values)[::-1] # Descending order
            
            top_features_indices = sorted_indices[:5] # Top 5 features
            for i in top_features_indices:
                feature_name = ecommerce_feature_names[i]
                shap_value = float(instance_shap_values[i])
                top_shap_features.append({"feature": feature_name, "value": shap_value})

        except Exception as shap_e:
            logging.error(f"Error calculating SHAP values for e-commerce prediction: {shap_e}", exc_info=True)
            # Continue without SHAP values if there's an error

        return jsonify({
            "user_id": data.get("user_id", "N/A"),
            "is_fraud": is_fraud,
            "fraud_probability": float(prediction_proba),
            "fraud_threshold_used": float(ecommerce_optimal_threshold),
            "shap_explanation": {
                "top_features": top_shap_features,
                "base_value": float(explainer.expected_value) if 'explainer' in locals() else None
            }
        })
    except ValueError as ve:
        logging.error(f"Invalid input data for e-commerce prediction: {ve}", exc_info=True)
        return jsonify({"error": f"Invalid input data: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"An internal error occurred during e-commerce prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route('/predict_bank_fraud', methods=['POST'])
def predict_bank_fraud():
    """Predicts fraud for a single bank transaction and provides SHAP explanations."""
    if not bank_model or not bank_preprocessor or not bank_feature_names:
        logging.error("Bank model or preprocessor/feature names not loaded. Service unavailable.")
        return jsonify({"error": "Service unavailable: Bank model not loaded."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON input."}), 400

        df_for_transformer = preprocess_bank_data_for_api(data)

        if df_for_transformer.empty:
            raise ValueError("Processed data DataFrame is empty after initial steps.")

        processed_data_array = bank_preprocessor.transform(df_for_transformer)
        
        if processed_data_array.ndim == 1:
            processed_data_array = processed_data_array.reshape(1, -1)

        prediction_proba = bank_model.predict_proba(processed_data_array)[:, 1][0]
        
        # Use the loaded optimal threshold
        is_fraud = int(prediction_proba >= bank_optimal_threshold)

        # --- SHAP Explanation ---
        top_shap_features = []
        try:
            X_processed_df_for_shap = pd.DataFrame(processed_data_array, columns=bank_feature_names)
            
            model_type_shap = 'tree_based' if 'LGBMClassifier' in str(type(bank_model)) or 'XGBClassifier' in str(type(bank_model)) else 'linear'
            
            explainer = shap.TreeExplainer(bank_model) if model_type_shap == 'tree_based' else shap.LinearExplainer(bank_model, X_processed_df_for_shap)
            instance_shap_values = explainer.shap_values(X_processed_df_for_shap.iloc[0])
            if isinstance(instance_shap_values, list) and len(instance_shap_values) == 2:
                instance_shap_values = instance_shap_values[1] # SHAP values for the positive class (fraud)

            abs_shap_values = np.abs(instance_shap_values)
            sorted_indices = np.argsort(abs_shap_values)[::-1]
            
            top_features_indices = sorted_indices[:5] # Top 5 features
            for i in top_features_indices:
                feature_name = bank_feature_names[i]
                shap_value = float(instance_shap_values[i])
                top_shap_features.append({"feature": feature_name, "value": shap_value})

        except Exception as shap_e:
            logging.error(f"Error calculating SHAP values for bank prediction: {shap_e}", exc_info=True)

        return jsonify({
            "transaction_id": data.get("transaction_id", "N/A"),
            "is_fraud": is_fraud,
            "fraud_probability": float(prediction_proba),
            "fraud_threshold_used": float(bank_optimal_threshold),
            "shap_explanation": {
                "top_features": top_shap_features,
                "base_value": float(explainer.expected_value) if 'explainer' in locals() else None
            }
        })
    except ValueError as ve:
        logging.error(f"Invalid input data for bank prediction: {ve}", exc_info=True)
        return jsonify({"error": f"Invalid input data: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"An internal error occurred during bank prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    # In a production environment, use a WSGI server like Gunicorn (gunicorn -w 4 app:app)
    # debug=True should NEVER be used in production.
    app.run(debug=False, host='0.0.0.0', port=5000)
