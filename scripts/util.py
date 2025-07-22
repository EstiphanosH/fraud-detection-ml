"""
Utility functions for fraud detection system
"""

import yaml
import json
import logging
import pandas as pd
from typing import Dict, Any

def configure_logging(level=logging.INFO, log_file=None):
    """Configure standardized logging format"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    # Suppress excessive logging from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data: dict, path: str):
    """Save data to JSON file with indentation"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_business_impact(y_true, y_pred, fp_cost=5, fn_cost=250):
    """
    Calculate business impact of fraud detection results
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        fp_cost: Cost of false positive ($)
        fn_cost: Cost of false negative ($)
    
    Returns:
        Dictionary with business metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "fraud_prevented": tp * fn_cost,
        "review_costs": fp * fp_cost,
        "false_negatives": fn * fn_cost,
        "net_savings": (tp * fn_cost) - (fp * fp_cost) - (fn * fn_cost),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

def prepare_api_data(transaction: Dict) -> Dict:
    """Prepare transaction data for API input"""
    return {
        "transaction_id": transaction.get("transaction_id", ""),
        "user_id": transaction["user_id"],
        "purchase_value": float(transaction["purchase_value"]),
        "signup_time": transaction["signup_time"],
        "purchase_time": transaction["purchase_time"],
        "device_id": transaction["device_id"],
        "ip_address": transaction["ip_address"],
        "browser": transaction["browser"],
        "age": int(transaction["age"]),
        "sex": transaction["sex"]
    }

def validate_transaction(transaction: Dict) -> bool:
    """Basic validation for transaction data"""
    required_fields = [
        'user_id', 'purchase_value', 'purchase_time',
        'device_id', 'browser', 'age', 'sex'
    ]
    for field in required_fields:
        if field not in transaction:
            return False
    return True