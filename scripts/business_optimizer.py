"""
Business rule engine for fraud decisions
Applies domain-specific rules beyond ML predictions
"""

def apply_business_rules(transaction: Dict, fraud_prob: float, config: Dict) -> str:
    """
    Apply business rules to determine final decision
    
    Decision Framework:
    1. Auto-decline: 
       - Fraud probability > decline_threshold OR
       - High-risk country AND new account
       
    2. Review:
       - Fraud probability > review_threshold OR
       - High-value + new account OR
       - Velocity exceeds threshold
       
    3. Pass: All other transactions
    
    Returns: 'decline', 'review', or 'pass'
    """
    # Auto-decline rules
    if fraud_prob > config['decline_threshold']:
        return "decline"
    
    if (transaction.get('country') in config['high_risk_countries'] and
        transaction.get('time_since_signup', 100) < config['new_account_threshold']):
        return "decline"
    
    # Review rules
    if fraud_prob > config['review_threshold']:
        return "review"
    
    if (transaction.get('purchase_value', 0) > config['high_value_threshold'] and
        transaction.get('time_since_signup', 100) < config['new_account_threshold']):
        return "review"
    
    if transaction.get('txn_velocity', 0) > config['velocity_threshold']:
        return "review"
    
    return "pass"

def calculate_business_value(decisions: pd.DataFrame) -> Dict:
    """
    Calculate business value of fraud detection
    
    Metrics:
    - Fraud prevented value
    - Review costs
    - Net savings
    - False negative rate
    """
    fraud_prevented = decisions[
        (decisions['decision'] == 'decline') & 
        (decisions['actual_fraud'] == 1)
    ]['purchase_value'].sum()
    
    review_costs = len(decisions[decisions['decision'] == 'review']) * 5
    false_negatives = decisions[
        (decisions['decision'].isin(['pass', 'review'])) & 
        (decisions['actual_fraud'] == 1)
    ]['purchase_value'].sum()
    
    return {
        "fraud_prevented": fraud_prevented,
        "review_costs": review_costs,
        "false_negatives": false_negatives,
        "net_savings": fraud_prevented - review_costs - false_negatives
    }