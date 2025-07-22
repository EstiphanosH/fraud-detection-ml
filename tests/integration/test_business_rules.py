import pytest
from scripts.business_optimizer import apply_business_rules, calculate_business_value

@pytest.fixture
def business_config():
    return {
        'decline_threshold': 0.9,
        'review_threshold': 0.7,
        'high_risk_countries': ['NG', 'RU', 'BY'],
        'new_account_threshold': 2,  # hours
        'high_value_threshold': 500,  # dollars
        'velocity_threshold': 5  # transactions/hour
    }

test_cases = [
    (0.95, {}, 'decline'),
    (0.85, {'country': 'NG', 'time_since_signup': 1.5}, 'decline'),
    (0.85, {'purchase_value': 600, 'time_since_signup': 1.0}, 'review'),
    (0.65, {'txn_velocity': 6}, 'review'),
    (0.65, {'purchase_value': 100, 'time_since_signup': 24}, 'pass'),
    (0.75, {'country': 'US'}, 'review')
]

@pytest.mark.parametrize("fraud_prob,transaction,expected", test_cases)
def test_business_rules(fraud_prob, transaction, expected, business_config):
    decision = apply_business_rules(transaction, fraud_prob, business_config)
    assert decision == expected

def test_business_value_calculation():
    decisions = pd.DataFrame({
        'decision': ['decline', 'review', 'pass', 'decline'],
        'actual_fraud': [1, 0, 1, 0],
        'purchase_value': [300, 150, 500, 200]
    })
    
    results = calculate_business_value(decisions)
    
    assert results['fraud_prevented'] == 300
    assert results['review_costs'] == 5 * 1
    assert results['false_negatives'] == 500
    assert results['net_savings'] == 300 - 5 - 500