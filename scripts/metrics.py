"""
Real-time business metrics tracking for fraud detection
Monitors key performance indicators and financial impact
"""

from prometheus_client import start_http_server, Gauge, Counter, Summary
import time

class BusinessMetrics:
    """
    Tracks business-critical metrics for fraud detection:
    
    Key Metrics:
    - Fraud prevented value ($)
    - False positive costs ($)
    - Model performance degradation
    - System latency
    - Error rates
    
    Business Goals:
    - Maintain precision > 85%
    - Keep false positive costs < $10k/month
    - Achieve 99.9% system uptime
    """
    
    def __init__(self):
        # Financial metrics
        self.fraud_prevented = Gauge('fraud_prevented_usd', 
                                     'Value of prevented fraud in USD')
        self.false_positive_costs = Gauge('false_positive_costs_usd', 
                                          'Cost of false positive reviews in USD')
        self.net_savings = Gauge('net_savings_usd', 
                                 'Net savings from fraud prevention')
        
        # Performance metrics
        self.prediction_latency = Summary('prediction_latency_seconds', 
                                          'Prediction latency in seconds')
        self.model_precision = Gauge('model_precision', 
                                     'Current model precision')
        self.model_recall = Gauge('model_recall', 
                                  'Current model recall')
        
        # Operational metrics
        self.prediction_count = Counter('predictions_total', 
                                        'Total predictions processed')
        self.error_count = Counter('prediction_errors_total', 
                                   'Total prediction errors')
        
        # Start metrics server
        start_http_server(8000)
    
    def update_prediction(self, transaction: dict, result: dict, actual: dict = None):
        """
        Update metrics based on prediction results
        
        :param transaction: Raw transaction data
        :param result: Prediction result from model
        :param actual: Actual fraud label (if available)
        """
        # Track prediction volume
        self.prediction_count.inc()
        
        # Update financial metrics
        if result['decision'] == 'decline':
            self.fraud_prevented.inc(transaction['purchase_value'])
            
        if result['decision'] == 'review':
            self.false_positive_costs.inc(self.config['review_cost'])
            
        # Calculate net savings
        current_prevented = self.fraud_prevented._value.get()
        current_costs = self.false_positive_costs._value.get()
        self.net_savings.set(current_prevented - current_costs)
        
        # Update model performance (if actual label available)
        if actual:
            precision, recall = self._calculate_performance(result, actual)
            self.model_precision.set(precision)
            self.model_recall.set(recall)
    
    def update_performance(self, X_test, y_test, model):
        """Periodically update model metrics"""
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        self.model_precision.set(precision)
        self.model_recall.set(recall)
    
    def log_latency(self, latency: float):
        """Record prediction latency"""
        self.prediction_latency.observe(latency)
    
    def log_error(self):
        """Increment error counter"""
        self.error_count.inc()
    
    def _calculate_performance(self, result, actual):
        """Calculate precision/recall from actual outcomes"""
        # Implementation for online learning systems
        # In production, this would update from ground truth feedback
        return 0.85, 0.75  # Placeholder