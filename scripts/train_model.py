"""
Business-optimized modeling with:
- Cost-sensitive learning
- Automated threshold optimization
- Comprehensive evaluation
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc
from imblearn.ensemble import BalancedRandomForestClassifier

class FraudModel:
    """
    Optimized fraud detector with business constraints
    Implements custom thresholding based on financial impact
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = BalancedRandomForestClassifier(
            n_estimators=200,
            sampling_strategy='all',
            replacement=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.calibrator = CalibratedClassifierCV(self.model, cv=5, method='isotonic')
        self.threshold = 0.5
        self.feature_importances_ = None
        
    def _optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Find optimal threshold based on financial impact"""
        y_proba = self.calibrator.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 100)
        costs = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            cost = fp * self.config['fp_cost'] + fn * self.config['fn_cost']
            costs.append(cost)
            
        self.threshold = thresholds[np.argmin(costs)]
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive business-aligned evaluation"""
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= self.threshold).astype(int)
        
        # Standard metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pr_auc = auc(precision_recall_curve(y_test, y_proba))
        
        # Business metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        savings = (tp * self.config['fn_cost']) - (fp * self.config['fp_cost'])
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "business_savings": savings,
            "threshold": self.threshold
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train with calibration and threshold optimization"""
        self.calibrator.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        
        # Optimize threshold using validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train
        )
        self._optimize_threshold(X_val, y_val)