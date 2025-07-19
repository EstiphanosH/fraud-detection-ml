"""
Fraud detection model with business-optimized training and evaluation
Includes cost-sensitive learning and threshold optimization
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

logger = logging.getLogger(__name__)

class FraudModel:
    """
    Business-optimized fraud detection model with:
    
    Key Features:
    - Cost-sensitive learning with business-defined weights
    - Threshold optimization for precision-recall tradeoff
    - SHAP explainability for business stakeholders
    - Model performance monitoring
    
    Business Considerations:
    - False Negative Cost: $250 (average fraud amount)
    - False Positive Cost: $5 (manual review cost)
    - Target precision: >85% to minimize operational costs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.threshold = 0.5
        self.explainer = None
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train model with business-optimized parameters"""
        logger.info("Training fraud detection model")
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.config['random_state']
        )
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(
            sampling_strategy='auto',
            random_state=self.config['random_state']
        )
        
        # Build model pipeline
        self.model = ImbPipeline([
            ('smote', smote),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                max_depth=12,
                random_state=self.config['random_state'],
                n_jobs=-1
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        self.feature_names = X.columns.tolist()
        
        # Optimize decision threshold
        self._optimize_threshold(X_val, y_val)
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model.named_steps['classifier'])
        
        logger.info("Model training complete")
        return self
    
    def _optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Find optimal threshold based on business costs"""
        logger.info("Optimizing decision threshold")
        
        # Get predicted probabilities
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate costs for different thresholds
        thresholds = np.linspace(0.1, 0.9, 50)
        costs = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            
            # Business cost calculation
            cost = (
                fp * self.config['fp_cost'] + 
                fn * self.config['fn_cost']
            )
            costs.append(cost)
        
        # Find threshold with minimum cost
        min_idx = np.argmin(costs)
        self.threshold = thresholds[min_idx]
        
        logger.info(f"Optimal threshold: {self.threshold:.3f} with cost ${costs[min_idx]:.2f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud with optimized threshold"""
        y_proba = self.model.predict_proba(X)[:, 1]
        return (y_proba >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def explain(self, X: pd.DataFrame) -> Dict:
        """Generate SHAP explanations for business users"""
        if self.explainer is None:
            raise RuntimeError("Model not trained")
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Format for business interpretation
        return {
            "expected_value": self.explainer.expected_value[1],
            "shap_values": shap_values[1].tolist(),
            "feature_names": self.feature_names
        }
    
    def save(self, path: str):
        """Save model with metadata"""
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'explainer': self.explainer,
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load trained model"""
        data = joblib.load(path)
        model = cls(data['config'])
        model.model = data['model']
        model.threshold = data['threshold']
        model.feature_names = data['feature_names']
        model.explainer = data['explainer']
        return model