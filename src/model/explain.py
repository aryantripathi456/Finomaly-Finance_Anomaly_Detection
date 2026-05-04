import shap
import pandas as pd
import numpy as np
import joblib
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = setup_logger(__name__)

class Explainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        try:
            self.model = joblib.load(settings.model_path)
            self.preprocessor = joblib.load(settings.preprocessor_path)
            logger.info("Successfully loaded artifacts for explainer.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            # Do not raise yet; user might just be importing
            
    def _get_feature_names(self):
        # We need to extract feature names from the ColumnTransformer
        try:
            numeric_features = self.preprocessor.named_steps['preprocessor'].transformers_[0][2]
            categorical_features = self.preprocessor.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            return list(numeric_features) + list(categorical_features)
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return None
            
    def explain(self, df: pd.DataFrame) -> list:
        if self.model is None:
            self._load_artifacts()
            
        X_processed = self.preprocessor.transform(df)
        
        # Initialize SHAP explainer
        if self.explainer is None:
            # TreeExplainer is fast for IsolationForest
            self.explainer = shap.TreeExplainer(self.model)
            
        shap_values = self.explainer.shap_values(X_processed)
        
        feature_names = self._get_feature_names()
        
        explanations = []
        for i in range(len(X_processed)):
            # Pair feature names with SHAP values
            feature_contributions = {}
            if feature_names is not None:
                for idx, val in enumerate(shap_values[i]):
                     # Filter out zero contributions for cleaner output
                     if abs(val) > 0.001:
                         feature_contributions[feature_names[idx]] = float(val)
            else:
                 feature_contributions = {f"Feature_{idx}": float(val) for idx, val in enumerate(shap_values[i])}
                 
            # Sort by absolute contribution
            sorted_contributions = dict(sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True))
            explanations.append(sorted_contributions)
            
        return explanations
