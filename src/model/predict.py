import pandas as pd
import numpy as np
import joblib
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = setup_logger(__name__)

class AnomalyPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        try:
            self.model = joblib.load(settings.model_path)
            self.preprocessor = joblib.load(settings.preprocessor_path)
            logger.info("Successfully loaded model and preprocessor artifacts.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
            
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict if transactions are anomalous.
        Returns dictionary with predictions (1 for normal, -1 for anomaly)
        and anomaly scores (lower is more anomalous).
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        predictions = self.model.predict(X_processed) # 1 normal, -1 anomaly
        scores = self.model.decision_function(X_processed)
        
        # Convert predictions to boolean (True if anomaly, False if normal)
        is_anomaly = (predictions == -1).tolist()
        
        return {
            "is_anomaly": is_anomaly,
            "scores": scores.tolist()
        }
