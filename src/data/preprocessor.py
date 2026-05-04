import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        
        # Ensure we have a DataFrame and convert timestamp appropriately if we get raw dicts/records
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if 'timestamp' in X.columns:
            # We assume 'timestamp' might be strings or datetime objects already
            # Convert specifically handling ISO8601 strings
            if X['timestamp'].dtype == 'O':
                 X['timestamp'] = pd.to_datetime(X['timestamp'])
            
            # Adding time-based features
            X['hour_of_day'] = X['timestamp'].dt.hour
            X['day_of_week'] = X['timestamp'].dt.dayofweek
            # Drop timestamp as we have extracted features
            X.drop('timestamp', axis=1, inplace=True)
            
        # Drop IDs or label columns during preprocessing for model input
        cols_to_drop = [col for col in ['transaction_id', 'is_anomaly'] if col in X.columns]
        if cols_to_drop:
            X.drop(cols_to_drop, axis=1, inplace=True)
            
        return X

def create_preprocessor() -> Pipeline:
    numeric_features = ['amount', 'hour_of_day', 'day_of_week']
    categorical_features = ['category', 'merchant']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

def save_preprocessor(preprocessor: Pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Saved preprocessor to {path}")

def load_preprocessor(path: str) -> Pipeline:
    preprocessor = joblib.load(path)
    logger.info(f"Loaded preprocessor from {path}")
    return preprocessor
