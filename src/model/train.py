import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
from src.utils.logger import setup_logger
from src.utils.config import settings
from src.data.loader import load_data
from src.data.preprocessor import create_preprocessor, save_preprocessor

logger = setup_logger(__name__)

def train_model(data_path: str = settings.data_path):
    logger.info(f"Starting model training pipeline.")
    
    # 1. Load Data
    df = load_data(data_path)
    
    # 2. Preprocess Data
    logger.info("Initializing preprocessor.")
    preprocessor = create_preprocessor()
    
    logger.info("Fitting preprocessor and transforming data.")
    X_processed = preprocessor.fit_transform(df)
    
    # Save preprocessor
    save_preprocessor(preprocessor, settings.preprocessor_path)
    
    # 3. Train Model
    logger.info(f"Training Isolation Forest with contamination={settings.contamination_rate}")
    model = IsolationForest(
        contamination=settings.contamination_rate,
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_processed)
    
    # 4. Save Model
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    joblib.dump(model, settings.model_path)
    logger.info(f"Model saved to {settings.model_path}")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
