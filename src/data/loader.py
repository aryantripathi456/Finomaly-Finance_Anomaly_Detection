import pandas as pd
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        
        # Convert timestamp strings to datetime objects
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
