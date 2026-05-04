from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Finance Anomaly Detector"
    logging_level: str = "INFO"
    
    # Model config
    contamination_rate: float = 0.05  # Approximate percentage of anomalies
    model_path: str = "models/isolation_forest.joblib"
    preprocessor_path: str = "models/preprocessor.joblib"
    
    # Data config
    data_path: str = "data/transactions.csv"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", protected_namespaces=())

settings = Settings()
