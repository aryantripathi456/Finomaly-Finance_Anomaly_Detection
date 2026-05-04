import pandas as pd
from src.data.generate_synthetic_data import generate_transactions
from src.data.preprocessor import create_preprocessor

def test_generate_transactions():
    df = generate_transactions(n_samples=100)
    assert len(df) == 100
    assert 'amount' in df.columns
    assert 'is_anomaly' in df.columns

def test_preprocessor():
    df = generate_transactions(n_samples=50)
    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(df)
    
    assert X_processed is not None
    assert X_processed.shape[0] == 50
    # Output array with more columns due to one-hot encoding
    assert X_processed.shape[1] > 3 
