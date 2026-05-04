import pytest
from src.data.generate_synthetic_data import generate_transactions
from src.model.predict import AnomalyPredictor
from src.model.explain import Explainer

def test_predictor_initialization():
    predictor = AnomalyPredictor()
    assert predictor.model is not None
    assert predictor.preprocessor is not None

def test_prediction():
    predictor = AnomalyPredictor()
    df = generate_transactions(n_samples=10)
    
    results = predictor.predict(df)
    
    assert "is_anomaly" in results
    assert "scores" in results
    assert len(results["is_anomaly"]) == 10
    assert len(results["scores"]) == 10

def test_explainer():
    explainer = Explainer()
    df = generate_transactions(n_samples=2)
    
    explanations = explainer.explain(df)
    
    assert len(explanations) == 2
    assert isinstance(explanations[0], dict)
