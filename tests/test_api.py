import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    payload = [
        {
            "transaction_id": "test_txn_1",
            "timestamp": "2023-10-01T12:00:00",
            "amount": 1500.0,
            "category": "rent",
            "merchant": "landlord"
        },
        {
            "transaction_id": "test_txn_2",
            "timestamp": "2023-10-02T15:30:00",
            "amount": 5.50,
            "category": "dining",
            "merchant": "starbucks"
        }
    ]
    
    response = client.post("/api/v1/predict?include_explanation=true", json=payload)
    assert response.status_code == 200
    
    results = response.json()
    assert len(results) == 2
    assert "is_anomaly" in results[0]
    assert "explanation" in results[0]
