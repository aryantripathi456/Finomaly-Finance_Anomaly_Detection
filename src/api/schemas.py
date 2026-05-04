from pydantic import BaseModel, Field
from datetime import datetime

class Transaction(BaseModel):
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    timestamp: datetime = Field(..., description="Timestamp of the transaction")
    amount: float = Field(..., description="Amount of the transaction")
    category: str = Field(..., description="Category of the transaction")
    merchant: str = Field(..., description="Merchant of the transaction")

class PredictionResponse(BaseModel):
    transaction_id: str
    is_anomaly: bool
    score: float
    explanation: dict = Field(default_factory=dict, description="SHAP explanation for the prediction")
