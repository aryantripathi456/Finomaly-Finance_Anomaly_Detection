from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import pandas as pd
from src.api.schemas import Transaction, PredictionResponse
from src.model.predict import AnomalyPredictor
from src.model.explain import Explainer
from src.model.train import train_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize models at startup
predictor = None
explainer = None

@router.on_event("startup")
async def startup_event():
    global predictor, explainer
    try:
        predictor = AnomalyPredictor()
        explainer = Explainer()
    except Exception as e:
        logger.error(f"Failed to initialize models on startup: {e}")

@router.post("/predict", response_model=List[PredictionResponse])
async def predict_anomalies(transactions: List[Transaction], include_explanation: bool = True):
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
        
    data = [t.model_dump() for t in transactions]
    df = pd.DataFrame(data)
    
    try:
        results = predictor.predict(df)
        
        explanations = []
        if include_explanation:
            if explainer is None or explainer.model is None:
                raise HTTPException(status_code=503, detail="Explainer is not initialized.")
            explanations = explainer.explain(df)
        else:
            explanations = [{} for _ in range(len(transactions))]
            
        response = []
        for i, txn in enumerate(transactions):
            response.append(PredictionResponse(
                transaction_id=txn.transaction_id,
                is_anomaly=results["is_anomaly"][i],
                score=results["scores"][i],
                explanation=explanations[i]
            ))
            
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(train_model)
        return {"message": "Model training triggered in background."}
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail=str(e))
