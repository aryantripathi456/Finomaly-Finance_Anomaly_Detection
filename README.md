# Finance Anomaly Detector

ML-powered personal finance anomaly detection system using IsolationForest and SHAP explainability.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic data:
```bash
python -m src.data.generate_synthetic_data
```

3. Train model:
```bash
python -m src.model.train
```

4. Run API Server:
```bash
uvicorn src.api.main:app --reload
```

5. Run Streamlit Frontend (in a new terminal):
```bash
source venv/bin/activate
streamlit run src/frontend/app.py
```

## Docker

You can run the application via Docker:
```bash
docker build -t finance-anomaly-detector .
docker run -p 8000:8000 finance-anomaly-detector
```

## API Endpoints
- `POST /api/v1/predict`
- `POST /api/v1/train`
- `GET /health`
