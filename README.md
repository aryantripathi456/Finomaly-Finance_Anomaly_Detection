# Finance Anomaly Detector

ML-powered personal finance anomaly detection system built with Isolation Forest and SHAP explainability.

This project trains an unsupervised anomaly detector on transaction data, exposes predictions through a FastAPI service, and provides a Streamlit UI for interactive single/batch analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Run the API](#run-the-api)
- [Run the Streamlit Frontend](#run-the-streamlit-frontend)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Overview

The system detects unusual financial transactions using an `IsolationForest` model trained on engineered features:

- Numeric: `amount`, `hour_of_day`, `day_of_week`
- Categorical: `category`, `merchant` (one-hot encoded)

It supports:

- Synthetic data generation for local experimentation
- Offline model training and artifact persistence (`joblib`)
- Online inference through FastAPI
- Optional SHAP explanations per prediction
- A Streamlit dashboard for end users

## Features

- Unsupervised anomaly detection using scikit-learn `IsolationForest`
- Feature engineering pipeline with timestamp decomposition
- Preprocessing and model artifact saving/loading
- REST API for health checks, prediction, and background retraining
- SHAP-based local explanations for prediction interpretability
- Streamlit interface for single transaction and batch CSV scoring
- Unit/API tests with `pytest`

## Project Structure

```text
finance_anomaly_detector/
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app + health route
│   │   ├── routes.py             # /predict and /train endpoints
│   │   └── schemas.py            # Pydantic request/response models
│   ├── data/
│   │   ├── generate_synthetic_data.py
│   │   ├── loader.py
│   │   └── preprocessor.py       # FeatureEngineer + sklearn pipeline
│   ├── model/
│   │   ├── train.py              # Training pipeline
│   │   ├── predict.py            # Runtime predictor
│   │   └── explain.py            # SHAP explainer wrapper
│   ├── frontend/
│   │   └── app.py                # Streamlit UI
│   └── utils/
│       ├── config.py             # Settings via environment variables
│       └── logger.py             # Shared logging setup
├── tests/
├── data/
├── models/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Architecture

1. Data is loaded from CSV (`src/data/loader.py`).
2. Preprocessing pipeline (`src/data/preprocessor.py`) performs:
	 - Timestamp parsing
	 - Feature extraction (`hour_of_day`, `day_of_week`)
	 - Dropping non-feature columns (`transaction_id`, `is_anomaly`)
	 - Scaling numeric and one-hot encoding categorical fields
3. `src/model/train.py` fits Isolation Forest and saves:
	 - `models/isolation_forest.joblib`
	 - `models/preprocessor.joblib`
4. FastAPI startup initializes predictor/explainer from saved artifacts.
5. `/api/v1/predict` returns anomaly flags, scores, and optional SHAP contributions.

## Tech Stack

- Python 3.12 (Docker image base)
- FastAPI + Uvicorn
- scikit-learn
- pandas + numpy
- SHAP
- Streamlit
- pytest + httpx

## Quickstart

### 1. Clone and enter the repository

```bash
git clone <your-repo-url>
cd finance_anomaly_detector
```

### 2. Create environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python -m src.data.generate_synthetic_data
```

### 4. Train the model

```bash
python -m src.model.train
```

### 5. Start API server

```bash
uvicorn src.api.main:app --reload
```

### 6. Start Streamlit UI (new terminal)

```bash
source venv/bin/activate
streamlit run src/frontend/app.py
```

## Configuration

Configuration is managed through `src/utils/config.py` using `pydantic-settings`.

Supported environment variables:

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Finance Anomaly Detector` | FastAPI app title |
| `LOGGING_LEVEL` | `INFO` | Logging level |
| `CONTAMINATION_RATE` | `0.05` | Expected anomaly fraction for Isolation Forest |
| `MODEL_PATH` | `models/isolation_forest.joblib` | Trained model artifact path |
| `PREPROCESSOR_PATH` | `models/preprocessor.joblib` | Preprocessor artifact path |
| `DATA_PATH` | `data/transactions.csv` | Training data path |

Example `.env`:

```env
APP_NAME=Finance Anomaly Detector
LOGGING_LEVEL=INFO
CONTAMINATION_RATE=0.05
MODEL_PATH=models/isolation_forest.joblib
PREPROCESSOR_PATH=models/preprocessor.joblib
DATA_PATH=data/transactions.csv
```

## Data Pipeline

Synthetic data generation script:

```bash
python -m src.data.generate_synthetic_data
```

What it generates:

- Columns: `transaction_id`, `timestamp`, `amount`, `category`, `merchant`, `is_anomaly`
- Normal + anomalous patterns (high/low amount outliers)
- Timestamp values over the past year

Output file:

- `data/transactions.csv`

## Model Training

Train and save artifacts:

```bash
python -m src.model.train
```

Artifacts created:

- `models/isolation_forest.joblib`
- `models/preprocessor.joblib`

Training details:

- Algorithm: `IsolationForest`
- Parameters: `n_estimators=100`, `random_state=42`, configurable `contamination`
- Feature flow: engineered + transformed through sklearn pipeline

## Run the API

```bash
uvicorn src.api.main:app --reload
```

Default URL: `http://127.0.0.1:8000`

Interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Run the Streamlit Frontend

```bash
streamlit run src/frontend/app.py
```

Frontend capabilities:

- Single transaction scoring form
- Batch CSV upload and scoring
- Visual anomaly flag and score display
- Downloadable result CSV

Note: The frontend expects backend API at `http://localhost:8000/api/v1/predict`.

## API Reference

### Health Check

- Method: `GET`
- Path: `/health`
- Response:

```json
{
	"status": "healthy"
}
```

### Predict Anomalies

- Method: `POST`
- Path: `/api/v1/predict`
- Query param: `include_explanation` (default `true`)

Request body (array of transactions):

```json
[
	{
		"transaction_id": "txn_001",
		"timestamp": "2026-05-04T10:15:00",
		"amount": 4999.99,
		"category": "shopping",
		"merchant": "example_store"
	},
	{
		"transaction_id": "txn_002",
		"timestamp": "2026-05-04T11:30:00",
		"amount": 12.75,
		"category": "dining",
		"merchant": "coffee_shop"
	}
]
```

Response body:

```json
[
	{
		"transaction_id": "txn_001",
		"is_anomaly": true,
		"score": -0.142,
		"explanation": {
			"amount": 0.83,
			"merchant_example_store": -0.21
		}
	},
	{
		"transaction_id": "txn_002",
		"is_anomaly": false,
		"score": 0.087,
		"explanation": {
			"amount": -0.11
		}
	}
]
```

### Trigger Background Training

- Method: `POST`
- Path: `/api/v1/train`
- Response:

```json
{
	"message": "Model training triggered in background."
}
```

### Example curl Requests

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/train"
```

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict?include_explanation=true" \
	-H "Content-Type: application/json" \
	-d '[
		{
			"transaction_id": "txn_100",
			"timestamp": "2026-05-04T12:00:00",
			"amount": 8500.00,
			"category": "shopping",
			"merchant": "online_store"
		}
	]'
```

## Testing

Run all tests:

```bash
pytest -q
```

Test suite includes:

- API route tests (`tests/test_api.py`)
- Data generation and preprocessing tests (`tests/test_data.py`)
- Predictor and explainer tests (`tests/test_model.py`)

Note: Some model tests require trained artifacts in `models/`.

## Docker

Build image:

```bash
docker build -t finance-anomaly-detector .
```

Run container:

```bash
docker run --rm -p 8000:8000 finance-anomaly-detector
```

Then access:

- API docs: `http://127.0.0.1:8000/docs`
