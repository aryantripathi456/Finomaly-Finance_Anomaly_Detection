# ML-Powered Personal Finance Anomaly Detector

This document outlines the architectural plan and implementation details for the ML-Powered Personal Finance Anomaly Detector.

## User Review Required

> [!IMPORTANT]
> Please review the proposed directory structure and the APIs for the ML model. Does this structure meet your expectations for a production-grade system? Are there any specific features or financial data fields you'd like to integrate? Let me know if you approve this plan to begin execution.

## Proposed Architecture & Folder Structure

We will adopt a modular structure to ensure the code is clean, production-ready, and independently testable. The architecture separates concerns into data handling, model training/inference, API routing, and utility services like logging and configuration.

### Directory Structure

```text
finance_anomaly_detector/
├── src/
│   ├── api/                  # FastAPI application and routing
│   │   ├── main.py           # Entry point for FastAPI
│   │   ├── routes.py         # API endpoints (e.g., /predict, /train)
│   │   └── schemas.py        # Pydantic models for request/response validation
│   ├── model/                # ML components
│   │   ├── train.py          # Model training pipeline (Isolation Forest/Autoencoder)
│   │   ├── predict.py        # Inference pipeline
│   │   └── explain.py        # SHAP integration for explainability
│   ├── data/                 # Data handling components
│   │   ├── loader.py         # Data loading utilities
│   │   └── preprocessor.py   # Feature engineering, scaling, encoding
│   ├── utils/                # Cross-cutting concerns
│   │   ├── logger.py         # Centralized structured logging
│   │   └── config.py         # Environment-based configuration (pydantic-settings)
├── tests/                    # Unit tests for each module
│   ├── test_api.py
│   ├── test_model.py
│   └── test_data.py
├── data/                     # Raw and processed datasets (ignored in git)
├── models/                   # Saved serialized models (ignored in git)
├── Dockerfile                # Docker configuration for deployment
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variables
└── README.md                 # Project documentation
```

### Key Components

1.  **FastAPI Application (`src/api`)**:
    *   Exposes endpoints for feeding transactions and detecting whether they are anomalous.
    *   Uses Pydantic schemas to validate incoming transaction data (e.g., amount, category, merchant, location, time).
2.  **ML Pipeline (`src/model` & `src/data`)**:
    *   **Algorithm**: We'll use an Unsupervised learning approach (e.g., `IsolationForest` from scikit-learn) since anomalies are typically rare and unlabeled in personal finance.
    *   **Preprocessing**: `preprocessor.py` will handle standardizing numerical features (amounts) and encoding categorical features (merchant type, category).
    *   **Explainability**: `explain.py` will use SHAP to provide a breakdown of *why* a specific transaction was flagged (e.g., "Amount significantly higher than usual for this category").
3.  **Cross-cutting & Best Practices**:
    *   **Configuration**: Environment values via `pydantic-settings` to avoid hardcoded values.
    *   **Logging**: A structured logger to log input payloads, model predictions, and any errors.
    *   **Reproducibility**: Setting constant random seeds for scikit-learn and using `joblib` or `pickle` for model versioning.

## Proposed Changes

If approved, I will sequentially:
1. Setup the workspace directory at `/home/aryan/.gemini/antigravity/scratch/finance_anomaly_detector`
2. Create the cross-cutting utility modules (logger, config).
3. Implement the data processing modules with Pytest coverage.
4. Implement the Scikit-learn model and SHAP explainers.
5. Create the FastAPI app and routes.
6. Package everything into a Docker container.

## Open Questions

1.  **Dataset Details**: Do you have a specific test dataset you would like to use for training the model, or should I generate a synthetic personal finance transaction dataset to demonstrate the complete pipeline?
2.  **Model Choice**: I am proposing `IsolationForest` as a starting point for unsupervised anomaly detection. Let me know if you have a different algorithm in mind.
3.  **Features**: Default features generally include `amount`, `transaction_timestamp`, `category`, and `merchant`. Are there context-specific features (e.g., location, user's average spending) you'd like to include?

## Verification Plan

### Automated Tests
*   Run Pytest to ensure data models, data preprocessor logic, and API endpoints function correctly in isolation. (`pytest tests/`)

### Docker Verification
*   Build the Docker container and run it locally to verify the containerization is robust.

### Manual Verification
*   Make queries to the FastAPI endpoints (`/predict`) utilizing both normal and synthetic anomalous transactions to view the predictions and SHAP explanations.
