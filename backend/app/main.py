"""
FastAPI Main Application for Employee Retention Predictor.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.routes.prediction_routes import router as prediction_router
from backend.app.routes.model_routes import router as model_router
from backend.app.routes.employee_routes import router as employee_router
from backend.app.routes.survey_routes import router as survey_router
from backend.app.database.connection import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize database on startup."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    yield


app = FastAPI(
    title="Employee Retention Predictor API",
    description="""
    ML-powered API for predicting employee attrition and generating retention recommendations.

    ## Features
    - **Attrition Prediction** — Predict the probability of an employee leaving using a trained RandomForest model
    - **Batch Predictions** — Evaluate attrition risk for multiple employees at once
    - **Employee Analytics** — Get risk distributions, department-level breakdowns, and organization insights
    - **Model Management** — Upload datasets, retrain models, and view performance metrics
    - **SHAP Explainability** — Understand which features drive each prediction
    - **Retention Recommendations** — Get actionable suggestions to reduce attrition risk

    ## ML Pipeline
    The API uses a **RandomForest classifier** trained on 19,650 employee records with 31 engineered features.
    The model achieves **97.5% ROC-AUC** and **88.1% accuracy** on held-out test data.
    """.strip(),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Employee Retention Predictor Team",
        "url": "https://github.com/employee-retention-predictor",
    },
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Local development server"},
    ],
    terms_of_service="https://github.com/employee-retention-predictor/terms",
)

# Configure OpenAPI tags
app.openapi_tags = [
    {
        "name": "Predictions",
        "description": "Predict employee attrition risk for individuals or batches. Includes SHAP explanations and retention recommendations.",
    },
    {
        "name": "Employee Analytics",
        "description": "Analytics endpoints for employee risk distribution, department-level risk, organization insights, and employee records.",
    },
    {
        "name": "Model Management",
        "description": "Manage ML models: upload training datasets, trigger retraining, view metrics, and inspect feature importance.",
    },
]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction_router)
app.include_router(model_router)
app.include_router(employee_router)
app.include_router(survey_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Employee Retention Predictor API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "POST /api/predict": "Predict employee attrition",
            "POST /api/upload-dataset": "Upload training dataset",
            "POST /api/train-model": "Train/retrain model",
            "GET /api/model-metrics": "Get model performance metrics",
            "GET /api/feature-importance": "Get SHAP feature importance",
            "GET /api/employee-risk": "Get employee risk distribution",
            "GET /api/department-risk": "Get department risk analysis",
            "GET /api/organization-insights": "Get organization insights",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "employee-retention-predictor"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
