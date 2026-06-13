"""
Model management routes for Employee Retention Predictor.
"""

import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks

from backend.app.services.ml_service import MLService
from backend.app.services.db_service import DBService
from backend.app.dependencies import get_db_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Model Management"])

ml_service = MLService()


def get_ml_service():
    ml_service.load_models()
    return ml_service


@router.post(
    "/upload-dataset",
    response_model=dict,
    summary="Upload Dataset",
    response_description="Dataset upload confirmation with row/column counts",
)
async def upload_dataset(file: UploadFile = File(..., description="CSV file to upload for training")):
    """
    Upload a CSV dataset for model training.

    The file is saved to the `ml_pipeline/data/uploads/` directory.
    Basic validation is performed (CSV parsing, row/column counting).
    Returns the filename, number of rows, columns, and column names.
    """
    try:
        # Save uploaded file
        upload_dir = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Basic validation
        import pandas as pd
        df = pd.read_csv(file_path)
        rows, cols = df.shape

        return {
            "status": "success",
            "filename": file.filename,
            "rows": rows,
            "columns": cols,
            "column_names": list(df.columns),
            "message": f"Dataset uploaded successfully. {rows} rows, {cols} columns."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/train-model",
    response_model=dict,
    summary="Train Model",
    response_description="Training trigger confirmation",
)
async def train_model(
    background_tasks: BackgroundTasks,
    use_hyperopt: bool = True,
):
    """
    Trigger model retraining in the background.

    Runs the full training pipeline (preprocessing → feature engineering → hyperparameter
    tuning → model training → evaluation). By default uses hyperparameter optimization
    with 10 iterations of RandomizedSearchCV.

    Training runs asynchronously in the background. Check server logs for progress.

    Parameters:
    - **use_hyperopt** (bool, default=True): Whether to use RandomizedSearchCV for hyperparameter tuning
    """
    try:
        from ml_pipeline.train_pipeline import TrainingPipeline

        def train():
            pipeline = TrainingPipeline()
            pipeline.run(use_hyperopt=use_hyperopt, n_iter=10)
            # Reload ML service
            ml_service.load_models()

        background_tasks.add_task(train)

        return {
            "status": "started",
            "message": "Model training started in background. Check logs for progress."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/model-metrics",
    response_model=dict,
    summary="Model Metrics",
    response_description="Model leaderboard with performance metrics for all trained models",
)
async def get_model_metrics(
    db_service: DBService = Depends(get_db_service),
):
    """
    Get model evaluation metrics and leaderboard.

    Returns a comparison of all trained models (RandomForest, XGBoost, LightGBM,
    LogisticRegression) with their accuracy, precision, recall, F1 score, and ROC-AUC.
    Identifies the best performing model.

    Queries the database first, falls back to the file-based model leaderboard.
    """
    # Try DB first
    leaderboard = db_service.get_model_leaderboard()
    best_info = db_service.get_best_model_info()

    if leaderboard:
        return {
            "leaderboard": leaderboard,
            "best_model": best_info["best_model"] if best_info else None,
            "best_roc_auc": best_info["best_roc_auc"] if best_info else None,
            "timestamp": None,
        }

    # Fallback to file-based metrics
    metrics = ml_service.get_model_metrics()
    return metrics


@router.get(
    "/feature-importance",
    response_model=dict,
    summary="Feature Importance",
    response_description="SHAP-based global feature importance for the best model",
)
async def get_feature_importance():
    """
    Get SHAP-based global feature importance.

    Returns the features that most influence the model's predictions, ranked by importance.
    Based on TreeExplainer analysis from the explainability module.
    """
    service = get_ml_service()
    importance = service.get_feature_importance()
    return importance


@router.get(
    "/model-info",
    response_model=dict,
    summary="Model Info",
    response_description="Current loaded model status and metadata",
)
async def get_model_info():
    """
    Get information about the currently loaded model.

    Returns whether the model is loaded, its name, and full metadata including
    the 31 training feature names, best ROC-AUC, and training timestamp.
    """
    service = get_ml_service()
    return {
        "model_loaded": service._is_loaded,
        "model_name": service.model_name,
        "model_meta": service.model_meta,
    }
