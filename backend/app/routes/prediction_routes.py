"""
Prediction routes for Employee Retention Predictor.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from backend.app.services.ml_service import MLService
from backend.app.services.db_service import DBService
from backend.app.dependencies import get_db_service
from backend.app.api_schemas import PredictResponse, PredictBatchResponse, PredictionsListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Predictions"])

# Singleton ML service
ml_service = MLService()
_models_loaded = False


def get_ml_service():
    """Get ML service instance."""
    global _models_loaded
    if not _models_loaded:
        _models_loaded = ml_service.load_models()
    return ml_service


class EmployeeInput(BaseModel):
    """Employee input for attrition prediction."""
    employee_id: Optional[str] = Field(None, description="Employee identifier for tracking prediction history", examples=["EMP0001"])
    age: Optional[int] = Field(None, ge=18, le=99, description="Employee age", examples=[35])
    gender: Optional[str] = Field(None, description="Employee gender", examples=["Male"])
    department: Optional[str] = Field(None, description="Department name", examples=["Engineering"])
    job_role: Optional[str] = Field(None, description="Job role", examples=["Software Engineer"])
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income", examples=[75000.0])
    job_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Job satisfaction (1-5)", examples=[3])
    environment_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Environment satisfaction (1-5)", examples=[3])
    work_life_balance: Optional[int] = Field(None, ge=1, le=5, description="Work-life balance (1-5)", examples=[3])
    distance_from_home: Optional[int] = Field(None, ge=0, description="Distance from home (km)", examples=[15])
    years_at_company: Optional[int] = Field(None, ge=0, description="Years at company", examples=[5])
    years_since_last_promotion: Optional[int] = Field(None, ge=0, description="Years since last promotion", examples=[2])
    overtime: Optional[str] = Field(None, description="Overtime (Yes/No)", examples=["No"])
    performance_rating: Optional[int] = Field(None, ge=1, le=5, description="Performance rating (1-5)", examples=[3])
    training_times_last_year: Optional[int] = Field(None, ge=0, description="Training times last year", examples=[2])
    education: Optional[str] = Field(None, description="Education level", examples=["Bachelor's"])
    marital_status: Optional[str] = Field(None, description="Marital status", examples=["Married"])
    num_projects: Optional[int] = Field(None, ge=0, description="Number of projects", examples=[4])
    avg_monthly_hours: Optional[int] = Field(None, ge=0, description="Average monthly hours worked", examples=[200])
    promotion_last_5years: Optional[int] = Field(None, ge=0, le=1, description="Promoted in last 5 years (0/1)", examples=[0])
    salary_level: Optional[str] = Field(None, description="Salary level (low/medium/high)", examples=["medium"])
    tenure_years: Optional[int] = Field(None, ge=0, description="Tenure in years", examples=[5])
    experience_years: Optional[int] = Field(None, ge=0, description="Years of experience", examples=[10])

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "gender": "Male",
                "department": "Engineering",
                "job_role": "Software Engineer",
                "monthly_income": 75000,
                "job_satisfaction": 2,
                "environment_satisfaction": 3,
                "work_life_balance": 2,
                "distance_from_home": 15,
                "years_at_company": 5,
                "years_since_last_promotion": 3,
                "overtime": "Yes",
                "performance_rating": 4,
                "training_times_last_year": 1,
                "education": "Master's",
                "marital_status": "Married",
                "num_projects": 6,
                "avg_monthly_hours": 220,
                "promotion_last_5years": 0,
                "salary_level": "medium",
                "tenure_years": 5,
                "experience_years": 8,
            }
        }
    }


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Attrition",
    response_description="Attrition prediction with probability, risk level, SHAP explanation, and retention recommendations",
)
async def predict(
    input_data: EmployeeInput,
    service: MLService = Depends(get_ml_service),
    db_service: DBService = Depends(get_db_service),
):
    """
    Predict employee attrition probability.

    Takes employee attributes (age, department, satisfaction, income, overtime, etc.)
    and returns:
    - **attrition_probability**: 0-100% probability of leaving
    - **prediction**: "Likely To Stay" or "Likely To Leave"
    - **risk_level**: Low (0-30%), Medium (31-70%), or High (71-100%)
    - **confidence**: Model confidence score
    - **shap_explanation**: Feature contributions driving this prediction
    - **recommendations**: Actionable retention recommendations
    - **prediction_id**: Database ID of saved prediction

    The prediction is powered by a RandomForest model trained on 31 engineered features
    from 19,650 employee records. Predictions are automatically saved to the database.
    """
    try:
        data = input_data.model_dump(exclude_none=True)
        result = service.predict(data)

        if "error" in result and result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Save prediction to database
        try:
            pred = db_service.save_prediction(
                employee_id=None,
                attrition_probability=result["attrition_probability"],
                prediction_label=result["prediction"],
                risk_level=result["risk_level"],
                confidence=result.get("confidence"),
                model_name=service.model_name or "RandomForest",
                shap_contributions=result.get("shap_explanation"),
                input_data=data,
            )
            result["prediction_id"] = pred.id
        except Exception as e:
            logger.warning(f"Failed to save prediction to DB: {e}")

        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=PredictBatchResponse,
    summary="Batch Prediction",
    response_description="List of attrition predictions for each input employee",
)
async def predict_batch(
    employees: List[EmployeeInput],
    service: MLService = Depends(get_ml_service),
    db_service: DBService = Depends(get_db_service),
):
    """
    Predict attrition for multiple employees in a single request.

    Accepts an array of employee inputs and returns predictions for each one.
    Useful for evaluating risk across a team, department, or the entire organization.
    Each prediction is saved to the database individually.
    """
    results = []
    for emp in employees:
        data = emp.model_dump(exclude_none=True)
        result = service.predict(data)
        if "error" not in result:
            try:
                db_service.save_prediction(
                    employee_id=None,
                    attrition_probability=result["attrition_probability"],
                    prediction_label=result["prediction"],
                    risk_level=result["risk_level"],
                    model_name=service.model_name,
                    input_data=data,
                )
            except Exception as e:
                logger.warning(f"Failed to save batch prediction: {e}")
        results.append(result)
    return {"predictions": results}


@router.get(
    "/predictions",
    response_model=PredictionsListResponse,
    summary="List Predictions",
    response_description="Recent prediction history from the database",
)
async def get_predictions(
    limit: int = 50,
    db_service: DBService = Depends(get_db_service),
):
    """
    Get recent predictions stored in the database.

    Returns the most recent predictions with their input data snapshots,
    probability scores, predictions, risk levels, and timestamps.
    Use this to review prediction history and track changes over time.
    """
    predictions = db_service.get_recent_predictions(limit=limit)
    return {
        "predictions": [
            {
                "id": p.id,
                "attrition_probability": p.attrition_probability,
                "prediction": p.prediction,
                "risk_level": p.risk_level,
                "confidence": p.confidence,
                "model_name": p.model_name,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "input_data": p.input_data,
            }
            for p in predictions
        ]
    }
