"""
Employee Survey routes for the Employee Self-Assessment Portal.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from backend.app.dependencies import get_db_service
from backend.app.services.db_service import DBService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Employee Portal"])


# ── Pydantic Schemas ──

class SurveySubmitRequest(BaseModel):
    employee_id: str = Field(..., description="Unique employee identifier (e.g., EMP001)")
    employee_name: Optional[str] = Field(None, description="Employee display name")
    department: Optional[str] = Field(None, description="Employee department")
    job_role: Optional[str] = Field(None, description="Employee job role")
    job_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="1=Very Dissatisfied, 5=Very Satisfied")
    work_life_balance: Optional[int] = Field(None, ge=1, le=5, description="1=Very Poor, 5=Excellent")
    stress_level: Optional[int] = Field(None, ge=1, le=5, description="1=No Stress, 5=Severe Stress")
    career_growth_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="1=Strongly Disagree, 5=Strongly Agree")
    manager_relationship: Optional[int] = Field(None, ge=1, le=5, description="1=Very Poor, 5=Excellent")
    engagement_score: Optional[int] = Field(None, ge=1, le=5, description="1=Not Engaged, 5=Fully Engaged")
    feedback_comment: Optional[str] = Field(None, max_length=2000, description="Open-ended feedback")


class SurveyResponseOut(BaseModel):
    id: int
    employee_id: str
    employee_name: Optional[str] = None
    department: Optional[str] = None
    job_role: Optional[str] = None
    job_satisfaction: Optional[int] = None
    work_life_balance: Optional[int] = None
    stress_level: Optional[int] = None
    career_growth_satisfaction: Optional[int] = None
    manager_relationship: Optional[int] = None
    engagement_score: Optional[int] = None
    feedback_comment: Optional[str] = None
    survey_score: Optional[float] = None
    status: str = "completed"
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SurveySubmitResponse(BaseModel):
    id: int
    employee_id: str
    survey_score: Optional[float] = None
    message: str


# ── Helper ──

def compute_survey_score(data: SurveySubmitRequest) -> Optional[float]:
    """Compute an overall wellbeing score from survey ratings."""
    fields = [
        data.job_satisfaction,
        data.work_life_balance,
        data.stress_level,
        data.career_growth_satisfaction,
        data.manager_relationship,
        data.engagement_score,
    ]
    adjusted = []
    for i, value in enumerate(fields):
        if value is None:
            continue
        if i == 2:  # stress_level (index 2) — invert so 5→1, 1→5
            adjusted.append(6 - value)
        else:
            adjusted.append(value)
    if not adjusted:
        return None
    return round(sum(adjusted) / len(adjusted) * 20, 1)  # scale to 0-100


# ── Routes ──

@router.get("/api/survey/employee/{employee_id}", response_model=List[SurveyResponseOut])
async def get_employee_surveys(
    employee_id: str,
    limit: int = Query(20, ge=1, le=100),
    db_service: DBService = Depends(get_db_service),
):
    """Get survey response history for a specific employee."""
    responses = db_service.get_survey_responses(employee_id, limit=limit)
    return responses


@router.post("/api/employee-survey", response_model=SurveySubmitResponse)
async def submit_survey(
    data: SurveySubmitRequest,
    db_service: DBService = Depends(get_db_service),
):
    """
    Submit an employee self-assessment survey response.

    Collects employee wellbeing data including job satisfaction, work-life balance,
    stress level, career growth perception, and engagement. The data is stored
    and later merged with HR records for ML-based attrition prediction.
    """
    survey_score = compute_survey_score(data)

    record_data = {
        "employee_id": data.employee_id,
        "employee_name": data.employee_name,
        "department": data.department,
        "job_role": data.job_role,
        "job_satisfaction": data.job_satisfaction,
        "work_life_balance": data.work_life_balance,
        "stress_level": data.stress_level,
        "career_growth_satisfaction": data.career_growth_satisfaction,
        "manager_relationship": data.manager_relationship,
        "engagement_score": data.engagement_score,
        "feedback_comment": data.feedback_comment,
        "survey_score": survey_score,
        "status": "completed",
    }

    try:
        response = db_service.save_survey_response(record_data)
        return SurveySubmitResponse(
            id=response.id,
            employee_id=response.employee_id,
            survey_score=response.survey_score,
            message="Survey submitted successfully. Thank you for your feedback!",
        )
    except Exception as e:
        logger.error(f"Failed to save survey response: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit survey. Please try again.")


@router.get("/api/survey-responses", response_model=dict)
async def list_survey_responses(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db_service: DBService = Depends(get_db_service),
):
    """List all survey responses (paginated)."""
    responses, total = db_service.get_all_survey_responses(skip=skip, limit=limit)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "responses": [
            SurveyResponseOut.model_validate(r).model_dump() for r in responses
        ],
    }
