"""
Employee & organization insight routes.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.app.dependencies import get_db_service
from backend.app.services.db_service import DBService
from backend.app.services.ml_service import MLService
from backend.app.routes.prediction_routes import get_ml_service
from backend.app.api_schemas import (
    EmployeeRiskResponse,
    DepartmentRiskResponse,
    OrganizationInsightsResponse,
    EmployeesResponse,
    EmployeeDetail as EmployeeDetailSchema,
    HighRiskEmployeesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Employee Analytics"])


@router.get(
    "/employee-risk",
    response_model=EmployeeRiskResponse,
    summary="Employee Risk Distribution",
    response_description="Risk distribution across the organization with high/medium/low breakdown",
)
async def get_employee_risk(
    db_service: DBService = Depends(get_db_service),
    ml_service: MLService = Depends(get_ml_service),
):
    """
    Get employee risk distribution from the database.

    Returns the total number of employees, how many are high/medium/low risk,
    the average attrition risk percentage, and a breakdown of the risk distribution.

    Falls back to ML service defaults if no data exists in the database.
    """
    dist = db_service.get_risk_distribution()
    total = dist["total_employees"]

    if total == 0:
        return ml_service._fallback_risk_distribution()

    avg_risk = round(dist["high_risk"] / total * 100, 1)

    return {
        "total_employees": total,
        "high_risk": dist["high_risk"],
        "medium_risk": dist["medium_risk"],
        "low_risk": dist["low_risk"],
        "high_risk_percentage": round(dist["high_risk"] / total * 100, 1),
        "average_attrition_risk": avg_risk,
        "risk_distribution": {
            "Low Risk (Stayed)": dist["low_risk"],
            "High Risk (Attrited)": dist["high_risk"],
        }
    }


@router.get(
    "/department-risk",
    response_model=DepartmentRiskResponse,
    summary="Department Risk Analysis",
    response_description="Risk scores for each department with highest risk identified",
)
async def get_department_risk(
    db_service: DBService = Depends(get_db_service),
    ml_service: MLService = Depends(get_ml_service),
):
    """
    Get department-level attrition risk analysis from database.

    Returns a list of departments with their risk scores, risk levels, and employee counts.
    Also identifies the highest-risk department across the organization.

    Falls back to ML service if no data in DB.
    """
    dept_risks = db_service.get_department_risk()

    # Fallback if no data
    if not dept_risks:
        dept_risks = ml_service.get_department_risk()

    return {
        "departments": dept_risks,
        "highest_risk_department": dept_risks[0]["department"] if dept_risks else None,
        "highest_risk_score": dept_risks[0]["risk_score"] if dept_risks else 0,
    }


@router.get(
    "/organization-insights",
    response_model=OrganizationInsightsResponse,
    summary="Organization Insights",
    response_description="Comprehensive organization-level attrition analysis with trends, department stats, and key insights",
)
async def get_organization_insights(
    db_service: DBService = Depends(get_db_service),
    ml_service: MLService = Depends(get_ml_service),
):
    """
    Get comprehensive organization-level attrition insights.

    Combines database data with ML forecasts to provide:
    - Attrition trend forecast and historical data
    - Department-level risk comparison
    - Overall organization risk score
    - Dynamic key insights derived from real data
    - Department statistics (count, attrition count, rate)
    - Aggregate feature statistics (avg age, income, satisfaction, tenure, work-life balance)
    """
    total = db_service.get_employee_count()
    left = db_service.get_attrition_count()
    avg_risk = round(left / total * 100, 1) if total > 0 else 35.2

    dept_risks = db_service.get_department_risk()
    dept_stats = db_service.get_department_stats()
    feature_stats = db_service.get_feature_stats()

    # Find highest risk department
    highest_dept = dept_risks[0]["department"] if dept_risks else "Unknown"
    highest_risk = dept_risks[0]["risk_score"] if dept_risks else 0

    # Generate dynamic insights from data
    key_insights = [
        f"{highest_dept} department has the highest attrition risk at {highest_risk}%",
    ]

    if feature_stats.get("avg_tenure_years", 0) > 3:
        key_insights.append(
            f"Average tenure is {feature_stats['avg_tenure_years']} years — monitor long-tenure retention"
        )
    else:
        key_insights.append("Employees with low tenure may need additional onboarding support")

    if feature_stats.get("avg_satisfaction", 0) < 3:
        key_insights.append(
            f"Average job satisfaction ({feature_stats['avg_satisfaction']}/5) indicates room for improvement"
        )
    else:
        key_insights.append(f"Job satisfaction is moderate ({feature_stats['avg_satisfaction']}/5)")

    key_insights.append(f"Total workforce: {total} employees with {avg_risk}% historical attrition rate")

    return {
        "attrition_trend": ml_service.get_attrition_forecast(),
        "department_comparison": dept_risks,
        "overall_risk_score": avg_risk,
        "trend_direction": "stable",
        "key_insights": key_insights,
        "department_stats": dept_stats,
        "feature_stats": feature_stats,
    }


@router.get(
    "/employees",
    response_model=EmployeesResponse,
    summary="List Employees",
    response_description="Paginated list of employees with total count",
)
async def get_employees(
    skip: int = Query(0, ge=0, description="Number of records to skip (offset)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of records to return"),
    db_service: DBService = Depends(get_db_service),
):
    """
    Get a paginated list of all employees.

    Supports pagination with `skip` and `limit` parameters.
    Returns basic employee info including ID, department, role, satisfaction, tenure, and attrition status.
    """
    employees = db_service.get_all_employees(skip=skip, limit=limit)
    total = db_service.get_employee_count()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "employees": [
            {
                "id": e.id,
                "employee_id": e.employee_id,
                "age": e.age,
                "gender": e.gender,
                "department": e.department,
                "job_role": e.job_role,
                "monthly_income": e.monthly_income,
                "job_satisfaction": e.job_satisfaction,
                "years_at_company": e.years_at_company,
                "education": e.education,
                "attrition": e.attrition,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in employees
        ],
    }


@router.get(
    "/employees/search",
    response_model=dict,
    summary="Search Employees",
    response_description="Filtered employee list with total count",
)
async def search_employees(
    q: str = Query("", description="Search query for ID, department, role, gender, education"),
    department: Optional[str] = Query(None, description="Filter by department"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    overtime: Optional[str] = Query(None, description="Filter by overtime (Yes/No)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (low/medium/high)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return"),
    db_service: DBService = Depends(get_db_service),
):
    """
    Search and filter employees.

    Supports full-text search across employee ID, department, role, gender,
    and education. Additional filters for department, gender, overtime, and risk level.
    Returns paginated results with total count.
    """
    employees, total = db_service.search_employees(
        q=q,
        department=department,
        gender=gender,
        overtime=overtime,
        risk_level=risk_level,
        skip=skip,
        limit=limit,
    )
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "employees": [
            {
                "id": e.id,
                "employee_id": e.employee_id,
                "age": e.age,
                "gender": e.gender,
                "department": e.department,
                "job_role": e.job_role,
                "monthly_income": e.monthly_income,
                "job_satisfaction": e.job_satisfaction,
                "work_life_balance": e.work_life_balance,
                "years_at_company": e.years_at_company,
                "years_since_last_promotion": e.years_since_last_promotion,
                "overtime": e.overtime,
                "education": e.education,
                "attrition": e.attrition,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in employees
        ],
    }


@router.get(
    "/employees/{employee_id}",
    response_model=EmployeeDetailSchema,
    summary="Get Employee",
    responses={
        200: {"description": "Employee details retrieved successfully"},
        404: {"description": "Employee not found"},
    },
)
async def get_employee(
    employee_id: int,
    db_service: DBService = Depends(get_db_service),
):
    """
    Get a single employee's complete details by database ID.

    Returns all available fields including satisfaction scores, work-life balance,
    performance rating, education, marital status, and attrition status.
    """
    emp = db_service.get_employee_by_id(employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {
        "id": emp.id,
        "employee_id": emp.employee_id,
        "age": emp.age,
        "gender": emp.gender,
        "department": emp.department,
        "job_role": emp.job_role,
        "monthly_income": emp.monthly_income,
        "job_satisfaction": emp.job_satisfaction,
        "environment_satisfaction": emp.environment_satisfaction,
        "work_life_balance": emp.work_life_balance,
        "distance_from_home": emp.distance_from_home,
        "years_at_company": emp.years_at_company,
        "years_since_last_promotion": emp.years_since_last_promotion,
        "overtime": emp.overtime,
        "performance_rating": emp.performance_rating,
        "education": emp.education,
        "marital_status": emp.marital_status,
        "attrition": emp.attrition,
    }


@router.get(
    "/high-risk-employees",
    response_model=HighRiskEmployeesResponse,
    summary="High-Risk Employees",
    response_description="List of employees with highest attrition risk signals",
)
async def get_high_risk_employees(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of high-risk employees to return"),
    db_service: DBService = Depends(get_db_service),
):
    """
    Get employees identified as having the highest attrition risk.

    Filters employees with low job satisfaction (≤2) and who have left (attrition=1).
    Returns focused info: ID, age, department, role, satisfaction, and tenure.
    """
    employees = db_service.get_high_risk_employees(limit=limit)
    return {
        "count": len(employees),
        "employees": [
            {
                "id": e.id,
                "employee_id": e.employee_id,
                "age": e.age,
                "department": e.department,
                "job_role": e.job_role,
                "job_satisfaction": e.job_satisfaction,
                "years_at_company": e.years_at_company,
            }
            for e in employees
        ],
    }


# ── Employee CRUD Schemas ──

class EmployeeCreate(BaseModel):
    """Schema for creating a new employee."""
    employee_id: Optional[str] = Field(None, description="Unique employee identifier")
    name: Optional[str] = Field(None, description="Employee name")
    age: Optional[int] = Field(None, ge=18, le=65, description="Employee age (18-65)")
    gender: Optional[str] = Field(None, description="Employee gender")
    department: Optional[str] = Field(None, description="Department name")
    job_role: Optional[str] = Field(None, description="Job role")
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income")
    job_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Job satisfaction (1-5)")
    environment_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Environment satisfaction (1-5)")
    work_life_balance: Optional[int] = Field(None, ge=1, le=5, description="Work-life balance (1-5)")
    distance_from_home: Optional[int] = Field(None, ge=0, description="Distance from home (km)")
    years_at_company: Optional[int] = Field(None, ge=0, description="Years at company")
    years_since_last_promotion: Optional[int] = Field(None, ge=0, description="Years since last promotion")
    overtime: Optional[str] = Field(None, description="Overtime (Yes/No)")
    performance_rating: Optional[int] = Field(None, ge=1, le=5, description="Performance rating (1-5)")
    training_times_last_year: Optional[int] = Field(None, ge=0, description="Training times last year")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    num_projects: Optional[int] = Field(None, ge=0, description="Number of projects")
    avg_monthly_hours: Optional[int] = Field(None, ge=0, description="Average monthly hours")
    promotion_last_5years: Optional[int] = Field(None, ge=0, le=1, description="Promoted in last 5 years")
    salary_level: Optional[str] = Field(None, description="Salary level (low/medium/high)")
    tenure_years: Optional[int] = Field(None, ge=0, description="Tenure in years")
    experience_years: Optional[int] = Field(None, ge=0, description="Years of experience")
    attrition: Optional[int] = Field(0, ge=0, le=1, description="0=Stayed, 1=Left")


class EmployeeUpdate(BaseModel):
    """Schema for updating an employee. All fields optional."""
    name: Optional[str] = Field(None, description="Employee name")
    age: Optional[int] = Field(None, ge=18, le=65, description="Employee age (18-65)")
    gender: Optional[str] = Field(None, description="Employee gender")
    department: Optional[str] = Field(None, description="Department name")
    job_role: Optional[str] = Field(None, description="Job role")
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income")
    job_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Job satisfaction (1-5)")
    environment_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Environment satisfaction (1-5)")
    work_life_balance: Optional[int] = Field(None, ge=1, le=5, description="Work-life balance (1-5)")
    distance_from_home: Optional[int] = Field(None, ge=0, description="Distance from home (km)")
    years_at_company: Optional[int] = Field(None, ge=0, description="Years at company")
    years_since_last_promotion: Optional[int] = Field(None, ge=0, description="Years since last promotion")
    overtime: Optional[str] = Field(None, description="Overtime (Yes/No)")
    performance_rating: Optional[int] = Field(None, ge=1, le=5, description="Performance rating (1-5)")
    training_times_last_year: Optional[int] = Field(None, ge=0, description="Training times last year")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    num_projects: Optional[int] = Field(None, ge=0, description="Number of projects")
    avg_monthly_hours: Optional[int] = Field(None, ge=0, description="Average monthly hours")
    promotion_last_5years: Optional[int] = Field(None, ge=0, le=1, description="Promoted in last 5 years")
    salary_level: Optional[str] = Field(None, description="Salary level (low/medium/high)")
    tenure_years: Optional[int] = Field(None, ge=0, description="Tenure in years")
    experience_years: Optional[int] = Field(None, ge=0, description="Years of experience")
    attrition: Optional[int] = Field(None, ge=0, le=1, description="0=Stayed, 1=Left")


# ── Employee CRUD Endpoints ──

@router.post(
    "/employees",
    response_model=dict,
    summary="Create Employee",
    response_description="Created employee record",
    status_code=201,
)
async def create_employee(
    data: EmployeeCreate,
    db_service: DBService = Depends(get_db_service),
):
    """
    Create a new employee record with all ML model features.

    Accepts employee details including personal info, employment info,
    satisfaction metrics, and compensation data. Returns the created record.
    """
    try:
        emp_data = data.model_dump(exclude_none=True)
        emp = db_service.create_employee(emp_data)
        return {
            "id": emp.id,
            "employee_id": emp.employee_id,
            "message": "Employee created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/employees/{employee_id}",
    response_model=dict,
    summary="Update Employee",
    response_description="Updated employee record",
)
async def update_employee(
    employee_id: int,
    data: EmployeeUpdate,
    db_service: DBService = Depends(get_db_service),
):
    """
    Update an existing employee record.

    All fields are optional. Only provided fields will be updated.
    Returns the updated employee data.
    """
    emp_data = data.model_dump(exclude_none=True)
    if not emp_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    emp = db_service.update_employee(employee_id, emp_data)
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {
        "id": emp.id,
        "employee_id": emp.employee_id,
        "message": "Employee updated successfully",
    }


@router.delete(
    "/employees/{employee_id}",
    response_model=dict,
    summary="Delete Employee",
    response_description="Deletion confirmation",
)
async def delete_employee(
    employee_id: int,
    db_service: DBService = Depends(get_db_service),
):
    """
    Delete an employee record from the database.

    Also removes associated predictions and recommendations.
    Returns a confirmation message.
    """
    deleted = db_service.delete_employee(employee_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found")
    return {"message": "Employee deleted successfully"}


@router.get(
    "/seed",
    summary="Seed Database",
    response_description="Seeding confirmation status",
)
async def seed_database():
    """
    Trigger database seeding from CSV files on the server.
    Useful for free tier instances where SSH/Shell console is unavailable.
    """
    try:
        from backend.scripts.seed_db import seed_from_employee_csv, seed_from_hr_churn_csv, seed_model_metrics
        from backend.app.database.connection import SessionLocal
        from backend.app.database.models import Base
        from pathlib import Path

        db = SessionLocal()
        
        # Clear existing data first
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(table.delete())
        db.commit()

        base_dir = Path(__file__).parent.parent.parent.parent
        employee_csv = base_dir / "Employee.csv"
        hr_churn_csv = base_dir / "hr_employee_churn_data.csv"

        total = 0

        # Seed from Employee.csv
        if employee_csv.exists():
            count = seed_from_employee_csv(db, str(employee_csv))
            total += count

        # Seed from hr_employee_churn_data.csv
        if hr_churn_csv.exists():
            count = seed_from_hr_churn_csv(db, str(hr_churn_csv))
            total += count

        # Seed model metrics
        seed_model_metrics(db)
        
        db.close()

        return {
            "status": "success",
            "message": f"Database successfully seeded with {total} employee records and leaderboard metrics.",
        }
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

