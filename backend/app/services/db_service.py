"""
Database service for querying employee data, predictions, and analytics
from the SQLite/Postgres database.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_, or_, text
from datetime import datetime, timezone

from backend.app.database.models import (
    Employee, Prediction, Recommendation, ModelMetric, TrainingRun, SurveyResponse
)


class DBService:
    """Service layer for database queries."""

    def __init__(self, db: Session):
        self.db = db

    # ── Employee Queries ──

    def get_employee_count(self) -> int:
        """Get total employee count."""
        return self.db.query(func.count(Employee.id)).scalar() or 0

    def get_attrition_count(self) -> int:
        """Get count of employees who left."""
        return self.db.query(func.count(Employee.id)).filter(
            Employee.attrition == 1
        ).scalar() or 0

    def get_retained_count(self) -> int:
        """Get count of employees who stayed."""
        return self.db.query(func.count(Employee.id)).filter(
            Employee.attrition == 0
        ).scalar() or 0

    def get_average_attrition_risk(self) -> float:
        """Get average attrition rate across all employees."""
        total = self.get_employee_count()
        if total == 0:
            return 0.0
        left = self.get_attrition_count()
        return round(left / total * 100, 1)

    def get_department_risk(self) -> List[Dict]:
        """Get risk breakdown by department."""
        results = self.db.query(
            Employee.department,
            func.count(Employee.id).label("employee_count"),
            func.avg(Employee.attrition).label("attrition_rate"),
            func.avg(Employee.job_satisfaction).label("avg_satisfaction"),
        ).group_by(Employee.department).having(
            Employee.department.isnot(None)
        ).all()

        departments = []
        for row in results:
            if not row.department:
                continue
            risk_score = round(float(row.attrition_rate or 0) * 100, 1)
            if risk_score > 60:
                risk_level = "High"
            elif risk_score > 30:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            departments.append({
                "department": row.department,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "employee_count": int(row.employee_count),
                "avg_satisfaction": round(float(row.avg_satisfaction or 0), 2),
            })

        departments.sort(key=lambda x: x["risk_score"], reverse=True)
        return departments

    def get_all_employees(self, skip: int = 0, limit: int = 100) -> List[Employee]:
        """Get paginated list of all employees."""
        return self.db.query(Employee).offset(skip).limit(limit).all()

    def get_employee_by_id(self, emp_id: int) -> Optional[Employee]:
        """Get a single employee by ID."""
        return self.db.query(Employee).filter(Employee.id == emp_id).first()

    # ── Prediction Queries ──

    def save_prediction(
        self,
        employee_id: Optional[int],
        attrition_probability: float,
        prediction_label: str,
        risk_level: str,
        confidence: Optional[float] = None,
        model_name: Optional[str] = None,
        shap_contributions: Optional[Dict] = None,
        input_data: Optional[Dict] = None,
    ) -> Prediction:
        """Save a prediction record to the database."""
        pred = Prediction(
            employee_id=employee_id,
            attrition_probability=attrition_probability,
            prediction=prediction_label,
            risk_level=risk_level,
            confidence=confidence,
            model_name=model_name,
            shap_contributions=shap_contributions,
            input_data=input_data,
        )
        self.db.add(pred)
        self.db.commit()
        self.db.refresh(pred)
        return pred

    def get_recent_predictions(self, limit: int = 50) -> List[Prediction]:
        """Get most recent predictions."""
        return self.db.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).limit(limit).all()

    # ── Analytics Queries ──

    def get_model_leaderboard(self) -> List[Dict]:
        """Get model leaderboard with metrics."""
        # Get the latest training run
        latest_run = self.db.query(TrainingRun).order_by(
            TrainingRun.created_at.desc()
        ).first()

        if not latest_run:
            return []

        metrics = self.db.query(ModelMetric).filter(
            ModelMetric.training_run_id == latest_run.id
        ).all()

        leaderboard = []
        for m in metrics:
            leaderboard.append({
                "model": m.model_name,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "roc_auc": m.roc_auc,
            })

        leaderboard.sort(key=lambda x: x["roc_auc"], reverse=True)
        return leaderboard

    def get_best_model_info(self) -> Optional[Dict]:
        """Get info about the best performing model."""
        leaderboard = self.get_model_leaderboard()
        if not leaderboard:
            return None
        best = leaderboard[0]
        return {
            "best_model": best["model"],
            "best_roc_auc": best["roc_auc"],
        }

    def get_feature_stats(self) -> Dict:
        """Get aggregate statistics about employee features."""
        stats = self.db.query(
            func.avg(Employee.age).label("avg_age"),
            func.avg(Employee.monthly_income).label("avg_income"),
            func.avg(Employee.job_satisfaction).label("avg_satisfaction"),
            func.avg(Employee.years_at_company).label("avg_tenure"),
            func.avg(Employee.work_life_balance).label("avg_wlb"),
        ).first()

        return {
            "avg_age": round(float(stats.avg_age or 0), 1),
            "avg_income": round(float(stats.avg_income or 0), 0),
            "avg_satisfaction": round(float(stats.avg_satisfaction or 0), 2),
            "avg_tenure_years": round(float(stats.avg_tenure or 0), 1),
            "avg_work_life_balance": round(float(stats.avg_wlb or 0), 2),
        }

    # ── Employee CRUD ──

    def create_employee(self, data: Dict[str, Any]) -> Employee:
        """Create a new employee record."""
        emp = Employee(**{k: v for k, v in data.items() if hasattr(Employee, k)})
        self.db.add(emp)
        self.db.commit()
        self.db.refresh(emp)
        return emp

    def update_employee(self, emp_id: int, data: Dict[str, Any]) -> Optional[Employee]:
        """Update an existing employee record."""
        emp = self.db.query(Employee).filter(Employee.id == emp_id).first()
        if not emp:
            return None
        for key, value in data.items():
            if hasattr(emp, key):
                setattr(emp, key, value)
        self.db.commit()
        self.db.refresh(emp)
        return emp

    def delete_employee(self, emp_id: int) -> bool:
        """Delete an employee record. Returns True if deleted."""
        emp = self.db.query(Employee).filter(Employee.id == emp_id).first()
        if not emp:
            return False
        self.db.delete(emp)
        self.db.commit()
        return True

    def search_employees(
        self,
        q: str = "",
        department: Optional[str] = None,
        gender: Optional[str] = None,
        overtime: Optional[str] = None,
        risk_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Employee], int]:
        """Search and filter employees. Returns (employees, total_count)."""
        query = self.db.query(Employee)

        if q:
            like = f"%{q}%"
            query = query.filter(
                or_(
                    Employee.employee_id.ilike(like),
                    Employee.department.ilike(like),
                    Employee.job_role.ilike(like),
                    Employee.gender.ilike(like),
                    Employee.education.ilike(like),
                    Employee.marital_status.ilike(like),
                )
            )
        if department:
            query = query.filter(Employee.department == department)
        if gender:
            query = query.filter(Employee.gender == gender)
        if overtime:
            query = query.filter(Employee.overtime == overtime)
        # Risk-level filtering is done client-side via ML batch predictions.
        # The DB doesn't store ML-predicted risk levels, so server-side
        # filtering by risk_level would be inaccurate.
        if risk_level:
            pass  # Handled client-side

        total = query.count()
        employees = query.order_by(Employee.id).offset(skip).limit(limit).all()
        return employees, total

    def get_high_risk_employees(self, limit: int = 50) -> List[Employee]:
        """Get employees most likely to leave based on features."""
        return self.db.query(Employee).filter(
            and_(
                Employee.job_satisfaction <= 2,
                Employee.attrition == 1,
            )
        ).order_by(Employee.job_satisfaction).limit(limit).all()

    def get_risk_distribution(self) -> Dict[str, int]:
        """
        Get feature-based risk distribution across ALL employees.

        Uses key attrition signals (job satisfaction, work-life balance, overtime)
        to categorize every employee into a risk bucket, so high+medium+low = total.
        """
        total = self.get_employee_count()

        # High risk: low satisfaction OR (overtime + poor WLB)
        high_count = self.db.query(func.count(Employee.id)).filter(
            or_(
                Employee.job_satisfaction <= 2,
                and_(Employee.overtime == 'Yes', Employee.work_life_balance <= 2),
            )
        ).scalar() or 0

        # Medium risk: moderate satisfaction (3) and not already high risk
        medium_count = self.db.query(func.count(Employee.id)).filter(
            Employee.job_satisfaction == 3,
            ~or_(
                Employee.job_satisfaction <= 2,
                and_(Employee.overtime == 'Yes', Employee.work_life_balance <= 2),
            ),
        ).scalar() or 0

        # Low risk: remaining (satisfaction >= 4)
        low_count = total - high_count - medium_count
        if low_count < 0:
            low_count = 0

        return {
            "total_employees": total,
            "high_risk": high_count,
            "medium_risk": medium_count,
            "low_risk": low_count,
        }

    def get_department_stats(self) -> List[Dict]:
        """Get detailed department statistics."""
        results = self.db.query(
            Employee.department,
            func.count(Employee.id).label("count"),
            func.avg(Employee.monthly_income).label("avg_income"),
            func.avg(Employee.age).label("avg_age"),
            func.avg(Employee.years_at_company).label("avg_tenure"),
            func.avg(Employee.job_satisfaction).label("avg_satisfaction"),
            func.sum(Employee.attrition).label("attrition_count"),
        ).group_by(Employee.department).having(
            Employee.department.isnot(None)
        ).all()

        departments = []
        for row in results:
            if not row.department:
                continue
            count = int(row.count)
            attrition_count = int(row.attrition_count or 0)
            departments.append({
                "department": row.department,
                "count": count,
                "attrition_count": attrition_count,
                "attrition_rate": round(attrition_count / count * 100, 1) if count > 0 else 0,
                "avg_income": round(float(row.avg_income or 0), 0),
                "avg_age": round(float(row.avg_age or 0), 1),
                "avg_tenure": round(float(row.avg_tenure or 0), 1),
                "avg_satisfaction": round(float(row.avg_satisfaction or 0), 2),
            })

        return sorted(departments, key=lambda x: x["attrition_rate"], reverse=True)

    # ── Survey Methods ──

    def save_survey_response(self, data: Dict[str, Any]) -> SurveyResponse:
        """Save an employee survey response."""
        resp = SurveyResponse(**{k: v for k, v in data.items() if hasattr(SurveyResponse, k)})
        self.db.add(resp)
        self.db.commit()
        self.db.refresh(resp)
        return resp

    def get_survey_responses(self, employee_id: str, limit: int = 20) -> List[SurveyResponse]:
        """Get survey responses for an employee."""
        return self.db.query(SurveyResponse).filter(
            SurveyResponse.employee_id == employee_id
        ).order_by(SurveyResponse.created_at.desc()).limit(limit).all()

    def get_all_survey_responses(self, skip: int = 0, limit: int = 50) -> tuple[List[SurveyResponse], int]:
        """Get paginated survey responses."""
        total = self.db.query(func.count(SurveyResponse.id)).scalar() or 0
        responses = self.db.query(SurveyResponse).order_by(
            SurveyResponse.created_at.desc()
        ).offset(skip).limit(limit).all()
        return responses, total
