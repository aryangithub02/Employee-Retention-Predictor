"""
Shared Pydantic response schemas with examples for all API endpoints.
Used to generate rich OpenAPI documentation in ReDoc/Swagger UI.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ─── Root & Health ─────────────────────────────────────────────

class RootResponse(BaseModel):
    """Root endpoint response."""
    service: str = Field("Employee Retention Predictor API", description="Service name")
    version: str = Field("1.0.0", description="API version")
    status: str = Field("operational", description="Service status")
    docs: str = Field("/docs", description="Swagger UI URL")
    endpoints: Dict[str, str] = Field(
        ...,
        description="Available endpoints",
        example={
            "POST /api/predict": "Predict employee attrition",
            "GET /api/employee-risk": "Get employee risk distribution",
            "GET /api/department-risk": "Get department risk analysis",
            "GET /api/organization-insights": "Get organization insights",
        },
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Health status")
    service: str = Field("employee-retention-predictor", description="Service name")


# ─── Employee Risk ─────────────────────────────────────────────

class RiskDistribution(BaseModel):
    """Risk distribution breakdown."""
    low_risk_stayed: int = Field(..., alias="Low Risk (Stayed)", description="Employees who stayed")
    high_risk_attrited: int = Field(..., alias="High Risk (Attrited)", description="Employees who left")


class EmployeeRiskResponse(BaseModel):
    """Employee risk distribution response."""
    total_employees: int = Field(19650, description="Total number of employees", ge=0)
    high_risk: int = Field(..., description="Number of high-risk employees", ge=0)
    medium_risk: int = Field(0, description="Number of medium-risk employees", ge=0)
    low_risk: int = Field(..., description="Number of low-risk employees", ge=0)
    high_risk_percentage: float = Field(..., description="Percentage of high-risk employees", ge=0, le=100)
    average_attrition_risk: float = Field(..., description="Average attrition risk percentage", ge=0, le=100)
    risk_distribution: RiskDistribution = Field(..., description="Risk distribution breakdown")


# ─── Department Risk ───────────────────────────────────────────

class DepartmentRisk(BaseModel):
    """Department-level risk data."""
    department: str = Field(..., description="Department name", example="Sales")
    risk_score: float = Field(..., description="Risk score percentage", ge=0, le=100, example=32.7)
    risk_level: str = Field(..., description="Risk level category", example="Medium")
    employee_count: int = Field(..., description="Number of employees in department", ge=0)


class DepartmentRiskResponse(BaseModel):
    """Department risk analysis response."""
    departments: List[DepartmentRisk] = Field(..., description="Department risk list")
    highest_risk_department: Optional[str] = Field(None, description="Department with highest risk")
    highest_risk_score: Optional[float] = Field(None, description="Highest risk score", ge=0, le=100)


# ─── Organization Insights ─────────────────────────────────────

class AttritionTrend(BaseModel):
    """Attrition trend data."""
    forecast: List[Any] = Field(default_factory=list, description="Forecasted attrition values")
    historical: List[Any] = Field(default_factory=list, description="Historical attrition values")


class DepartmentStat(BaseModel):
    """Department statistics."""
    department: str = Field(..., description="Department name")
    count: int = Field(..., description="Employee count")
    attrition_count: int = Field(..., description="Attrition count")
    attrition_rate: float = Field(..., description="Attrition rate percentage")


class FeatureStats(BaseModel):
    """Aggregate feature statistics."""
    avg_age: Optional[float] = Field(None, description="Average age")
    avg_income: Optional[float] = Field(None, description="Average monthly income")
    avg_satisfaction: Optional[float] = Field(None, description="Average job satisfaction")
    avg_tenure_years: Optional[float] = Field(None, description="Average tenure in years")
    avg_work_life_balance: Optional[float] = Field(None, description="Average work-life balance")


class OrganizationInsightsResponse(BaseModel):
    """Organization-level attrition insights response."""
    attrition_trend: AttritionTrend = Field(..., description="Attrition trend forecast")
    department_comparison: List[DepartmentRisk] = Field(..., description="Department risk comparison")
    overall_risk_score: float = Field(..., description="Overall organization risk score", ge=0, le=100)
    trend_direction: str = Field("stable", description="Trend direction", example="stable")
    key_insights: List[str] = Field(..., description="Key insights derived from data")
    department_stats: List[DepartmentStat] = Field(default_factory=list, description="Department statistics")
    feature_stats: FeatureStats = Field(default_factory=FeatureStats, description="Feature statistics")


# ─── Employees ─────────────────────────────────────────────────

class EmployeeSummary(BaseModel):
    """Brief employee record for list view."""
    id: int = Field(..., description="Database ID", example=1)
    employee_id: str = Field(..., description="Human-readable employee ID", example="EMP0001")
    age: Optional[int] = Field(None, description="Employee age", example=35)
    gender: Optional[str] = Field(None, description="Employee gender", example="Male")
    department: Optional[str] = Field(None, description="Department name", example="Engineering")
    job_role: Optional[str] = Field(None, description="Job role", example="Software Engineer")
    monthly_income: Optional[float] = Field(None, description="Monthly income", example=75000.0)
    job_satisfaction: Optional[int] = Field(None, description="Job satisfaction (1-5)", example=3)
    years_at_company: Optional[int] = Field(None, description="Years at company", example=5)
    education: Optional[str] = Field(None, description="Education level", example="Bachelor's")
    attrition: Optional[int] = Field(None, description="0=Stayed, 1=Left", example=0)
    created_at: Optional[str] = Field(None, description="Record creation timestamp")


class EmployeesResponse(BaseModel):
    """Paginated employee list response."""
    total: int = Field(..., description="Total number of employees", example=19650)
    skip: int = Field(..., description="Offset for pagination", example=0)
    limit: int = Field(..., description="Page size", example=100)
    employees: List[EmployeeSummary] = Field(..., description="List of employees")


class EmployeeDetail(BaseModel):
    """Full employee record for detail view."""
    id: int = Field(..., description="Database ID")
    employee_id: str = Field(..., description="Human-readable employee ID")
    age: Optional[int] = Field(None, description="Employee age")
    gender: Optional[str] = Field(None, description="Employee gender")
    department: Optional[str] = Field(None, description="Department name")
    job_role: Optional[str] = Field(None, description="Job role")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    job_satisfaction: Optional[int] = Field(None, description="Job satisfaction (1-5)")
    environment_satisfaction: Optional[int] = Field(None, description="Environment satisfaction (1-5)")
    work_life_balance: Optional[int] = Field(None, description="Work-life balance (1-5)")
    distance_from_home: Optional[int] = Field(None, description="Distance from home (km)")
    years_at_company: Optional[int] = Field(None, description="Years at company")
    years_since_last_promotion: Optional[int] = Field(None, description="Years since last promotion")
    overtime: Optional[str] = Field(None, description="Overtime (Yes/No)")
    performance_rating: Optional[int] = Field(None, description="Performance rating (1-5)")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    attrition: Optional[int] = Field(None, description="0=Stayed, 1=Left")


class HighRiskEmployee(BaseModel):
    """Employee with high attrition risk signals."""
    id: int = Field(..., description="Database ID")
    employee_id: str = Field(..., description="Employee ID")
    age: Optional[int] = Field(None, description="Employee age")
    department: Optional[str] = Field(None, description="Department name")
    job_role: Optional[str] = Field(None, description="Job role")
    job_satisfaction: Optional[int] = Field(None, description="Job satisfaction (1-5)")
    years_at_company: Optional[int] = Field(None, description="Years at company")


class HighRiskEmployeesResponse(BaseModel):
    """High-risk employees list response."""
    count: int = Field(..., description="Number of high-risk employees", ge=0)
    employees: List[HighRiskEmployee] = Field(..., description="List of high-risk employees")


# ─── Predictions ───────────────────────────────────────────────

class ShapExplanation(BaseModel):
    """SHAP explanation for a prediction."""
    base_value: Optional[float] = Field(None, description="Base prediction value")
    feature_contributions: Optional[Dict[str, Any]] = Field(
        None,
        description="Feature contributions to prediction (formatted strings for display)",
        example={"Overtime Risk": "+18.2%", "Satisfaction Score": "+15.1%"},
    )

    model_config = {"extra": "ignore"}


class RecommendationItem(BaseModel):
    """Single retention recommendation."""
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Priority level (high/medium/low)")
    category: str = Field(..., description="Recommendation category")


class Recommendations(BaseModel):
    """Retention recommendations for an employee."""
    recommendations: List[RecommendationItem] = Field(default_factory=list, description="List of recommendations")
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Identified risk factors with feature, impact, and direction")


class PredictResponse(BaseModel):
    """Prediction response with risk analysis."""
    attrition_probability: float = Field(..., description="Probability of attrition (0-100)", ge=0, le=100, example=72.3)
    prediction: str = Field(..., description="Prediction label", example="Likely To Leave")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)", example="High")
    confidence: Optional[float] = Field(None, description="Model confidence score", ge=0, le=1, example=0.845)
    shap_explanation: Optional[ShapExplanation] = Field(None, description="SHAP feature importance for this prediction")
    recommendations: Optional[Recommendations] = Field(None, description="Retention recommendations")
    prediction_id: Optional[int] = Field(None, description="Saved prediction database ID", example=42)


class PredictBatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictResponse] = Field(..., description="List of individual predictions")


class PredictionHistory(BaseModel):
    """A single prediction record."""
    id: int = Field(..., description="Prediction ID")
    attrition_probability: float = Field(..., description="Attrition probability")
    prediction: str = Field(..., description="Prediction label")
    risk_level: str = Field(..., description="Risk level")
    confidence: Optional[float] = Field(None, description="Model confidence")
    model_name: Optional[str] = Field(None, description="Model used")
    created_at: Optional[str] = Field(None, description="Timestamp")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data snapshot")


class PredictionsListResponse(BaseModel):
    """List of recent predictions."""
    predictions: List[PredictionHistory] = Field(..., description="Recent predictions")


# ─── Model Management ──────────────────────────────────────────

class ModelMetricsEntry(BaseModel):
    """Model performance metrics."""
    model: str = Field(..., description="Model name", example="RandomForest")
    accuracy: Optional[float] = Field(None, description="Accuracy score", example=0.881)
    precision: Optional[float] = Field(None, description="Precision score", example=0.644)
    recall: Optional[float] = Field(None, description="Recall score", example=0.952)
    f1_score: Optional[float] = Field(None, description="F1 score", example=0.768)
    roc_auc: Optional[float] = Field(None, description="ROC-AUC score", example=0.975)


class ModelMetricsResponse(BaseModel):
    """Model metrics and leaderboard response."""
    leaderboard: List[ModelMetricsEntry] = Field(..., description="Model comparison leaderboard")
    best_model: Optional[str] = Field(None, description="Best performing model name", example="RandomForest")
    best_roc_auc: Optional[float] = Field(None, description="Best model ROC-AUC", example=0.975)
    timestamp: Optional[str] = Field(None, description="Metrics timestamp")


class UploadDatasetResponse(BaseModel):
    """Dataset upload response."""
    status: str = Field("success", description="Upload status")
    filename: str = Field(..., description="Uploaded filename")
    rows: int = Field(..., description="Number of rows", ge=0)
    columns: int = Field(..., description="Number of columns", ge=0)
    column_names: List[str] = Field(..., description="Column names")
    message: str = Field(..., description="Status message")


class TrainModelResponse(BaseModel):
    """Training trigger response."""
    status: str = Field("started", description="Training status")
    message: str = Field(..., description="Status message")


class FeatureImportanceResponse(BaseModel):
    """SHAP feature importance response."""
    model_name: Optional[str] = Field(None, description="Model used for explanation")
    global_importance: Optional[List[Dict[str, Any]]] = Field(None, description="Global feature importance")
    summary_data: Optional[Dict[str, Any]] = Field(None, description="SHAP summary data")


class ModelInfoResponse(BaseModel):
    """Current model information."""
    model_loaded: bool = Field(..., description="Whether model is loaded in memory")
    model_name: Optional[str] = Field(None, description="Model name")
    model_meta: Optional[Dict[str, Any]] = Field(None, description="Model metadata including features")

    model_config = {"json_schema_extra": {
        "example": {
            "model_loaded": True,
            "model_name": "RandomForest",
            "model_meta": {
                "best_model_name": "RandomForest",
                "best_roc_auc": 0.975,
                "n_features": 31,
                "timestamp": "2026-06-12T22:20:30.433282",
            }
        }
    }}
