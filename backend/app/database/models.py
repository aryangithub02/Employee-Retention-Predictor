"""
SQLAlchemy database models for Employee Retention Predictor.
"""

from sqlalchemy import (Column, Integer, String, Float, Boolean,
                       DateTime, Text, JSON, ForeignKey, Date)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from backend.app.database.connection import Base


class Employee(Base):
    """Employee records."""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_id = Column(String(50), unique=True, index=True, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    job_role = Column(String(100), nullable=True)
    monthly_income = Column(Float, nullable=True)
    job_satisfaction = Column(Integer, nullable=True)
    environment_satisfaction = Column(Integer, nullable=True)
    work_life_balance = Column(Integer, nullable=True)
    distance_from_home = Column(Integer, nullable=True)
    years_at_company = Column(Integer, nullable=True)
    years_since_last_promotion = Column(Integer, nullable=True)
    overtime = Column(String(10), nullable=True)
    performance_rating = Column(Integer, nullable=True)
    training_times_last_year = Column(Integer, nullable=True)
    education = Column(String(50), nullable=True)
    marital_status = Column(String(20), nullable=True)
    num_companies_worked = Column(Integer, nullable=True)
    stock_option_level = Column(Integer, nullable=True)
    attrition = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    predictions = relationship("Prediction", back_populates="employee", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="employee", cascade="all, delete-orphan")


class Prediction(Base):
    """Prediction records."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=True)
    attrition_probability = Column(Float)
    prediction = Column(String(20))
    risk_level = Column(String(20))
    confidence = Column(Float, nullable=True)
    model_name = Column(String(50), nullable=True)
    model_version = Column(String(20), nullable=True)
    shap_contributions = Column(JSON, nullable=True)
    input_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    employee = relationship("Employee", back_populates="predictions")


class Dataset(Base):
    """Uploaded dataset records."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200))
    filename = Column(String(200))
    file_path = Column(String(500))
    rows = Column(Integer)
    columns = Column(Integer)
    target_column = Column(String(100), default="attrition")
    data_preview = Column(JSON, nullable=True)
    preprocessing_report = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TrainingRun(Base):
    """Training run records."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    run_name = Column(String(200))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    model_type = Column(String(100))
    best_model_name = Column(String(100))
    hyperparameters = Column(JSON, nullable=True)
    status = Column(String(20), default="completed")
    train_samples = Column(Integer)
    test_samples = Column(Integer)
    elapsed_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    metrics = relationship("ModelMetric", back_populates="training_run", cascade="all, delete-orphan")


class ModelMetric(Base):
    """Model evaluation metrics."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="CASCADE"))
    model_name = Column(String(100))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    cv_score = Column(Float, nullable=True)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    training_run = relationship("TrainingRun", back_populates="metrics")


class SurveyResponse(Base):
    """Employee self-assessment survey responses."""
    __tablename__ = "survey_responses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_id = Column(String(50), index=True, nullable=False)
    employee_name = Column(String(200), nullable=True)
    department = Column(String(100), nullable=True)
    job_role = Column(String(100), nullable=True)

    # Survey fields
    job_satisfaction = Column(Integer, nullable=True)
    work_life_balance = Column(Integer, nullable=True)
    stress_level = Column(Integer, nullable=True)
    career_growth_satisfaction = Column(Integer, nullable=True)
    manager_relationship = Column(Integer, nullable=True)
    engagement_score = Column(Integer, nullable=True)
    feedback_comment = Column(Text, nullable=True)

    # Metadata
    survey_score = Column(Float, nullable=True)
    status = Column(String(20), default="completed")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Recommendation(Base):
    """Retention recommendations."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"))
    title = Column(String(200))
    description = Column(Text)
    priority = Column(String(20))
    category = Column(String(50))
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    is_implemented = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    employee = relationship("Employee", back_populates="recommendations")
