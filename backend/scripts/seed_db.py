"""
Script to seed the database with employee data from CSV files.
Loads Employee.csv and hr_employee_churn_data.csv into the SQLite database.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app.database.connection import SessionLocal, init_db, engine
from backend.app.database.models import (
    Employee, Base, Dataset, TrainingRun, ModelMetric
)
from sqlalchemy.orm import Session
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global counter to ensure unique employee_ids across both CSVs
global_count = [0]


def seed_from_employee_csv(db: Session, csv_path: str) -> int:
    """Load Employee.csv into the employees table."""
    logger.info(f"Loading Employee data from {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} rows from Employee.csv")

    count = 0
    for _, row in df.iterrows():
        # Map columns to Employee model fields
        try:
            emp = Employee(
                age=int(row.get("Age", 0)),
                gender=row.get("Gender", ""),
                department=_map_department_from_city(row.get("City", "")),
                monthly_income=_map_payment_tier_to_income(row.get("PaymentTier", 2)),
                years_at_company=max(0, 2024 - int(row.get("JoiningYear", 2020))),
                education=_map_education(row.get("Education", "")),
                distance_from_home=10,
                job_satisfaction=3,
                environment_satisfaction=3,
                work_life_balance=3,
                performance_rating=3,
                years_since_last_promotion=0,
                overtime="No",
                training_times_last_year=2,
                marital_status="Single",
                num_companies_worked=1,
                stock_option_level=0,
                attrition=int(row.get("LeaveOrNot", 0)),
                employee_id=f"EMP{global_count[0]:04d}",
                job_role=_map_job_role(row.get("City", "")),
            )
            db.add(emp)
            count += 1
            global_count[0] += 1

            # Batch commit
            if global_count[0] % 500 == 0:
                db.commit()

        except Exception as e:
            logger.warning(f"Error processing row {count}: {e}")

    db.commit()
    logger.info(f"Inserted {count} employees from Employee.csv (global count: {global_count[0]})")
    return count


def seed_from_hr_churn_csv(db: Session, csv_path: str) -> int:
    """Load hr_employee_churn_data.csv into the employees table."""
    logger.info(f"Loading HR Churn data from {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} rows from hr_employee_churn_data.csv")

    count = 0
    for _, row in df.iterrows():
        try:
            satisfaction = row.get("satisfaction_level", 0.5)
            evaluation = row.get("last_evaluation", 0.5)
            tenure_raw = row.get("time_spend_company", 3)
            tenure = int(tenure_raw) if pd.notna(tenure_raw) else 3
            salary = row.get("salary", "medium")

            emp = Employee(
                age=tenure + 28,
                gender="Male",
                department=_map_department_from_salary(salary),
                monthly_income=_map_salary_to_income(salary),
                years_at_company=tenure,
                education="Bachelor's",
                distance_from_home=10,
                job_satisfaction=max(1, min(5, round(satisfaction * 4 + 1))),
                environment_satisfaction=max(1, min(5, round(satisfaction * 4 + 1))),
                work_life_balance=3,
                performance_rating=max(1, min(5, round(evaluation * 4 + 1))),
                years_since_last_promotion=tenure // 2,
                overtime="No",
                training_times_last_year=2,
                marital_status="Single",
                num_companies_worked=1,
                stock_option_level=0,
                attrition=int(row.get("left", 0)),
                employee_id=f"EMP{global_count[0]:04d}",
                job_role=_map_job_role_from_salary(salary),
            )
            db.add(emp)
            count += 1
            global_count[0] += 1

            # Commit in batches
            if global_count[0] % 500 == 0:
                db.commit()

        except Exception as e:
            logger.warning(f"Error processing row {count}: {e}")

    db.commit()
    logger.info(f"Inserted {count} employees from hr_employee_churn_data.csv (global count: {global_count[0]})")
    return count


def seed_model_metrics(db: Session):
    """Load model leaderboard metrics into the database."""
    leaderboard_path = Path(__file__).parent.parent.parent / "ml_pipeline" / "models" / "model_leaderboard.json"
    meta_path = Path(__file__).parent.parent.parent / "ml_pipeline" / "models" / "best_model_meta.json"

    if not leaderboard_path.exists():
        logger.warning("No model_leaderboard.json found, skipping model metrics seed")
        return

    with open(leaderboard_path) as f:
        leaderboard_data = json.load(f)

    # Create a training run record
    training_run = TrainingRun(
        run_name="Initial Model Training",
        model_type="ensemble",
        best_model_name=leaderboard_data.get("best_model", "RandomForest"),
        status="completed",
        train_samples=11731,  # ~80% of total
        test_samples=2933,    # ~20% of total
    )
    db.add(training_run)
    db.flush()

    # Add metrics for each model
    for model_data in leaderboard_data.get("leaderboard", []):
        metric = ModelMetric(
            training_run_id=training_run.id,
            model_name=model_data["model"],
            accuracy=model_data["accuracy"],
            precision=model_data["precision"],
            recall=model_data["recall"],
            f1_score=model_data["f1_score"],
            roc_auc=model_data["roc_auc"],
        )
        db.add(metric)

    # Create dataset record
    dataset = Dataset(
        name="Combined Employee Data",
        filename="Employee.csv, hr_employee_churn_data.csv",
        rows=14664,
        columns=21,
        target_column="attrition",
    )
    db.add(dataset)

    db.commit()
    logger.info("Model metrics and dataset record seeded")


def _map_department_from_city(city: str) -> str:
    """Map city to a reasonable department."""
    dept_map = {
        "Bangalore": "Engineering",
        "Pune": "Sales",
        "New Delhi": "Marketing",
        "Mumbai": "Finance",
    }
    return dept_map.get(str(city).strip(), "Engineering")


def _map_department_from_salary(salary: str) -> str:
    """Map salary level to department."""
    if salary == "high":
        return "Finance"
    elif salary == "low":
        return "Sales"
    else:
        return "Engineering"


def _map_job_role(city: str) -> str:
    """Map city to job role."""
    role_map = {
        "Bangalore": "Software Engineer",
        "Pune": "Sales Executive",
        "New Delhi": "Marketing Specialist",
        "Mumbai": "Financial Analyst",
    }
    return role_map.get(str(city).strip(), "Software Engineer")


def _map_job_role_from_salary(salary: str) -> str:
    """Map salary to job role."""
    if salary == "high":
        return "Senior Manager"
    elif salary == "low":
        return "Junior Associate"
    else:
        return "Software Engineer"


def _map_education(edu: str) -> str:
    """Map education value."""
    mapping = {
        "Bachelors": "Bachelor's",
        "Masters": "Master's",
        "PHD": "PhD",
    }
    return mapping.get(str(edu).strip(), "Bachelor's")


def _map_payment_tier_to_income(tier) -> float:
    """Map payment tier to monthly income."""
    tier = int(tier) if pd.notna(tier) else 2
    mapping = {1: 30000, 2: 50000, 3: 75000}
    return float(mapping.get(tier, 50000))


def _map_salary_to_income(salary: str) -> float:
    """Map salary level to income."""
    mapping = {"low": 30000, "medium": 50000, "high": 80000}
    return float(mapping.get(str(salary).strip(), 50000))


def main():
    """Run the seeding process."""
    logger.info("=" * 60)
    logger.info("SEEDING DATABASE")
    logger.info("=" * 60)

    # Initialize tables
    logger.info("Creating database tables...")
    init_db()

    # Create session
    db = SessionLocal()
    try:
        # Clear existing data
        logger.info("Clearing existing data...")
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(table.delete())
        db.commit()

        # Paths to CSVs
        base_dir = Path(__file__).parent.parent.parent
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

        logger.info(f"\\nTotal employees seeded: {total}")
        logger.info("Database seeding complete!")

    except Exception as e:
        logger.error(f"Error during seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
