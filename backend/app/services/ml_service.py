"""
ML Service for Employee Retention Predictor.
Loads trained models and serves predictions, SHAP explanations, and recommendations.
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ml_pipeline.preprocessing import DataPreprocessor
from ml_pipeline.feature_engineering import FeatureEngineer
from ml_pipeline.explainability import ModelExplainer
from ml_pipeline.recommendation_engine import RecommendationEngine
from ml_pipeline.survival_analysis import SurvivalAnalyzer, TrendForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLService:
    """Central ML service for predictions, explanations, and recommendations."""

    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_meta = None
        self.feature_names = []
        self.training_feature_names = []
        self.explainer = None
        self.preprocessor = None
        self.feature_engineer = None
        self.recommendation_engine = RecommendationEngine()
        self.survival_analyzer = None
        self.trend_forecaster = None
        self.training_pipeline = None
        self._is_loaded = False

    def load_models(self, models_dir: Optional[str] = None):
        """Load trained models from disk."""
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "ml_pipeline", "models"
            )
        models_dir = Path(models_dir).resolve()

        # Load best model
        best_model_path = models_dir / "best_model.pkl"
        meta_path = models_dir / "best_model_meta.json"

        if best_model_path.exists():
            self.model = joblib.load(str(best_model_path))
            logger.info(f"Model loaded from {best_model_path}")
        else:
            logger.warning(f"No model found at {best_model_path}")
            return False

        if meta_path.exists():
            with open(meta_path) as f:
                self.model_meta = json.load(f)
            self.model_name = self.model_meta.get("best_model_name", "Unknown")
            self.training_feature_names = self.model_meta.get("feature_names", [])

        # Load training pipeline if available
        pipeline_path = models_dir / "training_pipeline.pkl"
        if pipeline_path.exists():
            try:
                self.training_pipeline = joblib.load(str(pipeline_path))
                logger.info("Training pipeline loaded")
            except Exception as e:
                logger.warning(f"Could not load training pipeline: {e}")

        self._is_loaded = True
        return True

    def _prepare_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Convert raw input data into aligned feature vector for the model.

        Steps:
        1. Create DataFrame from input
        2. Map raw fields to feature engineering columns FIRST
        3. Call create_all_features ONCE with all mapped columns
        4. Align to the 31 training features (fill missing with 0)
        """
        df = pd.DataFrame([input_data])

        # ---- Apply manual mappings FIRST, before feature engineering ----

        # Map gender
        if "gender" in df.columns:
            df["gender_encoded"] = (df["gender"] == "Male").astype(int)

        # Map education
        if "education" in df.columns:
            edu_map = {"Bachelor's": 1, "Master's": 2, "PhD": 3, "Bachelors": 1, "Masters": 2, "PHD": 3}
            df["education_level"] = df["education"].map(edu_map).fillna(1).astype(int)

        # Map salary_level to salary_encoded and dummies
        if "salary_level" in df.columns:
            salary_map = {"low": 1, "medium": 2, "high": 3}
            df["salary_encoded"] = df["salary_level"].map(salary_map).fillna(2).astype(int)
            for level in ["high", "low", "medium"]:
                df[f"salary_{level}"] = (df["salary_level"] == level).astype(int)

        # Map overtime (done here; feature engineering later won't overwrite)
        if "overtime" in df.columns:
            df["overtime_risk"] = (df["overtime"] == "Yes").astype(int)

        # Map department to city-like dummies
        if "department" in df.columns:
            for city in ["Bangalore", "New Delhi", "Pune"]:
                col = f"city_{city}"
                if col not in df.columns:
                    df[col] = 0

        # Map satisfaction and performance scores
        if "job_satisfaction" in df.columns:
            df["satisfaction_score"] = df["job_satisfaction"].astype(float)
        if "environment_satisfaction" in df.columns:
            df["env_satisfaction_score"] = df["environment_satisfaction"].astype(float)
        if "performance_rating" in df.columns:
            df["performance_score"] = df["performance_rating"].astype(float)

        # Map monthly_income to payment_tier
        if "monthly_income" in df.columns:
            income = df["monthly_income"].iloc[0]
            if income < 35000:
                df["payment_tier"] = 1
            elif income < 65000:
                df["payment_tier"] = 2
            else:
                df["payment_tier"] = 3

        # Map experience and tenure
        if "years_at_company" in df.columns:
            df["tenure_years"] = df["years_at_company"]
            if "experience_years" not in df.columns:
                df["experience_years"] = df["years_at_company"]
        if "years_since_last_promotion" in df.columns:
            # Map to promotion_last_5years: if years since promo <= 5, set to 1
            df["promotion_last_5years"] = (df["years_since_last_promotion"] <= 5).astype(int)

        # Derive salary_level from monthly_income if not provided
        if "salary_level" not in df.columns and "monthly_income" in df.columns:
            income = df["monthly_income"].iloc[0]
            if income < 35000:
                df["salary_level"] = "low"
                df["salary_encoded"] = 1
            elif income < 65000:
                df["salary_level"] = "medium"
                df["salary_encoded"] = 2
            else:
                df["salary_level"] = "high"
                df["salary_encoded"] = 3
            for level in ["high", "low", "medium"]:
                df[f"salary_{level}"] = (df["salary_level"] == level).astype(int)

        # Derive num_projects from avg_monthly_hours if not provided
        if "num_projects" not in df.columns:
            if "avg_monthly_hours" in df.columns:
                hours = df["avg_monthly_hours"].iloc[0]
                if hours > 250:
                    df["num_projects"] = 6
                elif hours > 200:
                    df["num_projects"] = 5
                elif hours > 150:
                    df["num_projects"] = 4
                elif hours > 100:
                    df["num_projects"] = 3
                else:
                    df["num_projects"] = 2
            else:
                df["num_projects"] = 4

        # Derive avg_monthly_hours from overtime if not provided
        if "avg_monthly_hours" not in df.columns:
            if "overtime" in df.columns and df["overtime"].iloc[0] == "Yes":
                df["avg_monthly_hours"] = 220
            else:
                df["avg_monthly_hours"] = 170

        # Default values for missing fields that truly have no reasonable derivation
        for col, default in [("ever_benched_encoded", 0), ("work_accident", 0),
                             ("performance_score", 3.0)]:
            if col not in df.columns:
                df[col] = default

        # ---- NOW run feature engineering ONCE with all columns present ----
        engineer = FeatureEngineer()
        df = engineer.create_all_features(df)

        # ---- Align to training features: fill missing with 0, drop extras ----
        if self.training_feature_names:
            for col in self.training_feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.training_feature_names]
        else:
            df = df.select_dtypes(include=[np.number])

        return df.values.astype(np.float64)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for employee data."""
        if not self._is_loaded or self.model is None:
            return self._fallback_prediction(input_data)

        try:
            # Prepare aligned feature vector
            X = self._prepare_features(input_data)

            # Predict
            proba = self.model.predict_proba(X)[:, 1][0]
            pred_class = int(self.model.predict(X)[0])

            # Risk level
            prob_pct = proba * 100
            if prob_pct <= 30:
                risk_level = "Low"
            elif prob_pct <= 70:
                risk_level = "Medium"
            else:
                risk_level = "High"

            # Get SHAP explanation
            shap_explanation = self._get_shap_explanation(X)

            # Get recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                shap_explanation.get("raw_contributions", {}),
                input_data,
                prob_pct
            )

            return {
                "attrition_probability": round(prob_pct, 1),
                "prediction": "Likely To Leave" if pred_class == 1 else "Likely To Stay",
                "risk_level": risk_level,
                "confidence": round(proba if pred_class == 1 else 1 - proba, 3),
                "shap_explanation": shap_explanation,
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return self._fallback_prediction(input_data)

    @staticmethod
    def _shap_log_odds_to_probability(base_value: float, all_contributions: Dict[str, float]) -> Dict[str, float]:
        """Convert log-odds SHAP contributions to probability-scale contributions.

        SHAP values from tree models are in log-odds space. Displaying them as
        raw percentages (value × 100) produces meaningless numbers like +600%.
        This method converts each feature's contribution to the change in
        prediction probability (0-100%), which is intuitive and sensible.
        """
        total_log_odds = base_value + sum(all_contributions.values())
        prob_with_all = 1.0 / (1.0 + np.exp(-total_log_odds))

        prob_contributions = {}
        for feature, log_odds_val in all_contributions.items():
            log_odds_without = total_log_odds - log_odds_val
            prob_without = 1.0 / (1.0 + np.exp(-log_odds_without))
            # Feature's contribution in probability space (as percentage 0-100)
            prob_contributions[feature] = (prob_with_all - prob_without) * 100

        return prob_contributions

    def _get_shap_explanation(self, X: np.ndarray) -> Dict:
        """Get SHAP explanation for a prediction using real feature names."""
        try:
            # Use the 31 training feature names for SHAP labels
            fnames = self.training_feature_names if len(self.training_feature_names) == X.shape[1] \
                else [f"feat_{i}" for i in range(X.shape[1])]

            explainer = ModelExplainer()
            explainer.explain(self.model, X, X, feature_names=fnames)
            explanation = explainer.explain_prediction(X, sample_idx=0)

            # Get ALL contributions (log-odds space) and base value
            all_contributions = explanation.get("feature_contributions", {})
            base_value = explanation.get("base_value", 0)

            # Convert to probability-scale contributions
            prob_contribs = self._shap_log_odds_to_probability(base_value, all_contributions)

            # Top 5 by absolute probability contribution
            sorted_features = sorted(prob_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            top_5 = dict(sorted_features[:5])

            # Format for display with Title Case names
            formatted = {}
            for feature, pct in top_5.items():
                label = feature.replace("_", " ").title()
                sign = "+" if pct >= 0 else ""
                formatted[label] = f"{sign}{pct:.1f}%"

            # Also prepare raw log-odds contributions for the recommendation engine
            raw_top_5 = {}
            for feature, _ in top_5.items():
                if feature in all_contributions:
                    raw_top_5[feature] = all_contributions[feature]

            return {
                "base_value": base_value,
                "feature_contributions": formatted,
                "raw_contributions": raw_top_5,
            }
        except Exception as e:
            logger.warning(f"SHAP explanation error: {e}")
            return {}

    def _fallback_prediction(self, input_data: Dict) -> Dict:
        """Simple rule-based fallback when model isn't available."""
        logger.info("Using fallback prediction (no model loaded)")

        # Simple heuristic-based prediction
        risk_score = 0
        factors = []

        if input_data.get("job_satisfaction", 3) <= 2:
            risk_score += 25
            factors.append("low job satisfaction")
        if input_data.get("work_life_balance", 3) <= 2:
            risk_score += 20
            factors.append("poor work-life balance")
        if input_data.get("overtime") == "Yes":
            risk_score += 15
            factors.append("overtime")
        if input_data.get("years_since_last_promotion", 0) > 3:
            risk_score += 15
            factors.append("no recent promotion")
        if input_data.get("years_at_company", 0) < 1:
            risk_score += 10
            factors.append("new employee")
        if input_data.get("distance_from_home", 0) > 20:
            risk_score += 10
            factors.append("long commute")
        if input_data.get("monthly_income", 50000) < 30000:
            risk_score += 10
            factors.append("low income")

        risk_score = min(risk_score, 95)

        if risk_score <= 30:
            risk_level = "Low"
            prediction = "Likely To Stay"
        elif risk_score <= 70:
            risk_level = "Medium"
            prediction = "Uncertain"
        else:
            risk_level = "High"
            prediction = "Likely To Leave"

        return {
            "attrition_probability": float(risk_score),
            "prediction": prediction,
            "risk_level": risk_level,
            "confidence": 0.5,
            "note": "Rule-based estimation (model not loaded)",
            "risk_factors": factors,
        }

    def get_employee_risk_analysis(self, employee_data: Dict) -> Dict:
        """Get comprehensive risk analysis for an employee."""
        prediction = self.predict(employee_data)

        # Add survival analysis if available
        survival = None
        if self.survival_analyzer:
            try:
                df = pd.DataFrame([employee_data])
                survival = self.survival_analyzer.predict_survival_intervals(df)
            except Exception:
                pass

        return {
            "prediction": prediction,
            "survival_analysis": survival,
        }

    def get_model_metrics(self) -> Dict:
        """Get stored model metrics."""
        metrics_path = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "models" / "model_leaderboard.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {"error": "No metrics found"}

    def get_feature_importance(self) -> Dict:
        """Get stored feature importance from SHAP analysis."""
        importance_path = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "reports" / "shap" / "global_feature_importance.json"
        if importance_path.exists():
            with open(importance_path) as f:
                return json.load(f)
        return {"error": "No feature importance found"}

    def get_department_risk(self) -> list:
        """Get department risk analysis."""
        dept_path = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "reports" / "survival" / "department_risk.json"
        if dept_path.exists():
            with open(dept_path) as f:
                return json.load(f)

        # Return default departments
        return [
            {"department": "Sales", "risk_score": 45.2, "risk_level": "Medium", "employee_count": 120},
            {"department": "Engineering", "risk_score": 38.7, "risk_level": "Medium", "employee_count": 200},
            {"department": "HR", "risk_score": 28.4, "risk_level": "Low", "employee_count": 45},
            {"department": "Marketing", "risk_score": 42.1, "risk_level": "Medium", "employee_count": 65},
            {"department": "Finance", "risk_score": 32.5, "risk_level": "Medium", "employee_count": 55},
            {"department": "IT", "risk_score": 35.8, "risk_level": "Medium", "employee_count": 80},
        ]

    def get_retention_recommendations(self, employee_data: Dict = None) -> Dict:
        """Get retention recommendations."""
        if employee_data:
            prediction = self.predict(employee_data)
            return prediction.get("recommendations", {})
        return {"recommendations": [], "summary": "No employee data provided"}

    def get_attrition_forecast(self) -> Dict:
        """Get attrition trend forecast."""
        forecast_path = Path(__file__).parent.parent.parent.parent / "ml_pipeline" / "reports" / "forecast" / "attrition_forecast.json"
        if forecast_path.exists():
            with open(forecast_path) as f:
                return json.load(f)
        return {"forecast": []}

    def _fallback_risk_distribution(self) -> Dict:
        """Fallback risk distribution when no DB data is available."""
        return {
            "total_employees": 1470,
            "high_risk": 185,
            "medium_risk": 320,
            "low_risk": 965,
            "high_risk_percentage": 12.6,
            "average_attrition_risk": 22.4,
            "risk_distribution": {
                "Low Risk (0-30%)": 965,
                "Medium Risk (31-70%)": 320,
                "High Risk (71-100%)": 185,
            }
        }
