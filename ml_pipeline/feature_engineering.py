"""
Feature Engineering Module for Employee Retention Predictor.
Creates derived features from raw employee data.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates derived features for attrition prediction."""

    def __init__(self):
        self.engineered_features = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        # Reset engineered features list so cached instances don't accumulate
        self.engineered_features = []
        logger.info("Creating engineered features...")
        df = df.copy()

        # Promotion Gap
        df = self.create_promotion_gap(df)

        # Income Per Experience
        df = self.create_income_per_experience(df)

        # Satisfaction Index
        df = self.create_satisfaction_index(df)

        # Overtime Risk Score
        df = self.create_overtime_risk_score(df)

        # Experience Ratio
        df = self.create_experience_ratio(df)

        # Workload Score
        df = self.create_workload_score(df)

        # Tenure Category
        df = self.create_tenure_category(df)

        # Performance vs Satisfaction Gap
        df = self.create_performance_satisfaction_gap(df)

        # Interaction Features
        df = self.create_interaction_features(df)

        logger.info(f"Created {len(self.engineered_features)} engineered features")
        return df

    def create_promotion_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Years since last promotion vs total tenure."""
        if "tenure_years" in df.columns and "promotion_last_5years" in df.columns:
            df["promotion_gap"] = df["tenure_years"] * (1 - df["promotion_last_5years"])
            df["promotion_gap"] = df["promotion_gap"].clip(0)
            self.engineered_features.append("promotion_gap")
        return df

    def create_income_per_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """Income relative to experience."""
        if "payment_tier" in df.columns and "experience_years" in df.columns:
            df["income_per_experience"] = df["payment_tier"] / (df["experience_years"] + 1)
            self.engineered_features.append("income_per_experience")
        return df

    def create_satisfaction_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite satisfaction index."""
        sat_cols = [c for c in df.columns if "satisfaction" in c.lower()]
        if sat_cols:
            df["satisfaction_index"] = df[sat_cols].mean(axis=1)
            self.engineered_features.append("satisfaction_index")
        return df

    def create_overtime_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overtime risk based on hours worked.

        Uses a consistent fixed threshold (200h) for both training and inference
        to avoid train/inference distribution shift.
        Does NOT overwrite an existing overtime_risk column so that an explicit
        mapping (e.g. from the "overtime" Yes/No field) takes precedence.
        """
        if "avg_monthly_hours" in df.columns and "overtime_risk" not in df.columns:
            df["overtime_risk"] = (df["avg_monthly_hours"] > 200).astype(int)
        if "overtime_risk" in df.columns and "overtime_risk" not in self.engineered_features:
            self.engineered_features.append("overtime_risk")
        if "ever_benched_encoded" in df.columns:
            df["bench_risk"] = df["ever_benched_encoded"]
            self.engineered_features.append("bench_risk")
        return df

    def create_experience_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Experience relative to age."""
        if "tenure_years" in df.columns and "age" in df.columns:
            df["experience_ratio"] = df["tenure_years"] / (df["age"] + 1e-5)
            self.engineered_features.append("experience_ratio")
        return df

    def create_workload_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Workload based on projects and hours."""
        if "num_projects" in df.columns and "avg_monthly_hours" in df.columns:
            df["workload_score"] = df["num_projects"] * (df["avg_monthly_hours"] / 160)
            self.engineered_features.append("workload_score")
        elif "avg_monthly_hours" in df.columns:
            df["workload_score"] = df["avg_monthly_hours"] / 160
            self.engineered_features.append("workload_score")
        return df

    def create_tenure_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize tenure into groups."""
        if "tenure_years" in df.columns:
            df["tenure_category"] = pd.cut(
                df["tenure_years"],
                bins=[0, 1, 3, 5, 10, 50],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(2).astype(int)
            self.engineered_features.append("tenure_category")
        return df

    def create_performance_satisfaction_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gap between performance and satisfaction."""
        if "performance_score" in df.columns and "satisfaction_score" in df.columns:
            df["perf_sat_gap"] = df["performance_score"] - df["satisfaction_score"]
            self.engineered_features.append("perf_sat_gap")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables.

        Uses consistent fixed thresholds for both training and inference
        to avoid train/inference distribution shift.
        """
        # Low satisfaction (< 3) + high workload (> 5 projects * hours/160)
        if "satisfaction_score" in df.columns and "workload_score" in df.columns:
            sat_threshold = 3.0
            wl_threshold = 5.0
            df["low_sat_high_workload"] = (
                (df["satisfaction_score"] < sat_threshold).astype(int) &
                (df["workload_score"] > wl_threshold).astype(int)
            ).astype(int)
            self.engineered_features.append("low_sat_high_workload")

        # Long tenure (> 5 years) + no promotion (promotion_last_5years == 0)
        if "tenure_years" in df.columns and "promotion_last_5years" in df.columns:
            tenure_threshold = 5.0
            df["long_tenure_no_promo"] = (
                (df["tenure_years"] > tenure_threshold).astype(int) &
                (df["promotion_last_5years"] == 0).astype(int)
            ).astype(int)
            self.engineered_features.append("long_tenure_no_promo")

        return df

    def get_feature_names(self) -> list:
        """Get list of engineered feature names."""
        return self.engineered_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to engineer all features."""
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    from preprocessing import load_and_prepare_all_data
    df, prep = load_and_prepare_all_data()
    df = engineer_features(df)
    print(f"Features after engineering: {list(df.columns)}")
    print(f"Shape: {df.shape}")
