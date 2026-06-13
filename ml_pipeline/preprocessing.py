"""
Data Preprocessing Module for Employee Retention Predictor.
Handles both Employee.csv and hr_employee_churn_data.csv datasets.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Unified data preprocessor for employee retention datasets."""

    def __init__(self):
        self.dataset_name = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.target_col = None
        self.preprocessing_report = {}
        self.encoder = None
        self.scaler = None
        self.feature_names = []
        self.encoded_feature_names = []

    def detect_dataset(self, df: pd.DataFrame) -> str:
        """Detect which dataset is being used based on columns."""
        if "LeaveOrNot" in df.columns:
            return "employee_csv"
        elif "left" in df.columns:
            return "hr_churn"
        elif "Attrition" in df.columns:
            return "ibm_hr"
        return "unknown"

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data with encoding detection."""
        logger.info(f"Loading data from {filepath}")
        encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns with {enc} encoding")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not load {filepath} with any encoding")

    def generate_preprocessing_report(self, df: pd.DataFrame) -> dict:
        """Generate a comprehensive preprocessing report."""
        report = {
            "dataset": self.dataset_name,
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "missing_values": {},
            "duplicate_count": int(df.duplicated().sum()),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "class_distribution": {},
            "numerical_stats": {},
            "categorical_stats": {},
        }

        for col in df.columns:
            missing = int(df[col].isnull().sum())
            missing_pct = round(float(missing / len(df) * 100), 2)
            report["missing_values"][col] = {"count": missing, "percentage": missing_pct}

        if self.target_col and self.target_col in df.columns:
            dist = df[self.target_col].value_counts().to_dict()
            report["class_distribution"] = {str(k): int(v) for k, v in dist.items()}

        for col in self.numerical_cols:
            if col in df.columns:
                report["numerical_stats"][col] = {
                    "mean": round(float(df[col].mean()), 2),
                    "std": round(float(df[col].std()), 2),
                    "min": float(df[col].min()),
                    "25%": float(df[col].quantile(0.25)),
                    "50%": float(df[col].quantile(0.5)),
                    "75%": float(df[col].quantile(0.75)),
                    "max": float(df[col].max()),
                }

        for col in self.categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts().head(10).to_dict()
                report["categorical_stats"][col] = {str(k): int(v) for k, v in value_counts.items()}

        self.preprocessing_report = report
        return report

    def save_report(self, output_dir: str = "ml_pipeline/reports"):
        """Save preprocessing report to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "preprocessing_report.json")
        with open(path, "w") as f:
            json.dump(self.preprocessing_report, f, indent=2)
        logger.info(f"Preprocessing report saved to {path}")
        return path

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data - handle missing values, duplicates, invalid entries."""
        logger.info("Cleaning data...")
        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Drop columns that are all NaN
        df = df.dropna(axis=1, how="all")
        logger.info(f"Dropped fully empty columns. Remaining: {len(df.columns)}")

        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

        # Remove rows with infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df

    def prepare_datasets(self, df1_path: str, df2_path: str) -> tuple:
        """Load, detect, and prepare both datasets for unified processing."""
        # Load datasets
        df1 = self.load_data(df1_path) if os.path.exists(df1_path) else None
        df2 = self.load_data(df2_path) if os.path.exists(df2_path) else None

        processed_dfs = []

        if df1 is not None:
            logger.info(f"Processing Employee.csv...")
            df1_processed = self._process_employee_csv(df1.copy())
            if df1_processed is not None:
                processed_dfs.append(df1_processed)

        if df2 is not None:
            logger.info(f"Processing hr_employee_churn_data.csv...")
            df2_processed = self._process_hr_churn(df2.copy())
            if df2_processed is not None:
                processed_dfs.append(df2_processed)

        if not processed_dfs:
            raise ValueError("No datasets could be processed!")

        # Combine datasets
        combined = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} columns")

        return combined

    def _process_employee_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Employee.csv into unified format."""
        self.dataset_name = "employee_csv"

        # Map columns to standard names
        column_map = {
            "Education": "education",
            "JoiningYear": "joining_year",
            "City": "city",
            "PaymentTier": "payment_tier",
            "Age": "age",
            "Gender": "gender",
            "EverBenched": "ever_benched",
            "ExperienceInCurrentDomain": "experience_years",
            "LeaveOrNot": "attrition"
        }
        df = df.rename(columns=column_map)

        # Map variables
        df["attrition"] = df["attrition"].astype(int)

        # Map Education
        edu_map = {"Bachelors": 1, "Masters": 2, "PhD": 3}
        df["education_level"] = df["education"].map(edu_map).fillna(1).astype(int)

        # Map Gender
        df["gender_encoded"] = (df["gender"] == "Male").astype(int)

        # Map EverBenched
        df["ever_benched_encoded"] = (df["ever_benched"] == "Yes").astype(int)

        # Calculate derived features
        current_year = 2024
        df["tenure_years"] = current_year - df["joining_year"]

        # Satisfaction and performance are not in this dataset, use defaults
        df["satisfaction_score"] = 3.0  # Default median
        df["performance_score"] = 3.0   # Default median

        # City encoding
        city_dummies = pd.get_dummies(df["city"], prefix="city")
        df = pd.concat([df, city_dummies], axis=1)

        # Select final columns
        final_cols = [
            "age", "education_level", "payment_tier", "tenure_years",
            "experience_years", "gender_encoded", "ever_benched_encoded",
            "satisfaction_score", "performance_score", "attrition"
        ]
        # Add city dummies
        for col in city_dummies.columns:
            final_cols.append(col)

        return df[final_cols]

    def _process_hr_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process hr_employee_churn_data.csv into unified format."""
        self.dataset_name = "hr_churn"

        column_map = {
            "satisfaction_level": "satisfaction_score",
            "last_evaluation": "performance_score",
            "number_project": "num_projects",
            "average_montly_hours": "avg_monthly_hours",
            "time_spend_company": "tenure_years",
            "Work_accident": "work_accident",
            "promotion_last_5years": "promotion_last_5years",
            "salary": "salary_level",
            "left": "attrition"
        }
        df = df.rename(columns=column_map)

        df["attrition"] = df["attrition"].astype(int)

        # Encode salary
        salary_map = {"low": 1, "medium": 2, "high": 3}
        df["salary_encoded"] = df["salary_level"].map(salary_map).fillna(2).astype(int)

        # Satisfaction score is already 0-1 scale, scale to 1-5
        df["satisfaction_score"] = (df["satisfaction_score"] * 4 + 1).round(2)

        # Performance score is already 0-1 scale
        df["performance_score"] = (df["performance_score"] * 4 + 1).round(2)

        # Derived features
        df["experience_years"] = df["tenure_years"]  # proxy

        # Age is not in this dataset, estimate from tenure
        df["age"] = df["tenure_years"] + 30  # rough estimate

        # Other defaults
        df["education_level"] = 2  # Default
        df["payment_tier"] = 2     # Default
        df["gender_encoded"] = 0   # Default
        df["ever_benched_encoded"] = 0

        # Salary dummies
        salary_dummies = pd.get_dummies(df["salary_level"], prefix="salary")
        df = pd.concat([df, salary_dummies], axis=1)

        # Final columns
        final_cols = [
            "age", "education_level", "payment_tier", "tenure_years",
            "experience_years", "gender_encoded", "ever_benched_encoded",
            "satisfaction_score", "performance_score", "num_projects",
            "avg_monthly_hours", "work_accident", "promotion_last_5years",
            "salary_encoded", "attrition"
        ]
        for col in salary_dummies.columns:
            final_cols.append(col)

        return df[final_cols]

    def create_feature_pipeline(self, df: pd.DataFrame, target_col: str = "attrition"):
        """Create sklearn preprocessing pipeline."""
        self.target_col = target_col

        # Identify feature types
        feature_df = df.drop(columns=[target_col], errors="ignore")

        self.numerical_cols = feature_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self.categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Drop high-cardinality or ID columns
        for col in self.categorical_cols.copy():
            if feature_df[col].nunique() > 50:
                self.categorical_cols.remove(col)

        logger.info(f"Numerical features: {len(self.numerical_cols)}, Categorical features: {len(self.categorical_cols)}")

        # Build transformers
        transformers = []

        if self.numerical_cols:
            transformers.append(("num", StandardScaler(), self.numerical_cols))

        if self.categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols))

        self.encoder = ColumnTransformer(transformers=transformers, remainder="passthrough")
        return self.encoder

    def prepare_features(self, df: pd.DataFrame, target_col: str = "attrition", fit: bool = True):
        """Prepare features for model training."""
        self.target_col = target_col

        X = df.drop(columns=[target_col], errors="ignore")
        y = df[target_col] if target_col in df.columns else None

        # Identify column types for the combined dataset
        self.numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if fit:
            self.create_feature_pipeline(df, target_col)
            X_processed = self.encoder.fit_transform(X)
            self.feature_names = self._get_feature_names()
        else:
            X_processed = self.encoder.transform(X)

        return X_processed, y

    def _get_feature_names(self):
        """Get feature names after transformation."""
        names = []
        if self.numerical_cols:
            names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.encoder.named_transformers_["cat"]
            if hasattr(cat_encoder, "get_feature_names_out"):
                cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
                names.extend(cat_names)
        # Add remainder columns
        if "remainder" in self.encoder.named_transformers_:
            remainder = self.encoder.named_transformers_["remainder"]
            if hasattr(remainder, "get_feature_names_out"):
                names.extend(remainder.get_feature_names_out())
        self.encoded_feature_names = names
        return names

    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """Split data into train/test sets."""
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=stratify_param
        )
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        return X_train, X_test, y_train, y_test


def load_and_prepare_all_data():
    """Convenience function to load and prepare all datasets."""
    base_dir = Path(__file__).parent.parent
    preprocessor = DataPreprocessor()

    employee_path = base_dir / "Employee.csv"
    hr_churn_path = base_dir / "hr_employee_churn_data.csv"

    combined_df = preprocessor.prepare_datasets(
        str(employee_path) if employee_path.exists() else "",
        str(hr_churn_path) if hr_churn_path.exists() else ""
    )

    # Clean the combined data
    combined_df = preprocessor.clean_data(combined_df)

    # Generate report
    report = preprocessor.generate_preprocessing_report(combined_df)
    preprocessor.save_report(str(base_dir / "ml_pipeline" / "reports"))

    return combined_df, preprocessor


if __name__ == "__main__":
    df, prep = load_and_prepare_all_data()
    print(f"Prepared dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Attrition distribution:\n{df['attrition'].value_counts()}")
