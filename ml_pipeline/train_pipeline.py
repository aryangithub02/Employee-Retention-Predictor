"""
Master Training Pipeline for Employee Retention Predictor.
Orchestrates the entire ML lifecycle end-to-end.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.preprocessing import DataPreprocessor, load_and_prepare_all_data
from ml_pipeline.feature_engineering import FeatureEngineer, engineer_features
from ml_pipeline.eda import EDAnalyzer, run_eda
from ml_pipeline.training import ModelTrainer, train_models
from ml_pipeline.explainability import ModelExplainer, generate_explanations
from ml_pipeline.recommendation_engine import RecommendationEngine
from ml_pipeline.survival_analysis import SurvivalAnalyzer, TrendForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_pipeline/reports/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end ML training pipeline."""

    def __init__(self, output_dir: str = "ml_pipeline"):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_results = {}
        self.preprocessor = None
        self.feature_engineer = None
        self.trainer = None
        self.explainer = None
        self.engine = None

    def run(self, use_hyperopt: bool = True, n_iter: int = 10) -> dict:
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING EMPLOYEE RETENTION PREDICTOR TRAINING PIPELINE")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Load & Preprocess Data
        logger.info("\n[Step 1/7] Loading and preprocessing data...")
        df, self.preprocessor = load_and_prepare_all_data()
        logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")

        # Step 2: EDA
        logger.info("\n[Step 2/7] Running exploratory data analysis...")
        analyzer = EDAnalyzer()
        eda_results = analyzer.analyze(df)
        logger.info(f"EDA complete. Visualizations saved to {analyzer.output_dir}")

        # Step 3: Feature Engineering
        logger.info("\n[Step 3/7] Engineering features...")
        self.feature_engineer = FeatureEngineer()
        df = self.feature_engineer.create_all_features(df)
        logger.info(f"Feature engineering complete. {len(self.feature_engineer.get_feature_names())} new features")
        logger.info(f"Total features: {len(df.columns)}")

        # Step 4: Train-Test Split
        logger.info("\n[Step 4/7] Splitting data...")
        feature_cols = [c for c in df.columns if c != "attrition"]
        X = df[feature_cols].values
        y = df["attrition"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        # Step 5: Model Training
        logger.info("\n[Step 5/7] Training models...")
        self.trainer = ModelTrainer(str(self.models_dir))
        results = self.trainer.train_all(
            X_train, y_train, X_test, y_test,
            use_hyperopt=use_hyperopt,
            n_iter=n_iter
        )

        best_model = self.trainer.best_model
        best_model_name = self.trainer.best_model_name
        logger.info(f"Best model: {best_model_name}")

        # Step 6: Explainability
        logger.info("\n[Step 6/7] Generating SHAP explanations...")
        self.explainer = ModelExplainer(str(self.reports_dir / "shap"))
        feature_names = feature_cols  # Use original feature names
        importance = self.explainer.explain(
            best_model, X_train, X_test,
            feature_names=feature_names
        )
        logger.info("SHAP analysis complete")

        # Step 7: Survival Analysis & Forecasting
        logger.info("\n[Step 7/7] Running advanced analytics...")
        self.survival_analyzer = SurvivalAnalyzer(str(self.reports_dir / "survival"))
        cox_results = self.survival_analyzer.fit_cox_model(df)
        dept_risks = self.survival_analyzer.predict_department_attrition_risk(df)

        self.trend_forecaster = TrendForecaster(str(self.reports_dir / "forecast"))
        forecast = self.trend_forecaster.forecast_attrition_trend(df)

        # Compile final results
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.pipeline_results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "dataset": {
                "total_samples": len(df),
                "features": len(feature_cols),
                "attrition_rate": round(float(y.mean() * 100), 2),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
            "best_model": {
                "name": best_model_name,
                "roc_auc": self.trainer.best_score,
                "metrics": self.trainer.results.get(best_model_name, {}),
            },
            "leaderboard": self.trainer.leaderboard,
            "top_features": dict(list(importance.items())[:10]) if importance else {},
            "department_risks": dept_risks,
            "attrition_forecast": forecast,
            "cox_ph": cox_results,
        }

        # Save pipeline results
        self._save_results()

        # Print summary
        self._print_summary()

        logger.info(f"\nTraining pipeline completed in {elapsed:.1f} seconds!")
        return self.pipeline_results

    def _save_results(self):
        """Save pipeline results to JSON."""
        # Save a clean version (without large arrays)
        clean_results = self._clean_results(self.pipeline_results)
        path = self.reports_dir / "pipeline_results.json"
        with open(path, "w") as f:
            json.dump(clean_results, f, indent=2, default=str)
        logger.info(f"Pipeline results saved to {path}")

    def _clean_results(self, results: dict) -> dict:
        """Remove large arrays from results for clean JSON serialization."""
        if not isinstance(results, dict):
            return results
        cleaned = {}
        for k, v in results.items():
            if isinstance(v, dict):
                cleaned[k] = self._clean_results(v)
            elif isinstance(v, list):
                if len(v) > 100:
                    cleaned[k] = f"list[{len(v)} items - truncated]"
                else:
                    cleaned[k] = v
            elif isinstance(v, (str, int, float, bool)) or v is None:
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned

    def _print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 70)
        print(f"{'TRAINING PIPELINE SUMMARY':^70}")
        print("=" * 70)

        ds = self.pipeline_results.get("dataset", {})
        print(f"\n📊 Dataset: {ds.get('total_samples', 'N/A')} samples, {ds.get('features', 'N/A')} features")
        print(f"📈 Attrition Rate: {ds.get('attrition_rate', 'N/A')}%")

        bm = self.pipeline_results.get("best_model", {})
        print(f"\n🏆 Best Model: {bm.get('name', 'N/A')}")
        print(f"🎯 ROC-AUC: {bm.get('roc_auc', 'N/A'):.4f}")

        lb = self.pipeline_results.get("leaderboard", [])
        if lb:
            print(f"\n📋 Leaderboard:")
            for m in lb:
                print(f"   {m['model']:<25} | ROC-AUC: {m['roc_auc']:.4f} | F1: {m['f1_score']:.4f}")

        print(f"\n⏱️  Time: {self.pipeline_results.get('elapsed_seconds', 'N/A'):.1f}s")
        print("=" * 70)

    def predict(self, employee_data: dict) -> dict:
        """Make a prediction for a single employee."""
        if self.trainer is None or self.trainer.best_model is None:
            return {"error": "Pipeline not trained. Run train() first."}

        # Convert input to DataFrame
        df_input = pd.DataFrame([employee_data])

        # Engineer features
        if self.feature_engineer:
            df_input = self.feature_engineer.create_all_features(df_input)

        # Ensure all required features exist
        expected_features = self.trainer.X_train.shape[1] if hasattr(self.trainer, 'X_train') else 0
        if expected_features > 0 and df_input.shape[1] != expected_features:
            # Pad or truncate to match
            logger.warning(f"Feature mismatch: got {df_input.shape[1]}, expected {expected_features}")

        # Predict
        model = self.trainer.best_model
        try:
            proba = model.predict_proba(df_input.values)[:, 1][0]
            pred_class = int(model.predict(df_input.values)[0])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

        # Generate SHAP explanation
        shap_explanation = {}
        if self.explainer and self.explainer.explainer:
            try:
                shap_explanation = self.explainer.explain_prediction(df_input.values)
            except Exception:
                pass

        # Generate recommendations
        engine = RecommendationEngine()
        recommendations = engine.generate_recommendations(
            shap_explanation.get("feature_contributions", {}),
            employee_data,
            proba * 100
        )

        # Risk level
        prob_pct = proba * 100
        if prob_pct <= 30:
            risk_level = "Low"
        elif prob_pct <= 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "attrition_probability": round(prob_pct, 1),
            "prediction": "Likely To Leave" if pred_class == 1 else "Likely To Stay",
            "risk_level": risk_level,
            "confidence": round(proba if pred_class == 1 else 1 - proba, 3),
            "shap_explanation": shap_explanation,
            "recommendations": recommendations,
        }


def main():
    """Main entry point for training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Employee Retention Predictor models")
    parser.add_argument("--no-hyperopt", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of hyperopt iterations")
    args = parser.parse_args()

    pipeline = TrainingPipeline()
    results = pipeline.run(use_hyperopt=not args.no_hyperopt, n_iter=args.n_iter)

    # Save the pipeline object for later use
    pipeline_path = "ml_pipeline/models/training_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Training pipeline saved to {pipeline_path}")

    return results


if __name__ == "__main__":
    main()
