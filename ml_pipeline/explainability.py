"""
Explainable AI Module for Employee Retention Predictor.
Uses SHAP for model interpretability at global and local levels.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import logging
from pathlib import Path
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainer for attrition predictions."""

    def __init__(self, output_dir: str = "ml_pipeline/reports/shap"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.model = None

    def load_model(self, model_path: str = "ml_pipeline/models/best_model.pkl"):
        """Load trained model."""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        raise FileNotFoundError(f"Model not found at {model_path}")

    def explain(self, model, X_train: np.ndarray, X_test: np.ndarray,
                feature_names: list = None, model_type: str = "tree"):
        """Generate SHAP explanations for the model."""
        self.model = model
        self.feature_names = feature_names

        logger.info(f"Generating SHAP explanations using {model_type} explainer...")

        # Select appropriate explainer
        if model_type == "tree" or "XGBoost" in str(type(model)) or "RandomForest" in str(type(model)):
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                self.explainer = shap.Explainer(model, X_train)
        elif model_type == "linear" or "LogisticRegression" in str(type(model)):
            self.explainer = shap.LinearExplainer(model, X_train)
        else:
            self.explainer = shap.Explainer(model, X_train)

        # Calculate SHAP values on test set
        self.shap_values = self.explainer(X_test)

        logger.info(f"SHAP values shape: {self.shap_values.values.shape}")

        # Generate plots
        self._plot_summary()
        self._plot_feature_importance()
        self._plot_waterfall(sample_idx=0)
        self._plot_dependence()

        # Get global feature importance
        importance = self.get_global_feature_importance()
        return importance

    def _plot_summary(self):
        """Generate SHAP summary plot."""
        fig = plt.figure(figsize=(12, 8))

        shap.summary_plot(
            self.shap_values, feature_names=self.feature_names,
            show=False, max_display=15
        )
        plt.title("SHAP Summary Plot - Feature Impact on Predictions",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {path}")

    def _plot_feature_importance(self):
        """Generate SHAP feature importance bar plot."""
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, feature_names=self.feature_names,
            plot_type="bar", show=False, max_display=15
        )
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_feature_importance.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP feature importance saved to {path}")

    def _plot_waterfall(self, sample_idx: int = 0):
        """Generate SHAP waterfall plot for a single prediction."""
        try:
            fig = plt.figure(figsize=(12, 6))
            shap.waterfall_plot(
                self.shap_values[sample_idx],
                max_display=10,
                show=False
            )
            plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}",
                      fontsize=14, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"shap_waterfall_sample_{sample_idx}.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP waterfall plot saved to {path}")
        except Exception as e:
            logger.warning(f"Could not generate waterfall plot: {e}")

    def _plot_dependence(self):
        """Generate SHAP dependence plots for top features."""
        if self.feature_names is None or len(self.feature_names) == 0:
            return

        try:
            # Get top features
            mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-5:][::-1]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, idx in enumerate(top_indices[:5]):
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    ax = axes[i]
                    shap.dependence_plot(
                        idx, self.shap_values.values,
                        feature_names=self.feature_names,
                        ax=ax, show=False
                    )
                    ax.set_title(f"Dependence: {feature_name}", fontsize=11)
                    ax.grid(True, alpha=0.3)

            # Hide unused subplot
            if len(top_indices) < 6:
                axes[-1].set_visible(False)

            plt.suptitle("SHAP Dependence Plots - Top Features",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(self.output_dir, "shap_dependence_plots.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP dependence plots saved to {path}")
        except Exception as e:
            logger.warning(f"Could not generate dependence plots: {e}")

    def get_global_feature_importance(self) -> dict:
        """Get global feature importance from SHAP values."""
        if self.shap_values is None:
            return {}

        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        feature_names = self.feature_names or [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        # Ensure lengths match
        if len(feature_names) != len(mean_abs_shap):
            feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        importance_dict = {}
        for name, value in zip(feature_names, mean_abs_shap):
            importance_dict[name] = round(float(value), 4)

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        path = os.path.join(self.output_dir, "global_feature_importance.json")
        with open(path, "w") as f:
            json.dump(importance_dict, f, indent=2)
        logger.info(f"Global feature importance saved to {path}")

        return importance_dict

    def explain_prediction(self, sample: np.ndarray, sample_idx: int = 0) -> dict:
        """Explain a single prediction."""
        if self.explainer is None:
            return {"error": "Explainer not initialized. Run explain() first."}

        shap_val = self.shap_values[sample_idx]

        # Get top contributing features
        feature_effects = {}
        for i, name in enumerate(self.feature_names or []):
            if i < len(shap_val.values):
                effect = shap_val.values[i]
                feature_effects[name] = round(float(effect), 4)

        # Sort by absolute effect
        feature_effects = dict(
            sorted(feature_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Base value and prediction
        base_value = float(shap_val.base_values) if hasattr(shap_val, "base_values") else 0
        prediction = float(np.sum(shap_val.values) + base_value)

        explanation = {
            "base_value": round(base_value, 4),
            "prediction_value": round(prediction, 4),
            "feature_contributions": feature_effects,
            "top_contributors": dict(list(feature_effects.items())[:5]),
        }

        return explanation

    def get_top_features_contributing_to_attrition(self, top_n: int = 5) -> list:
        """Get top features pushing predictions toward attrition."""
        if self.shap_values is None:
            return []

        # For positive class (attrition=1), positive SHAP values push toward attrition
        mean_pos_shap = np.where(
            self.shap_values.values > 0,
            self.shap_values.values,
            0
        ).mean(axis=0)

        feature_names = self.feature_names or [f"Feature_{i}" for i in range(len(mean_pos_shap))]

        if len(feature_names) != len(mean_pos_shap):
            feature_names = [f"Feature_{i}" for i in range(len(mean_pos_shap))]

        # Sort by positive contribution
        indices = np.argsort(mean_pos_shap)[::-1][:top_n]

        result = []
        for idx in indices:
            if idx < len(feature_names):
                result.append({
                    "feature": feature_names[idx],
                    "contribution": round(float(mean_pos_shap[idx]), 4),
                })

        return result


def generate_explanations(model, X_train, X_test, feature_names):
    """Convenience function to generate SHAP explanations."""
    explainer = ModelExplainer()
    return explainer.explain(model, X_train, X_test, feature_names)


if __name__ == "__main__":
    from preprocessing import load_and_prepare_all_data
    from feature_engineering import engineer_features
    from sklearn.model_selection import train_test_split
    import joblib

    df, prep = load_and_prepare_all_data()
    df = engineer_features(df)
    feature_names = [c for c in df.columns if c != "attrition"]

    X = df.drop(columns=["attrition"]).values
    y = df["attrition"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_path = "ml_pipeline/models/best_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        explainer = ModelExplainer()
        importance = explainer.explain(model, X_train, X_test, feature_names)
        print("Top features:", dict(list(importance.items())[:10]))
