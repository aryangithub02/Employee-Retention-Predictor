"""
Model Training Module for Employee Retention Predictor.
Trains multiple models with hyperparameter tuning and evaluation.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import logging
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates multiple ML models for attrition prediction."""

    def __init__(self, output_dir: str = "ml_pipeline/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def define_models(self):
        """Define the models to train."""
        self.models = {
            "LogisticRegression": {
                "model": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
                "params": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                    "solver": ["lbfgs", "liblinear"],
                    "class_weight": [None, "balanced"],
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": [None, "balanced"],
                }
            },
            "XGBoost": {
                "model": None,  # Lazy import
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                }
            },
            "LightGBM": {
                "model": None,  # Lazy import
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [-1, 5, 7, 9],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_child_samples": [10, 20, 30],
                    "num_leaves": [31, 50, 70],
                }
            }
        }
        return self.models

    def _get_xgboost(self):
        """Lazy import XGBoost."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
        except ImportError:
            logger.warning("XGBoost not installed, skipping")
            return None

    def _get_lightgbm(self):
        """Lazy import LightGBM."""
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(random_state=42, verbosity=-1, force_col_wise=True)
        except ImportError:
            logger.warning("LightGBM not installed, skipping")
            return None

    def train_all(self, X_train, y_train, X_test, y_test, use_hyperopt: bool = True, n_iter: int = 20):
        """Train all defined models."""
        self.define_models()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Initialize XGBoost and LightGBM
        xgb_model = self._get_xgboost()
        lgb_model = self._get_lightgbm()
        if xgb_model:
            self.models["XGBoost"]["model"] = xgb_model
        if lgb_model:
            self.models["LightGBM"]["model"] = lgb_model

        # Remove models that couldn't be initialized
        self.models = {k: v for k, v in self.models.items() if v["model"] is not None}

        logger.info(f"Training {len(self.models)} models: {list(self.models.keys())}")

        for name, config in self.models.items():
            logger.info(f"Training {name}...")
            try:
                self._train_single(name, config, use_hyperopt, n_iter)
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        self._select_best_model()
        self._generate_report()
        self._save_models()
        return self.results

    def _train_single(self, name: str, config: dict, use_hyperopt: bool, n_iter: int):
        """Train a single model with optional hyperparameter tuning."""
        model = config["model"]
        params = config["params"]

        if use_hyperopt and params:
            logger.info(f"Running hyperparameter tuning for {name}...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                model, params, n_iter=min(n_iter, 10), cv=cv,
                scoring="roc_auc", random_state=42, n_jobs=-1,
                verbose=0
            )
            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
            logger.info(f"{name} best CV score: {cv_score:.4f}")
            logger.info(f"{name} best params: {best_params}")
        else:
            logger.info(f"Training {name} without hyperparameter tuning...")
            model.fit(self.X_train, self.y_train)
            best_model = model
            best_params = {}
            cv_score = None

        # Evaluate
        metrics = self._evaluate(best_model, name)
        metrics["cv_score"] = float(cv_score) if cv_score else None
        metrics["best_params"] = best_params

        self.trained_models[name] = best_model
        self.results[name] = metrics

        # Plot confusion matrix
        self._plot_confusion_matrix(name, metrics["y_pred"], metrics["y_prob"])
        self._plot_roc_curve(name, metrics["y_prob"])
        self._plot_precision_recall_curve(name, metrics["y_prob"])

    def _evaluate(self, model, name: str) -> dict:
        """Evaluate a trained model."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(self.y_test, y_prob)),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
        }

        logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, "
                   f"ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics

    def _select_best_model(self):
        """Select the best model based on ROC-AUC."""
        for name, metrics in self.results.items():
            score = metrics.get("roc_auc", 0)
            if score > self.best_score:
                self.best_score = score
                self.best_model = self.trained_models[name]
                self.best_model_name = name

        logger.info(f"Best model: {self.best_model_name} with ROC-AUC: {self.best_score:.4f}")

    def _plot_confusion_matrix(self, name: str, y_pred, y_prob):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                   xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
        ax.set_title(f"Confusion Matrix - {name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"confusion_matrix_{name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_roc_curve(self, name: str, y_prob):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc_score = roc_auc_score(self.y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.2, color="blue")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"roc_curve_{name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_precision_recall_curve(self, name: str, y_prob):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, "g-", linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2, color="green")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve - {name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"pr_curve_{name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

    def _generate_report(self):
        """Generate model leaderboard report."""
        leaderboard = []
        for name, metrics in self.results.items():
            leaderboard.append({
                "model": name,
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
            })

        leaderboard = sorted(leaderboard, key=lambda x: x["roc_auc"], reverse=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "best_model": self.best_model_name,
            "best_roc_auc": round(self.best_score, 4),
            "leaderboard": leaderboard,
        }

        path = os.path.join(self.output_dir, "model_leaderboard.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Model leaderboard saved to {path}")

        # Print leaderboard
        print("\n" + "=" * 80)
        print(f"{'Model Leaderboard':^80}")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
        print("-" * 80)
        for m in leaderboard:
            print(f"{m['model']:<25} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
                  f"{m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['roc_auc']:<12.4f}")
        print("=" * 80)
        print(f"Best Model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
        print("=" * 80)

        self.leaderboard = leaderboard
        return report

    def _save_models(self):
        """Save trained models."""
        # Save best model
        if self.best_model:
            best_path = os.path.join(self.output_dir, "best_model.pkl")
            joblib.dump(self.best_model, best_path)
            logger.info(f"Best model saved to {best_path}")

            # Save model metadata
            meta = {
                "best_model_name": self.best_model_name,
                "best_roc_auc": self.best_score,
                "timestamp": datetime.now().isoformat(),
            }
            meta_path = os.path.join(self.output_dir, "best_model_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Save all trained models
        for name, model in self.trained_models.items():
            path = os.path.join(self.output_dir, f"{name}.pkl")
            joblib.dump(model, path)

    def get_best_model(self):
        """Return the best trained model and its name."""
        return self.best_model, self.best_model_name


def train_models(X_train, y_train, X_test, y_test, use_hyperopt=True, n_iter=20):
    """Convenience function to train all models."""
    trainer = ModelTrainer()
    return trainer.train_all(X_train, y_train, X_test, y_test, use_hyperopt, n_iter)


if __name__ == "__main__":
    from preprocessing import load_and_prepare_all_data
    from feature_engineering import engineer_features
    from sklearn.model_selection import train_test_split

    df, prep = load_and_prepare_all_data()
    df = engineer_features(df)

    X = df.drop(columns=["attrition"])
    y = df["attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_models(X_train.values, y_train.values, X_test.values, y_test.values)
