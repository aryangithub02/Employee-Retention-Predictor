"""
Exploratory Data Analysis Module for Employee Retention Predictor.
Generates automated visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


class EDAnalyzer:
    """Automated EDA for employee retention datasets."""

    def __init__(self, output_dir: str = "ml_pipeline/reports/eda"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.insights = []

    def analyze(self, df: pd.DataFrame, target_col: str = "attrition"):
        """Run full EDA pipeline."""
        logger.info("Starting full EDA analysis...")

        self.df = df
        self.target_col = target_col

        results = {}

        results["class_distribution"] = self.plot_class_distribution()
        results["correlation"] = self.plot_correlation_matrix()
        results["numerical_distributions"] = self.plot_numerical_distributions()
        results["attrition_by_feature"] = self.plot_attrition_by_categories()
        results["boxplots"] = self.plot_boxplots_by_attrition()

        # Save insights
        self._save_insights()

        logger.info(f"EDA complete. Visualizations saved to {self.output_dir}")
        return results

    def plot_class_distribution(self):
        """Plot target class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        counts = self.df[self.target_col].value_counts()
        colors = ["#4CAF50", "#f44336"]
        axes[0].bar(["Stayed (0)", "Left (1)"], counts.values, color=colors, edgecolor="black", linewidth=1.2)
        axes[0].set_title("Attrition Class Distribution", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Count")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

        # Pie chart
        axes[1].pie(counts.values, labels=["Stayed", "Left"], autopct="%1.1f%%",
                    colors=colors, startangle=90, explode=(0, 0.05),
                    shadow=True, textprops={"fontsize": 12, "fontweight": "bold"})
        axes[1].set_title("Attrition Proportion", fontsize=14, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "class_distribution.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

        self.insights.append({
            "type": "class_distribution",
            "message": f"Class imbalance: {counts.get(0, 0)} stayed vs {counts.get(1, 0)} left "
                       f"({counts.get(1, 0) / sum(counts) * 100:.1f}% attrition rate)"
        })
        return path

    def plot_correlation_matrix(self):
        """Plot correlation heatmap."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "correlation_matrix.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

        # Find top features correlated with attrition
        if self.target_col in corr.columns:
            target_corr = corr[self.target_col].drop(self.target_col).abs().sort_values(ascending=False)
            top_features = target_corr.head(10)
            self.insights.append({
                "type": "correlation",
                "message": f"Top features correlated with attrition: {dict(zip(top_features.index, top_features.values.round(3)))}"
            })

        return path

    def plot_numerical_distributions(self):
        """Plot histograms for numerical features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col]

        # Limit to meaningful columns
        cols_to_plot = [c for c in numeric_cols if self.df[c].nunique() > 2][:12]

        if not cols_to_plot:
            return None

        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        colors = {"Stayed": "#4CAF50", "Left": "#f44336"}

        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            for label, color in colors.items():
                subset = self.df[self.df[self.target_col] == (1 if "Left" in label else 0)]
                ax.hist(subset[col].dropna(), bins=30, alpha=0.6, color=color,
                       label=label, density=True)
            ax.set_title(f"{col.replace('_', ' ').title()}", fontsize=11)
            ax.set_xlabel("")
            ax.legend(fontsize=8)

        # Hide unused subplots
        for j in range(len(cols_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Numerical Feature Distributions by Attrition", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "numerical_distributions.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def plot_attrition_by_categories(self):
        """Plot attrition rates by categorical features."""
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not cat_cols:
            return None

        n_cols = 2
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(cat_cols):
            ax = axes[i]
            crosstab = pd.crosstab(self.df[col], self.df[self.target_col], normalize="index") * 100
            crosstab.columns = ["Stayed %", "Left %"]
            crosstab = crosstab.sort_values("Left %", ascending=False)
            crosstab[["Stayed %", "Left %"]].plot(kind="bar", ax=ax, color=["#4CAF50", "#f44336"], edgecolor="black")
            ax.set_title(f"Attrition by {col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Percentage")
            ax.set_xlabel("")
            ax.legend(fontsize=9)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "attrition_by_categories.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def plot_boxplots_by_attrition(self):
        """Plot boxplots comparing numerical features by attrition status."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col and self.df[c].nunique() > 2]

        cols_to_plot = [c for c in numeric_cols if self.df[c].nunique() > 3][:8]

        if not cols_to_plot:
            return None

        n_cols = 2
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            sns.boxplot(x=self.df[self.target_col].astype(str), y=self.df[col],
                       palette={"0": "#4CAF50", "1": "#f44336"}, ax=ax)
            ax.set_title(f"{col.replace('_', ' ').title()} by Attrition", fontsize=12, fontweight="bold")
            ax.set_xlabel("Attrition (0=Stayed, 1=Left)")
            ax.set_ylabel("")

        plt.suptitle("Feature Comparison: Stayed vs Left", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "boxplots_by_attrition.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        return path

    def _save_insights(self):
        """Save EDA insights to JSON."""
        path = os.path.join(self.output_dir, "eda_insights.json")
        with open(path, "w") as f:
            json.dump(self.insights, f, indent=2)
        logger.info(f"EDA insights saved to {path}")


def run_eda(df: pd.DataFrame, target_col: str = "attrition"):
    """Convenience function to run EDA."""
    analyzer = EDAnalyzer()
    return analyzer.analyze(df, target_col)


if __name__ == "__main__":
    from preprocessing import load_and_prepare_all_data
    df, prep = load_and_prepare_all_data()
    run_eda(df)
