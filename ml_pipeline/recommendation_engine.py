"""
Intelligent Retention Recommendation Engine.
Generates dynamic recommendations based on SHAP values and feature importance.
"""

import pandas as pd
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates dynamic retention recommendations based on SHAP analysis."""

    # Map SHAP feature names → raw employee input field names for threshold checks
    FEATURE_FIELD_MAP = {
        "satisfaction_score": "job_satisfaction",
        "performance_score": "performance_rating",
        "overtime_risk": "overtime",
        "payment_tier": "monthly_income",
        "tenure_years": "years_at_company",
        "experience_years": "years_at_company",
        "promotion_gap": "years_since_last_promotion",
        "income_per_experience": "monthly_income",
        "avg_monthly_hours": "avg_monthly_hours",
        "num_projects": "num_projects",
        "work_accident": "work_accident",
        "workload_score": "num_projects",
        "perf_sat_gap": "job_satisfaction",
        "education_level": "education",
        "gender_encoded": "gender",
    }

    def __init__(self):
        self.recommendation_rules = self._build_rules()

    def _build_rules(self) -> dict:
        """Build recommendation rules for different feature contributions."""
        return {
            "satisfaction_score": {
                "condition": "low",
                "threshold": 2.5,
                "recommendations": [
                    {
                        "title": "Employee Engagement Program",
                        "description": "Implement personalized engagement activities and regular check-ins.",
                        "priority": "high",
                        "category": "engagement"
                    },
                    {
                        "title": "Work Environment Improvement",
                        "description": "Conduct stay interviews to understand dissatisfaction factors.",
                        "priority": "high",
                        "category": "environment"
                    }
                ]
            },
            "performance_score": {
                "condition": "low",
                "threshold": 2.0,
                "recommendations": [
                    {
                        "title": "Performance Support Plan",
                        "description": "Provide additional training and mentorship to improve performance.",
                        "priority": "medium",
                        "category": "development"
                    },
                    {
                        "title": "Skill Development Program",
                        "description": "Offer relevant skill-building workshops and courses.",
                        "priority": "medium",
                        "category": "training"
                    }
                ]
            },
            "performance_score": {
                "condition": "high_gap",
                "threshold": 0.5,
                "recommendations": [
                    {
                        "title": "Role Realignment",
                        "description": "Consider role adjustments to match employee capabilities with responsibilities.",
                        "priority": "medium",
                        "category": "role"
                    }
                ]
            },
            "promotion_gap": {
                "condition": "high",
                "threshold": 2.0,
                "recommendations": [
                    {
                        "title": "Career Growth Discussion",
                        "description": "Schedule a career path discussion to address promotion concerns.",
                        "priority": "high",
                        "category": "career"
                    },
                    {
                        "title": "Promotion Path Clarification",
                        "description": "Define clear criteria and timeline for next promotion.",
                        "priority": "high",
                        "category": "career"
                    }
                ]
            },
            "overtime_risk": {
                "condition": "high",
                "threshold": 0.5,
                "recommendations": [
                    {
                        "title": "Workload Reduction",
                        "description": "Review and redistribute workload to prevent burnout.",
                        "priority": "high",
                        "category": "workload"
                    },
                    {
                        "title": "Flexible Working Hours",
                        "description": "Introduce flexible schedules to improve work-life balance.",
                        "priority": "medium",
                        "category": "wellness"
                    }
                ]
            },
            "avg_monthly_hours": {
                "condition": "high",
                "threshold": 200,
                "recommendations": [
                    {
                        "title": "Sustainable Work Hours",
                        "description": "Set maximum work hour limits and monitor overtime.",
                        "priority": "high",
                        "category": "workload"
                    },
                    {
                        "title": "Team Expansion Review",
                        "description": "Assess if additional hires are needed to distribute workload.",
                        "priority": "medium",
                        "category": "staffing"
                    }
                ]
            },
            "workload_score": {
                "condition": "high",
                "threshold": 1.5,
                "recommendations": [
                    {
                        "title": "Resource Allocation Review",
                        "description": "Evaluate current resource allocation and redistribute tasks.",
                        "priority": "high",
                        "category": "workload"
                    },
                    {
                        "title": "Process Automation",
                        "description": "Identify manual processes that could be automated.",
                        "priority": "medium",
                        "category": "efficiency"
                    }
                ]
            },
            "tenure_years": {
                "condition": "high_without_promotion",
                "threshold": 5,
                "recommendations": [
                    {
                        "title": "Tenure Recognition Program",
                        "description": "Acknowledge long service with recognition and rewards.",
                        "priority": "medium",
                        "category": "retention"
                    },
                    {
                        "title": "Seniority-Based Benefits",
                        "description": "Review if benefits package reflects employee tenure.",
                        "priority": "medium",
                        "category": "compensation"
                    }
                ]
            },
            "income_per_experience": {
                "condition": "low",
                "threshold": 0.5,
                "recommendations": [
                    {
                        "title": "Compensation Review",
                        "description": "Conduct a market-based salary review for the employee.",
                        "priority": "high",
                        "category": "compensation"
                    },
                    {
                        "title": "Merit-Based Increase",
                        "description": "Consider performance-based salary adjustment.",
                        "priority": "high",
                        "category": "compensation"
                    }
                ]
            },
            "num_projects": {
                "condition": "high",
                "threshold": 6,
                "recommendations": [
                    {
                        "title": "Project Load Balancing",
                        "description": "Reduce number of concurrent projects to manageable levels.",
                        "priority": "medium",
                        "category": "workload"
                    }
                ]
            },
            "work_accident": {
                "condition": "high",
                "threshold": 0.5,
                "recommendations": [
                    {
                        "title": "Safety Program Review",
                        "description": "Review workplace safety protocols and provide additional training.",
                        "priority": "high",
                        "category": "safety"
                    }
                ]
            },
            "perf_sat_gap": {
                "condition": "high",
                "threshold": 2.0,
                "recommendations": [
                    {
                        "title": "Performance-Satisfaction Alignment",
                        "description": "Address disconnect between performance and job satisfaction.",
                        "priority": "medium",
                        "category": "engagement"
                    }
                ]
            },
            "bench_risk": {
                "condition": "high",
                "threshold": 0.5,
                "recommendations": [
                    {
                        "title": "Project Assignment Review",
                        "description": "Find suitable project assignments to prevent bench-related attrition.",
                        "priority": "high",
                        "category": "engagement"
                    },
                    {
                        "title": "Skill Utilization Assessment",
                        "description": "Ensure skills are being effectively utilized in current role.",
                        "priority": "medium",
                        "category": "development"
                    }
                ]
            },
            "payment_tier": {
                "condition": "low",
                "threshold": 2,
                "recommendations": [
                    {
                        "title": "Compensation Benchmarking",
                        "description": "Benchmark salary against industry standards for the role.",
                        "priority": "high",
                        "category": "compensation"
                    }
                ]
            }
        }

    def generate_recommendations(self, shap_contributions: dict,
                                 employee_data: dict = None,
                                 prediction_prob: float = 0.0) -> dict:
        """
        Generate retention recommendations dynamically from SHAP values.
        
        Args:
            shap_contributions: Dict of feature -> SHAP contribution value
            employee_data: Dict of employee feature values
            prediction_prob: Attrition probability (0-100)
            
        Returns:
            Dict with recommendations and reasoning
        """
        recommendations = []
        risk_factors = []
        positive_factors = []

        if not shap_contributions:
            return {
                "recommendations": [],
                "risk_factors": [],
                "positive_factors": [],
                "summary": "No SHAP data available for recommendations."
            }

        # Analyze each feature contribution
        for feature, contribution in shap_contributions.items():
            if contribution > 0.01:  # Positive contribution = pushes toward attrition
                risk_factors.append({
                    "feature": feature,
                    "impact": round(float(contribution * 100), 1),
                    "direction": "increases_attrition_risk"
                })

                # Find matching recommendation rules
                matched = self._match_recommendation(feature, contribution, employee_data)
                if matched:
                    recommendations.extend(matched)

            elif contribution < -0.01:  # Negative = reduces attrition risk
                positive_factors.append({
                    "feature": feature,
                    "impact": round(float(abs(contribution) * 100), 1),
                    "direction": "decreases_attrition_risk"
                })

        # Deduplicate recommendations
        seen_titles = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec["title"] not in seen_titles:
                seen_titles.add(rec["title"])
                unique_recommendations.append(rec)

        # Add high-level recommendations based on probability
        if prediction_prob > 70:
            unique_recommendations.insert(0, {
                "title": "Immediate Retention Intervention",
                "description": "Employee is at high risk of leaving. Schedule urgent retention discussion.",
                "priority": "critical",
                "category": "retention"
            })
        elif prediction_prob > 40:
            unique_recommendations.insert(0, {
                "title": "Proactive Retention Check-In",
                "description": "Employee shows moderate risk. Schedule a career discussion soon.",
                "priority": "high",
                "category": "retention"
            })

        # Generate summary
        summary = self._generate_summary(risk_factors, positive_factors, prediction_prob)

        result = {
            "recommendations": unique_recommendations[:8],
            "risk_factors": sorted(risk_factors, key=lambda x: x["impact"], reverse=True)[:5],
            "positive_factors": sorted(positive_factors, key=lambda x: x["impact"], reverse=True)[:3],
            "summary": summary,
        }

        return result

    def _match_recommendation(self, feature: str, contribution: float,
                               employee_data: dict = None) -> list:
        """Match feature contribution to recommendation rules.

        Uses the feature name from SHAP contributions (not raw employee_data keys)
        so that recommendations are driven by what the model says matters,
        not by trying to match employee_data field names.
        """
        matched = []

        # Normalise the SHAP feature name for matching
        shap_feat_lower = feature.lower()
        # Extract base word after splitting on '_' (e.g. overtime_risk -> overtime)
        base_word = shap_feat_lower.split("_")[0] if "_" in shap_feat_lower else shap_feat_lower

        # Also try to find the raw employee data value for this feature
        employee_value = None
        if employee_data:
            emp_key = self.FEATURE_FIELD_MAP.get(feature)
            if emp_key and emp_key in employee_data:
                employee_value = employee_data[emp_key]

        for rule_feature, rule in self.recommendation_rules.items():
            rule_feat_lower = rule_feature.lower()

            # Check if the rule feature name matches the SHAP feature name
            # Either as substring or via base word
            is_match = (
                rule_feat_lower in shap_feat_lower
                or base_word in rule_feat_lower
            )

            if is_match:
                should_apply = False

                # Try threshold check using employee data value first
                if employee_value is not None:
                    threshold = rule["threshold"]
                    if rule["condition"] == "low" and employee_value <= threshold:
                        should_apply = True
                    elif rule["condition"] == "high" and employee_value >= threshold:
                        should_apply = True
                    elif rule["condition"] == "high_gap":
                        # high_gap means perf_sat_gap is large
                        should_apply = abs(employee_value) >= threshold
                else:
                    # Fallback: apply if the SHAP contribution is significant
                    # Use a higher threshold (0.5) to avoid matching every rule
                    if contribution > 0.5:
                        should_apply = True

                if should_apply:
                    matched.extend(rule["recommendations"])

        return matched

    def _generate_summary(self, risk_factors: list, positive_factors: list,
                          probability: float) -> str:
        """Generate a human-readable summary of the analysis."""
        parts = []

        if probability > 70:
            parts.append("This employee shows a high probability of attrition.")
        elif probability > 40:
            parts.append("This employee shows a moderate probability of attrition.")
        else:
            parts.append("This employee shows a low probability of attrition.")

        if risk_factors:
            top_risks = [rf["feature"].replace("_", " ").title() for rf in risk_factors[:3]]
            parts.append(f"Key risk factors: {', '.join(top_risks)}.")

        if positive_factors:
            top_pos = [pf["feature"].replace("_", " ").title() for pf in positive_factors[:2]]
            parts.append(f"Protective factors: {', '.join(top_pos)}.")

        return " ".join(parts)

    def batch_generate(self, employees_data: list) -> list:
        """Generate recommendations for multiple employees."""
        results = []
        for emp in employees_data:
            result = self.generate_recommendations(
                emp.get("shap_contributions", {}),
                emp.get("employee_data", {}),
                emp.get("prediction_prob", 0)
            )
            result["employee_id"] = emp.get("employee_id", "unknown")
            results.append(result)
        return results


def get_recommendations(shap_contributions, employee_data=None, prediction_prob=0):
    """Convenience function to get retention recommendations."""
    engine = RecommendationEngine()
    return engine.generate_recommendations(shap_contributions, employee_data, prediction_prob)


if __name__ == "__main__":
    # Test with sample data
    sample_shap = {
        "satisfaction_score": 0.18,
        "promotion_gap": 0.15,
        "avg_monthly_hours": 0.10,
        "tenure_years": -0.05,
        "payment_tier": 0.08,
    }

    engine = RecommendationEngine()
    recs = engine.generate_recommendations(sample_shap, {"satisfaction_score": 2.0}, 75)
    print(json.dumps(recs, indent=2))
