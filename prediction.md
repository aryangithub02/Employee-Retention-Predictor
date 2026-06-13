# Prediction System — How It Works

## Overview

The prediction system uses a **RandomForest classifier** trained on 19,650 employee records to estimate the likelihood of employee attrition (an employee leaving the company). The output is a probability expressed as a percentage, along with a risk level and actionable recommendations.

---

## What the Percentage Represents

### Retention Probability vs Attrition Probability

The model internally calculates an **attrition probability** — the likelihood that the employee will leave.

| Concept | Meaning | Example |
|---|---|---|
| **Attrition Probability** | Chance the employee leaves | 6.8% |
| **Retention Probability** | Chance the employee stays | 93.2% (= 100% − 6.8%) |

> **In the UI, you see Retention Probability**, so higher numbers are better. A 93.2% retention probability means the model is 93.2% confident the employee will stay.

---

## How a Prediction is Made (Step by Step)

```
User Input (e.g., age, department, satisfaction, overtime...)
        │
        ▼
1. Feature Engineering ────────────────────────────────────────
   Raw inputs are transformed into 31 model features:
   - gender       → gender_encoded (0 or 1)
   - education    → education_level (1=Bachelor's, 2=Master's, 3=PhD)
   - salary_level → salary_encoded + salary_low/medium/high dummies
   - overtime     → overtime_risk (0 or 1)
   - monthly_income → payment_tier (1/2/3)
   - years_at_company → tenure_years
   - Derived features:
     • promotion_gap, income_per_experience, satisfaction_index
     • overtime_risk, bench_risk, experience_ratio, workload_score
     • tenure_category, perf_sat_gap
     • Interaction features: low_sat_high_workload, long_tenure_no_promo
        │
        ▼
2. Feature Alignment ──────────────────────────────────────────
   The engineered features are reindexed to match exactly the
   31 columns the model was trained on. Missing columns get 0.
        │
        ▼
3. Model Prediction ───────────────────────────────────────────
   RandomForest.predict_proba(X) returns [P(stay), P(leave)]
   We take P(leave) as the attrition probability.
        │
        ▼
4. Risk Level Assignment ──────────────────────────────────────
   attrition_probability ≤ 30%  →  Low Risk     →  "Likely To Stay"
   attrition_probability 31-70%  →  Medium Risk  →  "Moderate Risk"
   attrition_probability > 70%   →  High Risk    →  "Likely To Leave"
        │
        ▼
5. SHAP Explanation ───────────────────────────────────────────
   TreeExplainer identifies which features most influenced
   this prediction and their contribution (e.g., "+15% risk").
        │
        ▼
6. Recommendations ────────────────────────────────────────────
   Based on risk factors, personalized retention recommendations
   are generated (e.g., "Improve work-life balance", "Promotion review").
        │
        ▼
7. Saved to Database ──────────────────────────────────────────
   The prediction, input data, and SHAP explanation are persisted
   for future reference and analysis.
```

---

## The 31 Features Used for Prediction

The model considers these 31 pieces of information for every prediction:

| Category | Features |
|---|---|
| **Demographics** | age, education_level, gender_encoded |
| **Compensation** | payment_tier, salary_encoded, salary_high, salary_low, salary_medium, monthly_income |
| **Job Details** | department (as city dummies), job_role, num_projects, avg_monthly_hours |
| **Tenure & Experience** | tenure_years, experience_years, years_at_company, years_since_last_promotion |
| **Satisfaction** | satisfaction_score, job_satisfaction, environment_satisfaction, work_life_balance |
| **Performance** | performance_score, performance_rating, promotion_last_5years |
| **Risk Signals** | overtime_risk, ever_benched_encoded, work_accident, bench_risk |
| **Engineered Features** | promotion_gap, income_per_experience, satisfaction_index, overtime_risk, experience_ratio, workload_score, tenure_category, perf_sat_gap, low_sat_high_workload, long_tenure_no_promo |

---

## Interpreting the Results

### Example: 93.2% Retention Probability

```
┌──────────────────────────────────────────────────┐
│  Prediction Result                               │
│                                                  │
│           93.2%                                  │
│      Retention Probability                       │
│                                                  │
│      Likely To Stay  ●  Low Risk                 │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │██████████████████████████████████░░░░░░░│    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Key Factors:                                    │
│  Work Life Balance          -12.3% ← reduces risk│
│  Satisfaction Score         -8.1%  ← reduces risk│
│  Overtime Risk               +5.2% → increases    │
│  Promotion Gap               +4.1% → increases    │
└──────────────────────────────────────────────────┘
```

- **93.2%** means the model estimates a 93.2% probability the employee will stay (only 6.8% probability they'll leave)
- **"Likely To Stay"** because the probability of leaving is below 30%
- **"Low Risk"** confirmed by the same threshold
- **Key Factors** show a green "-" for features decreasing attrition risk and red "+" for features increasing it

### Risk Thresholds Reference

| Retention Probability | Attrition Probability | Risk Level | Prediction |
|---|---|---|---|
| **70% – 100%** | 0% – 30% | **Low** 🟢 | Likely To Stay |
| **30% – 69%** | 31% – 70% | **Medium** 🟡 | Moderate Risk |
| **0% – 29%** | 71% – 100% | **High** 🔴 | Likely To Leave |

---

## Model Performance

The RandomForest model achieves:
- **ROC-AUC**: 0.975 (excellent discrimination)
- **Accuracy**: 88.1%
- **Recall**: 95.2% (catches most employees who will leave)
- **F1 Score**: 0.768

---

## What Factors Increase Risk?

Based on SHAP analysis across all predictions, the top risk-increasing factors are:

1. **Low job satisfaction** (≤ 2/5) — strongest predictor
2. **Poor work-life balance** (≤ 2/5)
3. **Frequent overtime** (Yes)
4. **Long time since last promotion** (> 3 years)
5. **Low monthly income** relative to experience
6. **High number of projects** (> 6)
7. **Long commute** (> 20 km)

---

## Fallback Behavior

If the ML model is not loaded (e.g., first startup or model file missing), the system falls back to a **rule-based heuristic** that assigns risk points based on common HR risk factors. This fallback is less accurate but ensures the system never returns an error.

The prediction response includes `"note": "Rule-based estimation (model not loaded)"` when using the fallback.

---

## Technical Details

### API Endpoint

```
POST /api/predict
Content-Type: application/json

{
  "age": 35,
  "gender": "Male",
  "department": "Engineering",
  "monthly_income": 75000,
  "job_satisfaction": 2,
  "work_life_balance": 2,
  "overtime": "Yes",
  "years_at_company": 5,
  "years_since_last_promotion": 3,
  "num_projects": 6,
  "avg_monthly_hours": 220,
  "salary_level": "medium",
  ...
}
```

### Response Format

```json
{
  "attrition_probability": 6.8,
  "prediction": "Likely To Stay",
  "risk_level": "Low",
  "confidence": 0.932,
  "shap_explanation": {
    "base_value": 0.35,
    "feature_contributions": {
      "Work Life Balance": "-12.3%",
      "Satisfaction Score": "-8.1%",
      "Overtime Risk": "+5.2%"
    }
  },
  "recommendations": { ... },
  "prediction_id": 42
}
```

### Retention Probability (UI)

The UI converts the response to display **Retention Probability**:
```
Retention Probability = 100 − attrition_probability
```

This is purely a display transformation — the backend always returns `attrition_probability` so the risk levels and prediction labels remain accurate.

---

## Summary

| If you see... | It means... |
|---|---|
| **93.2%** | 93.2% chance the employee will stay (good) |
| **72.1%** | 72.1% chance the employee will stay (moderate) |
| **18.5%** | Only 18.5% chance the employee will stay (warning — high risk) |
| **"Likely To Stay"** | Attrition probability ≤ 30% (retention ≥ 70%) |
| **"Likely To Leave"** | Attrition probability > 70% (retention < 30%) |
