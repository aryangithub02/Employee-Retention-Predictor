# Employee Retention Predictor — Project Progress

> **ML-Powered Employee Attrition Prediction & Retention Analytics Platform**
> Last updated: June 13, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Tech Stack](#2-architecture--tech-stack)
3. [Data & Datasets](#3-data--datasets)
4. [ML Pipeline](#4-ml-pipeline)
5. [Model Performance & Accuracy](#5-model-performance--accuracy)
6. [SHAP Explainability & Feature Importance](#6-shap-explainability--feature-importance)
7. [Survival Analysis & Forecasting](#7-survival-analysis--forecasting)
8. [Retention Recommendation Engine](#8-retention-recommendation-engine)
9. [Backend API](#9-backend-api)
10. [Database Schema](#10-database-schema)
11. [Frontend Application](#11-frontend-application)
12. [User Roles & Workflows](#12-user-roles--workflows)
13. [Decision-Making Framework](#13-decision-making-framework)
14. [Project Structure](#14-project-structure)
15. [Setup Instructions](#15-setup-instructions)
16. [Current Status & Next Steps](#16-current-status--next-steps)

---

## 1. Project Overview

The **Employee Retention Predictor** is a full-stack, ML-powered platform designed to help HR teams proactively identify employees at risk of leaving and take data-driven action to improve retention. The system combines:

- **Machine Learning** — Trains and deploys multiple classifiers (LightGBM, RandomForest, XGBoost, LogisticRegression) to predict attrition probability from employee features
- **SHAP Explainability** — Provides transparent, per-prediction explanations of which features drive risk
- **Retention Recommendations** — Generates actionable, priority-ranked suggestions for each at-risk employee
- **Survival Analysis** — Uses Cox Proportional Hazards to estimate time-to-attrition
- **Trend Forecasting** — Uses Prophet to forecast future attrition rates
- **Real-time Dashboard** — Interactive analytics with charts, KPIs, and drill-downs
- **Employee Self-Assessment Portal** — Surveys to collect wellbeing data from employees

---

## 2. Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)            │
│  Vite + Tailwind CSS + Recharts + Lucide Icons             │
│  Port: 5173 → proxies /api to backend                       │
├─────────────────────────────────────────────────────────────┤
│                  Backend (FastAPI + Python)                   │
│  Uvicorn on port 8000                                        │
│  Routes: /api/predict, /api/employees, /api/model-*         │
│  Services: MLService (predictions), DBService (queries)     │
├─────────────────────────────────────────────────────────────┤
│              ML Pipeline (scikit-learn, SHAP, lifelines)     │
│  Preprocessing → Feature Engineering → Training → Explainer  │
│  Models saved as .pkl files in ml_pipeline/models/           │
├─────────────────────────────────────────────────────────────┤
│              Database (SQLite default / PostgreSQL optional)  │
│  Tables: employees, predictions, training_runs,             │
│          model_metrics, datasets, survey_responses,          │
│          recommendations                                     │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | React, TypeScript, Vite | React 18.3, TS 5.5, Vite 5.4 |
| UI Framework | Tailwind CSS | 3.4 |
| Charts | Recharts | 2.12 |
| HTTP Client | Axios | 1.7 |
| Icons | Lucide React | 0.400 |
| Backend | FastAPI + Uvicorn | 0.136, 0.49 |
| ORM | SQLAlchemy | 2.0 |
| ML Models | scikit-learn, XGBoost, LightGBM | 1.9, 3.2, 4.6 |
| Explainability | SHAP (TreeExplainer) | 0.52 |
| Survival Analysis | lifelines (Cox PH) | — |
| Forecasting | Prophet | 1.3 |
| Data Processing | pandas, numpy | 2.2, 2.2 |
| Visualization | matplotlib, seaborn | 3.10, 0.13 |

---

## 3. Data & Datasets

### Source Datasets

| Dataset | Records | Columns | Source |
|---------|---------|---------|--------|
| `Employee.csv` | 4,653 | 9 | Employee records with city, payment tier, education |
| `hr_employee_churn_data.csv` | 14,997 | 10 | HR churn data with satisfaction, projects, hours |

### Combined Dataset

After preprocessing and cleaning:
- **14,664 records** (duplicates removed, columns unified)
- **21 raw columns** → **31 engineered features** for training
- **Attrition rate: 20.71%** (3,037 left vs 11,627 stayed)
- **Class imbalance addressed** through model `class_weight` and stratified splits

### Key Employee Features Collected

| Category | Features |
|----------|----------|
| **Demographics** | Age, Gender, Education, Marital Status |
| **Employment** | Department, Job Role, Years at Company, Overtime, Monthly Hours, Projects |
| **Compensation** | Monthly Income, Salary Level (low/medium/high), Payment Tier |
| **Satisfaction** | Job Satisfaction (1-5), Environment Satisfaction, Work-Life Balance |
| **Performance** | Performance Rating, Years Since Last Promotion, Training Sessions |
| **Risk Signals** | Ever Benched, Work Accident, Commute Distance |

---

## 4. ML Pipeline

The ML pipeline (`ml_pipeline/`) is a 7-step end-to-end system:

### Step 1: Data Loading & Preprocessing (`preprocessing.py`)
- Loads both CSV datasets with encoding detection
- Maps column names to unified schema (e.g., `LeaveOrNot` → `attrition`, `left` → `attrition`)
- Handles missing values (median for numerical, mode for categorical)
- Removes duplicates and infinite values
- Combines both datasets into a single DataFrame
- Generates preprocessing report (saved to `reports/preprocessing_report.json`)

### Step 2: Exploratory Data Analysis (`eda.py`)
- Class distribution analysis (pie charts, bar charts)
- Correlation matrix heatmap
- Numerical feature distributions by attrition status
- Attrition rates by categorical features
- Boxplots comparing stayed vs. left employees
- Insights saved to `reports/eda/eda_insights.json`

**Key EDA Insights:**
- 11,627 stayed vs 3,037 left (20.7% attrition rate)
- Top correlations with attrition: satisfaction_score (0.33), tenure_years (0.229), education_level (0.164)

### Step 3: Feature Engineering (`feature_engineering.py`)
Creates 10+ derived features from raw data:

| Engineered Feature | Description |
|-------------------|-------------|
| `promotion_gap` | Tenure × (1 − promoted_in_5yr) — measures promotion drought |
| `income_per_experience` | Payment tier / (experience + 1) — income fairness ratio |
| `satisfaction_index` | Mean of all satisfaction-related columns |
| `overtime_risk` | Binary flag for excessive working hours |
| `bench_risk` | Ever-benched status (risk of disengagement) |
| `experience_ratio` | Tenure / age — career velocity metric |
| `workload_score` | Projects × (hours / 160) — workload intensity |
| `tenure_category` | Binned tenure (0–1, 1–3, 3–5, 5–10, 10+ years) |
| `perf_sat_gap` | Performance score − Satisfaction score — misalignment |
| `low_sat_high_workload` | Interaction: low satisfaction AND high workload |
| `long_tenure_no_promo` | Interaction: long tenure AND no promotion |

### Step 4: Train-Test Split
- **80/20 stratified split** (preserves attrition ratio)
- Training: 11,731 samples
- Testing: 2,933 samples

### Step 5: Model Training (`training.py`)
- 4 models trained with **RandomizedSearchCV** (3-fold StratifiedKFold)
- Hyperparameter tuning with 10 iterations per model
- Best model selected by **ROC-AUC** score
- Confusion matrices, ROC curves, and PR curves generated

### Step 6: SHAP Explainability (`explainability.py`)
- **TreeExplainer** for tree-based models (RandomForest, XGBoost, LightGBM)
- Global feature importance ranked by mean |SHAP value|
- Per-prediction explanations with feature contributions
- Summary plots, waterfall plots, dependence plots
- Results saved to `reports/shap/`

### Step 7: Survival Analysis & Forecasting (`survival_analysis.py`)
- **Cox Proportional Hazards** model (via lifelines library)
- Time-to-event modeling using tenure as time variable
- **Prophet**-based attrition trend forecasting (12-month horizon)
- Department-level risk scoring

---

## 5. Model Performance & Accuracy

### Model Leaderboard (Latest Run: June 13, 2026)

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🏆 1 | **LightGBM** | **95.19%** | **94.47%** | 81.55% | **0.875** | **0.9774** |
| 2 | RandomForest | 87.93% | 63.98% | **95.39%** | 0.766 | 0.9748 |
| 3 | XGBoost | 94.85% | 92.38% | 81.88% | 0.868 | 0.9746 |
| 4 | LogisticRegression | 85.92% | 70.73% | 54.53% | 0.616 | 0.9019 |

### Best Model: LightGBM

**LightGBM** was selected as the best model with:
- **ROC-AUC: 0.9774** (excellent discrimination between stayers and leavers)
- **Accuracy: 95.19%** (overall correct predictions)
- **Precision: 94.47%** (when it predicts someone will leave, it's right 94% of the time)
- **Recall: 81.55%** (catches 82% of actual leavers)
- **F1 Score: 0.875** (harmonic mean of precision and recall)

**Best Hyperparameters:**
```json
{
  "subsample": 0.6,
  "num_leaves": 70,
  "n_estimators": 100,
  "min_child_samples": 10,
  "max_depth": -1,
  "learning_rate": 0.05
}
```

**Confusion Matrix (LightGBM):**
```
              Predicted Stay  Predicted Leave
Actual Stay     2,297             29
Actual Leave      112            495
```

### Model Selection Rationale

- **LightGBM** chosen for best ROC-AUC (0.9774) — best overall at distinguishing classes
- **RandomForest** has highest recall (95.4%) — best at catching leavers, but lower precision
- Trade-off: LightGBM is more balanced; RandomForest is more aggressive (fewer false negatives, more false positives)

---

## 6. SHAP Explainability & Feature Importance

### Global Feature Importance (Top 15)

| Rank | Feature | Mean |SHAP Value| | Category |
|------|---------|------|-----------|----------|
| 1 | `perf_sat_gap` | 0.662 | Engineered — performance-satisfaction misalignment |
| 2 | `satisfaction_score` | 0.615 | Core — job satisfaction rating |
| 3 | `promotion_gap` | 0.525 | Engineered — time without promotion |
| 4 | `workload_score` | 0.359 | Engineered — projects × hours intensity |
| 5 | `education_level` | 0.294 | Demographic — education tier |
| 6 | `tenure_years` | 0.271 | Employment — years at company |
| 7 | `experience_years` | 0.213 | Employment — total experience |
| 8 | `avg_monthly_hours` | 0.130 | Employment — working hours |
| 9 | `payment_tier` | 0.126 | Compensation — salary tier |
| 10 | `experience_ratio` | 0.109 | Engineered — tenure/age ratio |
| 11 | `gender_encoded` | 0.056 | Demographic |
| 12 | `city_Bangalore` | 0.053 | Location |
| 13 | `city_New Delhi` | 0.047 | Location |
| 14 | `performance_score` | 0.039 | Performance rating |
| 15 | `num_projects` | 0.035 | Workload — number of projects |

### Key Takeaways

- **Engineered features dominate**: 4 of the top 5 features are engineered (not in raw data)
- **Satisfaction is king**: Job satisfaction + performance-satisfaction gap are the strongest predictors
- **Promotion matters**: The promotion gap (time without promotion × tenure) is the 3rd most important factor
- **Workload pressure**: High workload combined with low satisfaction is a critical risk signal

### Per-Prediction SHAP Explanations

For each prediction, the system shows:
1. **Base value** (log-odds baseline from training data)
2. **Top 5 feature contributions** (converted from log-odds to probability-scale %)
3. **Direction** (+% increases attrition risk, −% decreases it)

Example:
```
Feature Contributions:
  Overtime Risk          +18.2%
  Satisfaction Score     +15.1%
  Promotion Gap          +12.5%
  Work Life Balance      -8.3%
  Payment Tier           -5.1%
```

---

## 7. Survival Analysis & Forecasting

### Cox Proportional Hazards Model

- **Concordance Index: 0.9488** (excellent — 1.0 is perfect)
- **14,664 observations**, 3,037 events
- 15 features used for hazard modeling

**Hazard Ratios (risk-increasing factors have HR > 1):**

| Feature | Hazard Ratio | Interpretation |
|---------|-------------|----------------|
| education_level | 1.288 | Higher education → 29% higher hazard |
| ever_benched_encoded | 1.112 | Being benched → 11% higher hazard |
| performance_score | 1.062 | Higher performance → 6% higher hazard |
| avg_monthly_hours | 1.002 | More hours → slight hazard increase |
| age | 0.958 | Older → 4% lower hazard (protective) |
| payment_tier | 0.859 | Higher pay → 14% lower hazard |
| satisfaction_score | 0.587 | Higher satisfaction → 41% lower hazard |
| promotion_gap | 0.460 | Longer gap → 54% lower hazard (survival) |
| tenure_years | 0.398 | Longer tenure → 60% lower hazard |
| promotion_last_5years | 0.007 | Recent promotion → 99% lower hazard |

### Attrition Trend Forecast (Prophet)

- **24 months of historical data** (Jul 2024 – Jun 2026)
- **12-month forecast** (Jun 2026 – May 2027)
- **Trend direction: Decreasing** overall
- **Next month prediction: 27.8%** (elevated)
- **Next quarter prediction: 24.4%**

**Forecast Highlights:**
- Historical range: 16.2% – 29.1%
- Forecasted peak: 34.0% (October 2026)
- Forecasted low: 18.3% (March 2027)
- Seasonal pattern detected (yearly seasonality)

### Department Risk Analysis

| Department | Risk Score | Risk Level | Employee Count |
|-----------|-----------|------------|----------------|
| Engineering | 77.3% | 🔴 High | 2,565 |
| HR | 70.1% | 🔴 High | 1,607 |
| Marketing | 61.0% | 🔴 High | 1,906 |
| IT | 59.5% | 🟡 Medium | 3,543 |
| Finance | 28.9% | 🟢 Low | 3,518 |
| Sales | 23.9% | 🟢 Low | 1,280 |

---

## 8. Retention Recommendation Engine

The recommendation engine (`recommendation_engine.py`) generates dynamic, personalized retention suggestions based on:

1. **SHAP contributions** — Which features are driving attrition risk?
2. **Employee data** — What are the actual values for those features?
3. **Prediction probability** — How urgent is the intervention?

### Recommendation Categories

| Category | Trigger Features | Example Recommendations |
|----------|-----------------|----------------------|
| **Engagement** | Low satisfaction, bench_risk | Employee Engagement Program, Work Environment Improvement |
| **Career** | High promotion_gap | Career Growth Discussion, Promotion Path Clarification |
| **Workload** | High overtime_risk, workload_score, avg_monthly_hours | Workload Reduction, Flexible Working Hours, Team Expansion Review |
| **Compensation** | Low payment_tier, low income_per_experience | Compensation Review, Merit-Based Increase, Compensation Benchmarking |
| **Development** | Low performance_score, bench_risk | Performance Support Plan, Skill Development Program |
| **Safety** | High work_accident | Safety Program Review |
| **Retention** | High probability (>70%) | Immediate Retention Intervention (critical priority) |

### Priority Levels

- **Critical**: Probability > 70% — Immediate intervention needed
- **High**: Probability 40-70% — Proactive check-in recommended
- **Medium**: Feature-specific recommendations based on SHAP analysis
- **Low**: General wellness suggestions

---

## 9. Backend API

### API Endpoints Summary

| Method | Endpoint | Description | Tag |
|--------|----------|-------------|-----|
| `GET` | `/` | Service status & endpoint listing | Root |
| `GET` | `/health` | Health check | Root |
| `POST` | `/api/predict` | Single employee attrition prediction | Predictions |
| `POST` | `/api/predict/batch` | Batch predictions for multiple employees | Predictions |
| `GET` | `/api/predictions` | Recent prediction history | Predictions |
| `GET` | `/api/employee-risk` | Organization-wide risk distribution | Employee Analytics |
| `GET` | `/api/department-risk` | Department-level risk breakdown | Employee Analytics |
| `GET` | `/api/organization-insights` | Comprehensive org insights + forecast | Employee Analytics |
| `GET` | `/api/employees` | Paginated employee list | Employee Analytics |
| `GET` | `/api/employees/search` | Search & filter employees | Employee Analytics |
| `GET` | `/api/employees/{id}` | Single employee details | Employee Analytics |
| `POST` | `/api/employees` | Create new employee | Employee Analytics |
| `PUT` | `/api/employees/{id}` | Update employee record | Employee Analytics |
| `DELETE` | `/api/employees/{id}` | Delete employee record | Employee Analytics |
| `GET` | `/api/high-risk-employees` | High-risk employees list | Employee Analytics |
| `POST` | `/api/upload-dataset` | Upload CSV for retraining | Model Management |
| `POST` | `/api/train-model` | Trigger model retraining (background) | Model Management |
| `GET` | `/api/model-metrics` | Model leaderboard & performance | Model Management |
| `GET` | `/api/feature-importance` | SHAP feature importance | Model Management |
| `GET` | `/api/model-info` | Current model status & metadata | Model Management |
| `POST` | `/api/employee-survey` | Submit employee wellbeing survey | Employee Portal |
| `GET` | `/api/survey/employee/{id}` | Get survey history for employee | Employee Portal |
| `GET` | `/api/survey-responses` | List all survey responses | Employee Portal |

### Prediction API Flow

```
POST /api/predict
  ↓
Pydantic validation (EmployeeInput)
  ↓
MLService.predict(data)
  ├── Load model (lazy, first request only)
  ├── _prepare_features():
  │   ├── Map raw fields → training columns
  │   ├── FeatureEngineer.create_all_features()
  │   ├── Align to 31 training features
  │   └── Return np.ndarray (1, 31)
  ├── model.predict_proba(X) → attrition probability
  ├── Risk level: ≤30% Low, 31-70% Medium, >70% High
  ├── SHAP explanation (TreeExplainer)
  ├── RecommendationEngine.generate_recommendations()
  └── Save prediction to database
  ↓
JSON response with probability, risk, SHAP, recommendations
```

### Key API Features

- **Singleton ML model**: Loaded once at first request, reused for all subsequent predictions
- **Feature alignment**: Raw input is mapped and engineered to exactly match the 31 training features
- **Fallback prediction**: Rule-based heuristic if ML model fails to load
- **Prediction persistence**: Every prediction saved to database automatically
- **Background retraining**: Model retraining runs asynchronously via FastAPI BackgroundTasks
- **DB-first analytics**: Routes query database first, fall back to file-based data
- **Full OpenAPI docs**: Swagger UI at `/docs`, ReDoc at `/redoc`

---

## 10. Database Schema

**Default: SQLite** (`employee_retention.db`). PostgreSQL supported via environment variables.

### Tables

#### `employees` — Core employee records
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| employee_id | VARCHAR(50) UNIQUE | Human-readable ID (e.g., EMP0001) |
| age | INTEGER | Employee age |
| gender | VARCHAR(20) | Male/Female |
| department | VARCHAR(100) | Engineering, Sales, HR, etc. |
| job_role | VARCHAR(100) | Software Engineer, Sales Executive, etc. |
| monthly_income | FLOAT | Monthly salary |
| job_satisfaction | INTEGER | 1-5 scale |
| environment_satisfaction | INTEGER | 1-5 scale |
| work_life_balance | INTEGER | 1-5 scale |
| distance_from_home | INTEGER | Distance in km |
| years_at_company | INTEGER | Tenure |
| years_since_last_promotion | INTEGER | Promotion gap |
| overtime | VARCHAR(10) | Yes/No |
| performance_rating | INTEGER | 1-5 scale |
| education | VARCHAR(50) | Bachelor's/Master's/PhD |
| marital_status | VARCHAR(20) | Single/Married/Divorced |
| attrition | INTEGER | 0=Stayed, 1=Left |
| created_at | DATETIME | Record creation |
| updated_at | DATETIME | Last update |

#### `predictions` — Saved prediction history
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| employee_id | FK → employees.id | Nullable for ad-hoc predictions |
| attrition_probability | FLOAT | 0-100% probability |
| prediction | VARCHAR(20) | "Likely To Stay" / "Likely To Leave" |
| risk_level | VARCHAR(20) | Low / Medium / High |
| confidence | FLOAT | Model confidence score |
| model_name | VARCHAR(50) | Which model was used |
| shap_contributions | JSON | Feature importance breakdown |
| input_data | JSON | Snapshot of input data |
| created_at | DATETIME | Timestamp |

#### `training_runs` — Model training history
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Run identifier |
| best_model_name | VARCHAR(100) | Best model from this run |
| status | VARCHAR(20) | completed/failed |
| train_samples | INTEGER | Training set size |
| test_samples | INTEGER | Test set size |
| elapsed_seconds | FLOAT | Training duration |

#### `model_metrics` — Per-model evaluation scores
| Column | Type | Description |
|--------|------|-------------|
| training_run_id | FK → training_runs.id | Parent run |
| model_name | VARCHAR(100) | Model name |
| accuracy, precision, recall, f1_score, roc_auc | FLOAT | Metrics |

#### `survey_responses` — Employee self-assessment surveys
| Column | Type | Description |
|--------|------|-------------|
| employee_id | VARCHAR(50) | Employee identifier |
| job_satisfaction, work_life_balance, stress_level | INTEGER | 1-5 scale |
| career_growth_satisfaction, manager_relationship | INTEGER | 1-5 scale |
| engagement_score | INTEGER | 1-5 scale |
| feedback_comment | TEXT | Open-ended feedback |
| survey_score | FLOAT | Computed wellbeing score (0-100) |

#### `recommendations` — Retention action items
| Column | Type | Description |
|--------|------|-------------|
| employee_id | FK → employees.id | Target employee |
| title, description | TEXT | Recommendation details |
| priority | VARCHAR(20) | critical/high/medium/low |
| category | VARCHAR(50) | engagement/career/workload/etc. |
| is_implemented | BOOLEAN | Whether action was taken |

---

## 11. Frontend Application

### Pages & Features

| Page | Route | Description |
|------|-------|-------------|
| **Login** | `/` (when unauthenticated) | Role selection (HR Admin / Employee), authentication |
| **Dashboard** | `/` | KPI cards, department risk bar chart, risk distribution pie chart, progress bars |
| **Employee List** | `/employees` | Paginated table with search, filters (dept, gender, overtime, risk), CRUD modals, inline prediction |
| **Employee Portal** | `/portal` | Multi-step survey wizard, wellbeing score, survey history, draft saving |
| **Predict Attrition** | `/predict` | Full employee input form, real-time prediction, SHAP factors, recommendations |
| **Model Analytics** | `/analytics` | Model leaderboard table, radar chart comparison, SHAP feature importance bar chart |
| **Organization Insights** | `/insights` | Attrition trend forecast (area chart), department risk, key insights, survival analysis stats |

### UI Design System

- **Color Palette**: Signal Orange (#ff682c), Sienna Bronze (#816729), Carbon (#202020), Graphite (#4d4d4d)
- **Typography**: Space Grotesk (headings), Inter (body text)
- **Component Library**: Custom CSS classes (`.card`, `.btn-primary`, `.badge-high`, `.stat-card`, etc.)
- **Animations**: fadeIn, slideIn, scaleIn, slideUp with staggered delays
- **Risk Badges**: Color-coded (green=Low, amber=Medium, orange=High)
- **Responsive**: Mobile-friendly grid layouts with Tailwind breakpoints

### Frontend Tech Details

- **Build**: Vite with TypeScript compilation
- **Routing**: React Router v6 with protected routes
- **State**: React hooks (useState, useEffect, useCallback) + AuthContext
- **API**: Axios with `/api` base URL (proxied to localhost:8000)
- **Charts**: Recharts (BarChart, PieChart, AreaChart, RadarChart)
- **Icons**: Lucide React icon library

---

## 12. User Roles & Workflows

### Role: HR Administrator

**Login**: Username `hr`, Password `hr@078` (hardcoded in AuthContext)

**Available Pages**: Dashboard, Employees, Employee Portal, Predict Attrition, Model Analytics, Organization Insights

**Workflow:**
1. **Dashboard** → View KPIs (total employees, high risk count, avg attrition rate, retention rate)
2. **Employee List** → Search/filter employees, view profiles, run inline predictions, add/edit/delete employees
3. **Predict Attrition** → Enter any employee's details → Get prediction with probability, risk level, SHAP explanation, and retention recommendations
4. **Model Analytics** → View model leaderboard, compare algorithms, inspect SHAP feature importance
5. **Organization Insights** → View attrition trend forecast, department risk analysis, survival analysis, key insights
6. **Employee Portal** → View submitted surveys, access the same portal as employees

### Role: Employee

**Login**: Enter Employee ID (required), Name, Department, Job Role (optional)

**Available Pages**: Dashboard, My Survey (Employee Portal)

**Workflow:**
1. **Login** → Enter Employee ID → Access portal
2. **My Survey** → Complete 7-step wellbeing survey:
   - Step 1: Job Satisfaction (1-5 with emojis)
   - Step 2: Work-Life Balance (1-5)
   - Step 3: Stress Level (1-5, inverted for scoring)
   - Step 4: Career Growth Satisfaction (1-5)
   - Step 5: Manager Relationship (1-5)
   - Step 6: Engagement Score (1-5)
   - Step 7: Open Feedback (text, max 2000 chars)
3. **Review** → See summary of all ratings → Submit
4. **Result** → Get wellbeing score (0-100) based on survey responses
5. **History** → View past survey submissions and scores

### Survey Scoring

The wellbeing score is computed as:
```
Adjusted scores: invert stress_level (5→1, 1→5)
Average of all adjusted scores × 20 = score out of 100
```

- Score ≥ 70: Good wellbeing
- Score 40-69: Moderate — monitor
- Score < 40: Concerning — intervention needed

---

## 13. Decision-Making Framework

### How Predictions Drive Decisions

```
┌─────────────────────────────────────────────────────────┐
│              EMPLOYEE ATTRITION PREDICTION               │
│                                                         │
│  Input: Employee features (age, satisfaction, etc.)     │
│    ↓                                                    │
│  ML Model → Attrition Probability (0-100%)             │
│    ↓                                                    │
│  Risk Level Assignment:                                 │
│    ≤ 30%  → 🟢 LOW RISK                                │
│    31-70% → 🟡 MEDIUM RISK                             │
│    > 70%  → 🔴 HIGH RISK                               │
│    ↓                                                    │
│  SHAP Explainability:                                   │
│    → Which features drive the risk?                     │
│    → How much does each contribute?                     │
│    ↓                                                    │
│  Recommendation Engine:                                 │
│    → Personalized retention actions                     │
│    → Priority-ranked (critical → low)                   │
│    ↓                                                    │
│  HR Decision:                                           │
│    → Immediate intervention (critical)                  │
│    → Proactive check-in (high)                          │
│    → Monitor & plan (medium)                            │
│    → Standard engagement (low)                          │
└─────────────────────────────────────────────────────────┘
```

### Decision Thresholds

| Prediction | Retention Prob | Risk Level | Recommended Action |
|-----------|---------------|------------|-------------------|
| "Likely To Stay" | 70-100% | Low 🟢 | Standard engagement, periodic check-ins |
| "Moderate Risk" | 30-69% | Medium 🟡 | Proactive career discussion, monitor satisfaction |
| "Likely To Leave" | 0-29% | High 🔴 | Urgent retention intervention, stay interview |

### Top Attrition Risk Factors (from SHAP analysis)

1. **Performance-Satisfaction Gap** — When performance is high but satisfaction is low, the employee is likely feeling undervalued
2. **Low Job Satisfaction** — The single strongest predictor of attrition
3. **Promotion Gap** — Long tenure without promotion signals career stagnation
4. **High Workload** — Excessive projects and hours lead to burnout
5. **Overtime** — Regular overtime without flexibility increases departure risk
6. **Low Compensation** — Below-market pay relative to experience

### Organizational Insights for Leadership

- **Engineering** has the highest attrition risk (77.3%) — needs immediate attention
- **HR** department is second highest (70.1%) — possibly under-resourced
- **Sales** and **Finance** have the lowest risk — current retention programs may be effective
- **Attrition trend is decreasing** overall but peaks are forecasted in October 2026
- **Survival analysis** shows 45.3% of at-risk employees may leave within 12 months

---

## 14. Project Structure

```
Employee Retention Predictor/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, CORS, lifespan
│   │   ├── api_schemas.py             # Pydantic response schemas
│   │   ├── dependencies.py            # Dependency injection
│   │   ├── database/
│   │   │   ├── connection.py          # SQLAlchemy engine (SQLite/Postgres)
│   │   │   └── models.py             # 7 ORM models
│   │   ├── routes/
│   │   │   ├── prediction_routes.py   # /api/predict, /api/predict/batch
│   │   │   ├── employee_routes.py     # /api/employees, /api/employee-risk
│   │   │   ├── model_routes.py        # /api/model-metrics, /api/train-model
│   │   │   └── survey_routes.py       # /api/employee-survey
│   │   └── services/
│   │       ├── ml_service.py          # Model loading, prediction, SHAP
│   │       └── db_service.py          # Database query layer
│   ├── requirements.txt               # 20 Python dependencies
│   └── scripts/
│       └── seed_db.py                 # Database seeding
│
├── ml_pipeline/
│   ├── preprocessing.py               # Data loading, cleaning, unification
│   ├── feature_engineering.py         # 10+ derived features
│   ├── eda.py                         # Automated EDA & visualizations
│   ├── training.py                    # 4-model training with hyperopt
│   ├── train_pipeline.py              # 7-step orchestrator
│   ├── explainability.py              # SHAP analysis
│   ├── recommendation_engine.py       # Retention recommendations
│   ├── survival_analysis.py           # Cox PH + Prophet forecasting
│   ├── models/
│   │   ├── best_model.pkl             # Serialized LightGBM model
│   │   ├── best_model_meta.json       # Metadata + 31 feature names
│   │   ├── model_leaderboard.json     # All model metrics
│   │   ├── LightGBM.pkl, RandomForest.pkl, XGBoost.pkl, LogisticRegression.pkl
│   │   └── training_pipeline.pkl      # Serialized pipeline
│   └── reports/
│       ├── pipeline_results.json      # Full pipeline output
│       ├── preprocessing_report.json  # Data quality report
│       ├── eda/eda_insights.json      # EDA findings
│       ├── shap/global_feature_importance.json
│       ├── survival/cox_ph_results.json, department_risk.json
│       └── forecast/attrition_forecast.json
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # Router + Auth wrapper
│   │   ├── main.tsx                   # React DOM entry
│   │   ├── index.css                  # Global styles (CSS variables, animations)
│   │   ├── context/AuthContext.tsx     # Auth state (HR/Employee roles)
│   │   ├── components/
│   │   │   ├── Layout.tsx             # Sidebar navigation
│   │   │   └── StatCard.tsx           # Reusable KPI card
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx          # Analytics dashboard
│   │   │   ├── PredictionPage.tsx     # Attrition prediction form
│   │   │   ├── AnalyticsPage.tsx      # Model metrics & SHAP
│   │   │   ├── InsightsPage.tsx       # Org insights & forecasts
│   │   │   ├── EmployeeListPage.tsx   # Employee directory (CRUD)
│   │   │   ├── EmployeePortalPage.tsx # Employee survey portal
│   │   │   └── LoginPage.tsx          # Role selection + auth
│   │   └── services/api.ts            # Axios API client (all endpoints)
│   ├── package.json                   # React 18, Recharts, Axios, Lucide
│   ├── vite.config.ts                 # Dev server + /api proxy
│   ├── tailwind.config.js             # Custom design tokens
│   └── tsconfig.json
│
├── Employee.csv                       # Dataset 1 (4,653 records)
├── hr_employee_churn_data.csv         # Dataset 2 (14,997 records)
├── employee_retention.db              # SQLite database
├── prediction.md                      # Prediction system documentation
├── backend_ml.md                      # Backend & ML pipeline docs
└── progress.md                        # This document
```

---

## 15. Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- pip

### Quick Start

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Seed the database
python backend/scripts/seed_db.py

# 3. Start the backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Access Points
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Retrain Models

```bash
# Full pipeline with hyperparameter tuning
python -m ml_pipeline.train_pipeline

# Without hyperparameter tuning (faster)
python -m ml_pipeline.train_pipeline --no-hyperopt

# More tuning iterations
python -m ml_pipeline.train_pipeline --n-iter 20
```

### Database Configuration

```bash
# SQLite (default, no setup needed)
USE_SQLITE=true

# PostgreSQL
USE_SQLITE=false
DATABASE_URL=postgresql://user:pass@localhost:5432/employee_retention
```

---

## 16. Current Status & Next Steps

### ✅ Completed Features

- [x] Full ML pipeline (preprocessing → training → explainability → survival → forecasting)
- [x] 4 models trained and compared (LightGBM, RandomForest, XGBoost, LogisticRegression)
- [x] SHAP-based explainability for every prediction
- [x] Dynamic retention recommendation engine
- [x] Cox Proportional Hazards survival analysis
- [x] Prophet-based attrition trend forecasting
- [x] FastAPI backend with 20+ endpoints
- [x] SQLite database with 7 tables
- [x] Employee CRUD (Create, Read, Update, Delete)
- [x] Employee search and filtering (department, gender, overtime, risk level)
- [x] Batch prediction endpoint
- [x] Prediction history tracking
- [x] React frontend with 7 pages
- [x] HR Administrator login with full access
- [x] Employee self-assessment survey portal (7-step wizard)
- [x] Wellbeing score computation
- [x] Survey history and draft saving
- [x] Dashboard with KPIs, charts, and drill-downs
- [x] Model analytics page with leaderboard and SHAP importance
- [x] Organization insights with forecast visualization
- [x] Responsive design with animations
- [x] Comprehensive OpenAPI documentation
- [x] Fallback prediction when model is unavailable
- [x] Background model retraining

### 🔜 Potential Next Steps

- [ ] Add user authentication (JWT tokens) instead of hardcoded credentials
- [ ] Implement real-time WebSocket predictions
- [ ] Add email/Slack notifications for high-risk employees
- [ ] Build admin dashboard for model retraining controls
- [ ] Add A/B testing for different retention strategies
- [ ] Implement data validation and input sanitization
- [ ] Add unit and integration tests
- [ ] Deploy to production (Docker + cloud)
- [ ] Add role-based access control (RBAC)
- [ ] Integrate with real HR systems (Workday, BambooHR)
- [ ] Add more granular forecasting (weekly, by department)
- [ ] Implement model drift detection and auto-retraining

---

*This document provides a comprehensive overview of the Employee Retention Predictor project's architecture, implementation, and capabilities as of June 13, 2026.*
