# Backend & ML Pipeline Documentation

## Project Overview

**Employee Retention Predictor** — An ML-powered system that predicts employee attrition risk and generates personalized retention recommendations. The system combines:

- **ML Pipeline** (Python): Data preprocessing, feature engineering, model training, SHAP explainability, survival analysis, and trend forecasting
- **Backend API** (FastAPI): Serves predictions, model analytics, employee data management, and organization insights
- **Database** (SQLite/Postgres): Stores employee records, predictions, model metrics, and training history
- **Frontend** (React + TypeScript): Dashboard, prediction interface, analytics, employee directory, and organization insights

---

## Table of Contents

1. [ML Pipeline Architecture](#ml-pipeline-architecture)
2. [Backend Architecture](#backend-architecture)
3. [Setup & Commands](#setup--commands)
4. [API Endpoints](#api-endpoints)
5. [Data Flow](#data-flow)
6. [Database Schema](#database-schema)
7. [Project Structure](#project-structure)

---

## ML Pipeline Architecture

The ML pipeline is located in `ml_pipeline/` and orchestrates a complete ML lifecycle:

### Pipeline Steps

```
Step 1: Load & Preprocess Data
  └── ml_pipeline/preprocessing.py
  └── Reads Employee.csv + hr_employee_churn_data.csv
  └── Cleans, maps columns, handles missing values
  └── Combines datasets into unified format

Step 2: Exploratory Data Analysis (EDA)
  └── ml_pipeline/eda.py
  └── Class distribution, correlations, visualizations
  └── Reports saved to ml_pipeline/reports/eda/

Step 3: Feature Engineering
  └── ml_pipeline/feature_engineering.py
  └── Creates derived features:
      - promotion_gap, income_per_experience
      - satisfaction_index, overtime_risk
      - experience_ratio, workload_score
      - tenure_category, perf_sat_gap
      - interaction features (low_sat_high_workload, long_tenure_no_promo)
  └── Total: 31 features used for training

Step 4: Train-Test Split
  └── 80/20 stratified split (preserves attrition ratio)

Step 5: Model Training
  └── ml_pipeline/training.py
  └── 4 models trained with hyperparameter tuning:
      - RandomForest Classifier       (BEST: 0.975 ROC-AUC)
      - XGBoost                       (0.9746 ROC-AUC)
      - LightGBM                      (0.9738 ROC-AUC)
      - LogisticRegression            (0.9019 ROC-AUC)
  └── Uses RandomizedSearchCV with 3-fold StratifiedKFold
  └── Metric: ROC-AUC for model selection

Step 6: SHAP Explainability
  └── ml_pipeline/explainability.py
  └── TreeExplainer for tree-based models
  └── Summary plots, feature importance, waterfall plots
  └── Global feature importance saved JSON

Step 7: Survival Analysis & Forecasting
  └── ml_pipeline/survival_analysis.py
  └── Cox Proportional Hazards model (via lifelines)
  └── Prophet-based attrition trend forecasting
  └── Department risk analysis
```

### Trained Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **RandomForest** | 88.1% | 64.4% | 95.2% | 0.768 | **0.975** |
| XGBoost | 94.9% | 92.4% | 81.9% | 0.868 | 0.975 |
| LightGBM | 94.5% | 90.7% | 82.0% | 0.862 | 0.974 |
| LogisticRegression | 85.9% | 70.7% | 54.5% | 0.616 | 0.902 |

**Best model:** RandomForest (stored in `ml_pipeline/models/best_model.pkl`)

### The 31 Training Features

The model expects exactly 31 features in this order:

```
age                          (raw)
education_level              (encoded: 1=Bachelor's, 2=Master's, 3=PhD)
payment_tier                 (1/2/3 based on income)
tenure_years                 (years at company)
experience_years             (total experience)
gender_encoded               (1=Male, 0=Female)
ever_benched_encoded         (1=Yes, 0=No)
satisfaction_score           (1-5 scale)
performance_score            (1-5 scale)
city_Bangalore               (dummy encoded)
city_New Delhi               (dummy encoded)
city_Pune                    (dummy encoded)
num_projects                 (number of projects)
avg_monthly_hours            (average hours/month)
work_accident                (0/1)
promotion_last_5years        (0/1)
salary_encoded               (1=low, 2=medium, 3=high)
salary_high                  (dummy)
salary_low                   (dummy)
salary_medium                (dummy)
promotion_gap                (engineered)
income_per_experience        (engineered)
satisfaction_index           (engineered)
overtime_risk                (engineered)
bench_risk                   (engineered)
experience_ratio             (engineered)
workload_score               (engineered)
tenure_category              (engineered: 0-4)
perf_sat_gap                 (engineered)
low_sat_high_workload        (interaction feature)
long_tenure_no_promo         (interaction feature)
```

---

## Backend Architecture

The backend is a **FastAPI** application with the following layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
│                   (backend/app/main.py)                  │
├─────────────────────────────────────────────────────────┤
│                    Route Handlers                        │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐ │
│  │ Predictions  │  │ Employees  │  │ Model Management │ │
│  │ (prediction_ │  │ (employee_ │  │ (model_routes.py) │ │
│  │  routes.py)  │  │  routes.py)│  │                  │ │
│  └──────┬───────┘  └─────┬──────┘  └───────┬──────────┘ │
├─────────┼────────────────┼────────────────┼────────────┤
│         ▼                ▼                ▼             │
│              Service Layer                               │
│  ┌─────────────────┐  ┌────────────────────────┐        │
│  │   ML Service     │  │     DB Service          │       │
│  │ (ml_service.py)  │  │ (db_service.py)         │       │
│  │ - model loading  │  │ - employee queries      │       │
│  │ - predictions    │  │ - prediction storage    │       │
│  │ - SHAP expl.     │  │ - analytics queries     │       │
│  │ - fallback logic │  │ - model metrics         │       │
│  └────────┬─────────┘  └───────────┬─────────────┘       │
├───────────┼────────────────────────┼─────────────────────┤
│           ▼                        ▼                      │
│  ┌──────────────┐     ┌─────────────────────┐            │
│  │ ML Pipeline  │     │  SQLite/Postgres    │            │
│  │ (ml_pipeline/│     │  Database           │            │
│  │  *.py files) │     │  (employee_retention│            │
│  └──────────────┘     │   .db / connection) │            │
│                       └─────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Singleton ML Service**: Models are loaded once at first request (lazy loading) and reused
- **Feature Alignment**: Raw input → engineered features → aligned to 31 training features (fill missing with 0)
- **Fallback Prediction**: If the ML model fails, a rule-based heuristic is used
- **DB-First Queries**: Analytics routes query the database first, falling back to file-based data
- **Prediction Persistence**: Every prediction is saved to the database automatically

### Dependency Injection

FastAPI dependencies are used throughout:
- `get_ml_service()` — Returns the singleton ML service instance
- `get_db_service()` — Creates a fresh DBService with a SQLAlchemy session
- Shared via `backend/app/dependencies.py`

---

## Setup & Commands

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip

### Backend Setup

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Seed the database with employee data
python backend/scripts/seed_db.py

# 3. Start the backend server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload

#   Server starts at: http://localhost:8000
#   API docs at:      http://localhost:8000/docs
#   ReDoc at:         http://localhost:8000/redoc
```

### Frontend Setup

```bash
# 1. Install frontend dependencies
cd frontend
npm install

# 2. Start development server
npm run dev

#   Opens at: http://localhost:5173
#   Vite proxies /api requests to http://localhost:8000
```

### ML Pipeline (Retrain Models)

```bash
# Train all models (with hyperparameter tuning)
python -m ml_pipeline.train_pipeline

# Train without hyperparameter tuning
python -m ml_pipeline.train_pipeline --no-hyperopt

# Train with more tuning iterations
python -m ml_pipeline.train_pipeline --n-iter 20
```

### Database Management

```bash
# Seed/reset the database
python backend/scripts/seed_db.py

# Use SQLite by default (no setup needed)
# To use PostgreSQL, set environment variables:
#   DATABASE_URL=postgresql://user:pass@localhost:5432/employee_retention
#   USE_SQLITE=false
```

---

## API Endpoints

### Root & Health

#### `GET /`

Returns service status and available endpoints.

**Response:**
```json
{
  "service": "Employee Retention Predictor API",
  "version": "1.0.0",
  "status": "operational",
  "docs": "/docs",
  "endpoints": {
    "POST /api/predict": "Predict employee attrition",
    "POST /api/upload-dataset": "Upload training dataset",
    "POST /api/train-model": "Train/retrain model",
    "GET /api/model-metrics": "Get model performance metrics",
    "GET /api/feature-importance": "Get SHAP feature importance",
    "GET /api/employee-risk": "Get employee risk distribution",
    "GET /api/department-risk": "Get department risk analysis",
    "GET /api/organization-insights": "Get organization insights"
  }
}
```

#### `GET /health`

Simple health check.

**Response:**
```json
{ "status": "healthy", "service": "employee-retention-predictor" }
```

---

### Predictions

#### `POST /api/predict`

Predict attrition probability for an employee.

**Request Body:** (all fields optional)
```json
{
  "age": 35,
  "gender": "Male",
  "department": "Engineering",
  "job_role": "Software Engineer",
  "monthly_income": 75000,
  "job_satisfaction": 2,
  "environment_satisfaction": 3,
  "work_life_balance": 2,
  "distance_from_home": 15,
  "years_at_company": 5,
  "years_since_last_promotion": 3,
  "overtime": "Yes",
  "performance_rating": 4,
  "training_times_last_year": 1,
  "education": "Master's",
  "marital_status": "Married",
  "num_projects": 6,
  "avg_monthly_hours": 220,
  "promotion_last_5years": 0,
  "salary_level": "medium",
  "tenure_years": 5,
  "experience_years": 8
}
```

**Response:**
```json
{
  "attrition_probability": 72.3,
  "prediction": "Likely To Leave",
  "risk_level": "High",
  "confidence": 0.845,
  "shap_explanation": {
    "base_value": 0.35,
    "feature_contributions": {
      "Overtime Risk": "+18.2%",
      "Satisfaction Score": "+15.1%",
      "Promotion Gap": "+12.5%"
    }
  },
  "recommendations": {
    "recommendations": [
      {
        "title": "Employee Engagement Program",
        "description": "Implement personalized engagement activities and regular check-ins.",
        "priority": "high",
        "category": "engagement"
      }
    ],
    "risk_factors": [...]
  },
  "prediction_id": 42
}
```

**Prediction Flow:**
1. Raw input received → EmployeeInput validation (Pydantic)
2. Feature engineering via `FeatureEngineer.create_all_features()`
3. Raw fields mapped to training feature names (gender→gender_encoded, education→education_level, etc.)
4. Feature vector aligned to 31 training features (fill missing with 0)
5. RandomForest model predicts: `predict_proba(X)[:, 1]`
6. Risk level assigned: ≤30% Low, 31-70% Medium, >70% High
7. SHAP explanation generated (may be empty if explainer unavailable)
8. Retention recommendations generated based on risk factors
9. Prediction saved to database

**Risk Level Thresholds:**
- **Low** (0-30%): Likely To Stay
- **Medium** (31-70%): Moderate risk
- **High** (71-100%): Likely To Leave

#### `POST /api/predict/batch`

Predict for multiple employees at once.

```json
{
  "predictions": [
    { "attrition_probability": 72.3, "prediction": "Likely To Leave", ... },
    { "attrition_probability": 15.7, "prediction": "Likely To Stay", ... }
  ]
}
```

#### `GET /api/predictions?limit=50`

Get recent predictions stored in the database.

---

### Employee Analytics

#### `GET /api/employee-risk`

Risk distribution across the organization.

```json
{
  "total_employees": 19650,
  "high_risk": 5169,
  "medium_risk": 0,
  "low_risk": 14481,
  "high_risk_percentage": 26.3,
  "average_attrition_risk": 26.3,
  "risk_distribution": {
    "Low Risk (Stayed)": 14481,
    "High Risk (Attrited)": 5169
  }
}
```

#### `GET /api/department-risk`

Department-level risk breakdown. Queries DB first, falls back to file-based defaults.

```json
{
  "departments": [
    { "department": "Sales", "risk_score": 32.7, "risk_level": "Medium", "employee_count": 120 },
    { "department": "Marketing", "risk_score": 31.6, "risk_level": "Medium", "employee_count": 65 },
    { "department": "Engineering", "risk_score": 22.0, "risk_level": "Low", "employee_count": 200 },
    { "department": "Finance", "risk_score": 6.6, "risk_level": "Low", "employee_count": 55 }
  ],
  "highest_risk_department": "Sales",
  "highest_risk_score": 32.7
}
```

#### `GET /api/organization-insights`

Comprehensive insights combining DB data and ML forecasts.

```json
{
  "attrition_trend": { "forecast": [...], "historical": [...] },
  "overall_risk_score": 26.3,
  "trend_direction": "stable",
  "key_insights": [
    "Sales department has the highest attrition risk at 32.7%",
    "Average tenure is 4.2 years — monitor long-tenure retention",
    "Job satisfaction is moderate (3.1/5)"
  ],
  "department_stats": [...],
  "feature_stats": {
    "avg_age": 31.2,
    "avg_income": 49500,
    "avg_satisfaction": 3.1,
    "avg_tenure_years": 4.2,
    "avg_work_life_balance": 3.0
  }
}
```

---

### Employee Records

#### `GET /api/employees?skip=0&limit=50`

Paginated employee list.

#### `GET /api/employees/{id}`

Single employee details.

#### `GET /api/high-risk-employees?limit=50`

Employees with high attrition signals.

---

### Model Management

#### `GET /api/model-metrics`

Model leaderboard and best model info. Checks DB first, falls back to file.

```json
{
  "leaderboard": [
    { "model": "RandomForest", "accuracy": 0.881, "precision": 0.644, "recall": 0.952, "f1_score": 0.768, "roc_auc": 0.975 },
    { "model": "XGBoost", "accuracy": 0.949, "precision": 0.924, "recall": 0.819, "f1_score": 0.868, "roc_auc": 0.975 },
    { "model": "LightGBM", "accuracy": 0.945, "precision": 0.907, "recall": 0.820, "f1_score": 0.862, "roc_auc": 0.974 },
    { "model": "LogisticRegression", "accuracy": 0.859, "precision": 0.707, "recall": 0.545, "f1_score": 0.616, "roc_auc": 0.902 }
  ],
  "best_model": "RandomForest",
  "best_roc_auc": 0.975
}
```

#### `GET /api/feature-importance`

SHAP-based global feature importance from `ml_pipeline/reports/shap/global_feature_importance.json`.

#### `GET /api/model-info`

Current model status.

```json
{
  "model_loaded": true,
  "model_name": "RandomForest",
  "model_meta": {
    "best_model_name": "RandomForest",
    "best_roc_auc": 0.975,
    "timestamp": "2026-06-12T22:20:30.433282",
    "feature_names": [...31 features...],
    "n_features": 31
  }
}
```

#### `POST /api/upload-dataset`

Upload a CSV dataset (multipart form). Validates and saves to `ml_pipeline/data/uploads/`.

#### `POST /api/train-model?use_hyperopt=true`

Trigger model retraining in background. Uses `TrainingPipeline` from `ml_pipeline/train_pipeline.py`.

---

## Data Flow

### End-to-End Prediction Flow

```
User submits employee data (age, gender, department, salary, satisfaction, etc.)
        │
        ▼
FastAPI validates with Pydantic (EmployeeInput model)
        │
        ▼
MLService.predict() called
        │
        ├── Model loaded? ──No──► _fallback_prediction() (rule-based)
        │
        ▼ Yes
_prepare_features() executes:
  1. pd.DataFrame([input_data])
  2. FeatureEngineer.create_all_features(df)
  3. Map raw fields → training columns:
     - gender              → gender_encoded (1=Male, 0=Female)
     - education           → education_level (1=Bachelor's, 2=Master's, 3=PhD)
     - salary_level        → salary_encoded + dummies
     - job_satisfaction    → satisfaction_score
     - performance_rating  → performance_score
     - monthly_income      → payment_tier (1=<35k, 2=35-65k, 3=>65k)
     - overtime            → overtime_risk
     - years_at_company    → tenure_years, experience_years
     - (defaults for missing fields: ever_benched=0, work_accident=0, etc.)
  4. FeatureEngineer.create_all_features(df) again (picks up new columns)
  5. Reindex to 31 training feature columns (.reindex(columns=training_features, fill_value=0))
  6. Return np.ndarray shape (1, 31)
        │
        ▼
RandomForest predicts:
  - predict_proba(X)[:, 1] → attrition probability
  - predict(X)[0]          → class (0=Stay, 1=Leave)
        │
        ▼
Risk level assigned:
  ≤30%  → Low    → "Likely To Stay"
  31-70% → Medium → depends on class
  >70%  → High   → "Likely To Leave"
        │
        ▼
SHAP explanation attempted (may return {} if fails)
        │
        ▼
RecommendationEngine generates recommendations based on risk factors
        │
        ▼
Prediction saved to database (predictions table)
        │
        ▼
JSON response returned to frontend
```

### Analytics Data Flow

```
Dashboard/Insights requests
        │
        ▼
employee_routes.py handlers
        │
        ├── DBService queries SQLite:
        │   - get_employee_count()        → total employees
        │   - get_attrition_count()        → those who left
        │   - get_department_risk()        → department breakdown
        │   - get_department_stats()       → detailed department data
        │   - get_feature_stats()          → aggregate employee features
        │
        ├── MLService provides:
        │   - get_attrition_forecast()     → Prophet trend forecast
        │   - get_department_risk() fallback → if DB returns empty
        │
        └── Response assembled with dynamic insights
```

---

## Database Schema

**SQLite** by default (`employee_retention.db`). PostgreSQL supported via environment variables.

### Tables

#### `employees`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| employee_id | VARCHAR(50) UNIQUE | Human-readable ID (EMP0001) |
| age | INTEGER | Employee age |
| gender | VARCHAR(20) | Male/Female |
| department | VARCHAR(100) | Engineering, Sales, etc. |
| job_role | VARCHAR(100) | Software Engineer, etc. |
| monthly_income | FLOAT | Monthly salary |
| job_satisfaction | INTEGER | 1-5 scale |
| environment_satisfaction | INTEGER | 1-5 scale |
| work_life_balance | INTEGER | 1-5 scale |
| distance_from_home | INTEGER | km |
| years_at_company | INTEGER | Tenure |
| years_since_last_promotion | INTEGER | Years |
| overtime | VARCHAR(10) | Yes/No |
| performance_rating | INTEGER | 1-5 scale |
| education | VARCHAR(50) | Bachelor's/Master's/PhD |
| marital_status | VARCHAR(20) | Single/Married/Divorced |
| attrition | INTEGER | 0=Stayed, 1=Left |
| created_at | DATETIME | Record creation |
| updated_at | DATETIME | Last update |

#### `predictions`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| employee_id | FK → employees.id | Nullable for ad-hoc predictions |
| attrition_probability | FLOAT | 0-100% |
| prediction | VARCHAR(20) | Likely To Stay/Leave |
| risk_level | VARCHAR(20) | Low/Medium/High |
| confidence | FLOAT | Model confidence |
| model_name | VARCHAR(50) | RandomForest, etc. |
| shap_contributions | JSON | Feature importance |
| input_data | JSON | Raw prediction input |
| created_at | DATETIME | When predicted |

#### `training_runs`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| run_name | VARCHAR(200) | Training run identifier |
| best_model_name | VARCHAR(100) | Best performing model |
| status | VARCHAR(20) | completed/failed |
| train_samples | INTEGER | Training set size |
| test_samples | INTEGER | Test set size |
| elapsed_seconds | FLOAT | Training duration |
| created_at | DATETIME | When trained |

#### `model_metrics`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| training_run_id | FK → training_runs.id | Parent run |
| model_name | VARCHAR(100) | Model name |
| accuracy | FLOAT | Accuracy score |
| precision | FLOAT | Precision score |
| recall | FLOAT | Recall score |
| f1_score | FLOAT | F1 score |
| roc_auc | FLOAT | ROC-AUC score |

#### Additional tables: `datasets`, `recommendations`

---

## Project Structure

```
Employee Retention Predictor/
│
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── dependencies.py            # Shared DI (get_db_service)
│   │   ├── database/
│   │   │   ├── connection.py          # SQLAlchemy engine/session
│   │   │   └── models.py             # ORM models (Employee, Prediction, etc.)
│   │   ├── routes/
│   │   │   ├── prediction_routes.py   # POST /predict, /predict/batch, GET /predictions
│   │   │   ├── employee_routes.py     # GET /employee-risk, /department-risk, /organization-insights, /employees
│   │   │   └── model_routes.py        # GET /model-metrics, /feature-importance, POST /train-model
│   │   └── services/
│   │       ├── ml_service.py          # ML model loading, predictions, feature alignment
│   │       └── db_service.py          # Database query service
│   ├── scripts/
│   │   └── seed_db.py                 # Database seeding from CSVs
│   └── requirements.txt               # Python dependencies
│
├── ml_pipeline/
│   ├── preprocessing.py               # Data loading, cleaning, column mapping
│   ├── feature_engineering.py         # Derived feature creation
│   ├── training.py                    # Model training with hyperparameter tuning
│   ├── train_pipeline.py              # End-to-end training orchestrator
│   ├── explainability.py              # SHAP model explanation
│   ├── recommendation_engine.py       # Retention recommendation logic
│   ├── survival_analysis.py           # Cox PH + Prophet forecasting
│   ├── eda.py                         # Exploratory data analysis
│   ├── models/
│   │   ├── best_model.pkl             # Serialized RandomForest model
│   │   ├── best_model_meta.json       # Model metadata + 31 feature names
│   │   └── model_leaderboard.json     # Model comparison metrics
│   └── reports/                       # Generated reports & visualizations
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    # React Router setup
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx          # Analytics dashboard
│   │   │   ├── PredictionPage.tsx     # Attrition prediction form
│   │   │   ├── AnalyticsPage.tsx      # Model metrics & SHAP importance
│   │   │   ├── InsightsPage.tsx       # Organization insights & forecasts
│   │   │   └── EmployeeListPage.tsx   # Employee directory with search/filter
│   │   ├── components/
│   │   │   ├── Layout.tsx             # Sidebar navigation
│   │   │   └── StatCard.tsx           # Reusable stat card
│   │   └── services/
│   │       └── api.ts                 # API client (axios)
│   └── vite.config.ts                 # Proxy /api → localhost:8000
│
├── Employee.csv                       # Dataset 1: 4,653 records
├── hr_employee_churn_data.csv         # Dataset 2: 14,997 records
└── backend_ml.md                      # This documentation
```
