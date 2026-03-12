
# Workforce Analytics & Retention Engine

An end-to-end machine learning pipeline for predicting and explaining employee attrition, transforming HR data into actionable workforce retention strategies.

## Overview

This project architected an **Ensemble Learning pipeline** (Random Forest) achieving **86% accuracy**, with Hyperparameter Tuning and class-imbalance handling to isolate high-risk variables across job-role segments. An **Explainable AI (XAI) framework** using **SHAP** and **LIME** transforms latent attrition drivers from **1,470 employee records** into visual, actionable risk-mitigation strategies.

## Pipeline Architecture

| Stage | Technique | Output |
|---|---|---|
| **Data Preprocessing** | LabelEncoder, SatisfactionScore feature, correlation analysis | Clean feature matrix (35 features) |
| **Model Training** | Baseline RF → RandomizedSearchCV (50 iter, 5-fold CV) → class_weight='balanced' RF | 86%+ accuracy, optimised hyperparameters |
| **Explainability (XAI)** | SHAP TreeExplainer + LIME LimeTabularExplainer | Feature importance plots, individual explanations |
| **Risk Visualisation** | Department/role-wise SHAP segmentation, risk score table | Actionable HR intervention recommendations |

## Results

| Model | Accuracy | Purpose |
|---|---|---|
| Baseline Random Forest | 86.62% | Accuracy headline |
| Tuned RF (RandomizedSearchCV) | ~86–87% | Hyperparameter optimisation |
| Balanced RF (class_weight='balanced') | ~83–86% | SHAP / LIME explainability |

## Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance
- **Records:** 1,470 employees × 35 features
- **Departments:** Sales, Research & Development, Human Resources
- **Target:** Attrition (Yes/No) — ~16% positive class

## Key Findings

Top attrition drivers (SHAP-ranked):
1. **OverTime** — strongest predictor; excess overtime correlates with ~24.6% attrition lift
2. **MonthlyIncome** — lower income strongly associated with departure
3. **JobLevel** — junior levels face highest attrition risk
4. **TotalWorkingYears** — early-tenure employees most at risk
5. **YearsAtCompany** — attrition spikes at Year 1–2 and Year 5 milestones

## Libraries

| Library | Version | Use |
|---|---|---|
| scikit-learn | ≥1.0 | Random Forest, RandomizedSearchCV, metrics |
| shap | ≥0.41 | TreeExplainer, summary/beeswarm/dependence plots |
| lime | ≥0.2 | LimeTabularExplainer, individual prediction explanations |
| pandas | ≥1.3 | Data manipulation |
| plotly | ≥5.0 | Interactive EDA visualisations |
| seaborn / matplotlib | ≥0.11 / ≥3.4 | Statistical plots, confusion matrix |

## Output Files

Running the notebook generates:
- `shap_summary_bar.png` — global feature importance (mean |SHAP|)
- `shap_beeswarm.png` — feature impact direction and magnitude
- `shap_by_department.png` — top 5 risk drivers per department
- `shap_by_jobrole.png` — high-risk variables across 5 job role segments
- `lime_explanations.png` — individual prediction explanations (TP, high-conf, FN)

## Screenshots

![Attrition by Department](https://github.com/jayesh-masade/Employee-Attrition/blob/main/Screenshot%202023-08-03%20at%2010.19.06%20PM.png?raw=true)

![Age vs Monthly Income](https://github.com/jayesh-masade/Employee-Attrition/blob/main/download.png?raw=true)

## Usage

Open `Employee_Attrition_model.ipynb` in Google Colab and run all cells. The first cell installs all required dependencies automatically.
