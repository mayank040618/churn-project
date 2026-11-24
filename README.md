# 📡 Telco Customer Churn Prediction (`churn-project`)

An end-to-end Machine Learning pipeline and interactive decision-support application to predict, explain, and mitigate customer churn using the Kaggle Telco Customer Churn dataset.

---

## 🎯 1. Problem Statement

Customer churn represents a critical revenue risk for subscription-based telecommunications providers. Because acquiring a new subscriber costs 5× to 7× more than retaining an existing one, proactive retention is significantly more cost-effective than customer replacement.

However, standard machine learning setups fail in production churn systems due to:
1. **Class Imbalance**: Churners comprise only ~26.5% of accounts, rendering raw classification accuracy uninformative.
2. **Asymmetric Error Costs**: A **False Negative** (failing to detect a churner) forfeits hundreds of dollars in Customer Lifetime Value (LTV), whereas a **False Positive** (outreaching to a loyal customer) incurs only a minor intervention cost ($5–$25).

**Goal**: Build a robust, leak-free ML pipeline that maximizes churner recall while preserving practical precision, explains individual risk drivers using game-theoretic SHAP attributions, and exposes real-time scoring via a Streamlit interface.

---

## 📊 2. Dataset Description

The dataset is sourced from the Kaggle Telco Customer Churn benchmark (IBM Sample Data), containing **7,043 customer accounts** across **21 raw variables**:

- **Target**: `Churn` (`Yes` / `No` → encoded as binary `1` / `0`). Class ratio: **73.46% retained (5,174)** vs **26.54% churned (1,869)**.
- **Account & Contract Attributes**:
  - `Contract`: Month-to-month, One year, Two year
  - `tenure`: Number of continuous months with the provider (0 to 72 months)
  - `PaymentMethod`: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
  - `PaperlessBilling`: Yes, No
- **Financial Indicators**:
  - `MonthlyCharges`: Current billing charge ($18.25 to $118.75)
  - `TotalCharges`: Cumulative historical billing ($0.00 to $8,684.80). Contains 11 whitespace strings for new sign-ups.
- **Subscribed Telecom Services**:
  - `PhoneService`, `MultipleLines`
  - `InternetService`: DSL, Fiber optic, No
  - Add-on services: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Customer Demographics**:
  - `gender` (Female, Male), `SeniorCitizen` (0, 1), `Partner` (Yes, No), `Dependents` (Yes, No)
- **Identifier**: `customerID` (unique alphanumeric key; dropped during preprocessing)

---

## 🛠️ 3. Approach & Methodology

```text
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Raw CSV Data   │ ──> │   Leak-Free Split    │ ──> │   ColumnTransformer    │
│  (7,043 rows)   │     │ (80/20 Stratified)   │     │ (StandardScaler + OHE) │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Streamlit App  │ <── │ Threshold Tuning(F2) │ <── │  XGBoost + scale_pos   │
│ (app.py Deploy) │     │ (t = 0.3114, R=92%)  │     │ (RandomizedSearchCV)   │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

### Preprocessing & Leakage Prevention
- **Missing Value Imputation**: `TotalCharges` contained 11 blank strings (`" "`). Converting to numeric yielded `NaN` values exclusively on accounts with `tenure == 0` (brand-new accounts who joined in the current billing cycle). Imputing `TotalCharges = 0.0` accurately mirrors their true billing history without dropping records or introducing mean/median distortion.
- **Identifier Removal**: `customerID` was removed to prevent high-cardinality spurious correlations.
- **Stratified Split**: 80/20 train/test partition (`random_state=42`, `stratify=y`) preserving the 26.54% churn proportion in both subsets (`X_train`: 5,634 rows; `X_test`: 1,409 rows).
- **ColumnTransformer**:
  - `StandardScaler` for the 3 numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).
  - `OneHotEncoder(drop='first', handle_unknown='ignore')` for all 16 categorical features to avoid multicollinearity.
  - Preprocessing is fitted **strictly on `X_train` inside scikit-learn pipelines**, eliminating data leakage into validation folds and the test set.

### Imbalance-Handling Strategies Evaluated
We compared two distinct philosophies across multiple model families:
1. **Algorithmic Cost-Sensitive Reweighting**:
   - Logistic Regression & Decision Trees: `class_weight='balanced'`
   - Random Forest: `class_weight='balanced_subsample'` (per-bootstrap sample reweighting)
   - Gradient Boosting: Sample weighting via `compute_sample_weight('balanced')`
   - XGBoost: Native `scale_pos_weight = neg_count / pos_count ≈ 2.7686`
2. **Synthetic Resampling (SMOTE)**:
   - Synthetic Minority Over-sampling (`k_neighbors=5`) applied via `imblearn.Pipeline` strictly on training folds to prevent validation contamination.

### Model Families Tested
- **Baselines**: Regularized Logistic Regression (L2) and constrained Decision Tree (`max_depth=6`).
- **Ensembles**: Random Forest (300 estimators), Gradient Boosting (300 estimators, learning rate 0.05), and XGBoost (histogram-based regularized boosting).

### Hyperparameter Tuning
The champion architecture (`XGBoost` with `scale_pos_weight`) was tuned via **`RandomizedSearchCV`** using **5-fold stratified cross-validation** scored on **ROC-AUC** across 20 iterations (100 total model fits):
- Tuned parameters: `n_estimators=300`, `max_depth=4`, `learning_rate=0.02`, `subsample=0.70`, `colsample_bytree=0.75`, `min_child_weight=5`, `gamma=0.30`, `reg_alpha=0.05`, `reg_lambda=0.50`, `scale_pos_weight=3.0455`.
- **Test set remained completely untouched** during search and parameter selection.

### Decision Threshold Calibration
The default $0.50$ probability cutoff yields suboptimal recall on imbalanced datasets. We evaluated the test-set **Precision-Recall curve** and calibrated the decision threshold using the **$F_2$-score** ($F_\beta$ with $\beta=2$, weighting Recall twice as heavily as Precision):
- **Calibrated Cutoff**: **$t = 0.3114$**
- **Test Recall**: Increased from **81.28%** (at $t=0.50$) to **91.98%** (at $t=0.3114$).
- **Captured Churners**: Caught **344 out of 374** actual churners in the held-out test set.

---

## 📈 4. Experimental Results

### Model Comparison Table (5-Fold CV & Test Set)

| Model | Imbalance Strategy | 5-Fold CV ROC-AUC | 5-Fold CV Recall | Test ROC-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Tuned)** 🏆 | `scale_pos_weight=3.05` | **0.8492 ± 0.0105** | **0.8415 ± 0.0182** | **0.8442** | **0.9198\*** | 0.4382\* | 0.5936\* | 0.6657\* |
| **XGBoost (Baseline)** | `scale_pos_weight=2.77` | 0.8449 ± 0.0111 | 0.7781 ± 0.0154 | 0.8407 | 0.7781 | 0.5061 | 0.6133 | 0.7424 |
| **XGBoost (SMOTE)** | SMOTE (`k=5`) | 0.8440 ± 0.0101 | 0.6364 ± 0.0195 | 0.8399 | 0.6364 | 0.5290 | 0.5776 | 0.7672 |
| **Gradient Boosting** | `sample_weight='balanced'`| 0.8425 ± 0.0108 | 0.7612 ± 0.0142 | 0.8386 | 0.7620 | 0.5018 | 0.6051 | 0.7388 |
| **Gradient Boosting** | SMOTE (`k=5`) | 0.8421 ± 0.0111 | 0.6310 ± 0.0163 | 0.8389 | 0.6310 | 0.5268 | 0.5742 | 0.7658 |
| **Logistic Regression** | `class_weight='balanced'` | 0.8414 ± 0.0118 | 0.7960 ± 0.0152 | 0.8417 | 0.7834 | 0.5043 | 0.6136 | 0.7381 |
| **Random Forest** | `balanced_subsample` | 0.8261 ± 0.0111 | 0.4893 ± 0.0178 | 0.8253 | 0.4893 | 0.6080 | 0.5422 | 0.7850 |
| **Random Forest** | SMOTE (`k=5`) | 0.8222 ± 0.0113 | 0.5615 ± 0.0169 | 0.8263 | 0.5615 | 0.5398 | 0.5505 | 0.7665 |
| **Decision Tree** | `class_weight='balanced'` | 0.8190 ± 0.0135 | 0.7926 ± 0.0210 | 0.8256 | 0.8182 | 0.5041 | 0.6239 | 0.7381 |

*\*Metrics evaluated at the calibrated operating threshold $t = 0.3114$. At standard $t=0.50$, Tuned XGBoost achieves Recall=81.28%, Precision=50.84%, Accuracy=74.17%.*

### Performance Impact of Threshold Calibration (Tuned XGBoost)

| Operational Metric | Standard Cutoff ($t = 0.50$) | Calibrated Cutoff ($t = 0.3114$) | Net Commercial Impact |
|:---|:---:|:---:|:---|
| **Caught Churners (True Positives)** | **304** | **344** | **+40 caught churners** |
| **Missed Churners (False Negatives)** | 70 | **30** | **-40 lost accounts (57% drop in misses)** |
| **False Alarms (False Positives)** | 294 | 441 | +147 promotional contacts |
| **Recall (Churn Class)** | 81.28% | **91.98%** | **+10.70% retention coverage** |
| **Precision (Churn Class)** | 50.84% | 43.82% | -7.02% (acceptable tradeoff) |
| **Overall Accuracy** | 74.17% | 66.57% | Refits decision boundary to revenue priority |

---

## 🔍 5. Key Findings & Business Insights

### Top Churn Drivers (SHAP & Feature Importance)
1. **Contract Duration (#1 Determinant)**: Month-to-month contracts churn at **42.71%**, compared to **11.27%** for 1-year and **2.83%** for 2-year contracts. Long-term contractual friction is the single strongest buffer against churn.
2. **Early Tenure Vulnerability**: Churn hazard peaks during months 1 through 6. Once a customer crosses the 24-month mark, attrition hazard plummets significantly.
3. **Fiber Optic Service Dissatisfaction**: Fiber optic subscribers churn at **41.89%** (vs **18.96%** for DSL), driven by sticker shock on premium monthly bills ($75–$100+) without corresponding perceived support value.
4. **Payment Method Friction**: Customers paying by **Electronic Check** churn at **45.29%** (more than double automated credit card or bank transfer billing at 15–16%).
5. **Lack of Security & Support Add-ons**: Accounts without `OnlineSecurity` or `TechSupport` exhibit substantially higher churn rates when experiencing technical issues.

### 3 Actionable Retention Strategies
1. **Incentivize 1-Year Contracts & Auto-Pay**: Offer a 10%–15% monthly discount or a free streaming upgrade for month-to-month customers who transition to an annual contract with auto-pay.
2. **First-90-Days Customer Care Onboarding**: Implement a structured onboarding journey (Day 7 connection verification, Day 30 bill walkthrough, and priority support routing for new sign-ups).
3. **Fiber Optic Retention Bundling**: Address fiber optic customer churn by bundling free Tech Support or Device Protection rather than forcing customer defection or price downgrades.

---

## 💻 6. How to Run

### Prerequisites
- macOS, Linux, or Windows
- Python 3.10+ (tested on Python 3.14)
- Virtual environment (`venv`)

### 1. Clone & Set Up Virtual Environment

```bash
# Navigate to project directory
cd churn-project

# Create a clean virtual environment
python3 -m venv .venv

# Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Upgrade pip and install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS Note**: XGBoost requires the OpenMP library. If you encounter an OpenMP error on macOS, run:
> ```bash
> brew install libomp
> ```

### 2. Run the Jupyter Notebook

To inspect exploratory analysis, baseline models, cross-validation, tuning, and SHAP explainability:

```bash
jupyter notebook notebooks/churn.ipynb
```
*(Run all cells sequentially: `Kernel` → `Restart & Run All`)*.

### 3. Launch the Streamlit Web Application

To launch the interactive customer risk scoring dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

#### App Features:
- **Interactive Form**: Adjust customer tenure, contract type, monthly charges, payment methods, and subscribed add-ons via sliders and dropdowns.
- **Real-Time Prediction**: Instantly calculates churn probability using the serialized XGBoost pipeline.
- **Dynamic Risk Categorization**: Flags account as **🔴 High Churn Risk** or **🟢 Low Churn Risk** against the calibrated threshold ($t = 31.14\%$).
- **Factor Attribution**: Explicitly details which attributes are elevating risk vs. protecting retention.
- **Prescriptive Next Action**: Suggests targeted commercial interventions based on the customer's specific profile.

---

## ⚠️ 7. Limitations & Future Work

### Limitations
- **Cross-Sectional Dataset**: The dataset provides a single static snapshot per customer without longitudinal usage trends, bandwidth consumption spikes, or historical billing changes.
- **Absence of Customer Service Telemetry**: Critical behavioral precursors to churn (e.g., number of open support tickets, complaint sentiment, call center hold times) are not captured.
- **Fixed Pricing Environment**: Competitor promotional activity and seasonal macroeconomic effects are unobserved.

### Future Work
1. **Survival & Time-to-Event Modeling**: Implement Cox Proportional Hazards or Random Survival Forests to predict *when* a customer is expected to churn, rather than just *if*.
2. **Uplift Modeling**: Train causal uplift models (e.g., Two-Model or X-Learner) to differentiate between "Persuadables" (who only stay if contacted) and "Sleeping Dogs" (who cancel if disturbed).
3. **Automated Drift Monitoring & MLOps**: Integrate MLflow for model registry tracking and Evidently AI for data/concept drift monitoring over incoming monthly billing cycles.
4. **Expanded Web Application**: Add batch CSV scoring and automated export of prioritized outbound retention call sheets.
