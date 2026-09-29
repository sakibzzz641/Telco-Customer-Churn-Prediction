# Telco Customer Churn Prediction

**Predict which telecom customers are about to leave — before they leave — so the retention team can act first.**

_End-to-end binary classification: EDA, SMOTE-balanced pipelines, 6-model comparison, tuning, and a final interpretable Logistic Regression scoring 0.844 ROC-AUC on a held-out test set._

![Status](https://img.shields.io/badge/status-completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Pandas](https://img.shields.io/badge/pandas-3.0-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![XGBoost](https://img.shields.io/badge/xgboost-3.4-orange)
![SMOTE](https://img.shields.io/badge/imbalanced--learn-0.14-orange)

Author: **MD. Sakib Al Hasan** — Data Science Portfolio Project

**Summary.** This project builds a churn scorer on the classic IBM Telco dataset — 7,043 customers, 21 columns. Six models (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost) were compared inside a leakage-safe SMOTE pipeline. The "boring" model won: a tuned **Logistic Regression** hits **0.844 ROC-AUC** on the unseen test set, catches **77% of real churners** (recall), and — because it's linear — every single prediction can be explained to a business stakeholder coefficient-by-coefficient.

---

## 1. Problem Statement

Every churned customer is recurring revenue walked out the door, and telecoms know acquiring is 5–10x more expensive than retaining. Without a reliable early-warning signal, retention teams burn budget on blanket campaigns and still miss the customers who matter most — the ones quietly planning to cancel.

**CAN WE DETERMINE, FROM A CUSTOMER'S SERVICE, ACCOUNT AND BILLING PROFILE, WHICH CUSTOMERS ARE LIKELY TO CHURN, AND WHAT SHOULD BE DONE ABOUT IT?**

The dataset is IBM's publicly shared [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) sample data (also mirrored on IBM's demo-data repo). It contains **7,043 rows × 21 columns** — one record per customer, with demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), service subscriptions (internet, phone, online security/backup, tech support, streaming), account details (`Contract`, `PaperlessBilling`, `PaymentMethod`), billing (`MonthlyCharges`, `TotalCharges`, `tenure`), and the target `Churn`. It's a single snapshot at one point in time; tenures run from 0 to 72 months. The target is binary and imbalanced: **73.5% stayed, 26.5% churned**.

The success metric is **ROC-AUC and F1 (with recall on the churn class)** rather than accuracy, because a scanner that just predicts "everyone stays" would score 73.5% accurate and be completely useless — and because for a retention program, *not missing* an at-risk customer matters more than avoiding false positives.

## 2. Methodology

### 2.1 Initial inspection & cleaning
- **0 NaN values** anywhere in the file — rare and pleasant. But `TotalCharges` was stored as text, and **11 rows had blank strings**. All 11 are `tenure = 0` — brand-new customers with no accumulated bill yet — so the blanks were filled with `0.0` (verified: after conversion `TotalCharges.min() == 0`) and the column became numeric.
- **0 duplicate rows**; `customerID` dropped (it's an ID, not a feature → model inputs went from 21 to 20 columns).
- Categorical values like `No internet service` / `No phone service` in the add-on columns are legitimate levels (this customer simply doesn't have that service), not missing data — left untouched.

### 2.2 EDA — where the signal actually lives
The numbers that shaped everything later:

| Finding | Value |
|---|---|
| Churn rate overall | 26.5% |
| **Month-to-month** customers | **42.7% churn** (vs 11.3% one-year, **2.8% two-year**) |
| Share of churners on month-to-month | **88.6%** |
| **Electronic check** payers | **45.3% churn** (vs **16.0%** for auto-pay bank/credit card) |
| **Fiber optic** internet | **41.9% churn** (vs 19.0% DSL, 7.4% no internet) |
| Median tenure — churned vs stayed | 10 vs 38 months |
| Median monthly charge — churned vs stayed | $79.65 vs $64.43 |

Basically: **short-tenured, month-to-month, electronic-check, fiber-optic customers**. The churn profile is unmistakable and consistent across every chart.

- **Outliers:** IQR found **0 outliers** in all three numeric columns; skewness is mild (0.24 / −0.22 / 0.96). Nothing capped or winsorized — high bill values are real usage, not data errors.
- **Multicollinearity:** `TotalCharges` is a running total of monthly bills, so it's naturally collinear with `tenure` (VIF 9.53 vs 5.84 for tenure). Kept both — the logistic sanity check showed no AUC gain from dropping it (0.8076 → 0.8085), and trees handle collinearity fine anyway.
- **Feature engineering** — two small, hand-built additions that track engagement:
  - `ServiceCount` (0–6): how many internet add-ons the customer subscribes to. Churn rate falls almost monotonically with engagement: **45.8% at 1 add-on → 5.3% at 6 add-ons**. Engaged customers don't leave.
  - `AutoPay` (0/1): automatic vs manual payment. 16.0% vs 34.7% churn — the single strongest payment behavior.

### 2.3 Leakage-safe pipeline design
- Numeric columns → `StandardScaler`; categorical columns → `OneHotEncoder(handle_unknown='ignore')`, both inside a `ColumnTransformer`.
- SMOTE (synthetic oversampling of the churn class) lives **inside the pipeline** (`imblearn.Pipeline`), so during cross-validation it only ever sees the training folds — never validation/test. This is the only honest way to use SMOTE; oversampling the whole dataset would leak information.
- **70 / 15 / 15 stratified split** (train 4,930 / val 1,056 / test 1,057) preserved the 73.5:26.5 ratio in every fold. The test set was touched exactly once, at the very end.
- Train-set imbalance was **3,622 : 1,308 (73.5:26.5)** → after SMOTE **3,622 : 3,622 (50:50)**.

### 2.4 Model comparison — why the simplest model won

5-fold CV scores (SMOTE pipeline, train set):

| Model | CV ROC-AUC | CV F1 |
|---|---|---|
| **Logistic Regression** | **0.8454** | **0.6354** |
| Random Forest | 0.8419 | 0.6229 |
| XGBoost | 0.8396 | 0.6053 |
| Gradient Boosting | 0.8392 | 0.5978 |
| Decision Tree | 0.8195 | 0.6148 |
| KNN | 0.7828 | 0.5686 |

Held-out **validation** results told the same story: Logistic Regression **0.8455 ROC-AUC / 0.6293 F1** — best of all six — with recall 0.82. Gradient Boosting took the accuracy crown (0.7879) but only caught 63% of churners.

I expected a tree to win. It didn't, and the reason is defensible: SMOTE rebalances the classes, which is exactly what a linear model needs in order to stop ignoring the minority class. Logistic Regression then wins on every metric that matters for churn detection (AUC, F1, recall) *and* gives us coefficients.

### 2.5 Hyperparameter tuning
RandomizedSearchCV (3-fold, ROC-AUC) on the two finalists:
- **Logistic Regression** → `C=1.0`, solver `liblinear` (CV AUC 0.8436).
- **Gradient Boosting** → `learning_rate=0.03`, `n_estimators=250`, `max_depth=3`, `min_samples_leaf=20`, `subsample=0.8` (CV AUC 0.8475).

Both tuned pipelines were then scored once on the held-out test set.

## 3. Results

Final test-set comparison (1,057 never-seen customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Tuned Logistic Regression** | 0.7455 | 0.5144 | **0.7651** | **0.6152** | **0.8439** |
| Tuned Gradient Boosting | **0.7833** | **0.5793** | 0.6762 | 0.6240 | 0.8405 |

**Winner: Tuned Logistic Regression** — bolded row. It's a deliberate trade: GB is ~4 accuracy points and ~6.5 precision points more accurate, but LR catches **9 more out of every 100 churners** (recall 0.77 vs 0.68) while staying fully interpretable. For an early-warning system, catching churners is the whole job.

Confusion matrix on the 281 real churners in the test set:

```
                predicted
               No      Yes
actual  No    573      203
        Yes    66      215
```

**215 of 281 actual churners were caught (76.5%), 66 missed; 203 of 776 stayers were flagged as false alarms.**

**Honest assessment:** ROC-AUC 0.844 is solid, not exceptional — and I'm not going to pretend otherwise. Churn is driven by messy, real-world forces (competitor offers, personal finances, service complaints) that a static account snapshot simply cannot fully capture. A precision of 0.51 means about **1 in 2 flagged customers actually churns** at the default threshold — useful as a triage sieve, not as an oracle. A retention team should still qualify flagged accounts (and can raise the decision threshold to trade recall for precision) before spending offer budget.

**What drives churn (Logistic Regression coefficients, standardized numerics + one-hot dummies):**

| Pushes churn ↑ | Coefficient | Pulls churn ↓ | Coefficient |
|---|---|---|---|
| `TotalCharges` (1 std) | +1.005 | `tenure` (1 std) | −1.613 |
| Contract = Month-to-month | +0.684 | Contract = Two year | −0.747 |
| InternetService = Fiber optic | +0.547 | `MonthlyCharges` (1 std) | −0.566 |
| No OnlineSecurity | +0.272 | InternetService = DSL | −0.530 |
| No TechSupport | +0.203 | Paperless billing = No | −0.322 |
| Streaming (TV/Movies) = Yes | +0.160 / +0.197 | Has Dependents | −0.288 |
| Payment = Electronic check | +0.149 | Payment = Mailed check | −0.240 |

Mostly intuitive — with one honest caveat in the notebook: `tenure`/`MonthlyCharges`/`TotalCharges` are collinear (VIF ≈ 9.5), so their individual signs wobble a little; the categorical drivers (contract, fiber, payment method) are the ones the business story should rest on.

**The at-risk customer, in one sentence:** a month-to-month customer on fiber optic with no add-on services, paying by electronic check, who hasn't been around long. That profile overlaps the EDA so closely that the model is essentially formalizing what the data already shouted.

## 4. Business Implications

> **88.6% of all churners sit on month-to-month contracts, and 42.7% of month-to-month customers churn** — that contract type is where the entire churn problem concentrates.

| Segment / output | What to do (for a non-technical stakeholder) |
|---|---|
| **Month-to-month, < 12 months tenure, fiber** (top-risk) | Priority 1 for retention offers — offer a discounted 1-year contract or a loyalty credit before the next billing cycle |
| **Electronic check payers** (45.3% churn) | Push a one-time discount for switching to auto-pay bank/credit card; auto-pay churn is 16.0% |
| **Fiber optic with 0–1 add-ons** | Bundle online security/support or streaming into fiber plans — churn drops from ~46% at 1 add-on to 5% at 6 |
| **No internet / DSL** (7.4% / 19.0% churn) | Low priority — already loyal; defend with basic service quality |
| **Flagged by model at >0.5 threshold** | RM team calls before renewal: ~1 in 2 flagged customers is confirmed leaving |

**Suggested next step.** Before committing budget, validate the model against actual retention actions: track whether offered customers actually stayed 6 months post-offer, and compare win-back vs prevention rates. Also consider a threshold sweep to pick the operating point (e.g. top-decile risk) that maximizes "offers won per dollar", and feed complaint/NPS signals into the next iteration — a static-snapshot model will never capture complaints, and complaints are drowning signals in real churn.

## 5. Repository Structure

```
Telco-customer-churn/
├── README.md                                    # this file
├── requirements.txt                             # pinned packages used in the notebook
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # dataset
├── notebooks/
│   └── telco-churn-analysis.ipynb               # full EDA + modeling notebook (executed, all outputs saved)
└── models/
    └── model_pipeline.pkl                       # final fitted pipeline: preprocessor + SMOTE + tuned Logistic Regression
```

## 6. Environment & Reproducibility

- **Python 3.13.5**, run on Windows; notebook executed end-to-end with `nbconvert` (all 45 code cells, outputs saved).
- Install pinned deps, then open the notebook:
  ```bash
  pip install -r requirements.txt
  jupyter lab notebooks/telco-churn-analysis.ipynb
  ```
- The dataset lives in data/ — no download step needed. — no download step needed. To regenerate everything, run the notebook top to bottom; it rewrites `models/model_pipeline.pkl` as its final cell.

## 7. License

- **Dataset:** publicly shared IBM sample data ("Telco Customer Churn"), intended for tutorials, academic use and portfolio demonstrations; it is also mirrored on Kaggle. No proprietary or contractual restrictions are known for educational reuse — credit IBM/Kaggle as the source if you republish results.
- **Project code:** MIT — feel free to reuse, adapt and learn from this analysis with attribution.

## 8. Contact

- Author: **MD. Sakib Al Hasan**
- Open to discussions about data science, ML model building, and churn/pricing analytics.
- Email: [sakibzzz641@gmail.com](mailto:sakibzzz641@gmail.com)
- GitHub: [github.com/sakibzzz641](https://github.com/sakibzzz641)
- LinkedIn: [linkedin.com/in/sakibzzz641](https://www.linkedin.com/in/sakibzzz641/)

---

