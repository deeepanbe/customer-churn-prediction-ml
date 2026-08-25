# Customer Churn Prediction — ML Demo

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)

A churn-prediction pipeline built with pandas and scikit-learn: feature engineering, three-model comparison, feature importance, and a risk-scoring step that ranks customers by churn probability.

**This is a self-contained demo, not a production case study.** It ships with no dataset — running it generates a realistic 5,000-row synthetic customer base on the fly (seeded, so results are reproducible), which keeps the project runnable by anyone without a private data source. The methodology (feature engineering, model comparison, evaluation) is real and the same approach applies directly to a real telecom dataset (e.g. the public [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) by pointing `load_customer_data()` at a real CSV with the same column names.

## Run it yourself

```bash
pip install -r requirements.txt
python churn_predictor.py
```

No dataset needed — it's generated automatically on first run.

## What it does

1. Loads (or generates) customer records: tenure, monthly/total charges, contract type, payment method, internet service, tech support
2. Engineers 10 derived features (tenure buckets, spend-per-tenure ratios, contract/payment risk flags)
3. Trains and compares three models: Logistic Regression, Random Forest, Gradient Boosting
4. Picks the best by ROC-AUC, reports full precision/recall/F1
5. Ranks all customers by churn risk (Low / Medium / High) and lists the top 10 highest-risk

## Actual output (reproducible — run it and you'll get this)

| Model | ROC-AUC | Accuracy | Churn-class Precision | Churn-class Recall |
|---|---|---|---|---|
| **Random Forest (best)** | **0.974** | **0.92** | **0.85** | **0.69** |
| Gradient Boosting | 0.970 | 0.93 | 0.85 | 0.71 |
| Logistic Regression | 0.932 | 0.88 | 0.74 | 0.53 |

Top predictive features: tenure (in years and months), monthly charges, month-to-month contract flag, and charge-per-tenure ratio — together accounting for most of the model's decision-making.

**Being upfront about the recall number:** 69% recall on the churn class means roughly 3 in 10 actual churners are missed at this decision threshold. That's a realistic, honest number for an imbalanced classification problem — not the inflated "90%+ across the board" figures churn-prediction tutorials often quote. Threshold tuning or SMOTE-based resampling (not currently implemented) would be the natural next step to trade some precision for better recall, depending on the business cost of a missed churner vs. a false alarm.

## Stack

Python, pandas, NumPy, scikit-learn, joblib.

## Author

[Deepanraj Arumugam](https://deeepanbe.github.io) — Data Analyst / BI Developer
