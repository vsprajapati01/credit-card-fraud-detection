# Credit Card Fraud Detection

**Machine learning coursework project** — end-to-end fraud detection on a highly imbalanced credit-card transaction dataset, built with Python and scikit-learn.

## The problem
Credit card fraud is rare (< 1% of transactions), which makes it a classic **imbalanced classification** problem: a model that predicts "legitimate" every time looks 99%+ accurate while catching zero fraud. This project tackles that head-on.

## Dataset
[Credit Card Fraud Detection — Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European card transactions (Sept 2013), only 492 fraudulent (0.17%). Features V1–V28 are PCA-anonymized; `Time`, `Amount`, `Class` are raw.

> The CSV is ~143 MB — too large for GitHub, so it's intentionally **not** in this repo (see `.gitignore`). Download it free from Kaggle and place `creditcard.csv` in the project folder to run the pipeline.

## Methodology
1. **EDA** — transaction patterns, class distribution, feature correlations, amount/time analysis
2. **Preprocessing** — `StandardScaler` on Amount/Time, PCA-anonymized features kept as-is, `Pipeline`-based workflow
3. **Imbalance handling** — SMOTE oversampling and random undersampling, compared against the raw imbalanced baseline
4. **Modeling** — 9 configurations across 3 classifiers × 3 sampling strategies:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. **Evaluation** — stratified train/test split + cross-validation; compared on precision, recall, F1-score, ROC-AUC, precision-recall curves, and confusion matrices (accuracy alone is misleading on imbalanced data)
6. **Deployment demo** — best model + scaler saved with `joblib`; an interactive script loads them and scores new transactions live with a tunable fraud-probability threshold

## Key takeaways
- Recall matters more than accuracy here: missing a fraud costs far more than flagging a legit transaction
- Proper sampling strategy changed results more than switching algorithms
- Precision-recall curves tell the real story on skewed data; ROC curves can look deceptively good

## Run it
```bash
pip install -r requirements.txt
# place creditcard.csv in the project folder, then:
python CCdetection19.py        # full pipeline: EDA → train → evaluate → save model
# or open CCdetection19.ipynb for the notebook walkthrough
```

## Files
- `CCdetection19.ipynb` — full notebook with outputs and charts
- `CCdetection19.py` — the same pipeline as a script
- `fraud_model.pkl` — trained best model (joblib)
- `scaler.pkl` — fitted scaler for live predictions
- `requirements.txt` — dependencies

*Coursework project, Spring 2026.*
