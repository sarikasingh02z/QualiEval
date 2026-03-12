#  QualiEval — Data Quality Inspection for Network Traffic Anomaly Detection

**What is QualiEval?**
QualiEval is an interactive 6-tab web app that lets you upload any tabular dataset and instantly see how data quality issues — missing values, class imbalance, leakage, outliers — affect your ML models.
Instead of just reading about data quality problems, this tool lets you see them happen in real time.

**The Experiment Behind It**
I trained models on two versions of the same dataset:

Clean version — UNSW-NB15 network intrusion dataset as-is
Faulty version — same dataset with deliberately injected faults:

**Fault**        **Amount**           **Why It Hurts**
Missing Values     15%              Forces models to guess or drop signal 
Label Noise        12%              Model trains on lies — can't detect it
Extreme Outliers    8%              Drags decision boundary the wrong way 
Duplicate Rows      5%              Model overfits to patterns that aren't real
Sensor Drift       20%              Gradual shifts that go completely unnoticed

Results on 175,341 test samples:
**Model**       **Clean F1**    **Faulty F1**    **Drop**
Random Forest     94.86%           ~91.8%           ~3%
XGBoost           94.09%           ~92.5%           ~3%
The models didn't crash. They degraded quietly — which is the real danger.

**The 6-Tab Framework*8
Tab 1 — Upload & Inspect
Upload any CSV and instantly get:
-Missing value detection with severity ratings
-Class imbalance analysis with ratio scoring
-Data leakage detection (correlation scan + future-keyword scan)
-Outlier detection using IQR method
Tab 2 — Model Risk Assessment
Scores Logistic Regression, Random Forest, and XGBoost (0–100) based on your specific dataset's issues — not generic advice.
Tab 3 — Model Explanation
Explains exactly why each model succeeds or fails on your data, with the reasoning tailored to what the inspection found.
Tab 4 — Stress Tests
Shows estimated performance drops per model across 4 fault scenarios — calculated from your actual dataset profile, not hardcoded values.
Tab 5 — Actionable Fixes
Prioritized fix plan (CRITICAL → HIGH → MEDIUM → STANDARD) with actual code snippets for each issue found.
Tab 6 — Final Recommendation
Real results from training on UNSW-NB15 with threshold sensitivity analysis across 3 scenarios:

-Security (threshold = 0.10) — catch everything, tolerate false alarms
-Balanced (threshold = 0.52) — best overall F1
-Operations (threshold = 0.90) — high precision, fewer false positives

**Architecture**
STAGE 1 — Data Quality Audit (Isolation Forest)
    ↓ unsupervised — no labels needed
    "Is this data point statistically weird?"
    Clean anomaly rate: 1.2% → Faulty: 4.7%
    Proves fault injection worked ✓

STAGE 2 — Attack Classification (Random Forest + XGBoost)
    ↓ supervised — trained on labeled data
    "Is this network traffic an attack?"
    RF ROC-AUC: 0.9856 | XGB ROC-AUC: 0.9849

STAGE 3 — Robustness Comparison
    ↓ clean vs faulty performance gap
    XGBoost Robustness Score: 97.2/100
    Random Forest Robustness Score: 96.8/100

Key insight: Isolation Forest is NOT a classifier here. It's a data quality auditor — trained only on normal samples to flag suspicious records before the supervised models ever see the data.

**Feature Selection**
Used a hybrid voting approach combining 3 methods:

Manual selection (domain knowledge) — 2 votes
Mutual Information — 1 vote
Random Forest Importance — 1 vote

Result: 42 features → 18 features with no F1 loss (slight improvement on cross-validation)

**Dataset**
Trained and tested on UNSW-NB15 — a network intrusion detection dataset created by the Australian Centre for Cyber Security (ACCS).
-247,872 total samples (train + test)
-49 features covering network traffic attributes
-Binary classification: Normal (0) vs Attack (1)

**What I Learned**
-Data quality > model choice — the same XGBoost model on clean vs faulty data told two completely different stories
-Threshold tuning matters as much as the model — default 0.5 is rarely optimal
-Isolation Forest has a specific job — it's not a classifier, it's an auditor. Using it as a classifier gave 32% accuracy. Using it correctly gave meaningful data quality signals.
-Quiet degradation is the real risk — models don't fail loudly, they fail silently

Feedback Welcome
This is a student project — I'm still learning and would genuinely appreciate any feedback, suggestions, or critique. Feel free to open an issue or reach out!
