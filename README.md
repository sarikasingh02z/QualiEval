#  QualiEval — Data Quality Inspection for Network Traffic Anomaly Detection

QualiEval is a 6-tab Gradio app that lets you upload any tabular dataset and see how data quality issues affect your ML models. Built on a controlled experiment: same dataset, same models, same test set — only the training data changes.
**The Experiment**
Trained three models on two versions of UNSW-NB15 (247,872 total samples, binary classification: Normal vs Attack):
Clean version — dataset as-is, properly preprocessed. Faulty version — same dataset with 6 independently injected faults using fixed seed=42 for reproducibility: 15% missing values, 8% outliers, 12% label noise, 5% duplicate rows, 20% sensor drift across rate/sload/dload columns, and Gaussian noise (10% of column std) across 6 base features.
**Results on 175,341 held-out test samples**
Tree-based models collapsed. XGBoost F1 dropped 13.65 points from 91.69% to 78.04%, with ROC-AUC falling from 0.985 to 0.271 — below random chance. Random Forest dropped 11.56 points from 91.68% to 80.13%, ROC-AUC falling from 0.986 to 0.440. Logistic Regression dropped only 3.11 points from 92.53% to 89.42%, ROC-AUC staying at 0.890. Robustness scores: LR 96.6/100, RF 87.4/100, XGBoost 85.1/100.
The ROC-AUC collapse for tree models isn't just "worse performance" — it means their predictions became actively anti-correlated with ground truth under combined data corruption. LR's smooth weighted decision boundary averaged out corruption instead of flipping at disrupted split thresholds.
Isolation Forest, trained on 37,000 normal-only samples as a data quality auditor, showed negligible anomaly rate shift (0.58%→0.64%) despite the supervised models collapsing — a documented blind spot of unsupervised auditors under distributed corruption.
**Feature Selection**
Hybrid pipeline combining domain knowledge (2 votes), Mutual Information (1 vote), and RF importance (1 vote) — reduced 42 features to 15 with 0.28% F1 trade-off. Selection fitted on training data only. Leaking engineered features (byte_ratio) dropped before selection ran.
**Threshold Optimization**
Security Mode (θ=0.10): 99.53% recall — catch everything, tolerate false alarms. Operations Mode (θ=0.90): 99.86% precision — minimize false positives. Tuned across 50 values on validation set only. Test set used once for final evaluation.
**The 6-Tab App**
**Tab 1** — Upload & Inspect: missing value detection, class imbalance analysis, data leakage scan, outlier detection. **Tab 2** — Model Risk Assessment: scores LR, RF, XGBoost 0-100 based on your specific dataset's issues. **Tab 3** — Model Explanation: explains why each model succeeds or fails given what the inspection found. **Tab 4** — Stress Tests: estimated performance drops across fault scenarios calculated from your dataset profile. **Tab 5** — Actionable Fixes: prioritized fix plan (CRITICAL→HIGH→MEDIUM→STANDARD) with code snippets. **Tab 6** — Final Recommendation: full clean vs faulty comparison with robustness scores and threshold sensitivity across all three deployment scenarios.
**What This Project Found**
The model with the best clean-data performance (XGBoost, F1 91.69%) had the largest collapse under corruption. The simplest model (LR) was the most robust. Model selection on clean data alone doesn't tell you which model will hold up when your production data degrades — and in real systems, it always eventually does.
**Dataset:** UNSW-NB15, Australian Centre for Cyber Security. 247,872 samples, 49 features, binary target.
Student project. Results are from controlled experiments on UNSW-NB15 — generalization to other datasets and fault combinations not guaranteed. Feedback welcome.
