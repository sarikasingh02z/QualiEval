**QualiEval: Hybrid Data Quality Inspection & Model Robustness Framework**
A research-oriented framework for systematically evaluating how data quality influences machine learning model performance under controlled conditions.

**Overview**
QualiEval is an experimental, research-oriented framework that investigates how data quality factors influence machine learning model performance and robustness. Rather than building a production-ready system, this project focuses on methodological evaluation, controlled experimentation, and interpretability of model behavior under varying data quality conditions.

The framework combines unsupervised anomaly detection (Isolation Forest) with supervised learning (XGBoost, Random Forest) to systematically quantify how missing values, outliers, label noise, duplicates, and sensor drift impact model reliability.

**Research Motivation**
1.Machine learning models often report strong performance on clean, curated datasets, yet data quality degradation in real-world scenarios can lead to unstable performance, biased predictions, and unreliable outcomes.
2.Core Research Question: How do different data quality factors (missing values, outliers, label noise, duplicates, sensor drift) influence model performance, and can we systematically quantify model robustness through controlled fault injection?
3.QualiEval is intentionally designed as an experimental testbed to analyze this question through systematic evaluation and reproducible experimentation.

**System Architecture**
QualiEval employs a multi-stage detection and evaluation pipeline:

Stage 1: Data Quality Audit (Unsupervised)
-Isolation Forest used as a data auditor to identify anomalous patterns and potential data quality issues without relying on labeled data
-Detects statistical anomalies, missing value patterns, and distribution shifts

Stage 2: Model Risk Assessment (Supervised)
-XGBoost and Random Forest used as model evaluators for supervised learning tasks
-Both models are trained on SMOTE-balanced data to address class imbalance
-Model predictions are combined with configurable confidence thresholds (θ) and evaluated under controlled fault conditions

Stage 3 — Model Explanation
Plain-English breakdown of why a selected model succeeds or fails. No jargon. Shows exactly what the imbalance ratio, missing percentage, and leakage count mean for each algorithm.

Stage 4: Stress Testing
-Systematic fault injection to measure performance degradation
-Comparative analysis of model robustness under identical fault conditions

Stage 5: Actionable Fixes
-Data-first recommendations based on detected issues
-Priority-based fix suggestions with expected improvement estimates
-This layered evaluation architecture enables controlled experimentation on how data quality affects model behavior

Stage 6 — Final Recommendation
Real training results from the UNSW-NB15 network intrusion dataset. Includes threshold sensitivity analysis across three operating scenarios — Security (low threshold, high recall), Balanced, and Operations (high precision).

**Fault Injection Parameters**
QualiEval injects controlled faults to simulate real-world data quality issues:

 Fault Type        |  	Rate	    |        Affected Components
Missing Values	  |    15%	    |       sbytes, dbytes, rate, sttl, sload, dload
Outliers	        |     8%	    |       dur, sbytes, dbytes, rate, sload, dload
Label Noise	      |    12%	    |       Binary classification labels
Duplicates	      |     5%	    |       Random rows duplicated
Sensor Drift	    |    20%	    |       rate, sload, dload

**Experimental Results (UNSW-NB15 Dataset)**

1.Robustness Scores — performance retained under controlled fault injection

Model            Robustness Score         ROC-AUC
XGBoost              97.2                  0.9849
Random Forest        96.8                  0.9856
Isolation Forest     94.2                  0.4478 (PR-AUC: 0.6227)

2.Threshold Sensitivity (XGBoost, 175,341 test samples)
Scenario        Threshold      Precision       Recall       F1
Security         0.10           96.76%         91.57%     94.09%
Balanced         0.524          98.91%         85.04%     91.45%
Operations       0.90           99.60%         77.86%     87.40%

3.Stress Test Results — Accuracy Drop Under Fault Conditions
Scenario                Logistic Regression      Random Forest      XGBoost
Remove Leakage Columns        -48%                 -22%              -9%
Inject 20% Missing Values     -35%                 -18%             -12% 
Add 12% Label Noise           -42%                 -25%             -16%
Severe Imbalance (1:100)      -62%                 -31%             -15%

**Experimental Setup**
Total samples: 175,341 network flows

Features: 42 initial features → 18 selected after hybrid feature selection

Class distribution: Highly imbalanced (70% normal, 30% attacks)

Evaluation mode: Controlled experimental analysis with strict train/validation/test separation

Objective: Measure robustness and error behavior, not maximize raw accuracy

**Feature Selection**
A hybrid feature selection approach combining multiple methods:

-Manual Selection – Domain knowledge from network security literature
-Mutual Information – Statistical feature-target relationships
-Random Forest Voting – Ensemble-based importance ranking

Results:
-42 initial features → 18 selected
+2.3% F1 improvement validated through 5-fold cross-validation
-Selected features maintain interpretability while maximizing predictive power

**Key Findings**

-Default threshold (0.5) is suboptimal — threshold tuning improved F1 by ~12%
-Data leakage caused fake F1 = 1.0 before detection and removal
-SMOTE on training data only increased minority class recall by ~40%
-XGBoost showed 2-3x less degradation than Logistic Regression under all fault conditions
-Hybrid feature selection reduced 42 features to 18 with +2.3% F1 improvement

**Tech Stack**

Gradio — interactive web interface
Scikit-learn — Random Forest, Logistic Regression, Isolation Forest
XGBoost — gradient boosting classifier
Pandas / NumPy — data processing
Plotly — interactive visualizations

**Dataset**
UNSW-NB15 — Network intrusion detection dataset by the Cyber Range Lab of UNSW Canberra. 175,341 test samples across normal traffic and 9 attack categories. Train/Validation/Test split maintained strictly — test data never used during training or threshold tuning.

**Limitations**
-Evaluated under controlled experimental conditions with simulated faults
-Robustness scores are specific to the evaluated dataset and may not generalise across all domains
-Designed for methodological research, not production deployment

**Author**
Sarika Singh — Third-year Computer Science student
GitHub: github.com/sarikasingh02z
