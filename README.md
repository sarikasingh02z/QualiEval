# 🔬 QualiEval — Data Quality Inspection for Network Traffic Anomaly Detection

QualiEval is a **Gradio-based interactive framework** designed to audit dataset quality before modeling, detect data risks (missing values, outliers, class imbalance, leakage), and recommend the most robust machine learning model for **network traffic anomaly detection**.

It bridges the gap between exploratory data analysis (EDA) and model selection — helping security analysts and data scientists avoid silent failures caused by poor data quality.

---

## 🚀 Live Demo (Hugging Face)

👉 [Insert Hugging Face Space Link Here]

---

## 🧠 Why QualiEval?

In network intrusion detection, **data quality directly impacts security outcomes**:

| Problem | Consequence |
|--------|-------------|
| Missing values | Biased models, missed attacks |
| Class imbalance | Model ignores minority attacks |
| Data leakage | Overly optimistic, fails in production |
| Outliers | False positives / false negatives |

QualiEval catches these issues **before you train a single model**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Data Quality Audit** | Missing %, column types, class imbalance, outliers, leakage candidates |
| ⚠️ **Model Risk Assessment** | Predicts model-specific risk (LR, RF, XGB) based on detected issues |
| 🧪 **Stress Tests** | Simulates performance drop under real-world data problems |
| 🛠️ **Actionable Fixes** | Generates concrete steps: imputation, SMOTE, leakage removal |
| 🎚️ **Threshold Optimization** | Visual F1/Precision/Recall curves with optimal cutoff for different scenarios |
| 🔁 **State Persistence** | Full session memory across tabs — no re-upload needed |

---

## 📊 Model Performance (on Test Data)

### Random Forest
| Scenario | Threshold | Accuracy | Precision | Recall | F1 Score |
|----------|-----------|----------|-----------|--------|----------|
| Default | 0.50 | 89.28% | 98.96% | 85.14% | 91.54% |
| **Security** 🛡️ | **0.10** | **93.05%** | **95.50%** | **94.23%** | **94.86%** |
| Operations ⚙️ | 0.90 | 83.00% | 99.87% | 75.12% | 85.75% |

### XGBoost
| Scenario | Threshold | Accuracy | Precision | Recall | F1 Score |
|----------|-----------|----------|-----------|--------|----------|
| Default | 0.50 | 89.32% | 98.88% | 85.29% | 91.58% |
| **Security** 🛡️ | **0.10** | **92.17%** | **96.76%** | **91.57%** | **94.09%** |
| Operations ⚙️ | 0.90 | 84.71% | 99.60% | 77.86% | 87.40% |

### Isolation Forest (Anomaly Detection)
- ROC-AUC: 0.4478 (trained on normal samples only)

---

## 🖥️ Screenshots

| Data Quality Audit | Model Risk Assessment |
|--------------------|----------------------|
| *[Add screenshot]* | *[Add screenshot]* |

| Stress Tests | Threshold Optimization |
|--------------|------------------------|
| *[Add screenshot]* | *[Add screenshot]* |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/qualieval.git
cd qualieval

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
