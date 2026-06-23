import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

print("QUALIEVAL — Data Quality Inspection & Model Risk Framework")


MODEL_RESULTS = {
    "random_forest": {
        "roc_auc_clean": 0.9858, "roc_auc_faulty": 0.4397,
        "robustness": 84.6, "train_samples": 65865, "test_samples": 175341,
        "clean": {
            "security":   {"threshold": 0.10, "accuracy": 93.19, "precision": 95.16, "recall": 94.82, "f1": 94.99},
            "balanced":   {"threshold": 0.48, "accuracy": 89.45, "precision": 98.95, "recall": 85.41, "f1": 91.68},
            "operations": {"threshold": 0.90, "accuracy": 82.33, "precision": 99.88, "recall": 74.13, "f1": 85.10},
        },
        "faulty": {
            "security":   {"threshold": 0.10, "accuracy": 68.06, "precision": 68.06, "recall": 100.00, "f1": 81.00},
            "balanced":   {"threshold": 0.48, "accuracy": 67.14, "precision": 68.10, "recall": 97.32,  "f1": 80.13},
            "operations": {"threshold": 0.90, "accuracy": 36.63, "precision": 98.37, "recall": 7.02,   "f1": 13.10},
        },
    },
    "xgboost": {
        "roc_auc_clean": 0.9854, "roc_auc_faulty": 0.2707,
        "robustness": 85.1, "train_samples": 65865, "test_samples": 175341,
        "clean": {
            "security":   {"threshold": 0.10, "accuracy": 92.37, "precision": 96.71, "recall": 91.92, "f1": 94.25},
            "balanced":   {"threshold": 0.48, "accuracy": 89.46, "precision": 98.94, "recall": 85.42, "f1": 91.69},
            "operations": {"threshold": 0.90, "accuracy": 85.46, "precision": 99.58, "recall": 78.97, "f1": 88.09},
        },
        "faulty": {
            "security":   {"threshold": 0.10, "accuracy": 68.06, "precision": 68.06, "recall": 99.99, "f1": 80.99},
            "balanced":   {"threshold": 0.48, "accuracy": 64.14, "precision": 66.92, "recall": 93.59, "f1": 78.04},
            "operations": {"threshold": 0.90, "accuracy": 33.47, "precision": 51.30, "recall": 44.14, "f1": 47.46},
        },
    },
    "logistic_regression": {
        "roc_auc_clean": 0.9244, "roc_auc_faulty": 0.8904,
        "robustness": 96.6, "train_samples": 65865, "test_samples": 175341,
        "clean": {
            "security":   {"threshold": 0.10, "accuracy": 83.92, "precision": 81.73, "recall": 98.37, "f1": 89.28},
            "balanced":   {"threshold": 0.26, "accuracy": 89.50, "precision": 89.67, "recall": 95.59, "f1": 92.53},
            "operations": {"threshold": 0.90, "accuracy": 58.33, "precision": 99.18, "recall": 39.10, "f1": 56.09},
        },
        "faulty": {
            "security":   {"threshold": 0.10, "accuracy": 77.12, "precision": 75.61, "recall": 97.97, "f1": 85.35},
            "balanced":   {"threshold": 0.26, "accuracy": 84.70, "precision": 84.44, "recall": 95.03, "f1": 89.42},
            "operations": {"threshold": 0.90, "accuracy": 56.76, "precision": 99.00, "recall": 36.85, "f1": 53.70},
        },
    },
    "isolation_forest": {
        "roc_auc_clean": 0.4478, "roc_auc_faulty": 0.2166,
        "pr_auc_clean": 0.6227, "pr_auc_faulty": 0.5382,
        "anomaly_rate_clean_pct": 0.58, "anomaly_rate_faulty_pct": 0.64,
        "train_samples": 37000, "contamination": 0.01,
        "note": "Unsupervised data quality auditor, not a classifier. Anomaly rate barely shifted "
                "under fault injection despite a ~12pt F1 drop in RF/XGBoost — a blind spot, not a benefit.",
    },
}

ROBUSTNESS_RATING = lambda score: (
    "Excellent" if score >= 95 else
    "Good"      if score >= 90 else
    "Fair"      if score >= 85 else
    "Poor"
)

# DATA QUALITY INSPECTION


def run_stage1(file, target_column, task_type="classification"):
    EMPTY = (None, None, None, None, None, None, None,
             "Upload a CSV file first.",
             None, None, None, None, None)
    if file is None:
        return EMPTY
    try:
        path = file.name if hasattr(file, 'name') else file
        df   = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        target_column = (target_column or "").strip()

        col_lower = {c.lower(): c for c in df.columns}
        if target_column not in df.columns:
            if target_column.lower() in col_lower:
                target_column = col_lower[target_column.lower()]
            else:
                target_column = df.columns[-1]

        dataset_info = {
            "Basic Info": {
                "Rows":          f"{len(df):,}",
                "Columns":       len(df.columns),
                "Memory":        f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB",
                "Target Column": target_column,
            },
            "Data Types": {
                "Numerical":   int(len(df.select_dtypes(include=['int64','float64']).columns)),
                "Categorical": int(len(df.select_dtypes(include=['object','category']).columns)),
            },
            "Columns (first 10)": list(df.columns[:10]) + (["..."] if len(df.columns) > 10 else [])
        }

        missing     = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_summary = {
            "Missing Overview": {
                "Total Missing":        f"{int(missing.sum()):,}",
                "Missing %":            f"{missing_pct.mean():.2f}%",
                "Columns with Missing": int((missing_pct > 0).sum()),
            }
        }
        critical_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
        if not critical_missing.empty:
            missing_summary["Critical Columns (>5% missing)"] = {
                k: f"{v:.2f}%" for k, v in critical_missing.items()
            }

        fig_missing = _plot_missing(missing_pct)

        imbalance_info = {}
        fig_imbalance  = _plot_imbalance(None)
        if task_type == "classification" and target_column in df.columns:
            cc = df[target_column].value_counts()
            if len(cc) >= 2:
                majority = int(cc.max()); minority = int(cc.min())
                ratio    = majority / minority if minority > 0 else float('inf')
                severity = ("CRITICAL" if ratio > 100 else
                            "HIGH"     if ratio > 20  else
                            "MEDIUM"   if ratio > 5   else "LOW")
                imbalance_info = {
                    "Target":       target_column,
                    "Ratio":        f"1:{ratio:.2f}",
                    "Severity":     severity,
                    "Distribution": {
                        f"Class '{cls}'": f"{cnt:,} ({cnt/len(df)*100:.1f}%)"
                        for cls, cnt in cc.items()
                    }
                }
                fig_imbalance = _plot_imbalance(cc)
            else:
                imbalance_info = {"Info": "Only 1 class found - check target column"}

        leakage = []
        future_keywords = ['success', 'result', 'outcome', 'final', 'approved',
                           'confirmed', 'completed', 'resolved', 'closed', 'settled']
        if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) > 1:
                corr = df[num_cols].corr()[target_column].drop(target_column, errors='ignore')
                for col, val in corr[abs(corr) > 0.7].items():
                    leakage.append({"column": col, "type": "High Correlation",
                                    "correlation": round(float(val), 3),
                                    "risk": "CRITICAL"})
        for col in df.columns:
            if col == target_column:
                continue
            if any(kw in col.lower() for kw in future_keywords):
                if not any(l["column"] == col for l in leakage):
                    leakage.append({"column": col, "type": "Future-looking name",
                                    "correlation": "N/A", "risk": "HIGH"})
        if not leakage:
            leakage = [{"status": "No significant leakage detected"}]

        outliers = []
        for col in df.select_dtypes(include=[np.number]).columns[:20]:
            s = df[col].dropna()
            if len(s) < 4: continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                n = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum())
                if n > 0:
                    pct = round(n / len(df) * 100, 2)
                    outliers.append({"column": col, "outliers": n, "percentage": pct,
                                     "risk": "HIGH" if pct > 10 else
                                             "MEDIUM" if pct > 5 else "LOW"})
        if not outliers:
            outliers = [{"status": "No significant outliers detected"}]

        status = (f"Inspection complete — {len(df):,} rows x {len(df.columns)} cols | "
                  f"Target: '{target_column}' | "
                  f"Issues: {len([l for l in leakage if 'status' not in l])} leakage, "
                  f"{len(critical_missing)} critical missing")

        return (dataset_info, missing_summary, imbalance_info, leakage, outliers,
                fig_missing, fig_imbalance, status,
                dataset_info, missing_summary, imbalance_info, leakage, outliers)

    except Exception as e:
        import traceback; traceback.print_exc()
        return (None, None, None, None, None, None, None,
                f"Error: {e}", None, None, None, None, None)


def _plot_missing(missing_pct):
    fig = go.Figure()
    data = missing_pct[missing_pct > 0].sort_values(ascending=False).head(15)
    if len(data) > 0:
        fig.add_trace(go.Bar(
            x=list(data.values), y=list(data.index), orientation='h',
            marker_color='#e53e3e',
            text=[f"{v:.1f}%" for v in data.values], textposition='auto'
        ))
        fig.update_layout(title="Missing Values by Column (%)", height=400,
                          xaxis_title="Missing %")
    else:
        fig.add_annotation(text="No missing values found",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="Missing Values", height=400)
    return fig


def _plot_imbalance(class_counts):
    fig = go.Figure()
    if class_counts is not None and len(class_counts) > 0:
        total  = class_counts.sum()
        colors = ['#48bb78', '#f56565', '#4299e1', '#ed8936', '#9f7aea']
        fig.add_trace(go.Bar(
            x=[str(x) for x in class_counts.index],
            y=list(class_counts.values),
            marker_color=colors[:len(class_counts)],
            text=[f"{v:,} ({v/total*100:.1f}%)" for v in class_counts.values],
            textposition='auto'
        ))
        fig.update_layout(title="Class Distribution", height=400,
                          xaxis_title="Class", yaxis_title="Count")
    else:
        fig.add_annotation(text="No classification target available",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Class Distribution", height=400)
    return fig

# MODEL RISK ASSESSMENT

def run_stage2(dataset_info, missing_summary, imbalance_info, leakage):
    if dataset_info is None:
        return None, None, "Run the inspection first (Upload & Inspect tab)."

    risks = {
        "Logistic Regression": {"risk": "LOW",  "score": 100, "factors": []},
        "Random Forest":       {"risk": "LOW",  "score": 100, "factors": []},
        "XGBoost":             {"risk": "LOW",  "score": 100, "factors": []},
    }

    if missing_summary:
        try:
            pct = float(
                missing_summary.get("Missing Overview", {})
                               .get("Missing %", "0%").replace("%", "")
            )
            if pct > 20:
                risks["Logistic Regression"]["risk"]  = "HIGH"
                risks["Logistic Regression"]["score"] -= 40
                risks["Logistic Regression"]["factors"].append(f">{pct:.0f}% missing — LR cannot handle NaN")
                risks["Random Forest"]["risk"]         = "MEDIUM"
                risks["Random Forest"]["score"]       -= 20
                risks["Random Forest"]["factors"].append("Missing values reduce tree quality")
                risks["XGBoost"]["score"]             -= 10
                risks["XGBoost"]["factors"].append("XGBoost handles NaN natively — minimal impact")
            elif pct > 10:
                risks["Logistic Regression"]["risk"]  = "MEDIUM"
                risks["Logistic Regression"]["score"] -= 20
                risks["Logistic Regression"]["factors"].append(f"{pct:.0f}% missing — imputation required")
                risks["Random Forest"]["score"]       -= 10
        except: pass

    if imbalance_info and "Ratio" in imbalance_info:
        try:
            ratio_str = imbalance_info["Ratio"].replace("1:", "")
            ratio = float(ratio_str)
            if ratio > 100:
                risks["Logistic Regression"]["risk"]  = "CRITICAL"
                risks["Logistic Regression"]["score"] -= 55
                risks["Logistic Regression"]["factors"].append(f"1:{ratio:.0f} ratio - always predicts majority class")
                risks["Random Forest"]["risk"]         = "HIGH"
                risks["Random Forest"]["score"]       -= 30
                risks["Random Forest"]["factors"].append(f"1:{ratio:.0f} ratio — needs class_weight='balanced'")
                risks["XGBoost"]["score"]             -= 15
                risks["XGBoost"]["factors"].append(f"1:{ratio:.0f} ratio — use scale_pos_weight")
            elif ratio > 20:
                risks["Logistic Regression"]["risk"]  = "HIGH"
                risks["Logistic Regression"]["score"] -= 35
                risks["Logistic Regression"]["factors"].append(f"1:{ratio:.0f} — minority class underrepresented")
                risks["Random Forest"]["risk"]         = "MEDIUM"
                risks["Random Forest"]["score"]       -= 15
            elif ratio > 5:
                risks["Logistic Regression"]["score"] -= 15
                risks["Logistic Regression"]["factors"].append(f"Mild imbalance 1:{ratio:.0f}")
        except: pass

    if leakage and "status" not in leakage[0]:
        for m in risks:
            risks[m]["risk"]  = "CRITICAL"
            risks[m]["score"] -= 50
            risks[m]["factors"].append("Data leakage - inflated, unrealistic accuracy")

    for m in risks:
        risks[m]["score"] = max(0, risks[m]["score"])

    for m in risks:
        s = risks[m]["score"]
        if risks[m]["risk"] not in ["CRITICAL"]:
            risks[m]["risk"] = ("LOW" if s >= 80 else
                                "MEDIUM" if s >= 60 else
                                "HIGH"   if s >= 40 else "CRITICAL")

    best       = max(risks, key=lambda m: risks[m]["score"])
    worst      = min(risks, key=lambda m: risks[m]["score"])
    risk_chart = _plot_risk_chart(risks)

    rec = (
        f"### Recommended: **{best}** (Score: {risks[best]['score']}/100)\n\n"
        f"**Why {best}?**\n"
        + ("XGBoost handles class imbalance natively (scale_pos_weight) and missing values — "
           "making it a strong choice when data has quality issues, though see the Final "
           "Recommendation tab for this project's robustness findings under fault injection.\n\n"
           if best == "XGBoost" else
           "Random Forest is robust to outliers and handles moderate imbalance well with "
           "class_weight='balanced'.\n\n"
           if best == "Random Forest" else
           "Logistic Regression is appropriate when data is clean and balanced — and in this "
           "project's own stress tests, it also proved the most resilient to data corruption "
           "overall. See the Final Recommendation tab.\n\n")
        + f"**Lowest risk score here: {worst}** (Score: {risks[worst]['score']}/100) — "
        + (", ".join(risks[worst]["factors"][:2]) if risks[worst]["factors"] else "lowest score for this data profile")
        + "\n\n*Note: this score reflects sensitivity to missing values, imbalance, and leakage "
          "found in your uploaded data. It does not measure robustness to broader data corruption "
          "— see the Final Recommendation tab for that finding.*"
    )
    return risks, risk_chart, rec


def _plot_risk_chart(risks):
    models = list(risks.keys())
    scores = [risks[m]["score"] for m in models]
    colors = ["#48bb78" if s >= 80 else "#f6ad55" if s >= 60 else
              "#f56565" if s >= 40 else "#9b2c2c" for s in scores]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=scores,
        marker_color=colors,
        text=[f"{s}/100\n{risks[m]['risk']}" for m, s in zip(models, scores)],
        textposition='auto'
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="green",  annotation_text="Safe (80)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Caution (60)")
    fig.add_hline(y=40, line_dash="dash", line_color="red",    annotation_text="Risky (40)")
    fig.update_layout(title="Algorithm Risk Scores", height=400,
                      yaxis_range=[0, 110], yaxis_title="Safety Score / 100")
    return fig

#MODEL EXPLANATION

def run_stage3(model_name, imbalance_info, missing_summary, leakage):
    rf  = MODEL_RESULTS["random_forest"]
    xgb = MODEL_RESULTS["xgboost"]
    lr  = MODEL_RESULTS["logistic_regression"]

    ratio_str = "N/A"
    ratio     = 1.0
    if imbalance_info and "Ratio" in imbalance_info:
        ratio_str = imbalance_info["Ratio"]
        try: ratio = float(ratio_str.replace("1:", ""))
        except: pass

    missing_pct = "0%"
    if missing_summary and "Missing Overview" in missing_summary:
        missing_pct = missing_summary["Missing Overview"].get("Missing %", "0%")

    has_leakage = leakage and "status" not in leakage[0]
    n_leakage   = len(leakage) if has_leakage else 0

    if model_name == "Logistic Regression":
        lr_clean_f1  = lr["clean"]["balanced"]["f1"]
        lr_faulty_f1 = lr["faulty"]["balanced"]["f1"]
        lines = [
            "## Logistic Regression — Lower Ceiling, Higher Floor",
            "",
            f"Trained on {lr['train_samples']:,} samples. "
            f"Clean-data F1: {lr_clean_f1}% (balanced scenario), ROC-AUC {lr['roc_auc_clean']:.4f}.",
            "",
            f"**Weakness 1 — Class Imbalance ({ratio_str})**",
            "   With a strong class imbalance, Logistic Regression's linear decision boundary "
            "can lean toward the majority class unless class_weight='balanced' is used (as it was here).",
            "",
            f"**Weakness 2 — Missing Values ({missing_pct})**",
            "   Logistic Regression cannot handle NaN values natively — imputation is required "
            "before training, unlike XGBoost which handles missing values internally.",
            "",
        ]
        if has_leakage:
            lines += [
                f"**Weakness 3 — Data Leakage ({n_leakage} column(s) flagged)**",
                "   Leakage columns give the model the answer during training, inflating accuracy "
                "in a way that won't hold on real data. Remove leakage columns before training, "
                "regardless of model choice.",
                "",
            ]
        lines += [
            "**This project's own finding — robustness under data corruption:**",
            f"   When trained on deliberately corrupted data (missing values, label noise, outliers, "
            f"duplicates, sensor drift, Gaussian noise), Logistic Regression's F1 dropped from "
            f"{lr_clean_f1}% to {lr_faulty_f1}% — about 3 points. Its ROC-AUC stayed strong "
            f"({lr['roc_auc_clean']:.3f} → {lr['roc_auc_faulty']:.3f}). Random Forest and XGBoost, "
            f"despite scoring higher on clean data, dropped much further and their ROC-AUC fell to "
            f"near or below random chance under the same corruption (see Final Recommendation tab).",
            "",
            "**Takeaway:** Logistic Regression has a lower ceiling on clean data but degrades far "
            "more gracefully when the underlying data quality is compromised — a real tradeoff, "
            "not just a weaker model.",
        ]
        return "\n".join(lines)

    elif model_name == "Random Forest":
        rf_clean_f1  = rf["clean"]["balanced"]["f1"]
        rf_faulty_f1 = rf["faulty"]["balanced"]["f1"]
        lines = [
            "## Random Forest — Strong on Clean Data, Fragile Under Corruption",
            "",
            f"Trained on {rf['train_samples']:,} samples. "
            f"Clean-data F1: {rf_clean_f1}% (balanced scenario), ROC-AUC {rf['roc_auc_clean']:.4f}.",
            "",
            f"**Weakness 1 — Imbalance ({ratio_str})**",
            "   Random Forest biases toward the majority class unless class_weight='balanced' "
            "is applied — used here instead of SMOTE, since SMOTE adds limited benefit at mild "
            "imbalance levels and can introduce synthetic noise.",
            "",
            f"**Weakness 2 — Missing values ({missing_pct})**",
            "   Scikit-learn's Random Forest cannot handle NaN natively — imputation is required.",
            "",
        ]
        if has_leakage:
            lines += [
                f"**Weakness 3 — Leakage ({n_leakage} columns)**",
                "   Feature importance will rank leakage columns at the top, inflating reported "
                "performance. Remove before training.",
                "",
            ]
        lines += [
            "**This project's own finding — fragility under data corruption:**",
            f"   On deliberately corrupted training data, Random Forest's F1 dropped from "
            f"{rf_clean_f1}% to {rf_faulty_f1}% — roughly 12 points. More notably, its ROC-AUC "
            f"collapsed from {rf['roc_auc_clean']:.3f} to {rf['roc_auc_faulty']:.3f}, near random "
            f"chance. Tree-based splits learned on clean feature thresholds become misleading once "
            f"those thresholds are crossed by injected noise and outliers.",
            "",
            f"**Result on test set (balanced scenario, clean data):** "
            f"Precision {rf['clean']['balanced']['precision']}% | "
            f"Recall {rf['clean']['balanced']['recall']}% | F1 {rf_clean_f1}%",
        ]
        return "\n".join(lines)

    elif model_name == "XGBoost":
        xgb_clean_f1  = xgb["clean"]["balanced"]["f1"]
        xgb_faulty_f1 = xgb["faulty"]["balanced"]["f1"]
        lines = [
            "## XGBoost — Best on Clean Data, Most Fragile Under Corruption",
            "",
            f"Trained on {xgb['train_samples']:,} samples. "
            f"Clean-data F1: {xgb_clean_f1}% (balanced scenario), ROC-AUC {xgb['roc_auc_clean']:.4f} "
            f"— the strongest clean-data result of the three models tested.",
            "",
            f"**1. Class Imbalance ({ratio_str})**",
            "   XGBoost's native scale_pos_weight parameter handles class imbalance directly, "
            "computed from the actual training class distribution rather than a fixed value.",
            "",
            f"**2. Missing Values ({missing_pct})**",
            "   XGBoost handles NaN natively — it learns the optimal split direction for missing "
            "values without imputation.",
            "",
        ]
        if has_leakage:
            lines += [
                f"**3. Leakage ({n_leakage} columns)**",
                "   Leakage columns inflate clean-data accuracy in any model — remove before "
                "training regardless of algorithm choice.",
                "",
            ]
        lines += [
            "**This project's own finding — the largest fragility gap of the three models:**",
            f"   XGBoost had the best clean-data F1 ({xgb_clean_f1}%) but the largest drop under "
            f"corruption — down to {xgb_faulty_f1}%, a ~14 point fall. Its ROC-AUC on faulty data "
            f"fell to {xgb['roc_auc_faulty']:.3f} — below random chance, meaning predictions were "
            f"no longer reliably correlated with the true label at all.",
            "",
            f"**Real test set results (clean data):**",
            f"| Scenario | Threshold | Precision | Recall | F1 |",
            f"|---|---|---|---|---|",
            f"| Security | {xgb['clean']['security']['threshold']} | {xgb['clean']['security']['precision']}% | {xgb['clean']['security']['recall']}% | {xgb['clean']['security']['f1']}% |",
            f"| Balanced | {xgb['clean']['balanced']['threshold']} | {xgb['clean']['balanced']['precision']}% | {xgb['clean']['balanced']['recall']}% | {xgb['clean']['balanced']['f1']}% |",
            f"| Operations | {xgb['clean']['operations']['threshold']} | {xgb['clean']['operations']['precision']}% | {xgb['clean']['operations']['recall']}% | {xgb['clean']['operations']['f1']}% |",
        ]
        return "\n".join(lines)

    return "Select a model above to see the explanation."

# STRESS TESTS

def run_stage4(missing_summary, imbalance_info, leakage):
    missing_pct  = 0.0
    ratio        = 1.0
    n_leakage    = 0
    has_leakage  = False

    if missing_summary and "Missing Overview" in missing_summary:
        try:
            missing_pct = float(
                missing_summary["Missing Overview"]
                               .get("Missing %", "0%").replace("%", "")
            )
        except: pass

    if imbalance_info and "Ratio" in imbalance_info:
        try:
            ratio = float(imbalance_info["Ratio"].replace("1:", ""))
        except: pass

    if leakage and "status" not in leakage[0]:
        has_leakage = True
        n_leakage   = len(leakage)

    leak_lr  = min(70, 30 + n_leakage * 9)  if has_leakage else 15
    leak_rf  = min(45, 15 + n_leakage * 5)  if has_leakage else 8
    leak_xgb = min(25, 6  + n_leakage * 3)  if has_leakage else 4

    miss_lr  = min(60, max(10, missing_pct * 2.0))
    miss_rf  = min(40, max(6,  missing_pct * 1.1))
    miss_xgb = min(25, max(4,  missing_pct * 0.7))

    import math
    ratio_factor = math.log10(max(ratio, 1.1))
    imb_lr  = min(75, max(10, ratio_factor * 22))
    imb_rf  = min(50, max(8,  ratio_factor * 12))
    imb_xgb = min(30, max(5,  ratio_factor * 6))

    # Noise sensitivity reference values are based on this project's own measured
    # robustness gap (LR 96.6/100, RF 84.6/100, XGB 85.1/100) rather than arbitrary guesses.
    noise_lr  = 3
    noise_rf  = 12
    noise_xgb = 14

    def r(v): return round(v, 1)

    scenarios = {
        f"Remove Leakage ({n_leakage} col{'s' if n_leakage != 1 else ''})": {
            "Logistic Regression": -r(leak_lr),
            "Random Forest":       -r(leak_rf),
            "XGBoost":             -r(leak_xgb),
        },
        f"Inject {r(missing_pct)}% Missing": {
            "Logistic Regression": -r(miss_lr),
            "Random Forest":       -r(miss_rf),
            "XGBoost":             -r(miss_xgb),
        },
        "Combined Fault Injection (measured)": {
            "Logistic Regression": -noise_lr,
            "Random Forest":       -noise_rf,
            "XGBoost":             -noise_xgb,
        },
        f"Severe Imbalance (1:{r(ratio)})": {
            "Logistic Regression": -r(imb_lr),
            "Random Forest":       -r(imb_rf),
            "XGBoost":             -r(imb_xgb),
        },
    }

    fig = go.Figure()
    colors = {
        "Logistic Regression": "#e53e3e",
        "Random Forest":       "#4299e1",
        "XGBoost":             "#48bb78"
    }
    for model, color in colors.items():
        drops = [scenarios[s][model] for s in scenarios]
        fig.add_trace(go.Bar(
            name=model,
            x=list(scenarios.keys()),
            y=drops,
            marker_color=color,
            text=[f"{abs(d)}%" for d in drops],
            textposition='auto'
        ))

    max_drop = max(abs(v) for s in scenarios.values() for v in s.values())

    fig.update_layout(
        title="Performance Drop Under Stress Conditions (Based on Your Dataset)",
        barmode='group', height=450,
        yaxis_range=[-(max_drop + 8), 5],
        yaxis_title="Performance Drop (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        annotations=[dict(
            text=f"Leakage/missing/imbalance bars estimated from your uploaded dataset's profile "
                 f"({r(missing_pct)}% missing, 1:{r(ratio)} imbalance, {n_leakage} leakage column(s)). "
                 f"'Combined Fault Injection' bar uses this project's own measured F1 drop from "
                 f"controlled experiments on UNSW-NB15.",
            xref="paper", yref="paper", x=0, y=-0.18,
            showarrow=False, font=dict(size=11, color="#666")
        )]
    )

    summary = {
        "Dataset Profile Used": {
            "Missing %":        f"{r(missing_pct)}%",
            "Imbalance Ratio":  f"1:{r(ratio)}",
            "Leakage Columns":  n_leakage,
        },
        f"Leakage Removal ({n_leakage} cols)": {
            "LR":  f"-{r(leak_lr)}%", "RF":  f"-{r(leak_rf)}%",
            "XGB": f"-{r(leak_xgb)}%", "Winner": "XGBoost"
        },
        f"Missing Values ({r(missing_pct)}%)": {
            "LR":  f"-{r(miss_lr)}%", "RF":  f"-{r(miss_rf)}%",
            "XGB": f"-{r(miss_xgb)}%", "Winner": "XGBoost"
        },
        "Combined Fault Injection (measured, this project)": {
            "LR":  f"-{noise_lr}%", "RF":  f"-{noise_rf}%",
            "XGB": f"-{noise_xgb}%", "Winner": "Logistic Regression"
        },
        f"Imbalance (1:{r(ratio)})": {
            "LR":  f"-{r(imb_lr)}%", "RF":  f"-{r(imb_rf)}%",
            "XGB": f"-{r(imb_xgb)}%", "Winner": "XGBoost"
        },
        "Note": "Leakage/missing/imbalance bars are estimated from your dataset profile. "
                "Combined Fault Injection uses this project's own measured results — Logistic "
                "Regression was the most robust model under realistic combined data corruption, "
                "despite weaker clean-data performance.",
    }
    return summary, fig

# ACTIONABLE FIXES

def run_stage5(missing_summary, imbalance_info, leakage):
    fixes = []

    if leakage and "status" not in leakage[0]:
        cols = [x["column"] for x in leakage if "column" in x]
        fixes.append({
            "priority": "CRITICAL",
            "issue": f"Data Leakage ({len(cols)} column(s))",
            "action": f"Remove column(s): {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}",
            "why": "Leakage inflates accuracy to 99%+ during training but model fails on real data entirely.",
            "expected_improvement": "Realistic accuracy (drop expected — that's normal and healthy)",
            "code_hint": f"df = df.drop({cols[:3]}, axis=1)"
        })

    if imbalance_info and "Ratio" in imbalance_info:
        try:
            ratio = float(imbalance_info["Ratio"].replace("1:", ""))
            if ratio > 5:
                priority = "CRITICAL" if ratio > 100 else "HIGH"
                fixes.append({
                    "priority": priority,
                    "issue": f"Class Imbalance (1:{ratio:.0f})",
                    "action": "Use class_weight='balanced' first; only try SMOTE if that's insufficient",
                    "why": f"1:{ratio:.0f} ratio causes models to under-predict the minority class. "
                           f"At mild-to-moderate imbalance, class_weight is often cleaner than SMOTE "
                           f"since it avoids synthetic samples that can distort the decision boundary.",
                    "expected_improvement": "+30-40% minority class recall",
                    "code_hint": "RandomForestClassifier(class_weight='balanced')\n"
                                 "# or for XGBoost: scale_pos_weight = n_negative / n_positive"
                })
        except: pass

    if missing_summary and "Missing Overview" in missing_summary:
        try:
            pct = float(
                missing_summary["Missing Overview"].get("Missing %", "0%").replace("%", "")
            )
            if pct > 5:
                priority = "CRITICAL" if pct > 20 else "HIGH" if pct > 10 else "MEDIUM"
                fixes.append({
                    "priority": priority,
                    "issue": f"Missing Values ({pct:.1f}% average)",
                    "action": "Impute numerical with median; categorical with 'unknown'",
                    "why": "Missing values break Logistic Regression entirely and reduce RF/XGB quality.",
                    "expected_improvement": f"+{min(20, int(pct/2))}% model stability",
                    "code_hint": "from sklearn.impute import SimpleImputer\nSimpleImputer(strategy='median').fit_transform(X)"
                })
        except: pass

    fixes.append({
        "priority": "STANDARD",
        "issue": "Threshold Selection",
        "action": "Tune decision threshold on validation set — not just use 0.5",
        "why": "Default 0.5 threshold rarely optimal. Security use-case: lower threshold. Operations: higher threshold.",
        "expected_improvement": "+5-15% F1 score depending on scenario",
        "code_hint": "thresholds = np.linspace(0.1, 0.9, 50)\nbest_t = thresholds[np.argmax([f1_score(y_val, (proba>=t).astype(int)) for t in thresholds])]"
    })
    fixes.append({
        "priority": "STANDARD",
        "issue": "Train/Validation/Test Split",
        "action": "Never touch test data until final evaluation",
        "why": "Using test data for threshold tuning or model selection leaks information and inflates reported performance.",
        "expected_improvement": "Trustworthy, publishable results",
        "code_hint": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)"
    })
    fixes.append({
        "priority": "STANDARD",
        "issue": "Model Selection Under Data Risk",
        "action": "Don't pick a model on clean-data accuracy alone — check robustness to corruption",
        "why": "This project found the model with the best clean-data F1 (XGBoost) had the largest "
               "performance collapse under realistic data corruption, while a simpler model "
               "(Logistic Regression) degraded far more gracefully.",
        "expected_improvement": "Avoids deploying a model that looks best in testing but fails first in production",
        "code_hint": "# Compare F1 / ROC-AUC on both clean and deliberately-corrupted training data\n"
                     "# before choosing a model for a production system"
    })

    return fixes if fixes else [{"priority": "NONE", "issue": "No critical issues found", "action": "Your data looks clean!", "why": "", "expected_improvement": "N/A", "code_hint": ""}]

#FINAL RECOMMENDATION

def build_stage6_robustness_plot():
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    scores = [
        MODEL_RESULTS["logistic_regression"]["robustness"],
        MODEL_RESULTS["random_forest"]["robustness"],
        MODEL_RESULTS["xgboost"]["robustness"],
    ]
    colors = ['#48bb78', '#4299e1', '#e53e3e']  # LR green (best), RF blue, XGB red (worst)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=scores,
        marker_color=colors,
        text=[f"{s}/100" for s in scores],
        textposition='auto'
    ))
    fig.add_hline(y=95, line_dash="dash", line_color="green",  annotation_text="Excellent (95)")
    fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Good (90)")
    fig.add_hline(y=85, line_dash="dash", line_color="red",    annotation_text="Fair (85)")
    fig.update_layout(title="Robustness Score — (Faulty F1 / Clean F1) x 100, Measured on Test Set",
                      height=400, yaxis_range=[0, 105], yaxis_title="Robustness / 100")
    return fig


def build_stage6_clean_vs_faulty_f1_plot():
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    clean  = [MODEL_RESULTS["logistic_regression"]["clean"]["balanced"]["f1"],
              MODEL_RESULTS["random_forest"]["clean"]["balanced"]["f1"],
              MODEL_RESULTS["xgboost"]["clean"]["balanced"]["f1"]]
    faulty = [MODEL_RESULTS["logistic_regression"]["faulty"]["balanced"]["f1"],
              MODEL_RESULTS["random_forest"]["faulty"]["balanced"]["f1"],
              MODEL_RESULTS["xgboost"]["faulty"]["balanced"]["f1"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Clean", x=models, y=clean, marker_color="#48bb78",
                          text=[f"{v}%" for v in clean], textposition="auto"))
    fig.add_trace(go.Bar(name="Faulty", x=models, y=faulty, marker_color="#f56565",
                          text=[f"{v}%" for v in faulty], textposition="auto"))
    fig.update_layout(title="F1 Score — Clean vs Faulty Training Data (Test Set, Balanced Threshold)",
                      barmode="group", height=420, yaxis_title="F1 Score (%)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def build_stage6_threshold_plot():
    rf  = MODEL_RESULTS["random_forest"]["clean"]
    xgb = MODEL_RESULTS["xgboost"]["clean"]
    lr  = MODEL_RESULTS["logistic_regression"]["clean"]
    fig = go.Figure()

    for label, model, color in [("Random Forest", rf, "#4299e1"),
                                  ("XGBoost", xgb, "#48bb78"),
                                  ("Logistic Regression", lr, "#e53e3e")]:
        pts = [model["security"], model["balanced"], model["operations"]]
        fig.add_trace(go.Scatter(
            x=[p["threshold"] for p in pts],
            y=[p["f1"] for p in pts],
            name=f"{label} F1", mode='lines+markers',
            line=dict(color=color, width=3),
            marker=dict(size=10)
        ))

    fig.add_vline(x=0.10, line_dash="dash", line_color="gray", annotation_text="Security t=0.10")
    fig.add_vline(x=0.90, line_dash="dash", line_color="gray", annotation_text="Operations t=0.90")
    fig.update_layout(title="Threshold Sensitivity — Clean Test Data, F1 by Scenario",
                      height=450, xaxis_title="Decision Threshold",
                      yaxis_title="F1 Score (%)", yaxis_range=[0, 102],
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig

# GRADIO INTERFACE

rf  = MODEL_RESULTS["random_forest"]
xgb = MODEL_RESULTS["xgboost"]
lr  = MODEL_RESULTS["logistic_regression"]
iso = MODEL_RESULTS["isolation_forest"]

with gr.Blocks(title="QualiEval", theme=gr.themes.Soft()) as demo:

    s_dataset   = gr.State()
    s_missing   = gr.State()
    s_imbalance = gr.State()
    s_leakage   = gr.State()
    s_outliers  = gr.State()

    gr.HTML("""
    <div style="text-align:center; padding:24px 0 8px;">
        <h1 style="font-size:2.4em; margin:0;">QualiEval</h1>
        <p style="color:#666; margin-top:6px; font-size:1.1em;">
            Upload any tabular dataset and get a 6-stage data quality and model risk report
        </p>
        <p style="color:#999; font-size:0.9em; margin-top:4px;">
            Upload &rarr; Inspect &rarr; Risk Assessment &rarr; Explanation &rarr; Stress Tests &rarr; Fixes &rarr; Recommendation
        </p>
    </div>
    """)

    with gr.Tabs():

        with gr.TabItem("Upload & Inspect"):
            gr.Markdown("""
            ### Upload
            Upload your CSV, select your target column and task type, then click **Run Inspection**.
            ### Data Quality Inspection
            Checks: **Missing values** · **Class imbalance** · **Data leakage** · **Outliers**
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    file_input  = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
                    target_col  = gr.Textbox(label="Target Column Name",
                                             placeholder="e.g. is_fraud, Label, target, y",
                                             value="Label")
                    task_type   = gr.Radio(["classification", "regression"],
                                           value="classification", label="Task Type")
                    inspect_btn = gr.Button("Run Inspection", variant="primary", size="lg")
                    status_out  = gr.Markdown("*Upload a CSV and click Run Inspection.*")
                with gr.Column(scale=2):
                    dataset_out = gr.JSON(label="Dataset Overview")

            with gr.Row():
                with gr.Column():
                    missing_out    = gr.JSON(label="Missing Values")
                    missing_plot   = gr.Plot()
                with gr.Column():
                    imbalance_out  = gr.JSON(label="Class Imbalance")
                    imbalance_plot = gr.Plot()
            with gr.Row():
                with gr.Column():
                    leakage_out = gr.JSON(label="Data Leakage")
                with gr.Column():
                    outlier_out = gr.JSON(label="Outliers")

        with gr.TabItem("Model Risk Assessment"):
            gr.Markdown("""
            ### Algorithm Risk Assessment
            Rates Logistic Regression, Random Forest, and XGBoost for risk on your specific data.
            *Run the inspection first, then click the button below.*
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    risk_btn  = gr.Button("Run Risk Assessment", variant="primary", size="lg")
                    risk_out  = gr.JSON(label="Risk Scores by Model")
                    risk_rec  = gr.Markdown()
                with gr.Column(scale=1):
                    risk_plot = gr.Plot()

        with gr.TabItem("Model Explanation"):
            gr.Markdown("""
            ### Why Models Succeed or Fail
            Select a model to see exactly how it performs on this project's UNSW-NB15 experiments,
            and how that connects to the risk factors found in your uploaded data.
            """)
            model_selector = gr.Dropdown(
                ["Logistic Regression", "Random Forest", "XGBoost"],
                value="Logistic Regression",
                label="Select Model to Explain"
            )
            explain_btn = gr.Button("Explain This Model", variant="primary")
            explain_out = gr.Markdown()

        with gr.TabItem("Stress Tests"):
            gr.Markdown("""
            ### Proof of Model Breaking
            Shows estimated performance drop based on the actual issues found in your dataset,
            plus this project's own measured drop under combined fault injection.
            """)
            stress_btn  = gr.Button("Run Stress Tests", variant="primary")
            stress_out  = gr.JSON(label="Performance Drop by Scenario")
            stress_plot = gr.Plot()

        with gr.TabItem("Actionable Fixes"):
            gr.Markdown("""
            ### Data-First Fixes
            Specific, prioritised fixes for the issues found in the inspection.
            """)
            fixes_btn = gr.Button("Generate Fix Plan", variant="primary")
            fixes_out = gr.JSON(label="Recommended Fixes (Prioritised)")

            gr.Markdown("""
            ---
            #### Fixes Applied in This Project's Training Run
            | Fix | Detail | Result |
            |---|---|---|
            | Remove label leakage | engineered leakage features dropped before feature selection | F1 realistic, not inflated |
            | class_weight='balanced' (RF) / scale_pos_weight (XGB) | No SMOTE — mild 55/45 imbalance | Robust under imbalance without synthetic noise |
            | Train/Val/Test split | 65,865 train / 16,467 val / 175,341 test | No leakage to test |
            | Threshold tuning on val | t=0.10 (security), t≈0.5 (balanced), t=0.90 (operations) | Tuned per deployment scenario |
            | 6-fault injection (seed=42) | missing, label noise, outliers, duplicates, drift, Gaussian noise | Reproducible robustness comparison |
            """)

        with gr.TabItem("Final Recommendation"):
            gr.Markdown("""
            ### Final Recommendation
            Results from training on the UNSW-NB15 network intrusion dataset. All numbers below
            are from the actual test set (175,341 held-out samples), evaluated once after threshold
            tuning on a separate validation set.
            """)
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
**The headline result: the best clean-data model was the most fragile one.**

On clean training data, XGBoost scored highest — F1 of {xgb['clean']['balanced']['f1']}% and
ROC-AUC of {xgb['roc_auc_clean']:.4f}, narrowly ahead of Random Forest
({rf['clean']['balanced']['f1']}% F1) and Logistic Regression ({lr['clean']['balanced']['f1']}% F1).

After training the same three models on data with six realistic injected faults — missing
values, label noise, outliers, duplicates, sensor drift, and Gaussian noise — the ranking
inverted. XGBoost's F1 fell to {xgb['faulty']['balanced']['f1']}% and its ROC-AUC collapsed to
{xgb['roc_auc_faulty']:.4f}, below random chance. Random Forest fell to
{rf['faulty']['balanced']['f1']}% F1 with ROC-AUC of {rf['roc_auc_faulty']:.4f}. Logistic
Regression, the weakest model on clean data, dropped only to {lr['faulty']['balanced']['f1']}%
F1 and kept a strong ROC-AUC of {lr['roc_auc_faulty']:.4f}.

**Robustness scores** (Faulty F1 / Clean F1 x 100): Logistic Regression {lr['robustness']}/100,
XGBoost {xgb['robustness']}/100, Random Forest {rf['robustness']}/100.

**Recommendation:** there is no single "best" model — there is a tradeoff. XGBoost gives the
highest performance ceiling on clean data. Logistic Regression gives the most predictable,
graceful degradation when data quality cannot be guaranteed. For a production system where
data corruption risk is real and hard to fully eliminate, that tradeoff should be made
deliberately, not by default.

Isolation Forest was used separately as an unsupervised data quality auditor, not a classifier.
Notably, its anomaly rate barely shifted under the same fault injection
({iso['anomaly_rate_clean_pct']}% to {iso['anomaly_rate_faulty_pct']}%) even though it caused a
~12-14 point F1 drop in the supervised models — a blind spot worth knowing about if you're
relying on an anomaly detector to flag data quality problems before they reach your classifier.
---
**Test Set Results — Clean Data (Balanced Scenario)**
| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | {xgb['clean']['balanced']['threshold']} | {xgb['clean']['balanced']['precision']}% | {xgb['clean']['balanced']['recall']}% | {xgb['clean']['balanced']['f1']}% | {xgb['roc_auc_clean']:.4f} |
| Random Forest | {rf['clean']['balanced']['threshold']} | {rf['clean']['balanced']['precision']}% | {rf['clean']['balanced']['recall']}% | {rf['clean']['balanced']['f1']}% | {rf['roc_auc_clean']:.4f} |
| Logistic Regression | {lr['clean']['balanced']['threshold']} | {lr['clean']['balanced']['precision']}% | {lr['clean']['balanced']['recall']}% | {lr['clean']['balanced']['f1']}% | {lr['roc_auc_clean']:.4f} |

**Test Set Results — Faulty Data (Balanced Scenario)**
| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | {xgb['faulty']['balanced']['threshold']} | {xgb['faulty']['balanced']['precision']}% | {xgb['faulty']['balanced']['recall']}% | {xgb['faulty']['balanced']['f1']}% | {xgb['roc_auc_faulty']:.4f} |
| Random Forest | {rf['faulty']['balanced']['threshold']} | {rf['faulty']['balanced']['precision']}% | {rf['faulty']['balanced']['recall']}% | {rf['faulty']['balanced']['f1']}% | {rf['roc_auc_faulty']:.4f} |
| Logistic Regression | {lr['faulty']['balanced']['threshold']} | {lr['faulty']['balanced']['precision']}% | {lr['faulty']['balanced']['recall']}% | {lr['faulty']['balanced']['f1']}% | {lr['roc_auc_faulty']:.4f} |
                    """)
                with gr.Column():
                    gr.Plot(build_stage6_robustness_plot())

            gr.Markdown("### Clean vs Faulty F1 — All Three Models")
            gr.Plot(build_stage6_clean_vs_faulty_f1_plot())

            gr.Markdown("### Threshold Sensitivity (Clean Test Data)")
            gr.Plot(build_stage6_threshold_plot())

    # EVENT WIRING

    inspect_btn.click(
        run_stage1,
        inputs=[file_input, target_col, task_type],
        outputs=[dataset_out, missing_out, imbalance_out, leakage_out, outlier_out,
                 missing_plot, imbalance_plot, status_out,
                 s_dataset, s_missing, s_imbalance, s_leakage, s_outliers]
    )

    risk_btn.click(
        run_stage2,
        inputs=[s_dataset, s_missing, s_imbalance, s_leakage],
        outputs=[risk_out, risk_plot, risk_rec]
    )

    explain_btn.click(
        run_stage3,
        inputs=[model_selector, s_imbalance, s_missing, s_leakage],
        outputs=[explain_out]
    )

    stress_btn.click(
        run_stage4,
        inputs=[s_missing, s_imbalance, s_leakage],
        outputs=[stress_out, stress_plot]
    )

    fixes_btn.click(
        run_stage5,
        inputs=[s_missing, s_imbalance, s_leakage],
        outputs=[fixes_out]
    )

# LAUNCH

demo.launch(server_name="0.0.0.0", server_port=7860)