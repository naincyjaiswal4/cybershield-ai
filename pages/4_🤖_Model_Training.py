"""
CyberShield AI - Machine Learning Model Training & Evaluation Hub
Implements Random Forest, Support Vector Machine (SVM), Decision Tree, Logistic Regression, and XGBoost.
Displays Accuracy, Precision, Recall, F1-Score, ROC-AUC, Training Time, Confusion Matrix (TP/TN/FP/FN), and ROC Curves.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card,
    get_plotly_theme_layout
)
from utils.preprocessor import CyberDataPreprocessor
from models.ml_models import CyberMLModelHub

st.set_page_config(page_title="Model Training - CyberShield AI", page_icon="🤖", layout="wide")
apply_cyber_theme()

render_header(
    title="Machine Learning Threat Classification",
    subtitle="Train, optimize, and evaluate supervised machine learning models for multi-class intrusion detection.",
    status_tag="Supervised ML"
)

# Ensure preprocessed data exists
if "preprocessed_data" not in st.session_state or st.session_state.preprocessed_data is None:
    if "current_dataset" in st.session_state:
        pre = CyberDataPreprocessor(scaling_method="StandardScaler", n_features=20)
        target = "attack_category" if "attack_category" in st.session_state.current_dataset.columns else "label"
        st.session_state.preprocessed_data = pre.fit_transform(st.session_state.current_dataset, target_col=target)
        st.session_state.preprocessor_obj = pre
    else:
        st.warning("Please load a dataset first.")
        st.stop()

data = st.session_state.preprocessed_data
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
classes = data['target_classes']

hub = CyberMLModelHub()
plotly_layout = get_plotly_theme_layout()

# Training Mode Controls
st.subheader("Model Configuration & Hyperparameters")

col_m, col_p1, col_p2 = st.columns([1, 1, 1])

with col_m:
    selected_model = st.selectbox(
        "Algorithm Architecture:",
        [
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Decision Tree",
            "Logistic Regression",
            "XGBoost / Gradient Boosting",
            "Train All Models Concurrently"
        ]
    )

with col_p1:
    if selected_model == "Random Forest":
        n_est = st.slider("Number of Trees (n_estimators):", 20, 200, 100, 10)
        max_d = st.slider("Max Tree Depth:", 5, 30, 15, 1)
        params = {"n_estimators": n_est, "max_depth": max_d}
    elif selected_model == "Support Vector Machine (SVM)":
        c_val = st.slider("Regularization (C):", 0.1, 10.0, 1.0, 0.5)
        kernel_val = st.selectbox("Kernel Function:", ["rbf", "linear"])
        params = {"C": c_val, "kernel": kernel_val}
    elif selected_model == "Decision Tree":
        max_d = st.slider("Max Depth:", 3, 25, 12, 1)
        crit = st.selectbox("Split Criterion:", ["gini", "entropy"])
        params = {"max_depth": max_d, "criterion": crit}
    elif selected_model == "Logistic Regression":
        c_val = st.slider("Inverse Regularization (C):", 0.01, 10.0, 1.0, 0.1)
        max_it = st.slider("Max Iterations:", 100, 2000, 1000, 100)
        params = {"C": c_val, "max_iter": max_it}
    elif selected_model == "XGBoost / Gradient Boosting":
        n_est = st.slider("Number of Estimators:", 30, 200, 100, 10)
        max_d = st.slider("Max Depth:", 3, 10, 6, 1)
        lr = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01)
        params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}
    else:
        st.caption("Training with standard balanced hyperparameters across all algorithms.")
        params = {}

with col_p2:
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        train_btn = st.button("Train Selected Model", use_container_width=True)
    with col_btn2:
        train_all_btn = st.button("Train All Models", use_container_width=True)

# Train Process
if train_btn or train_all_btn:
    if train_all_btn or selected_model == "Train All Models Concurrently":
        models_to_train = [
            "Random Forest", "Support Vector Machine (SVM)", "Decision Tree",
            "Logistic Regression", "XGBoost / Gradient Boosting"
        ]
    else:
        models_to_train = [selected_model]
    
    with st.spinner("Fitting algorithms, measuring execution time, and calculating metrics..."):
        for m_name in models_to_train:
            res = hub.train_and_evaluate(m_name, X_train, y_train, X_test, y_test, class_names=classes, params=params)
            st.session_state.trained_models[m_name] = res["model"]
            st.session_state.model_metrics[m_name] = res
            
        st.success(f"Successfully trained and evaluated {len(models_to_train)} model(s).")

# Display Results
if st.session_state.get("model_metrics"):
    metrics_hub = st.session_state.model_metrics
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Evaluation Performance & Metrics")
    
    display_model = st.selectbox(
        "Select Model to Inspect:",
        options=list(metrics_hub.keys()),
        index=len(metrics_hub) - 1
    )
    
    res = metrics_hub[display_model]
    
    # Check for single-class dataset validation
    is_single_class = len(classes) < 2 or len(np.unique(y_train)) < 2
    if is_single_class:
        st.warning("Feature Importance and Classification Metrics require at least 2 target classes.")
        st.info("ROC-AUC unavailable for single-class datasets.")

    auc_str = f"{res['roc_auc']:.3f}" if res.get('roc_auc') is not None else "N/A"
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric_card("Accuracy", f"{res['accuracy']*100:.2f}%", "Overall Correctness", "green")
    with c2:
        render_metric_card("Precision", f"{res['precision']*100:.2f}%", "Weighted Precision", "blue")
    with c3:
        render_metric_card("Recall", f"{res['recall']*100:.2f}%", "Threat Capture Rate", "purple")
    with c4:
        render_metric_card("F1-Score", f"{res['f1']*100:.2f}%", "Harmonic Balance", "orange")
    with c5:
        render_metric_card("ROC-AUC", auc_str, "Discrimination Area", "blue")
    with c6:
        render_metric_card("Training Time", f"{res.get('training_time', 0):.3f}s", "Fit Elapsed Time", "blue")

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Confusion Matrix (TP, TN, FP, FN Breakdown)
    st.subheader(f"Confusion Matrix & Prediction Breakdown ({display_model})")
    
    col_cm, col_cm_stats = st.columns([1, 1])
    
    with col_cm:
        cm = res["confusion_matrix"]
        cm_classes = classes if len(classes) == cm.shape[0] else [f"Class {i}" for i in range(cm.shape[0])]
        fig_cm = px.imshow(
            cm,
            x=cm_classes,
            y=cm_classes,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Category", y="Actual Category")
        )
        fig_cm.update_layout(**plotly_layout)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm_stats:
        st.markdown("#### Classification Metrics Breakdown")
        
        c_tp, c_tn = st.columns(2)
        with c_tp:
            render_metric_card("True Positives (TP)", f"{res['tp']:,}", "Correctly Identified Attacks", "green")
        with c_tn:
            render_metric_card("True Negatives (TN)", f"{res['tn']:,}", "Correctly Identified Normal", "blue")
            
        c_fp, c_fn = st.columns(2)
        with c_fp:
            render_metric_card("False Positives (FP)", f"{res['fp']:,}", "False Alarms (Clean Flagged)", "orange")
        with c_fn:
            render_metric_card("False Negatives (FN)", f"{res['fn']:,}", "Missed Attack Invasions", "red")
            
        # Interpretation Section
        st.markdown("""
        <div class="content-card" style="margin-top: 8px;">
            <div style="font-size: 13px; font-weight: 600; color: #1E293B; margin-bottom: 4px;">Confusion Matrix Interpretation:</div>
            <div style="font-size: 12px; color: #475569; line-height: 1.5;">
                • <b>High TP & TN:</b> Demonstrates that the classifier accurately distinguishes between benign network operations and active threat vectors.<br>
                • <b>Low FN:</b> Minimizes the risk of severe network breaches slipping undetected.<br>
                • <b>Low FP:</b> Prevents alert fatigue for cybersecurity analysts in the SOC.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. ROC Curve & AUC Scores
    st.subheader(f"Receiver Operating Characteristic (ROC) & AUC Curves")
    
    col_roc_plot, col_roc_info = st.columns([3, 2])
    
    with col_roc_plot:
        roc_curves = res.get("roc_curves", {})
        if roc_curves:
            fig_roc = go.Figure()
            for cls_name, curve_data in roc_curves.items():
                fig_roc.add_trace(go.Scatter(
                    x=curve_data["fpr"],
                    y=curve_data["tpr"],
                    mode='lines',
                    name=f"{cls_name} (AUC = {curve_data['auc']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(color='#94A3B8', dash='dash')))
            fig_roc.update_layout(
                title=f"Multi-Class ROC Curves - {display_model}",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity / Recall)",
                **plotly_layout
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        elif res.get("roc_auc") is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name=f"Overall (AUC = {res['roc_auc']:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(color='#94A3B8', dash='dash')))
            fig_roc.update_layout(
                title=f"ROC Curves - {display_model}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **plotly_layout
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC-AUC unavailable for single-class datasets.")

    with col_roc_info:
        st.markdown("#### Detailed Per-Class Classification Report")
        report_data = res["classification_report"]
        report_rows = []
        for cls in classes:
            if cls in report_data:
                report_rows.append({
                    "Attack Category": cls,
                    "Precision": f"{report_data[cls]['precision']*100:.1f}%",
                    "Recall": f"{report_data[cls]['recall']*100:.1f}%",
                    "F1-Score": f"{report_data[cls]['f1-score']*100:.1f}%",
                    "Support": int(report_data[cls]['support'])
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

        if res.get("feature_importance") is not None and not is_single_class:
            st.markdown("#### Key Predictive Features")
            fi_df = pd.DataFrame({
                "Feature": data["feature_names"][:len(res["feature_importance"])],
                "Weight": res["feature_importance"]
            }).sort_values(by="Weight", ascending=True).tail(5)
            
            fig_fi = px.bar(fi_df, x="Weight", y="Feature", orientation="h", color="Weight", color_continuous_scale="Blues")
            fig_fi.update_layout(**plotly_layout)
            fig_fi.update_layout(showlegend=False, height=180)
            st.plotly_chart(fig_fi, use_container_width=True)
        elif is_single_class:
            st.caption("Feature Importance requires at least 2 target classes.")
else:
    st.info("Click 'Train Selected Model' or 'Train All Models' above to begin model fitting.")
