"""
CyberShield AI - Autoencoder Anomaly Detection Hub
Unsupervised intrusion detection trained exclusively on Normal baseline traffic.
Flags zero-day attacks and stealth network anomalies via Mean Squared Reconstruction Error.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card,
    get_plotly_theme_layout
)
from utils.preprocessor import CyberDataPreprocessor
from models.autoencoder import CyberAutoencoder

st.set_page_config(page_title="Anomaly Detection - CyberShield AI", page_icon="🔍", layout="wide")
apply_cyber_theme()

render_header(
    title="Unsupervised Autoencoder Anomaly Detection",
    subtitle="Detect zero-day attacks, unknown botnets, and abnormal network flow behaviors via latent reconstruction error.",
    status_tag="Zero-Day Detection"
)

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
plotly_layout = get_plotly_theme_layout()

# Robust Normal class indexing
normal_idx = None
for idx, c in enumerate(classes):
    c_str = str(c).strip().lower()
    if c_str in ['normal', 'benign', '0', 'clean', 'legitimate', 'safe', 'no_attack']:
        normal_idx = idx
        break

if normal_idx is None:
    for idx, c in enumerate(classes):
        if 'norm' in str(c).strip().lower() or 'benign' in str(c).strip().lower():
            normal_idx = idx
            break

if normal_idx is None:
    normal_idx = 0

X_train_normal = X_train[y_train == normal_idx]
if len(X_train_normal) == 0:
    st.warning("No baseline 'Normal' traffic flows detected in training set. Using general baseline for autoencoder.")
    X_train_normal = X_train

y_test_binary = (y_test != normal_idx).astype(int)

# Autoencoder Configuration
st.subheader("Autoencoder Parameters & Baseline Setup")

c1, c2, c3, c4 = st.columns(4)
with c1:
    latent_dim = st.selectbox("Latent Dimension (Bottleneck):", [4, 6, 8, 12], index=1)
    hidden_dim = st.selectbox("Hidden Layer Dimension:", [10, 14, 20, 32], index=1)
with c2:
    lr = st.select_slider("Learning Rate:", options=[0.005, 0.01, 0.02, 0.05], value=0.01)
    ae_epochs = st.slider("Training Epochs:", 10, 60, 35, 5)
with c3:
    st.caption("Training Normal Baseline Subsets:")
    st.write(f"Normal Flows: **{X_train_normal.shape[0]:,} samples**")
    st.write(f"Testing Flows: **{X_test.shape[0]:,} samples**")
with c4:
    st.markdown("<br>", unsafe_allow_html=True)
    train_ae_btn = st.button("Train Autoencoder", use_container_width=True)

# Training Execution
if train_ae_btn:
    with st.spinner("Training Autoencoder on normal baseline network flows..."):
        ae = CyberAutoencoder(
            input_dim=X_train.shape[1],
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=ae_epochs,
            batch_size=64
        )
        history = ae.fit(X_train_normal)
        eval_res = ae.evaluate(X_test, y_test_binary)
        
        st.session_state.autoencoder_model = ae
        st.session_state.autoencoder_eval = eval_res
        st.success("Autoencoder training complete. Anomaly reconstruction baseline established.")

# Dynamic Evaluation & Visualization
if "autoencoder_eval" in st.session_state and st.session_state.autoencoder_eval is not None:
    ae = st.session_state.autoencoder_model
    eval_res = st.session_state.autoencoder_eval
    
    # Check for single-class dataset validation
    is_single_class = len(np.unique(y_test_binary)) < 2
    if is_single_class:
        st.warning("Feature Importance and Classification Metrics require at least 2 target classes.")
        st.info("ROC-AUC unavailable for single-class datasets.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Decision Threshold Calibration")
    
    col_thresh, col_rec = st.columns([2, 1])
    with col_thresh:
        curr_threshold = st.slider(
            "Reconstruction Error Decision Threshold (τ):",
            min_value=float(np.min(eval_res["errors"])),
            max_value=float(np.percentile(eval_res["errors"], 99.5)),
            value=float(eval_res.get("threshold", ae.threshold)),
            step=float(max(1e-5, (np.percentile(eval_res["errors"], 99.5) - np.min(eval_res["errors"])) / 100)),
            format="%.4f"
        )
    with col_rec:
        st.markdown("<br>", unsafe_allow_html=True)
        recalc_btn = st.button("Recalculate Metrics", use_container_width=True)

    # Live update if threshold changed or button clicked
    if recalc_btn or abs(curr_threshold - float(eval_res.get("threshold", ae.threshold))) > 1e-5:
        eval_res = ae.evaluate(X_test, y_test_binary, custom_threshold=curr_threshold)
        st.session_state.autoencoder_eval = eval_res
        st.rerun()

    # Dynamic metric formatting directly from evaluation results
    acc_disp = f"{eval_res['accuracy']*100:.2f}%"
    prec_disp = f"{eval_res['precision']*100:.2f}%"
    rec_disp = f"{eval_res['recall']*100:.2f}%"
    f1_disp = f"{eval_res['f1']*100:.2f}%"
    auc_disp = f"{eval_res['roc_auc']:.3f}" if eval_res.get('roc_auc') is not None else "N/A"

    # Display Dynamic Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Accuracy", acc_disp, f"TP={eval_res['tp']}, TN={eval_res['tn']}", "green")
    with m2:
        render_metric_card("Precision", prec_disp, f"FP={eval_res['fp']}", "blue")
    with m3:
        render_metric_card("Recall", rec_disp, f"FN={eval_res['fn']}", "purple")
    with m4:
        render_metric_card("F1-Score", f1_disp, "Harmonic Mean", "orange")
    with m5:
        render_metric_card("ROC-AUC", auc_disp, "Area Under Curve", "blue")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations: Error Distribution & ROC Curve
    col_err_plot, col_roc = st.columns([3, 2])
    
    with col_err_plot:
        n_show = min(350, len(eval_res["errors"]))
        indices = np.arange(n_show)
        sample_errors = eval_res["errors"][:n_show]
        sample_labels = ["Normal" if y == 0 else "Anomaly / Attack" for y in y_test_binary[:n_show]]
        
        scatter_df = pd.DataFrame({
            "Packet Index": indices,
            "Reconstruction Error (MSE)": sample_errors,
            "Class": sample_labels
        })
        
        fig_scatter = px.scatter(
            scatter_df, x="Packet Index", y="Reconstruction Error (MSE)",
            color="Class",
            color_discrete_map={"Normal": "#16A34A", "Anomaly / Attack": "#DC2626"},
            title="Reconstruction Error per Packet vs Decision Threshold"
        )
        fig_scatter.add_hline(y=curr_threshold, line_dash="dash", line_color="#2563EB", annotation_text=f"Threshold (τ={curr_threshold:.4f})")
        fig_scatter.update_layout(**plotly_layout)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_roc:
        if eval_res.get("fpr") is not None and eval_res.get("tpr") is not None:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=eval_res["fpr"], y=eval_res["tpr"], mode='lines', name=f'ROC Curve (AUC = {auc_disp})', line=dict(color='#2563EB', width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance Level', line=dict(color='#94A3B8', dash='dash')))
            fig_roc.update_layout(
                title="Autoencoder ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                **plotly_layout
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC-AUC unavailable for single-class datasets.")
else:
    st.info("Click 'Train Autoencoder' above to build the baseline normal profile.")
