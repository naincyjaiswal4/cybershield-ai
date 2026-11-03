"""
CyberShield AI - Deep Learning Threat Classification Module
Implements a Deep Neural Network (MLP): Input Layer -> Dense(ReLU) -> Dropout -> Dense(ReLU) -> Dropout -> Softmax.
Shows training/validation loss curves, accuracy trajectories, and multi-class probability scores.
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
from models.deep_learning import CyberDeepNeuralNetwork

st.set_page_config(page_title="Deep Learning - CyberShield AI", page_icon="🧠", layout="wide")
apply_cyber_theme()

render_header(
    title="Deep Neural Network (MLP) Classifier",
    subtitle="Deep Multi-Layer Perceptron architecture with ReLU activations, Inverted Dropout regularization, and Softmax classification.",
    status_tag="Deep Learning"
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
classes = data.get('target_classes', [])

if y_train is None or len(classes) == 0:
    st.info("ℹ️ The active dataset contains **unlabeled network telemetry** without ground-truth attack labels.\n\nDeep Neural Network classification requires labeled targets. For unlabeled datasets, please use the **Unsupervised Autoencoder Anomaly Detection** module (Page 6) or automated anomaly scoring on Page 1.")
    st.stop()
plotly_layout = get_plotly_theme_layout()

# Architecture & Training Hyperparameters
st.subheader("Network Architecture & Training Parameters")

c1, c2, c3, c4 = st.columns(4)
with c1:
    h1 = st.selectbox("Hidden Layer 1 Nodes:", [64, 128, 256], index=1)
    h2 = st.selectbox("Hidden Layer 2 Nodes:", [32, 64, 128], index=1)
with c2:
    dropout_rate = st.slider("Dropout Rate:", 0.0, 0.5, 0.25, 0.05)
    lr = st.select_slider("Learning Rate (η):", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)
with c3:
    epochs = st.slider("Training Epochs:", 10, 80, 40, 5)
    batch_size = st.selectbox("Batch Size:", [32, 64, 128], index=1)
with c4:
    st.markdown("<br>", unsafe_allow_html=True)
    train_dl_btn = st.button("Train Neural Network", use_container_width=True)

# Architecture Blueprint Card
st.markdown(f"""
<div class="content-card" style="margin-top: 10px;">
    <div style="font-size: 14px; font-weight: 600; color: #1E293B; margin-bottom: 4px;">Network Layer Blueprint</div>
    <div style="font-size: 13px; color: #475569;">
        <b>Input:</b> {X_train.shape[1]} features &nbsp;➔&nbsp; 
        <b>Dense 1:</b> {h1} units (ReLU) &nbsp;➔&nbsp; 
        <b>Dropout:</b> {dropout_rate*100:.0f}% &nbsp;➔&nbsp; 
        <b>Dense 2:</b> {h2} units (ReLU) &nbsp;➔&nbsp; 
        <b>Dropout:</b> {dropout_rate*100:.0f}% &nbsp;➔&nbsp; 
        <b>Output:</b> Softmax ({len(classes)} classes: {', '.join(classes)})
    </div>
</div>
""", unsafe_allow_html=True)

# Training Execution
if train_dl_btn:
    with st.spinner(f"Training Deep Neural Network across {epochs} epochs with mini-batch optimization..."):
        dnn = CyberDeepNeuralNetwork(
            input_dim=X_train.shape[1],
            hidden_layers=(h1, h2),
            num_classes=len(classes),
            dropout_rate=dropout_rate,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )
        history = dnn.fit(X_train, y_train, X_test, y_test)
        eval_res = dnn.evaluate(X_test, y_test, class_names=classes)
        
        st.session_state.dnn_model = dnn
        st.session_state.dnn_eval = eval_res
        st.session_state.model_metrics["Deep Neural Network (MLP)"] = eval_res
        st.session_state.trained_models["Deep Neural Network (MLP)"] = dnn
        st.success("Deep Neural Network trained and evaluated successfully.")

# Display Training Results & Curves
if "dnn_eval" in st.session_state and st.session_state.dnn_eval is not None:
    eval_res = st.session_state.dnn_eval
    history = eval_res["history"]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Training Convergence & Performance Summary")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Accuracy", f"{eval_res['accuracy']*100:.2f}%", "Testing Set Accuracy", "green")
    with m2:
        render_metric_card("Precision", f"{eval_res['precision']*100:.2f}%", "Weighted Precision", "blue")
    with m3:
        render_metric_card("Recall", f"{eval_res['recall']*100:.2f}%", "Threat Capture Rate", "purple")
    with m4:
        render_metric_card("F1-Score", f"{eval_res['f1']*100:.2f}%", "Harmonic Balance", "orange")

    # Curves
    col_loss, col_acc = st.columns(2)
    
    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"], mode='lines', name='Training Loss', line=dict(color='#2563EB', width=2.5)))
        fig_loss.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"], mode='lines+markers', name='Validation Loss', line=dict(color='#DC2626', width=2)))
        fig_loss.update_layout(
            title="Loss Convergence (Cross-Entropy)",
            xaxis_title="Epoch", yaxis_title="Loss Value",
            **plotly_layout
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_acc:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=history["epoch"], y=[a*100 for a in history["train_acc"]], mode='lines', name='Training Accuracy', line=dict(color='#16A34A', width=2.5)))
        fig_acc.add_trace(go.Scatter(x=history["epoch"], y=[a*100 for a in history["val_acc"]], mode='lines+markers', name='Validation Accuracy', line=dict(color='#F59E0B', width=2)))
        fig_acc.update_layout(
            title="Accuracy Trajectory (%)",
            xaxis_title="Epoch", yaxis_title="Accuracy (%)",
            **plotly_layout
        )
        st.plotly_chart(fig_acc, use_container_width=True)
else:
    st.info("Click 'Train Neural Network' to launch model training.")
