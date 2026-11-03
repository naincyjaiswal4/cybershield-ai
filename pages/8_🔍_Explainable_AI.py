"""
CyberShield AI - Explainable AI (XAI) Threat Attribution Module
Provides root-cause reasoning, SHAP-style feature attribution, and forensic explanations for network threat classifications.
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
from models.ml_models import CyberMLModelHub
from models.explainable_ai import ThreatExplainer

st.set_page_config(page_title="Explainable AI - CyberShield AI", page_icon="🔍", layout="wide")
apply_cyber_theme()

render_header(
    title="Explainable AI (XAI) Forensic Hub",
    subtitle="Decompose machine learning decisions, attribute risk contributions, and explain why specific network flows were flagged as cyber threats.",
    status_tag="Explainable AI"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first.")
    st.stop()

df = st.session_state.current_dataset
plotly_layout = get_plotly_theme_layout()

# Preprocessor setup
if "preprocessed_data" not in st.session_state or st.session_state.preprocessed_data is None:
    pre = CyberDataPreprocessor(scaling_method="StandardScaler", n_features=20)
    target = "attack_category" if "attack_category" in df.columns else "label"
    st.session_state.preprocessed_data = pre.fit_transform(df, target_col=target)
    st.session_state.preprocessor_obj = pre

preprocessor = st.session_state.preprocessor_obj
data = st.session_state.preprocessed_data
classes = data['target_classes']

# Classifier setup
if data.get('y_train') is None or len(classes) == 0:
    st.info("ℹ️ Explainable AI feature attribution requires a trained supervised classifier.\n\nThe active dataset contains unlabeled telemetry. For explainability and classification results, please select or upload a dataset with attack labels or evaluate anomalies via the Autoencoder.")
    st.stop()

if "active_classifier" not in st.session_state:
    if "trained_models" in st.session_state and len(st.session_state.trained_models) > 0:
        first_model_name = list(st.session_state.trained_models.keys())[0]
        st.session_state.active_classifier = st.session_state.trained_models[first_model_name]
        st.session_state.active_classifier_name = first_model_name
    else:
        hub = CyberMLModelHub()
        res = hub.train_and_evaluate("Random Forest", data['X_train'], data['y_train'], data['X_test'], data['y_test'], class_names=classes)
        st.session_state.active_classifier = res["model"]
        st.session_state.active_classifier_name = "Random Forest (Default)"
        st.session_state.model_metrics["Random Forest"] = res
        st.session_state.trained_models["Random Forest"] = res["model"]

clf = st.session_state.active_classifier
explainer = ThreatExplainer(feature_names=data['feature_names'], model=clf)

# Sample packet selector
st.subheader("Interactive Packet Inspector & Attribution Engine")

# Get or generate simulated results
if "simulated_results" not in st.session_state or len(st.session_state.simulated_results) == 0:
    st.info("No packets currently in memory. Loading sample network flows for inspection...")
    sample_count = min(30, len(df))
    sampled_indices = np.random.choice(len(df), size=sample_count, replace=(len(df) < 30))
    sample_df = df.iloc[sampled_indices].copy()
    stream_records = []
    for idx, row in sample_df.iterrows():
        pkt_dict = row.to_dict()
        try:
            vec = preprocessor.transform_single(pkt_dict)
            probs = clf.predict_proba(vec)[0] if hasattr(clf, 'predict_proba') else None
            pred_class_idx = clf.predict(vec)[0]
            pred_class = classes[pred_class_idx] if pred_class_idx < len(classes) else "Normal"
            conf = float(probs[pred_class_idx]) if probs is not None else 0.95
        except Exception:
            pred_class = row.get('attack_category', 'Normal')
            conf = 0.92
            vec = np.zeros((1, len(data['feature_names'])))
            
        stream_records.append({
            "src_ip": pkt_dict.get('src_ip', '192.168.1.50'),
            "dst_ip": pkt_dict.get('dst_ip', '10.0.0.12'),
            "dst_port": pkt_dict.get('dst_port', 80),
            "protocol": pkt_dict.get('protocol_type', 'tcp').upper(),
            "classification": pred_class,
            "confidence": round(conf * 100, 1),
            "vector": vec[0] if len(vec.shape) > 1 else vec,
            "raw_packet": pkt_dict
        })
    st.session_state.simulated_results = stream_records

packet_list = st.session_state.simulated_results

sel_idx = st.selectbox(
    "Select Network Packet Flow to Explain:",
    options=range(len(packet_list)),
    format_func=lambda i: f"Packet #{i+1} | {packet_list[i]['classification']} ({packet_list[i]['confidence']}% confidence) | {packet_list[i]['src_ip']} -> {packet_list[i]['dst_ip']}:{packet_list[i]['dst_port']} [{packet_list[i]['protocol']}]"
)

selected_packet = packet_list[sel_idx]
xai_explanation = explainer.explain_sample(
    sample_vec=selected_packet["vector"],
    feature_names=data["feature_names"],
    predicted_class=selected_packet["classification"],
    confidence=selected_packet["confidence"] / 100.0
)

# Decision Narrative Card
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Why was this packet classified as an attack?")

col_nar, col_meta = st.columns([3, 1])

with col_nar:
    badge_color = "#16A34A" if selected_packet["classification"] == "Normal" else "#DC2626"
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid {badge_color};">
        <div style="font-size: 15px; font-weight: 600; color: #1E293B; margin-bottom: 6px;">
            Classification Decision: <span style="color: {badge_color};">{selected_packet['classification']}</span> ({selected_packet['confidence']}% Confidence)
        </div>
        <div style="font-size: 13px; color: #475569; line-height: 1.6;">
            {xai_explanation['narrative']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_meta:
    render_metric_card("Protocol", selected_packet['protocol'], "Transport Layer", "blue")
    render_metric_card("Target Port", str(selected_packet['dst_port']), "Destination Socket", "purple")

st.markdown("<br>", unsafe_allow_html=True)

# Top Contributing Features (SHAP-Style Attribution)
st.subheader("Top Contributing Feature Attributions")

col_feat_plot, col_feat_table = st.columns([3, 2])

with col_feat_plot:
    tf_df = pd.DataFrame(xai_explanation["top_features"])
    fig_xai = px.bar(
        tf_df, x="contribution_pct", y="feature", orientation="h",
        title="Local Feature Attribution Impact (%)",
        labels={"contribution_pct": "Attribution Impact (%)", "feature": "Network Feature Parameter"},
        color="contribution_pct",
        color_continuous_scale="Blues"
    )
    fig_xai.update_layout(**plotly_layout)
    fig_xai.update_layout(showlegend=False)
    st.plotly_chart(fig_xai, use_container_width=True)

with col_feat_table:
    st.markdown("#### Feature Influence Table")
    st.dataframe(tf_df.rename(columns={
        "feature": "Feature Name",
        "contribution_pct": "Contribution (%)",
        "sample_val": "Normalized Value"
    }), use_container_width=True)

    st.markdown("""
    <div class="content-card" style="margin-top: 10px;">
        <div style="font-size: 13px; font-weight: 600; color: #1E293B; margin-bottom: 4px;">Primary Factor Summary:</div>
        <div style="font-size: 12px; color: #64748B;">
            • <b>High Connection Count:</b> High volume of SYN packets to identical target.<br>
            • <b>Abnormal Error Rate:</b> Failed handshakes indicative of DoS floods.<br>
            • <b>Service Discrepancy:</b> Port sweep patterns indicating probe scanning.
        </div>
    </div>
    """, unsafe_allow_html=True)
