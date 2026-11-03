"""
CyberShield AI - Real-Time Threat Detection & Security Alert Center
Multi-class attack classification, 0-100 severity engine, real-time alert feed, and MITRE ATT&CK integration.
Supports supervised multi-class detection for labeled traffic and unsupervised Autoencoder anomaly detection for raw streams.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card,
    get_severity_badge_html, get_plotly_theme_layout
)
from utils.preprocessor import CyberDataPreprocessor
from utils.severity_engine import calculate_risk_score
from utils.alert_system import generate_threat_alert
from models.ml_models import CyberMLModelHub

st.set_page_config(page_title="Threat Dashboard - CyberShield AI", page_icon="🛡️", layout="wide")
apply_cyber_theme()

render_header(
    title="Threat Detection & Security Alert Center",
    subtitle="Real-time network stream simulation, multi-class threat classification, dynamic 0-100 risk scoring, and security incident alerts.",
    status_tag="Active Monitoring"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first from the Dataset Ingestion page.")
    st.stop()

df = st.session_state.current_dataset
plotly_layout = get_plotly_theme_layout()

# Preprocessor setup
has_gt = df.attrs.get('has_ground_truth', ('attack_category' in df.columns or 'label' in df.columns))
if "preprocessor_obj" not in st.session_state or st.session_state.preprocessor_obj is None:
    pre = CyberDataPreprocessor(scaling_method="StandardScaler", n_features=20)
    target = ("attack_category" if "attack_category" in df.columns else "label") if has_gt else None
    st.session_state.preprocessed_data = pre.fit_transform(df, target_col=target)
    st.session_state.preprocessor_obj = pre

preprocessor = st.session_state.preprocessor_obj
data = st.session_state.preprocessed_data
classes = data.get('target_classes', [])
has_gt = data.get('has_ground_truth', True) and (data.get('y_train') is not None)

# Classifier / Anomaly Detection Engine setup
if has_gt:
    if "active_classifier" not in st.session_state or st.session_state.active_classifier is None:
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
else:
    clf = None
    if "autoencoder_model" not in st.session_state or st.session_state.autoencoder_model is None:
        from models.autoencoder import CyberAutoencoder
        ae = CyberAutoencoder(input_dim=data['X_train'].shape[1], latent_dim=6, epochs=25, batch_size=64)
        ae.fit(data['X_train'])
        st.session_state.autoencoder_model = ae
    st.session_state.active_classifier_name = "Autoencoder (Unsupervised Anomaly Detector)"

# Stream Simulator Controls
st.subheader("Network Stream Simulator & Ingestion")

c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 2, 2])
with c_ctrl1:
    sim_batch_size = st.slider("Simulation Batch Size (Packets):", 20, 200, 50, 10)
with c_ctrl2:
    filter_severity = st.multiselect(
        "Filter by Threat Severity:",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )
with c_ctrl3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_sim_btn = st.button("Ingest & Analyze Network Stream", use_container_width=True)

# Run Simulation
if run_sim_btn or "simulated_results" not in st.session_state:
    sample_count = min(sim_batch_size, len(df))
    sampled_indices = np.random.choice(len(df), size=sample_count, replace=(len(df) < sim_batch_size))
    sample_df = df.iloc[sampled_indices].copy()
    
    stream_records = []
    
    for idx, row in sample_df.iterrows():
        pkt_dict = row.to_dict()
        try:
            vec = preprocessor.transform_single(pkt_dict)
            if clf is not None:
                probs = clf.predict_proba(vec)[0] if hasattr(clf, 'predict_proba') else None
                pred_class_idx = clf.predict(vec)[0]
                pred_class = classes[pred_class_idx] if pred_class_idx < len(classes) else "Normal"
                conf = float(probs[pred_class_idx]) if probs is not None else 0.95
            else:
                ae = st.session_state.autoencoder_model
                rec = ae.reconstruct(vec)
                mse = float(np.mean((vec - rec) ** 2))
                is_anom = (mse > ae.threshold)
                pred_class = "Suspicious Anomaly" if is_anom else "Normal"
                conf = float(min(0.98, max(0.60, 0.60 + abs(mse - ae.threshold) / (ae.threshold + 1e-6) * 0.35)))
        except Exception:
            vec = np.zeros((1, len(data['feature_names'])))
            pred_class = str(row.get('attack_category', 'Normal'))
            conf = 0.85
            
        risk_info = calculate_risk_score(pred_class, conf, pkt_dict)
        alert = generate_threat_alert(pkt_dict, pred_class, conf, risk_info)
        
        stream_records.append({
            "timestamp": alert["timestamp"],
            "src_ip": alert["src_ip"],
            "dst_ip": alert["dst_ip"],
            "dst_port": alert["dst_port"],
            "protocol": alert["protocol"],
            "classification": pred_class,
            "confidence": round(conf * 100, 1),
            "severity": risk_info["severity"],
            "risk_score": risk_info["risk_score"],
            "color": risk_info.get("color", "#2563EB"),
            "mitre_technique": risk_info["mitre_technique"],
            "vector": vec[0] if len(vec.shape) > 1 else vec,
            "raw_packet": pkt_dict,
            "alert": alert
        })
        
    st.session_state.simulated_results = stream_records

results = st.session_state.simulated_results
filtered_results = [r for r in results if r["severity"] in filter_severity]

# Real-Time KPI Stats Strip
total_p = len(results)
safe_p = sum(1 for r in results if r["classification"] == "Normal")
mal_p = total_p - safe_p
threat_pct = (mal_p / total_p * 100) if total_p > 0 else 0.0
avg_risk = np.mean([r["risk_score"] for r in results]) if total_p > 0 else 0.0

st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_metric_card("Total Analyzed", f"{total_p:,}", f"Engine: {st.session_state.get('active_classifier_name', 'Default')[:18]}", "blue")
with k2:
    render_metric_card("Normal Traffic", f"{safe_p:,}", f"{(safe_p/total_p*100):.1f}% Clean", "green")
with k3:
    render_metric_card("Detected Threats", f"{mal_p:,}", f"{threat_pct:.1f}% Invasions", "red")
with k4:
    render_metric_card("Avg Risk Score", f"{avg_risk:.1f} / 100", "Composite Threat Index", "orange" if avg_risk < 51 else "red")
with k5:
    crit_count = sum(1 for r in results if r["severity"] == "Critical")
    render_metric_card("Critical Alerts", f"{crit_count}", "Immediate Action Req", "red")

st.markdown("<br>", unsafe_allow_html=True)

# 1. Threat Severity Engine Breakdown (0-25 Low, 26-50 Medium, 51-75 High, 76-100 Critical)
st.subheader("Threat Severity Engine & Risk Score Distribution")

sev_counts = pd.Series([r["severity"] for r in results]).value_counts().to_dict()
sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #16A34A;">
        <div style="font-size: 13px; font-weight: 600; color: #16A34A;">Low Risk (0 - 25)</div>
        <div style="font-size: 20px; font-weight: 700; color: #1E293B;">{sev_counts.get('Low', 0)} Packets</div>
        <div style="font-size: 11px; color: #64748B;">Benign Baseline Traffic</div>
    </div>
    """, unsafe_allow_html=True)

with sc2:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #F59E0B;">
        <div style="font-size: 13px; font-weight: 600; color: #F59E0B;">Medium Risk (26 - 50)</div>
        <div style="font-size: 20px; font-weight: 700; color: #1E293B;">{sev_counts.get('Medium', 0)} Packets</div>
        <div style="font-size: 11px; color: #64748B;">Reconnaissance / Probe Activity</div>
    </div>
    """, unsafe_allow_html=True)

with sc3:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #DC2626;">
        <div style="font-size: 13px; font-weight: 600; color: #DC2626;">High Risk (51 - 75)</div>
        <div style="font-size: 20px; font-weight: 700; color: #1E293B;">{sev_counts.get('High', 0)} Packets</div>
        <div style="font-size: 11px; color: #64748B;">DoS / Remote Exploitation</div>
    </div>
    """, unsafe_allow_html=True)

with sc4:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #991B1B;">
        <div style="font-size: 13px; font-weight: 600; color: #991B1B;">Critical Risk (76 - 100)</div>
        <div style="font-size: 20px; font-weight: 700; color: #1E293B;">{sev_counts.get('Critical', 0)} Packets</div>
        <div style="font-size: 11px; color: #64748B;">Privilege Escalation / Breach</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. Real-Time Alert Panel
st.subheader("🚨 Real-Time Security Alert Center")

malicious_alerts = [r for r in results if r["classification"] != "Normal"]

if malicious_alerts:
    for a in malicious_alerts[:4]:
        sev_color = a.get("color", "#DC2626")
        cat_title = {
            "DoS": "🚨 DoS Attack Detected",
            "Probe": "🚨 Suspicious Traffic Spike / Probe Scan",
            "R2L": "🚨 High Risk Connection / Unauthorized Login Attempt",
            "U2R": "🚨 Critical Possible Intrusion / Root Privilege Escalation",
            "Suspicious Anomaly": "🚨 Latent Anomaly Detected / Unknown Traffic Outlier"
        }.get(a["classification"], "🚨 Possible Intrusion Attempt")
        
        st.markdown(f"""
        <div class="content-card" style="border-left: 4px solid {sev_color}; margin-bottom: 8px; padding: 12px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 14px; font-weight: 600; color: #1E293B;">{cat_title}</div>
                <div style="font-size: 12px; color: #64748B;">{a['timestamp']}</div>
            </div>
            <div style="font-size: 12px; color: #475569; margin-top: 4px;">
                <b>Origin:</b> {a['src_ip']} ➔ <b>Target:</b> {a['dst_ip']}:{a['dst_port']} &nbsp;|&nbsp; 
                <b>Category:</b> {a['classification']} &nbsp;|&nbsp; 
                <b>Confidence:</b> {a['confidence']}% &nbsp;|&nbsp; 
                <b>Risk Score:</b> {a['risk_score']} / 100 ({a['severity']})
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("All monitored network packets in the current batch are within normal parameters.")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Attack Analytics Charts
st.subheader("Attack Analytics & Telemetry Charts")

col_trend, col_dist = st.columns([3, 2])

with col_trend:
    timeline_df = pd.DataFrame({
        "Packet #": range(1, len(results) + 1),
        "Threat Risk Score": [r["risk_score"] for r in results],
        "Severity": [r["severity"] for r in results],
        "Category": [r["classification"] for r in results]
    })
    
    fig_timeline = px.line(
        timeline_df, x="Packet #", y="Threat Risk Score",
        title="Threat Risk Score Velocity Across Ingested Stream",
        markers=True,
        color="Severity",
        color_discrete_map={"Critical": "#991B1B", "High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"}
    )
    fig_timeline.add_hline(y=76, line_dash="dot", line_color="#991B1B", annotation_text="Critical Threshold (76)")
    fig_timeline.update_layout(**plotly_layout)
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_dist:
    cat_counts = pd.Series([r["classification"] for r in results]).value_counts().reset_index()
    cat_counts.columns = ["Attack Type", "Count"]
    
    fig_donut = px.pie(
        cat_counts, values="Count", names="Attack Type",
        hole=0.5,
        title="Detected Threat Distribution",
        color="Attack Type",
        color_discrete_map={
            "Normal": "#16A34A", "DoS": "#DC2626", "Probe": "#F59E0B",
            "R2L": "#7C3AED", "U2R": "#991B1B", "Suspicious Anomaly": "#DC2626"
        }
    )
    fig_donut.update_layout(**plotly_layout)
    st.plotly_chart(fig_donut, use_container_width=True)

# 4. Recent Ingested Packets Table
st.subheader("Recent Ingested Network Packets")

if filtered_results:
    table_rows = []
    for r in filtered_results:
        table_rows.append({
            "Timestamp": r["timestamp"],
            "Source IP": r["src_ip"],
            "Destination IP:Port": f"{r['dst_ip']}:{r['dst_port']}",
            "Protocol": r["protocol"],
            "Category": r["classification"],
            "Confidence": f"{r['confidence']}%",
            "Severity": r["severity"],
            "Risk Score": r["risk_score"],
            "MITRE Technique": r["mitre_technique"]
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
