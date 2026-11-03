"""
CyberShield AI - Main Hub & Dashboard Application
AI-Based Cyber Threat Detection System using Machine Learning & Deep Learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card,
    get_plotly_theme_layout, get_current_theme
)
from utils.dataset_loader import load_benchmark_dataset, get_dataset_summary
from data.dataset_generator import ensure_datasets

# Page Configuration
st.set_page_config(
    page_title="CyberShield AI - Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Clean Academic Theme
apply_cyber_theme()

# Ensure benchmark data is generated
ensure_datasets()

# Initialize session state for cross-page persistence
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = load_benchmark_dataset("NSL-KDD")
    st.session_state.dataset_name = "NSL-KDD Benchmark"

if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}

if "live_alerts" not in st.session_state:
    st.session_state.live_alerts = []

if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = False

# Top Header
render_header(
    title="CyberShield AI",
    subtitle="Machine Learning Based Cyber Threat Detection System",
    status_tag="System Operational"
)

# Presentation Mode Switcher
col_pres_left, col_pres_right = st.columns([3, 1])
with col_pres_right:
    is_pres = st.toggle("🎯 Presentation Mode", value=st.session_state.presentation_mode)
    if is_pres != st.session_state.presentation_mode:
        st.session_state.presentation_mode = is_pres
        st.rerun()

# Project Overview Card
st.markdown("""
<div class="content-card" style="margin-bottom: 20px;">
    <div style="font-size: 15px; font-weight: 600; color: #1E293B; margin-bottom: 6px;">Project Overview</div>
    <div style="font-size: 14px; color: #475569; line-height: 1.6;">
        This system analyzes network traffic data using machine learning and deep learning techniques to detect cyber threats, 
        classify attacks, identify anomalies, and generate security insights.
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
df_current = st.session_state.current_dataset
summary = get_dataset_summary(df_current)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Total Records", f"{summary['rows']:,}", "Network Samples Analysed", "blue")
with col2:
    normal_pct = 0
    if 'Normal' in summary['class_distribution']:
        normal_pct = (summary['class_distribution']['Normal'] / summary['rows']) * 100
    render_metric_card("Normal Traffic", f"{normal_pct:.1f}%", "Healthy Network Activity", "green")
with col3:
    mal_count = summary['rows'] - summary['class_distribution'].get('Normal', 0)
    render_metric_card("Threats Detected", f"{mal_count:,}", "Potential Attacks Found", "red")
with col4:
    feat_count = summary['columns']
    render_metric_card("Feature Count", f"{feat_count}", "Network Parameters", "purple")

st.markdown("<br>", unsafe_allow_html=True)

# Academic Project Cards
st.subheader("Project Foundation & Scope")

c_p1, c_p2 = st.columns(2)

with c_p1:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #2563EB; margin-bottom: 6px;">Problem Statement</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.5;">
            Traditional signature-based Intrusion Detection Systems (IDS) struggle to identify zero-day exploits, polymorphic 
            threats, and stealth low-and-slow reconnaissance scans. This project develops an intelligent multi-tier AI system 
            capable of detecting known and novel cyber intrusions with high accuracy and low false-alarm rates.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #16A34A; margin-bottom: 6px;">Project Objectives</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.5;">
            1. Preprocess and extract optimal features from standardized benchmark telemetry (NSL-KDD & CICIDS2017).<br>
            2. Benchmark supervised ML classifiers (Random Forest, SVM, Decision Tree, Logistic Regression, XGBoost).<br>
            3. Implement Deep Neural Networks (MLP) and Unsupervised Autoencoders for anomaly detection.<br>
            4. Ingest multi-page cyber incident & threat intelligence PDF reports for automated IoC mining.
            5. Provide Explainable AI (XAI) feature attribution and automated incident response mitigation playbooks.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_p2:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #7C3AED; margin-bottom: 6px;">Technologies & Frameworks Used</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.5;">
            • <b>Core Platform:</b> Python 3.10+, Streamlit<br>
            • <b>Machine Learning:</b> Scikit-learn, XGBoost, NumPy, Pandas<br>
            • <b>Deep Learning & Anomaly:</b> Multi-Layer Perceptron (MLP), Autoencoder Reconstruction Models<br>
            • <b>PDF Intelligence & Reports:</b> PyMuPDF, pdfplumber, Plotly Express, Seaborn, FPDF2
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #D97706; margin-bottom: 6px;">Dataset & Key Features</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.5;">
            • <b>Primary Dataset:</b> NSL-KDD (41 features + label + attack category taxonomy).<br>
            • <b>Cross-Evaluation:</b> CICIDS2017 flow telemetry dataset support.<br>
            • <b>Key Capabilities:</b> 5-Class attack classification, 0-100 Threat Severity Engine, Multi-Page Threat Intelligence PDF Processing & IoC Forensics, and real-time stream simulation.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Workflow Diagram Card
st.subheader("System Architecture & Processing Workflow")
st.markdown("""
<div class="content-card" style="text-align: center; padding: 22px;">
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px;">
        <span class="badge badge-low" style="padding: 8px 14px; font-size: 13px;">Data & PDF Ingestion</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-low" style="padding: 8px 14px; font-size: 13px;">Data Preprocessing</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-low" style="padding: 8px 14px; font-size: 13px;">Feature Engineering</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-normal" style="padding: 8px 14px; font-size: 13px;">Model Training</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-high" style="padding: 8px 14px; font-size: 13px;">Threat Detection</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-medium" style="padding: 8px 14px; font-size: 13px;">Dashboard Visualization</span>
        <span style="color: #94A3B8; font-weight: bold;">➔</span>
        <span class="badge badge-normal" style="padding: 8px 14px; font-size: 13px;">Report Generation</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Overview Interactive Charts
st.subheader("Dataset Distribution Overview")

col_ch1, col_ch2 = st.columns([3, 2])
plotly_layout = get_plotly_theme_layout()

with col_ch1:
    if 'attack_category' in df_current.columns:
        cat_counts = df_current['attack_category'].value_counts().reset_index()
        cat_counts.columns = ['Attack Category', 'Count']
        
        color_map = {
            'Normal': '#16A34A',
            'DoS': '#DC2626',
            'Probe': '#F59E0B',
            'R2L': '#7C3AED',
            'U2R': '#991B1B'
        }
        
        fig = px.bar(
            cat_counts, x='Attack Category', y='Count',
            color='Attack Category',
            color_discrete_map=color_map,
            title="Distribution of Network Traffic Categories"
        )
        fig.update_layout(**plotly_layout)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col_ch2:
    if 'protocol_type' in df_current.columns:
        proto_counts = df_current['protocol_type'].value_counts().reset_index()
        proto_counts.columns = ['Protocol', 'Count']
        
        fig_donut = px.pie(
            proto_counts, values='Count', names='Protocol',
            hole=0.55,
            title="Protocol Distribution (TCP / UDP / ICMP)",
            color_discrete_sequence=['#2563EB', '#7C3AED', '#F59E0B']
        )
        fig_donut.update_layout(**plotly_layout)
        st.plotly_chart(fig_donut, use_container_width=True)

# Presentation Mode Summary Block (if active)
if st.session_state.presentation_mode:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Presentation Summary")
    
    perf_summary = "Ensemble models achieve high F1-Score with low inference latency"
    if st.session_state.get("model_metrics"):
        f1_scores = [m.get("f1", 0) * 100 for m in st.session_state.model_metrics.values() if isinstance(m, dict) and "f1" in m]
        if f1_scores:
            perf_summary = f"Trained architectures achieve a peak F1-Score of <b>{max(f1_scores):.2f}%</b>"
            
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #2563EB;">
        <div style="font-size: 16px; font-weight: 600; color: #1E293B; margin-bottom: 6px;">Evaluation Highlights for Faculty & Reviewers</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.6;">
            • <b>Machine Learning Performance:</b> {perf_summary} on current network telemetry.<br>
            • <b>Zero-Day Resilience:</b> The Unsupervised Autoencoder flags unknown attack profiles through reconstruction error deviation.<br>
            • <b>Explainability:</b> Decision trees and feature deviation algorithms explain <i>why</i> packets are flagged in real-time.<br>
            • <b>Operational Actionability:</b> Dynamic 0-100 severity scores trigger immediate mitigation playbooks (e.g. firewall drop rules).
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ CyberShield AI")
    st.caption("AI-Based Intrusion Detection System")
    
    st.markdown("---")
    st.markdown("**Active Dataset:**")
    st.info(f"**{st.session_state.dataset_name}**\n\nTotal Samples: `{len(df_current):,}`")
    
    st.markdown("---")
    st.markdown("### 🔄 Switch Dataset")
    selected_ds = st.selectbox(
        "Select Baseline:",
        ["NSL-KDD Benchmark", "CICIDS2017 Sample"],
        index=0 if "NSL-KDD" in st.session_state.dataset_name else 1
    )
    if st.button("Load Selected Dataset", use_container_width=True):
        if "NSL-KDD" in selected_ds:
            st.session_state.current_dataset = load_benchmark_dataset("NSL-KDD")
            st.session_state.dataset_name = "NSL-KDD Benchmark"
        else:
            st.session_state.current_dataset = load_benchmark_dataset("CICIDS2017")
            st.session_state.dataset_name = "CICIDS2017 Sample"
        st.session_state.preprocessed_data = None
        st.success(f"Loaded {selected_ds}")
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 12px;">
        <div style="font-size: 11px; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Console Theme</div>
        <div style="font-size: 13px; font-weight: 600; color: #F8FAFC; margin-top: 2px;">🛡️ Cyber SOC Dark Navy</div>
        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">High-Contrast Presentation Mode Active</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Final Year AIML Mini-Project | IT Department")
