"""
CyberShield AI - Dataset Upload & Inspection Module
Allows uploading custom CSV network captures or using pre-loaded NSL-KDD / CICIDS2017 datasets.
Displays rows, columns, missing values, statistics, and column distributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.ui_components import apply_cyber_theme, render_header, render_metric_card
from utils.dataset_loader import load_benchmark_dataset, parse_uploaded_dataset, get_dataset_summary

st.set_page_config(page_title="Dataset Analysis - CyberShield AI", page_icon="📊", layout="wide")
apply_cyber_theme()

render_header(
    title="Dataset Upload & Telemetry Analysis",
    subtitle="Ingest, inspect, and analyze network flow records from NSL-KDD, CICIDS2017, and custom network CSV captures.",
    status_tag="Dataset Ingestion"
)

# Dataset source selector
upload_tab, benchmark_tab = st.tabs(["Upload Custom CSV File", "Load Standard Benchmark Datasets"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV File:",
        type=["csv"],
        help="Upload standard NSL-KDD, CICIDS2017, or custom network log CSV."
    )
    if uploaded_file is not None:
        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('last_uploaded_sig') != file_signature:
            uploaded_file.seek(0)
            df_uploaded, err = parse_uploaded_dataset(uploaded_file)
            if err:
                st.error(f"Error parsing uploaded file: {err}")
                st.session_state.current_dataset = None
            else:
                st.session_state.last_uploaded_sig = file_signature
                st.session_state.current_dataset = df_uploaded
                st.session_state.dataset_name = f"Uploaded: {uploaded_file.name}"
                st.session_state.preprocessed_data = None
                st.session_state.preprocessor_obj = None
                st.session_state.autoencoder_model = None
                st.session_state.autoencoder_eval = None
                st.session_state.trained_models = {}
                st.session_state.model_metrics = {}
                st.session_state.simulated_results = []

        if st.session_state.get('current_dataset') is not None and st.session_state.get('last_uploaded_sig') == file_signature:
            df_curr = st.session_state.current_dataset
            det_col = df_curr.attrs.get('detected_target_col', 'attack')
            all_cols = list(df_curr.columns)
            target_candidates = [c for c in all_cols if c not in ['attack_category', 'label']]
            
            # Default to det_col if present
            def_idx = target_candidates.index(det_col) if det_col in target_candidates else 0
            
            col_target, col_stats = st.columns([2, 3])
            with col_target:
                selected_target = st.selectbox(
                    "🎯 Ground Truth Target Column:",
                    options=target_candidates,
                    index=def_idx,
                    help="Select the column representing ground-truth attack or traffic type."
                )
            
            if selected_target != det_col:
                from utils.dataset_loader import normalize_attack_category
                df_curr['attack_category'] = df_curr[selected_target].apply(normalize_attack_category)
                df_curr['label'] = df_curr['attack_category'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'benign', '0'] else 1)
                df_curr.attrs['detected_target_col'] = selected_target
                st.session_state.current_dataset = df_curr
                st.session_state.preprocessed_data = None
                st.session_state.preprocessor_obj = None
                st.session_state.autoencoder_model = None
                st.session_state.autoencoder_eval = None
                st.session_state.trained_models = {}
                st.session_state.model_metrics = {}
                st.session_state.simulated_results = []
                st.rerun()

            classes_found = list(df_curr['attack_category'].value_counts().items())
            class_str = ", ".join([f"**{k}** ({v:,})" for k, v in classes_found])
            with col_stats:
                st.success(f"Loaded '{uploaded_file.name}' ({len(df_curr):,} rows)")
                st.info(f"Target: **{selected_target}** | Classes: {class_str}")

with benchmark_tab:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load NSL-KDD Primary Benchmark (5,000 Samples)", use_container_width=True):
            st.session_state.current_dataset = load_benchmark_dataset("NSL-KDD")
            st.session_state.dataset_name = "NSL-KDD Benchmark"
            st.session_state.preprocessed_data = None
            st.session_state.preprocessor_obj = None
            st.session_state.autoencoder_model = None
            st.session_state.autoencoder_eval = None
            st.session_state.trained_models = {}
            st.session_state.model_metrics = {}
            st.session_state.simulated_results = []
            st.success("Loaded NSL-KDD Benchmark!")
            st.rerun()
    with col2:
        if st.button("Load CICIDS2017 Cross-Evaluation Benchmark", use_container_width=True):
            st.session_state.current_dataset = load_benchmark_dataset("CICIDS2017")
            st.session_state.dataset_name = "CICIDS2017 Benchmark"
            st.session_state.preprocessed_data = None
            st.session_state.preprocessor_obj = None
            st.session_state.autoencoder_model = None
            st.session_state.autoencoder_eval = None
            st.session_state.trained_models = {}
            st.session_state.model_metrics = {}
            st.session_state.simulated_results = []
            st.success("Loaded CICIDS2017 Benchmark!")
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

df = st.session_state.get("current_dataset", load_benchmark_dataset("NSL-KDD"))
summary = get_dataset_summary(df)

# Dataset Summary KPIs
st.subheader("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Total Records", f"{summary['rows']:,}", "Network Flow Rows", "blue")
with c2:
    render_metric_card("Total Features", f"{summary['columns']}", f"{summary['numeric_count']} Numeric, {summary['categorical_count']} Categorical", "purple")
with c3:
    render_metric_card("Missing Values", f"{summary['missing_values']}", "Null Cells Detected", "green" if summary['missing_values'] == 0 else "orange")
with c4:
    render_metric_card("Duplicate Rows", f"{summary['duplicates']}", "Identical Records", "blue")

st.markdown("<br>", unsafe_allow_html=True)

# Sample Records Table
st.subheader("Sample Records Table")
st.caption(f"Currently inspecting: **{st.session_state.get('dataset_name', 'Active Dataset')}**")

col_prev1, col_prev2 = st.columns([1, 4])
with col_prev1:
    preview_rows = st.selectbox("Rows to display:", [10, 25, 50, 100], index=0)
with col_prev2:
    search_col = st.multiselect("Select columns to view:", options=list(df.columns), default=list(df.columns)[:10])

st.dataframe(df[search_col].head(preview_rows), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dataset Statistics & Schema
st.subheader("Feature Information & Data Statistics")
tab_stats, tab_missing, tab_types = st.tabs(["Descriptive Statistics", "Missing Values Analysis", "Feature Schema"])

with tab_stats:
    st.dataframe(df.describe().T, use_container_width=True)

with tab_missing:
    missing_df = pd.DataFrame({
        'Feature': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing Percentage (%)': (df.isnull().sum().values / len(df)) * 100
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if missing_df.empty:
        st.success("Clean dataset: 0 missing or null values detected across all features.")
    else:
        st.dataframe(missing_df, use_container_width=True)

with tab_types:
    types_df = pd.DataFrame({
        'Feature Name': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Sample Value': [str(df[col].iloc[0]) for col in df.columns]
    })
    st.dataframe(types_df, use_container_width=True)
