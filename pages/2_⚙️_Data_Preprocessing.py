"""
CyberShield AI - Data Preprocessing & Feature Engineering Module
Performs missing value imputation, label encoding, scaling, train-test splitting, and feature selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card,
    get_plotly_theme_layout
)
from utils.preprocessor import CyberDataPreprocessor

st.set_page_config(page_title="Data Preprocessing - CyberShield AI", page_icon="⚙️", layout="wide")
apply_cyber_theme()

render_header(
    title="Data Preprocessing & Feature Engineering",
    subtitle="Encode categorical attributes, normalize numeric scales, split training/testing subsets, and select optimal features.",
    status_tag="Preprocessing Pipeline"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first from the Home or Dataset Analysis page.")
    st.stop()

df = st.session_state.current_dataset

# Preprocessing Pipeline Settings
st.subheader("Pipeline Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    scaling_choice = st.selectbox(
        "Feature Scaling Algorithm:",
        ["StandardScaler (Z-Score)", "MinMaxScaler (0 to 1)", "RobustScaler (Median-IQR)"],
        index=0
    )
    scale_method = scaling_choice.split(" ")[0]

with col2:
    target_choice = st.selectbox(
        "Target Classification Variable:",
        ["attack_category (5-Class: Normal, DoS, Probe, R2L, U2R)", "label (Binary: 0-Normal, 1-Attack)"] if 'attack_category' in df.columns else ["label"],
        index=0
    )
    target_col = "attack_category" if "attack_category" in target_choice else "label"

with col3:
    enable_feat_selection = st.checkbox("Enable Feature Selection (SelectKBest)", value=True)
    n_features = st.slider("Number of Top Features (k):", min_value=5, max_value=min(40, len(df.columns) - 1), value=20, step=1) if enable_feat_selection else None

# Action Button
if st.button("Execute Preprocessing Pipeline", use_container_width=True):
    with st.spinner("Processing network packets, encoding categorical variables, and selecting features..."):
        preprocessor = CyberDataPreprocessor(scaling_method=scale_method, n_features=n_features)
        
        try:
            processed = preprocessor.fit_transform(df, target_col=target_col, select_features=enable_feat_selection)
            st.session_state.preprocessed_data = processed
            st.session_state.preprocessor_obj = preprocessor
            st.success("Preprocessing pipeline executed successfully. Training and testing datasets are ready.")
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

# Display Results if Available
if "preprocessed_data" in st.session_state and st.session_state.preprocessed_data is not None:
    data = st.session_state.preprocessed_data
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Preprocessing Output Summary")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Training Samples", f"{data['X_train'].shape[0]:,}", "80% Stratified Split", "blue")
    with c2:
        render_metric_card("Testing Samples", f"{data['X_test'].shape[0]:,}", "20% Stratified Split", "purple")
    with c3:
        render_metric_card("Selected Features", f"{data['X_train'].shape[1]}", "Normalized Columns", "green")
    with c4:
        render_metric_card("Target Classes", f"{len(data['target_classes'])}", f"{', '.join(data['target_classes'][:3])}...", "orange")

    # Selected Features Table & Distribution
    t1, t2 = st.tabs(["Selected Features List", "Training Class Distribution"])
    
    with t1:
        feat_df = pd.DataFrame({
            "Index": range(1, len(data['feature_names']) + 1),
            "Feature Name": data['feature_names'],
            "Status": ["Retained for Training"] * len(data['feature_names'])
        })
        st.dataframe(feat_df, use_container_width=True)

    with t2:
        unique, counts = np.unique(data['y_train'], return_counts=True)
        class_names = [data['target_classes'][u] for u in unique]
        split_df = pd.DataFrame({'Class': class_names, 'Count': counts})
        
        fig = px.pie(
            split_df, values='Count', names='Class',
            title="Class Distribution in Training Split",
            color_discrete_sequence=['#16A34A', '#DC2626', '#F59E0B', '#7C3AED', '#991B1B'],
            hole=0.45
        )
        fig.update_layout(**get_plotly_theme_layout())
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click 'Execute Preprocessing Pipeline' above to prepare training datasets.")
