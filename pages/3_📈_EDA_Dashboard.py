"""
CyberShield AI - Exploratory Data Analysis (EDA) Dashboard
Interactive cybersecurity visualizations for protocol distributions, attack types, correlation heatmaps, and feature importance.
Adaptive across NSL-KDD, CICIDS2017, and custom network telemetry datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

from utils.ui_components import (
    apply_cyber_theme, render_header, get_plotly_theme_layout
)

st.set_page_config(page_title="EDA Dashboard - CyberShield AI", page_icon="📈", layout="wide")
apply_cyber_theme()

render_header(
    title="Exploratory Data Analysis",
    subtitle="Statistical analysis of network flow features, attack categories, protocol distributions, and correlation matrices.",
    status_tag="Exploratory Analysis"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first from the Home or Dataset Analysis page.")
    st.stop()

df = st.session_state.current_dataset
plotly_layout = get_plotly_theme_layout()

# Top Visual Overview Tabs
tab_threats, tab_protocols, tab_correlations, tab_importance, tab_flows = st.tabs([
    "Attack Distribution",
    "Protocol & Services",
    "Correlation Matrix",
    "Feature Importance",
    "Traffic Flow Metrics"
])

# 1. Attack Category Breakdown
with tab_threats:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if 'attack_category' in df.columns:
            counts = df['attack_category'].value_counts().reset_index()
            counts.columns = ['Category', 'Packets']
            
            fig_bar = px.bar(
                counts, x='Category', y='Packets', color='Category',
                text='Packets',
                title="Traffic Volume by Attack Category",
                color_discrete_map={'Normal': '#16A34A', 'DoS': '#DC2626', 'Probe': '#F59E0B', 'R2L': '#7C3AED', 'U2R': '#991B1B'}
            )
            fig_bar.update_layout(**plotly_layout)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ℹ️ Unlabeled Network Telemetry: No ground-truth attack category labels present in this dataset. Threat distribution can be analyzed via the Anomaly Detection module.")
            
    with col2:
        if 'attack_type' in df.columns:
            sub_counts = df[df['attack_type'] != 'normal']['attack_type'].value_counts().head(8).reset_index()
            sub_counts.columns = ['Attack Sub-Type', 'Count']
            
            fig_sub = px.pie(
                sub_counts, values='Count', names='Attack Sub-Type',
                title="Top Attack Sub-Types",
                color_discrete_sequence=px.colors.qualitative.Prism,
                hole=0.4
            )
            fig_sub.update_layout(**plotly_layout)
            st.plotly_chart(fig_sub, use_container_width=True)
        else:
            cat_target = 'attack_category' if 'attack_category' in df.columns else 'label' if 'label' in df.columns else None
            if cat_target:
                p_counts = df[cat_target].value_counts().reset_index()
                p_counts.columns = ['Class', 'Count']
                fig_pie_gen = px.pie(
                    p_counts, values='Count', names='Class',
                    title="Target Class Proportion",
                    hole=0.4
                )
                fig_pie_gen.update_layout(**plotly_layout)
                st.plotly_chart(fig_pie_gen, use_container_width=True)
            else:
                st.caption("Detailed attack sub-types not available in this telemetry schema.")

# 2. Protocol & Service Telemetry
with tab_protocols:
    c1, c2 = st.columns(2)
    
    with c1:
        proto_col = next((c for c in ['protocol_type', 'protocol', 'proto'] if c in df.columns), None)
        if proto_col:
            if 'attack_category' in df.columns:
                ct = pd.crosstab(df[proto_col], df['attack_category']).reset_index()
                ct_melted = pd.melt(ct, id_vars=[proto_col], value_name='Count')
                fig_proto = px.bar(
                    ct_melted, x=proto_col, y='Count', color='attack_category',
                    barmode='group',
                    title=f"Protocols ({proto_col}) vs Attack Categories",
                    labels={proto_col: 'Network Protocol', 'Count': 'Number of Packets'},
                    color_discrete_map={'Normal': '#16A34A', 'DoS': '#DC2626', 'Probe': '#F59E0B', 'R2L': '#7C3AED', 'U2R': '#991B1B'}
                )
            else:
                p_df = df[proto_col].value_counts().reset_index()
                p_df.columns = [proto_col, 'Count']
                fig_proto = px.bar(p_df, x=proto_col, y='Count', color=proto_col, title=f"Protocol Distribution ({proto_col})")
            fig_proto.update_layout(**plotly_layout)
            st.plotly_chart(fig_proto, use_container_width=True)
        else:
            st.info("No specific protocol type column ('protocol_type' / 'proto') detected in active schema.")
            
    with c2:
        srv_col = next((c for c in ['service', 'dst_port', 'destination_port', 'port'] if c in df.columns), None)
        if srv_col:
            top_services = df[srv_col].value_counts().head(8).index
            df_top_srv = df[df[srv_col].isin(top_services)]
            
            fig_srv = px.histogram(
                df_top_srv, x=srv_col, color='attack_category' if 'attack_category' in df.columns else None,
                title=f"Top Targeted Services / Ports ({srv_col})",
                labels={srv_col: 'Service / Port'},
                color_discrete_map={'Normal': '#16A34A', 'DoS': '#DC2626', 'Probe': '#F59E0B', 'R2L': '#7C3AED', 'U2R': '#991B1B'}
            )
            fig_srv.update_layout(**plotly_layout)
            st.plotly_chart(fig_srv, use_container_width=True)
        else:
            st.info("No specific network service or destination port column detected in active schema.")

# 3. Correlation Heatmap
with tab_correlations:
    numeric_df = df.select_dtypes(include=[np.number])
    selected_num_cols = st.multiselect(
        "Select numerical features for correlation matrix:",
        options=list(numeric_df.columns),
        default=list(numeric_df.columns)[:min(10, len(numeric_df.columns))]
    )
    
    if len(selected_num_cols) > 1:
        corr = numeric_df[selected_num_cols].corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Heatmap"
        )
        fig_heat.update_layout(**plotly_layout)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Select at least 2 features to view the correlation heatmap.")

# 4. Feature Importance Ranking
with tab_importance:
    st.markdown("#### Random Forest Feature Importance Analysis")
    st.caption("Measures Gini impurity reduction across ensemble trees to identify primary predictive features.")
    
    if st.button("Calculate Feature Importance Ranking"):
        with st.spinner("Fitting Random Forest feature importance estimator..."):
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            if 'label' in numeric_cols:
                y = df['label']
                X = df[numeric_cols].drop(columns=['label', 'dst_port'], errors='ignore')
            elif 'attack_category' in df.columns:
                y = df['attack_category'].astype('category').cat.codes
                X = df[numeric_cols].drop(columns=['dst_port'], errors='ignore')
            else:
                y = None
                X = df[numeric_cols]
                
            if y is None or len(np.unique(y)) < 2:
                st.warning("Supervised Feature Importance requires at least 2 target classes. For unlabeled telemetry, use variance metrics.")
            else:
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X.fillna(0), y)
                
                imp_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance Score': rf.feature_importances_
                }).sort_values(by='Importance Score', ascending=True).tail(12)
                
                fig_imp = px.bar(
                    imp_df, x='Importance Score', y='Feature', orientation='h',
                    title="Top 12 Features by Gini Importance",
                    color='Importance Score',
                    color_continuous_scale="Blues"
                )
                fig_imp.update_layout(**plotly_layout)
                fig_imp.update_layout(showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)

# 5. Traffic Flow Metrics
with tab_flows:
    c1, c2 = st.columns(2)
    sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 0 else df
    
    # Adaptive Flow Scatter 1
    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        x1, y1, title1, log_s1 = 'src_bytes', 'dst_bytes', "Source Bytes vs Destination Bytes (Log Scale)", True
    elif 'total_fwd_packets' in df.columns and 'total_backward_packets' in df.columns:
        x1, y1, title1, log_s1 = 'total_fwd_packets', 'total_backward_packets', "Total Fwd Packets vs Backward Packets", False
    else:
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        x1, y1, title1, log_s1 = (num_cols[0], num_cols[1], f"{num_cols[0]} vs {num_cols[1]}", False) if len(num_cols) >= 2 else (None, None, "", False)

    with c1:
        if x1 and y1:
            fig_scatter = px.scatter(
                sample_df, x=x1, y=y1,
                color='attack_category' if 'attack_category' in df.columns else None,
                log_x=log_s1, log_y=log_s1,
                title=title1,
                color_discrete_map={'Normal': '#16A34A', 'DoS': '#DC2626', 'Probe': '#F59E0B', 'R2L': '#7C3AED', 'U2R': '#991B1B'}
            )
            fig_scatter.update_layout(**plotly_layout)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Insufficient numeric features to render flow scatter plot.")
            
    # Adaptive Flow Scatter 2
    if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
        x2, y2, title2 = 'serror_rate', 'rerror_rate', "SYN Error Rate vs REJ Error Rate"
    elif 'flow_duration' in df.columns and 'flow_bytes_s' in df.columns:
        x2, y2, title2 = 'flow_duration', 'flow_bytes_s', "Flow Duration vs Flow Bytes/s"
    else:
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        x2, y2, title2 = (num_cols[-2], num_cols[-1], f"{num_cols[-2]} vs {num_cols[-1]}") if len(num_cols) >= 4 else (None, None, "")

    with c2:
        if x2 and y2:
            fig_err = px.scatter(
                sample_df, x=x2, y=y2,
                color='attack_category' if 'attack_category' in df.columns else None,
                title=title2,
                color_discrete_map={'Normal': '#16A34A', 'DoS': '#DC2626', 'Probe': '#F59E0B', 'R2L': '#7C3AED', 'U2R': '#991B1B'}
            )
            fig_err.update_layout(**plotly_layout)
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.caption("Secondary flow correlation attributes not detected.")
