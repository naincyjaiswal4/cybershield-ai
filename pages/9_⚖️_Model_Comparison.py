"""
CyberShield AI - Model Comparison & Benchmarking Hub
Compares Random Forest, SVM, Decision Tree, Logistic Regression, XGBoost, and Deep Neural Networks.
Automatically highlights the best performing model with multi-dimensional radar maps and leaderboard rankings.
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
from models.deep_learning import CyberDeepNeuralNetwork

st.set_page_config(page_title="Model Comparison - CyberShield AI", page_icon="⚖️", layout="wide")
apply_cyber_theme()

render_header(
    title="Model Benchmarking & Comparison",
    subtitle="Comparative evaluation of accuracy, precision, recall, F1-scores, AUC scores, and execution times across all machine learning and deep learning models.",
    status_tag="Model Leaderboard"
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

# Train All Models for Full Benchmark Button
if st.button("Train & Benchmark All 6 Model Architectures", use_container_width=True) or len(st.session_state.get("model_metrics", {})) < 2:
    hub = CyberMLModelHub()
    with st.spinner("Benchmarking Random Forest, SVM, Decision Tree, Logistic Regression, XGBoost, and Deep Neural Network..."):
        for m_name in [
            "Random Forest", "Support Vector Machine (SVM)", "Decision Tree",
            "Logistic Regression", "XGBoost / Gradient Boosting"
        ]:
            if m_name not in st.session_state.model_metrics:
                res = hub.train_and_evaluate(m_name, X_train, y_train, X_test, y_test, class_names=classes)
                st.session_state.model_metrics[m_name] = res
                st.session_state.trained_models[m_name] = res["model"]
                
        if "Deep Neural Network (MLP)" not in st.session_state.model_metrics:
            dnn = CyberDeepNeuralNetwork(input_dim=X_train.shape[1], num_classes=len(classes), epochs=30, batch_size=64)
            dnn.fit(X_train, y_train, X_test, y_test)
            eval_dnn = dnn.evaluate(X_test, y_test, class_names=classes)
            st.session_state.model_metrics["Deep Neural Network (MLP)"] = eval_dnn
            st.session_state.trained_models["Deep Neural Network (MLP)"] = dnn

metrics_dict = st.session_state.get("model_metrics", {})

# Benchmark Comparison Table
st.subheader("Model Performance Comparison Leaderboard")

comp_rows = []
for name, m in metrics_dict.items():
    comp_rows.append({
        "Model": name,
        "Accuracy (%)": round(m.get("accuracy", 0) * 100, 2),
        "Precision (%)": round(m.get("precision", 0) * 100, 2),
        "Recall (%)": round(m.get("recall", 0) * 100, 2),
        "F1-Score (%)": round(m.get("f1", 0) * 100, 2),
        "AUC Score": round(m.get("roc_auc", 0), 3) if m.get("roc_auc") is not None else 0.0,
        "Training Time (s)": round(m.get("training_time", 0.05), 3)
    })

df_comp = pd.DataFrame(comp_rows)
if not df_comp.empty:
    df_comp = df_comp.sort_values(by="F1-Score (%)", ascending=False).reset_index(drop=True)
    st.dataframe(df_comp, use_container_width=True)
    
    # Best Model Callout
    best_model_name = df_comp.iloc[0]["Model"]
    best_f1 = df_comp.iloc[0]["F1-Score (%)"]
    best_acc = df_comp.iloc[0]["Accuracy (%)"]
    
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid #16A34A; margin-top: 14px;">
        <div style="font-size: 15px; font-weight: 600; color: #16A34A; margin-bottom: 4px;">🏆 Best Performing Model: {best_model_name}</div>
        <div style="font-size: 13px; color: #475569;">
            Achieved the highest overall F1-Score of <b>{best_f1}%</b> and Accuracy of <b>{best_acc}%</b> with balanced precision and recall. Recommended for primary production SOC deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Visual Comparison Charts: Grouped Bar & Radar Chart
st.markdown("<br>", unsafe_allow_html=True)
c_bar, c_radar = st.columns([3, 2])

with c_bar:
    if not df_comp.empty:
        melted_df = pd.melt(df_comp, id_vars=["Model"], value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"], var_name="Metric", value_name="Score")
        fig_bars = px.bar(
            melted_df, x="Model", y="Score", color="Metric",
            barmode="group",
            title="Comparison of Core Performance Metrics Across Models",
            color_discrete_sequence=['#2563EB', '#16A34A', '#DC2626', '#F59E0B']
        )
        fig_bars.update_layout(**plotly_layout)
        st.plotly_chart(fig_bars, use_container_width=True)

with c_radar:
    if not df_comp.empty:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig_radar = go.Figure()
        
        for idx, row in df_comp.iterrows():
            scores = [row['Accuracy (%)'], row['Precision (%)'], row['Recall (%)'], row['F1-Score (%)']]
            scores.append(scores[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Model']
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[70, 100])),
            showlegend=True,
            title="Multi-Dimensional Radar Capability Comparison",
            **plotly_layout
        )
        st.plotly_chart(fig_radar, use_container_width=True)
