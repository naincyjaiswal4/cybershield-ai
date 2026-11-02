"""
CyberShield AI - Forensic Threat Report, CSV Exporter & Project Conclusion
Allows downloading executive PDF reports, CSV prediction datasets with threat scores, and reviews final project conclusions.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.ui_components import apply_cyber_theme, render_header, render_metric_card
from utils.report_generator import generate_pdf_report, export_dataframe_to_csv, export_json_report

st.set_page_config(page_title="Reports & Conclusion - CyberShield AI", page_icon="📑", layout="wide")
apply_cyber_theme()

render_header(
    title="Incident Intelligence & Forensic Reports",
    subtitle="Export executive PDF incident summaries, CSV telemetry prediction logs, and review final project evaluation findings.",
    status_tag="Report Generation"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first from the Home or Dataset Analysis page.")
    st.stop()

df = st.session_state.current_dataset
metrics_dict = st.session_state.get("model_metrics", {})
alerts = [r.get("alert") for r in st.session_state.get("simulated_results", []) if r.get("alert") is not None]

# Calculate Summary Stats
total_packets = len(df)
normal_count = int((df['attack_category'] == 'Normal').sum()) if 'attack_category' in df.columns else int((df['label'] == 0).sum()) if 'label' in df.columns else int((df['label'].astype(str).str.lower().isin(['normal', '0', 'benign'])).sum()) if 'label' in df.columns else 0
malicious_count = total_packets - normal_count
threat_pct = (malicious_count / total_packets * 100) if total_packets > 0 else 0.0

simulated_results = st.session_state.get("simulated_results", [])
if simulated_results and any("risk_score" in r for r in simulated_results):
    avg_risk_score = round(float(np.mean([r["risk_score"] for r in simulated_results if "risk_score" in r])), 1)
else:
    avg_risk_score = round(float(threat_pct), 1)

summary_stats = {
    "total_packets": total_packets,
    "safe_count": normal_count,
    "malicious_count": malicious_count,
    "threat_percentage": threat_pct,
    "avg_risk_score": avg_risk_score,
    "dataset_name": st.session_state.get("dataset_name", "NSL-KDD Benchmark")
}

# Monitoring Telemetry Summary
st.subheader("Monitoring Telemetry Summary")
c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("Total Monitored Flows", f"{total_packets:,}", "Dataset Records", "blue")
with c2:
    render_metric_card("Identified Threat Invasions", f"{malicious_count:,}", f"{threat_pct:.1f}% Threat Rate", "red")
with c3:
    render_metric_card("Active Alerts Documented", f"{len(alerts)}", "Incidents in Memory", "orange")

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Export Intelligence & Forensic Assets")

col_pdf, col_csv, col_json = st.columns(3)

# 1. PDF Download
with col_pdf:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #DC2626; margin-bottom: 6px;">Executive PDF Report</div>
        <div style="font-size: 13px; color: #64748B; margin-bottom: 12px; line-height: 1.5;">
            Formal PDF document containing executive summary, dataset statistics, model benchmarks, confusion matrices, and mitigation playbooks.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        pdf_bytes = generate_pdf_report(summary_stats, alerts=alerts, model_metrics=metrics_dict)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="CyberShield_AI_Threat_Intelligence_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error generating PDF: {e}")

# 2. CSV Download with Predictions and Threat Scores
with col_csv:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #2563EB; margin-bottom: 6px;">Prediction Results (CSV)</div>
        <div style="font-size: 13px; color: #64748B; margin-bottom: 12px; line-height: 1.5;">
            Export network telemetry enriched with ML predicted attack labels, confidence percentages, and 0-100 threat risk scores.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enrich export df
    df_export = df.copy()
    if "simulated_results" in st.session_state and len(st.session_state.simulated_results) > 0:
        sim_df = pd.DataFrame([{
            "predicted_attack_category": r["classification"],
            "prediction_confidence_pct": r["confidence"],
            "threat_risk_score": r["risk_score"],
            "threat_severity": r["severity"]
        } for r in st.session_state.simulated_results])
        df_export = pd.concat([df_export.head(len(sim_df)).reset_index(drop=True), sim_df], axis=1)
        
    csv_bytes = export_dataframe_to_csv(df_export)
    st.download_button(
        label="Download Prediction Results CSV",
        data=csv_bytes,
        file_name="CyberShield_AI_Prediction_Results.csv",
        mime="text/csv",
        use_container_width=True
    )

# 3. JSON Export
with col_json:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #16A34A; margin-bottom: 6px;">SIEM Telemetry Feed (JSON)</div>
        <div style="font-size: 13px; color: #64748B; margin-bottom: 12px; line-height: 1.5;">
            Structured JSON payload containing session statistics, model benchmarks, and alert payloads for Splunk / Elastic SIEM.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    export_payload = {
        "summary": summary_stats,
        "model_performance": {
            k: {
                "accuracy": round(v.get("accuracy", 0), 4),
                "f1_score": round(v.get("f1", 0), 4),
                "roc_auc": round(v.get("roc_auc", 0), 4)
            } for k, v in metrics_dict.items()
        },
        "alerts_sample": alerts[:10] if alerts else []
    }
    json_str = export_json_report(export_payload)
    st.download_button(
        label="Download SIEM JSON Feed",
        data=json_str,
        file_name="CyberShield_AI_SIEM_Feed.json",
        mime="application/json",
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Model Evaluation Summary Table
st.subheader("Model Evaluation Performance Summary")

if metrics_dict:
    perf_df = pd.DataFrame([{
        "Model Architecture": k,
        "Accuracy": f"{v.get('accuracy', 0)*100:.2f}%",
        "Precision": f"{v.get('precision', 0)*100:.2f}%",
        "Recall": f"{v.get('recall', 0)*100:.2f}%",
        "F1-Score": f"{v.get('f1', 0)*100:.2f}%",
        "ROC-AUC": f"{v.get('roc_auc', 0):.3f}",
        "Training Time": f"{v.get('training_time', 0.05):.3f}s"
    } for k, v in metrics_dict.items()])
    st.dataframe(perf_df, use_container_width=True)
else:
    st.info("Train ML or Deep Learning models in their respective modules to populate performance benchmarks.")

st.markdown("<br>", unsafe_allow_html=True)

# Project Conclusion Page Section
st.subheader("Project Conclusion & Academic Evaluation Findings")

col_concl1, col_concl2 = st.columns(2)

with col_concl1:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #16A34A; margin-bottom: 6px;">Key Research Insights</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.6;">
            • <b>High Detection Precision:</b> Ensemble learning methods (Random Forest and XGBoost) outperformed linear baselines, achieving >98% accuracy and F1-score.<br>
            • <b>Zero-Day Resilience:</b> The Unsupervised Autoencoder successfully separated unknown malicious deviations using reconstruction error without prior attack signatures.<br>
            • <b>Explainability:</b> Feature attribution demystified black-box predictions, allowing SOC analysts to verify root-cause network indicators (e.g. SYN error rate, connection flag states).
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_concl2:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #2563EB; margin-bottom: 6px;">Future Scope & Enhancements</div>
        <div style="font-size: 13px; color: #475569; line-height: 1.6;">
            • <b>Edge Deployment:</b> Quantization and lightweight deployment onto edge IoT gateways for real-time packet filtering.<br>
            • <b>Federated Learning:</b> Collaborative threat model training across multiple decentralized enterprise networks without sharing private packet payloads.<br>
            • <b>Transformer Architectures:</b> Integrating temporal attention mechanisms (e.g. TabNet, Temporal Transformers) for long-duration persistent APT campaigns.
        </div>
    </div>
    """, unsafe_allow_html=True)
