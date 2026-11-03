"""
CyberShield AI - Security Reports & Forensic Intelligence Export Hub
Generates multi-page PDF Executive Summary Reports, CSV Prediction Results, and JSON Incident Feeds.
Integrates ML benchmarks, anomaly detection scores, alert telemetry, and PDF forensics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.ui_components import (
    apply_cyber_theme, render_header, render_metric_card
)
from utils.report_generator import (
    generate_pdf_report, export_dataframe_to_csv, export_alerts_to_json
)

st.set_page_config(page_title="Security Reports - CyberShield AI", page_icon="📑", layout="wide")
apply_cyber_theme()

render_header(
    title="Executive Reports & Forensic Asset Export",
    subtitle="Generate audit-ready PDF executive briefings, download ML prediction telemetry, and export real-time incident alerts.",
    status_tag="Compliance & Forensics"
)

if "current_dataset" not in st.session_state:
    st.warning("Please load a dataset first from the Home or Dataset Analysis page.")
    st.stop()

df = st.session_state.current_dataset
metrics_dict = st.session_state.get("model_metrics", {})
alerts = [r.get("alert") for r in st.session_state.get("simulated_results", []) if r.get("alert") is not None]

# Calculate Summary Stats
total_packets = len(df)
if "auto_threat_results" in st.session_state and st.session_state.auto_threat_results is not None:
    normal_count = st.session_state.auto_threat_results.get("normal_count", 0)
    malicious_count = st.session_state.auto_threat_results.get("threat_count", 0)
else:
    normal_count = int((df['attack_category'] == 'Normal').sum()) if 'attack_category' in df.columns else int((df['label'] == 0).sum()) if 'label' in df.columns else 0
    malicious_count = total_packets - normal_count
threat_pct = (malicious_count / max(1, total_packets) * 100) if total_packets > 0 else 0.0

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
        pdf_intel = st.session_state.get('pdf_analysis_result', {}).get('analysis') if st.session_state.get('pdf_analysis_result') else None
        if pdf_intel and 'filename' not in pdf_intel:
            pdf_intel['filename'] = st.session_state.get('pdf_analysis_result', {}).get('filename', 'Threat Report.pdf')
        pdf_bytes = generate_pdf_report(summary_stats, alerts=alerts, model_metrics=metrics_dict, pdf_intel=pdf_intel)
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
            "predicted_attack_category": r.get("classification", "Normal"),
            "prediction_confidence_pct": r.get("confidence", 90.0),
            "threat_risk_score": r.get("risk_score", 10.0 if r.get("classification", "Normal") == "Normal" else 75.0),
            "threat_severity": r.get("severity", "Low" if r.get("classification", "Normal") == "Normal" else "High")
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

# 3. JSON Real-Time Incident Stream
with col_json:
    st.markdown("""
    <div class="content-card">
        <div style="font-size: 15px; font-weight: 600; color: #F59E0B; margin-bottom: 6px;">Incident Telemetry (JSON)</div>
        <div style="font-size: 13px; color: #64748B; margin-bottom: 12px; line-height: 1.5;">
            Export live SIEM/SOC alert payloads with IP origins, MITRE ATT&CK techniques, and recommended mitigation actions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    json_bytes = export_alerts_to_json(alerts)
    st.download_button(
        label="Download Alert Stream JSON",
        data=json_bytes,
        file_name="CyberShield_AI_Incident_Stream.json",
        mime="application/json",
        use_container_width=True
    )
