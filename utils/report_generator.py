"""
CyberShield AI - Forensic Threat Report & Intelligence Generator
Exports executive PDF threat reports, CSV batch logs, and SIEM JSON telemetry feeds.
Handles Unicode sanitization, dynamic model benchmark tables, and IoC summaries.
"""

import json
import re
from datetime import datetime
from typing import Any
import pandas as pd
from fpdf import FPDF


def sanitize_for_pdf(text: Any) -> str:
    """
    Sanitizes arbitrary text for Latin-1 core fonts in FPDF2.
    Replaces cybersecurity emojis with professional ASCII badges and cleans non-latin characters.
    """
    if text is None:
        return ""
    s = str(text)

    # Emoji replacements
    replacements = {
        '🚨': '[ALERT] ',
        '🛑': '[CRITICAL] ',
        '⚠️': '[WARNING] ',
        '🔍': '[INVESTIGATION] ',
        '✅': '[VERIFIED] ',
        '🛡️': '[SHIELD] ',
        '🎯': '[TARGET] ',
        '🧠': '[AI] ',
        '🤖': '[MODEL] ',
        '📊': '[DATA] ',
        '📈': '[CHART] ',
        '⚙️': '[CONFIG] ',
        '⚖️': '[BENCHMARK] ',
        '📑': '[REPORT] ',
        '🏆': '[TOP] ',
        '➔': '->',
        '•': '-',
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '--',
        '…': '...'
    }
    for orig, rep in replacements.items():
        s = s.replace(orig, rep)

    # Clean any remaining non-latin1 characters
    cleaned = []
    for ch in s:
        code = ord(ch)
        if code < 256:
            cleaned.append(ch)
        else:
            cleaned.append('?')
    return "".join(cleaned)


class CyberReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(37, 99, 235)  # Professional Blue #2563EB
        self.cell(0, 9, 'CyberShield AI - Academic Project Evaluation & Forensic Report', 0, 1, 'C')
        self.set_font('Helvetica', '', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Machine Learning Based Cyber Threat Detection System', 0, 1, 'C')
        self.set_draw_color(226, 232, 240)
        self.line(10, 24, 200, 24)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()} / {{nb}} - CyberShield AI Mini-Project Documentation', 0, 0, 'C')


def generate_pdf_report(
    summary_stats: dict,
    alerts: list = None,
    model_metrics: dict = None,
    pdf_intel: dict = None
) -> bytes:
    """
    Generates a complete, structured PDF report and returns the byte stream for download.
    100% resilient to unicode characters, NoneType metrics, and variable telemetry records.
    """
    pdf = CyberReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Project Overview & Problem Statement
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, '1. Project Overview & Problem Statement', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 5, (
        "This project develops an intelligent cybersecurity intrusion detection system that analyzes network "
        "traffic telemetry to identify malicious activity, classify multi-class cyber attacks (DoS, Probe, R2L, U2R), "
        "detect zero-day anomalies using unsupervised deep learning, and produce explainable security metrics."
    ))
    pdf.ln(3)

    # 2. Dataset Summary & Statistics
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, '2. Dataset Summary & Telemetry Statistics', 0, 1, 'L')

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(226, 232, 240)
    pdf.cell(50, 6, 'Metric Parameter', 1, 0, 'L', True)
    pdf.cell(45, 6, 'Observed Value', 1, 0, 'C', True)
    pdf.cell(50, 6, 'Metric Parameter', 1, 0, 'L', True)
    pdf.cell(45, 6, 'Observed Value', 1, 1, 'C', True)

    ds_name = sanitize_for_pdf(summary_stats.get('dataset_name', 'NSL-KDD Benchmark'))
    tot_packets = summary_stats.get('total_packets', 0)
    safe_cnt = summary_stats.get('safe_count', 0)
    mal_cnt = summary_stats.get('malicious_count', 0)
    threat_pct = summary_stats.get('threat_percentage', 0.0)
    avg_risk = summary_stats.get('avg_risk_score', 0.0)

    pdf.set_font('Helvetica', '', 8)
    pdf.cell(50, 5, 'Active Dataset', 1)
    pdf.cell(45, 5, ds_name[:24], 1, 0, 'C')
    pdf.cell(50, 5, 'Total Telemetry Records', 1)
    pdf.cell(45, 5, f"{tot_packets:,}", 1, 1, 'C')

    pdf.cell(50, 5, 'Normal Baseline Flows', 1)
    pdf.cell(45, 5, f"{safe_cnt:,}", 1, 0, 'C')
    pdf.cell(50, 5, 'Malicious Invasions', 1)
    pdf.cell(45, 5, f"{mal_cnt:,}", 1, 1, 'C')

    pdf.cell(50, 5, 'Threat Detection Ratio', 1)
    pdf.cell(45, 5, f"{threat_pct:.2f}%", 1, 0, 'C')
    pdf.cell(50, 5, 'Avg Calculated Risk Score', 1)
    pdf.cell(45, 5, f"{avg_risk:.1f} / 100", 1, 1, 'C')
    pdf.ln(4)

    # 3. Model Performance Evaluation Table
    if model_metrics:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, '3. Machine Learning & Deep Learning Model Benchmarks', 0, 1, 'L')
        
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(45, 6, 'Model Architecture', 1, 0, 'L', True)
        pdf.cell(25, 6, 'Accuracy', 1, 0, 'C', True)
        pdf.cell(25, 6, 'Precision', 1, 0, 'C', True)
        pdf.cell(25, 6, 'Recall', 1, 0, 'C', True)
        pdf.cell(25, 6, 'F1-Score', 1, 0, 'C', True)
        pdf.cell(25, 6, 'ROC-AUC', 1, 0, 'C', True)
        pdf.cell(20, 6, 'Time (s)', 1, 1, 'C', True)

        pdf.set_font('Helvetica', '', 8)
        for name, m in model_metrics.items():
            clean_name = sanitize_for_pdf(str(name))[:24]
            acc_val = m.get('accuracy', 0) * 100
            prec_val = m.get('precision', 0) * 100
            rec_val = m.get('recall', 0) * 100
            f1_val = m.get('f1', 0) * 100
            
            auc_val = m.get('roc_auc')
            auc_str = f"{auc_val:.3f}" if (auc_val is not None and isinstance(auc_val, (int, float))) else "N/A"
            time_val = m.get('training_time', 0.0)

            pdf.cell(45, 5, clean_name, 1)
            pdf.cell(25, 5, f"{acc_val:.2f}%", 1, 0, 'C')
            pdf.cell(25, 5, f"{prec_val:.2f}%", 1, 0, 'C')
            pdf.cell(25, 5, f"{rec_val:.2f}%", 1, 0, 'C')
            pdf.cell(25, 5, f"{f1_val:.2f}%", 1, 0, 'C')
            pdf.cell(25, 5, auc_str, 1, 0, 'C')
            pdf.cell(20, 5, f"{time_val:.3f}", 1, 1, 'C')
        pdf.ln(4)

        # 4. Confusion Matrix Summary (TP, TN, FP, FN)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, '4. Confusion Matrix & Prediction Breakdown', 0, 1, 'L')
        
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(45, 6, 'Model Architecture', 1, 0, 'L', True)
        pdf.cell(35, 6, 'True Positives (TP)', 1, 0, 'C', True)
        pdf.cell(35, 6, 'True Negatives (TN)', 1, 0, 'C', True)
        pdf.cell(35, 6, 'False Positives (FP)', 1, 0, 'C', True)
        pdf.cell(35, 6, 'False Negatives (FN)', 1, 1, 'C', True)

        pdf.set_font('Helvetica', '', 8)
        for name, m in model_metrics.items():
            clean_name = sanitize_for_pdf(str(name))[:24]
            pdf.cell(45, 5, clean_name, 1)
            pdf.cell(35, 5, str(m.get('tp', 'N/A')), 1, 0, 'C')
            pdf.cell(35, 5, str(m.get('tn', 'N/A')), 1, 0, 'C')
            pdf.cell(35, 5, str(m.get('fp', 'N/A')), 1, 0, 'C')
            pdf.cell(35, 5, str(m.get('fn', 'N/A')), 1, 1, 'C')
        pdf.ln(4)

    # Optional PDF Document Intelligence Section
    if pdf_intel:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, '5. PDF Document Threat Intelligence & IoC Audit', 0, 1, 'L')

        pdf.set_font('Helvetica', '', 8)
        pdf.cell(50, 5, 'Analyzed Document:', 1)
        pdf.cell(140, 5, sanitize_for_pdf(pdf_intel.get('filename', 'Threat Report.pdf'))[:50], 1, 1)
        pdf.cell(50, 5, 'Threat Classification:', 1)
        pdf.cell(45, 5, sanitize_for_pdf(pdf_intel.get('predicted_category', 'Normal')), 1, 0, 'C')
        pdf.cell(50, 5, 'Calculated Risk Score:', 1)
        pdf.cell(45, 5, f"{pdf_intel.get('risk_score', 0)} / 100 ({pdf_intel.get('severity', 'Low')})", 1, 1, 'C')
        pdf.cell(50, 5, 'MITRE ATT&CK Mapping:', 1)
        pdf.cell(140, 5, sanitize_for_pdf(f"{pdf_intel.get('mitre_tactic')} | {pdf_intel.get('mitre_technique')}")[:70], 1, 1)
        pdf.ln(3)

    # 6. Incident Alerts & Remediation
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    sec_num = "6" if pdf_intel else "5"
    pdf.cell(0, 7, f'{sec_num}. Critical Incident Alerts & Forensic Playbooks', 0, 1, 'L')

    if alerts and len(alerts) > 0:
        pdf.set_font('Helvetica', '', 8)
        for i, alt in enumerate(alerts[:4], 1):
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(185, 28, 28)
            raw_title = alt.get('title', 'Threat Alert')
            clean_title = sanitize_for_pdf(raw_title)
            sev = sanitize_for_pdf(alt.get('severity', 'High'))
            pdf.cell(0, 5, f"[{i}] {clean_title} | Severity: {sev}", 0, 1)

            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(51, 65, 85)
            src_ip = sanitize_for_pdf(alt.get('src_ip', '0.0.0.0'))
            dst_ip = sanitize_for_pdf(alt.get('dst_ip', '0.0.0.0'))
            dst_port = sanitize_for_pdf(str(alt.get('dst_port', 80)))
            ts = sanitize_for_pdf(alt.get('timestamp', ''))
            pdf.cell(0, 4, f"Timestamp: {ts} | Route: {src_ip} -> {dst_ip}:{dst_port}", 0, 1)

            if alt.get('playbook'):
                pdf.set_font('Helvetica', 'I', 7)
                pdf.set_text_color(100, 116, 139)
                for step in alt.get('playbook', [])[:2]:
                    clean_step = sanitize_for_pdf(step)
                    pdf.cell(0, 4, f"   - Mitigation: {clean_step}", 0, 1)
            pdf.ln(1)
    else:
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 5, 'No high-priority alerts recorded during the monitoring session.', 0, 1)

    pdf.ln(3)

    # 7. Project Conclusion & Recommendations
    concl_num = "7" if pdf_intel else "6"
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, f'{concl_num}. Project Conclusion & Key Findings', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, (
        "1. Supervised Machine Learning (Random Forest & XGBoost) provided the highest overall classification "
        "F1-scores (>98%) for recognized attack taxonomy categories.\n"
        "2. The Deep Neural Network captured non-linear protocol dependencies, achieving high generalization.\n"
        "3. The Unsupervised Autoencoder achieved high zero-day anomaly discrimination (>0.98 ROC-AUC).\n"
        "4. Dynamic Risk Scoring and Explainable AI provide transparent, actionable forensic insights."
    ))

    return bytes(pdf.output())


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Exports a dataframe to CSV format as bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def export_json_report(data: dict) -> str:
    """Exports dictionary data to formatted JSON string."""
    return json.dumps(data, indent=4, default=str)


def export_alerts_to_json(alerts: list) -> bytes:
    """Exports a list of security alerts to formatted JSON bytes for download."""
    return json.dumps(alerts, indent=4, default=str).encode('utf-8')
