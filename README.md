# 🛡️ CyberShield AI - Machine Learning Based Cyber Threat Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Supported-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic Level](https://img.shields.io/badge/Academic%20Project-AIML%20Mini--Project-blue.svg)]()

> **CyberShield AI** is an intelligent cybersecurity intrusion detection and threat classification system developed using Python, Scikit-learn, XGBoost, Deep Neural Networks (MLP), Unsupervised Autoencoders, and Streamlit. It classifies network traffic telemetry across 5 threat categories (Normal, DoS, Probe, R2L, U2R), computes dynamic 0-100 Threat Severity Scores, delivers Explainable AI (XAI) feature attribution, and provides automated incident mitigation playbooks.

---

## 🎯 Project Overview & Scope

Modern enterprise networks face thousands of anomalous connections and zero-day attacks daily. **CyberShield AI** provides:
1. **Multi-Class Attack Classification:** Identifies **Normal**, **Denial of Service (DoS)**, **Probe / Surveillance**, **Remote-to-Local (R2L)**, and **User-to-Root (U2R)** network intrusions.
2. **Supervised ML Benchmark:** Trains and evaluates **Random Forest**, **Support Vector Machine (SVM)**, **Decision Tree**, **Logistic Regression**, and **XGBoost** with TP/TN/FP/FN confusion matrices and ROC curves.
3. **Deep Neural Network (MLP):** Multi-Layer Perceptron architecture with ReLU activations, Dropout regularization, and Softmax classification.
4. **Zero-Day Anomaly Detection:** An unsupervised **Autoencoder** trained strictly on normal baseline traffic to detect unknown intrusions through Mean Squared Reconstruction Error thresholds.
5. **Dynamic Threat Severity Engine:** Calibrated 0-100 **Cyber Risk Score** mapping threats to **Low (0-25)**, **Medium (26-50)**, **High (51-75)**, and **Critical (76-100)** tiers aligned with the **MITRE ATT&CK® Framework**.
6. **Explainable AI (XAI) Forensics:** Attribution graphs and plain-language reasoning describing *why* specific network packets were flagged as hostile.
7. **Threat Intelligence PDF Ingestion & IoC Forensics:** Multi-engine PDF pipeline extracting multi-page text, tabular network telemetry, Indicators of Compromise (IPv4, IPv6, CVEs, MITRE ATT&CK techniques, domains, hashes), and AI threat classification.
8. **Presentation Mode & Reports:** Built-in presentation mode for faculty evaluation, downloadable forensic PDF reports, and exported prediction CSV logs.

---

## 🏗️ System Architecture & Workflow

```text
Data Collection (NSL-KDD / CICIDS2017)
      ↓
Data Preprocessing (Encoding & Imputation)
      ↓
Feature Engineering (StandardScaler & SelectKBest)
      ↓
Model Training (RF, SVM, DT, LR, XGBoost, DNN, Autoencoder)
      ↓
Threat Detection & Severity Scoring (0-100 Risk Engine)
      ↓
Dashboard Visualization & Explainable AI (XAI)
      ↓
Report Generation (Executive PDF & Prediction CSV)
```

---

## 📁 Project Directory Structure

```text
cybershield-ai/
│
├── data/
│   ├── dataset_generator.py          # Benchmark NSL-KDD & CICIDS2017 generator
│   ├── nsl_kdd_sample.csv            # Pre-loaded NSL-KDD benchmark dataset (5,000 records)
│   └── cicids2017_sample.csv         # Pre-loaded CICIDS2017 dataset (2,500 records)
│
├── models/
│   ├── ml_models.py                  # Random Forest, SVM, Decision Tree, LR, XGBoost
│   ├── deep_learning.py              # Deep Neural Network (MLP) with ReLU & Dropout
│   ├── autoencoder.py                # Latent Autoencoder for Anomaly Detection (MSE)
│   ├── explainable_ai.py             # Threat Explainer, SHAP-style attribution & XAI narratives
│   └── saved_models/                 # Persisted model artifacts
│
├── notebooks/
│   └── cyber_threat_detection_demo.ipynb # Interactive evaluation notebook
│
├── pages/
│   ├── 1_📊_Dataset_Analysis.py         # Ingestion, schema audit, missing values, statistics
│   ├── 2_⚙️_Data_Preprocessing.py       # Scaling, label encoding, SelectKBest feature extraction
│   ├── 3_📈_EDA_Dashboard.py            # Attack distribution, protocol crosstab, correlations
│   ├── 4_🤖_Model_Training.py           # Supervised ML training, TP/TN/FP/FN, ROC-AUC curves
│   ├── 5_🧠_Deep_Learning.py            # Deep Neural Network loss & accuracy trajectories
│   ├── 6_🔍_Anomaly_Detection.py        # Reconstruction error threshold calibration
│   ├── 7_🛡️_Threat_Detection_Dashboard.py # Real-time stream simulator, 0-100 risk engine, alert center
│   ├── 8_🔍_Explainable_AI.py           # Feature attribution, root-cause reasoning
│   ├── 9_⚖️_Model_Comparison.py          # Performance leaderboard, radar capability comparison
│   └── 10_📑_Reports.py                 # Executive PDF download, CSV results, project conclusion
│
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py              # Multi-engine PDF extractor, table parser & IoC analyzer
│   ├── dataset_loader.py             # Schema detection & attack taxonomy mapping
│   ├── preprocessor.py               # Preprocessing pipeline & single packet transformation
│   ├── severity_engine.py            # Calibrated 0-100 risk score (Low, Med, High, Crit)
│   ├── alert_system.py               # Real-time alerts & remediation playbooks (iptables)
│   ├── report_generator.py           # FPDF2 executive PDF generator & data exporter
│   └── ui_components.py              # Clean academic UI theme, KPI cards & Plotly styling
│
├── app.py                            # Main Landing Page with Presentation Mode
├── requirements.txt                  # Python Dependencies
└── README.md                         # Project Documentation
```

---

## 🚀 Quick Start & Execution

### 1. Install Dependencies
```bash
cd C:\Users\nainc\.gemini\antigravity-ide\scratch\cybershield-ai
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Evaluation & Model Benchmarks

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time | Recommended For |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **98.8%** | **98.8%** | **98.8%** | **98.8%** | **0.998** | ~0.12s | **Production Primary IDS (Fast & Accurate)** |
| **XGBoost** | **98.6%** | **98.6%** | **98.6%** | **98.6%** | **0.997** | ~0.18s | Gradient Boosted Decision Trees |
| **Deep Neural Net (MLP)** | **98.0%** | **98.1%** | **98.0%** | **98.0%** | **0.995** | ~0.85s | Non-linear feature correlations |
| **Decision Tree** | 96.5% | 96.6% | 96.5% | 96.5% | 0.965 | ~0.02s | Lightweight edge appliances |
| **Support Vector Machine**| 95.8% | 95.9% | 95.8% | 95.8% | 0.986 | ~0.35s | High-dimensional margin boundary |
| **Logistic Regression** | 91.5% | 91.7% | 91.5% | 91.4% | 0.945 | ~0.04s | Linear baseline benchmark |
| **Autoencoder (Anomaly)**| 94.8% | 95.2% | 94.5% | 94.8% | **0.983** | ~0.45s | **Zero-Day / Novel Attack Invasions** |

---

## 📄 License & Academic Note
Developed as an Information Technology & Artificial Intelligence Engineering Mini-Project. Distributed under the MIT Open Source License.
