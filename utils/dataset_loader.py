"""
CyberShield AI - Dataset Loader, Profile Detector & Schema Normalizer
Supports CSV and Excel (.xlsx, .xls) uploads, auto-detects NSL-KDD, CICIDS2017,
and custom cybersecurity network telemetry, handles labeled and unlabeled streams.
"""

import os
import io
import pandas as pd
import numpy as np
from data.dataset_generator import ensure_datasets, ATTACK_CATEGORIES, generate_nsl_kdd_dataset, generate_cicids2017_sample


CATEGORY_MAP = {k.lower(): v for k, v in ATTACK_CATEGORIES.items()}
CATEGORY_MAP.update({
    'normal': 'Normal', 'benign': 'Normal', '0': 'Normal', '0.0': 'Normal',
    'clean': 'Normal', 'legitimate': 'Normal', 'safe': 'Normal', 'no_attack': 'Normal',
    'dos': 'DoS', 'ddos': 'DoS', 'denial_of_service': 'DoS', 'bot': 'DoS',
    'probe': 'Probe', 'surveillance': 'Probe', 'portscan': 'Probe', 'scan': 'Probe',
    'r2l': 'R2L', 'remote_to_local': 'R2L', 'bruteforce': 'R2L', 'brute_force': 'R2L',
    'u2r': 'U2R', 'user_to_root': 'U2R', 'privilege_escalation': 'U2R'
})


def normalize_attack_category(val):
    s = str(val).strip().lower()
    if s in CATEGORY_MAP:
        return CATEGORY_MAP[s]
    # Substring heuristics
    if any(k in s for k in ['norm', 'benign', 'clean', 'safe', 'legit']):
        return 'Normal'
    if any(k in s for k in ['ddos', 'dos', 'flood', 'bot', 'drop']):
        return 'DoS'
    if any(k in s for k in ['probe', 'scan', 'sweep', 'nmap', 'recon']):
        return 'Probe'
    if any(k in s for k in ['r2l', 'pass', 'guess', 'brute', 'ftp', 'ssh', 'phf', 'warez']):
        return 'R2L'
    if any(k in s for k in ['u2r', 'root', 'overflow', 'priv', 'escal', 'buffer']):
        return 'U2R'
    # Fallback: if numeric 1 or generic attack
    if s in ['1', '1.0', 'attack', 'malicious', 'anomaly', 'threat']:
        return 'DoS'
    return str(val).strip().capitalize()


def detect_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Analyzes dataframe schema to recognize standard benchmarks or custom telemetry signatures.
    """
    cols_clean = [str(c).lower().replace(' ', '_').replace('-', '_').strip() for c in df.columns]
    
    nsl_kdd_signatures = {
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'diff_srv_rate'
    }
    cicids_signatures = {
        'destination_port', 'flow_duration', 'total_fwd_packets', 'total_backward_packets',
        'total_length_of_fwd_packets', 'flow_bytes_s', 'flow_packets_s', 'fwd_packet_length_max'
    }
    network_keywords = {'ip', 'port', 'proto', 'protocol', 'packet', 'byte', 'bytes', 'duration', 'rate', 'flow', 'tcp', 'udp'}

    nsl_match = sum(1 for sig in nsl_kdd_signatures if any(sig in c for c in cols_clean))
    cicids_match = sum(1 for sig in cicids_signatures if any(sig in c for c in cols_clean))
    net_match = sum(1 for kw in network_keywords if any(kw in c for c in cols_clean))

    if nsl_match >= 3:
        return {
            "type": "NSL-KDD Network Benchmark",
            "confidence": "High",
            "is_benchmark": True,
            "description": "Standard DARPA/NSL-KDD benchmark containing TCP/UDP connection records and multi-class intrusion labels.",
            "recommended_model": "Random Forest / XGBoost"
        }
    elif cicids_match >= 3:
        return {
            "type": "CICIDS2017 Flow Benchmark",
            "confidence": "High",
            "is_benchmark": True,
            "description": "Canadian Institute for Cybersecurity (CICIDS2017) bidirectional flow captures with advanced statistical features.",
            "recommended_model": "Random Forest / Deep Neural Network"
        }
    elif net_match >= 2:
        return {
            "type": "Custom Network Traffic Telemetry",
            "confidence": "Medium",
            "is_benchmark": False,
            "description": "Network telemetry containing IP/Port/Flow metrics. Ready for automated preprocessing and intrusion evaluation.",
            "recommended_model": "Autoencoder (Anomaly) / Random Forest"
        }
    else:
        return {
            "type": "Generic Tabular Telemetry",
            "confidence": "Informative",
            "is_benchmark": False,
            "description": "General numerical/categorical telemetry dataset. Automatic encoding, scaling, and anomaly scoring enabled.",
            "recommended_model": "Autoencoder Anomaly Detector"
        }


def load_benchmark_dataset(dataset_type="NSL-KDD"):
    """
    Loads benchmark datasets (NSL-KDD or CICIDS2017) from the data directory.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_type == "NSL-KDD":
        path = os.path.join(data_dir, "nsl_kdd_sample.csv")
        if not os.path.exists(path):
            ensure_datasets()
        df = pd.read_csv(path)
    elif dataset_type == "CICIDS2017":
        path = os.path.join(data_dir, "cicids2017_sample.csv")
        if not os.path.exists(path):
            ensure_datasets()
        df = pd.read_csv(path)
    elif dataset_type == "Quick-Demo":
        demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_datasets")
        path = os.path.join(demo_dir, "3_Custom_Network_Traffic_FastDemo.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = generate_nsl_kdd_dataset(n_samples=100)
    else:
        df = generate_nsl_kdd_dataset(n_samples=3000)

    # Set benchmark metadata
    df.attrs['has_ground_truth'] = True
    df.attrs['detected_target_col'] = 'attack_category' if 'attack_category' in df.columns else 'label'
    df.attrs['profile'] = detect_dataset_profile(df)
        
    return df


def parse_uploaded_dataset(uploaded_file, target_column_override=None):
    """
    Parses an uploaded CSV or Excel dataset file, auto-detects schema, validates
    integrity, recognizes benchmark patterns, and normalizes label columns.
    """
    try:
        # Determine file name for extension check
        filename = getattr(uploaded_file, 'name', '')
        if isinstance(uploaded_file, str):
            filename = uploaded_file

        # Check empty file size if file-like
        if hasattr(uploaded_file, 'seek') and hasattr(uploaded_file, 'tell'):
            uploaded_file.seek(0, 2)
            sz = uploaded_file.tell()
            uploaded_file.seek(0)
            if sz == 0:
                return None, "Empty dataset: The uploaded file contains no data rows. Please upload a valid CSV or Excel file."
        is_excel = filename.lower().endswith(('.xlsx', '.xls'))

        # Reading mechanism with fallbacks
        if is_excel:
            df = pd.read_excel(uploaded_file)
        else:
            # Try reading as CSV with UTF-8, then fallback to Latin-1, then Excel
            try:
                if hasattr(uploaded_file, 'seek'):
                    uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    if hasattr(uploaded_file, 'seek'):
                        uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except Exception:
                    if hasattr(uploaded_file, 'seek'):
                        uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file)

        if df is None or df.empty or len(df) == 0:
            return None, "Empty dataset: The uploaded file contains no data rows. Please upload a valid CSV or Excel file."

        if len(df.columns) < 2:
            return None, f"Unsupported dataset structure: Found only {len(df.columns)} column. Cybersecurity telemetry requires at least 2 feature columns."

        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        cols_normalized = {str(c).lower().replace(' ', '_').replace('-', '_').strip(): c for c in df.columns}
        
        detected_col = target_column_override
        
        if not detected_col:
            # Priority candidate target column names
            priority_names = [
                'attack', 'attacks', 'attack_type', 'attack_cat', 'attacktype', 'attack_name',
                'label', 'labels', 'class', 'classes', 'target', 'targets',
                'attack_category', 'classification', 'category', 'categories',
                'threat_type', 'threat', 'outcome', 'result', 'status', 'activity'
            ]
            matching_cols = []
            for p in priority_names:
                if p in cols_lower:
                    matching_cols.append(cols_lower[p])
                elif p in cols_normalized:
                    matching_cols.append(cols_normalized[p])
            
            # Prioritize candidates that contain multiple distinct classes (e.g. Normal + Attacks)
            multi_class = [c for c in matching_cols if df[c].dropna().nunique() > 1]
            if multi_class:
                detected_col = multi_class[0]
            elif matching_cols:
                detected_col = matching_cols[0]
                    
        # Value-based heuristic if still not found
        if not detected_col:
            heuristic_multi = []
            heuristic_single = []
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10:
                    sample_vals = [str(x).lower().strip() for x in df[c].dropna().head(50)]
                    if any(v in ['normal', 'benign', 'dos', 'probe', 'r2l', 'u2r', 'attack', 'anomaly', 'malicious'] for v in sample_vals):
                        if df[c].dropna().nunique() > 1:
                            heuristic_multi.append(c)
                        else:
                            heuristic_single.append(c)
            if heuristic_multi:
                detected_col = heuristic_multi[0]
            elif heuristic_single:
                detected_col = heuristic_single[0]
                        
        # Last column fallback ONLY if column name strongly indicates a label and is not a network feature
        if not detected_col and len(df.columns) > 1:
            last_col = df.columns[-1]
            last_col_clean = str(last_col).lower().strip()
            label_indicators = ['label', 'labels', 'class', 'classes', 'target', 'attack', 'attack_type', 'threat', 'verdict']
            network_features = ['flag', 'proto', 'service', 'port', 'ip', 'state', 'rate', 'byte', 'packet']
            if any(term in last_col_clean for term in label_indicators) and not any(term in last_col_clean for term in network_features):
                detected_col = last_col

        # Set labels and ground-truth metadata
        if detected_col and detected_col in df.columns:
            df['attack_category'] = df[detected_col].apply(normalize_attack_category)
            df['label'] = df['attack_category'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'benign', '0'] else 1)
            df.attrs['has_ground_truth'] = True
            df.attrs['detected_target_col'] = detected_col
        elif 'attack_category' in df.columns:
            df['attack_category'] = df['attack_category'].apply(normalize_attack_category)
            df['label'] = df['attack_category'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'benign', '0'] else 1)
            df.attrs['has_ground_truth'] = True
            df.attrs['detected_target_col'] = 'attack_category'
        elif 'label' in df.columns:
            df['attack_category'] = df['label'].apply(normalize_attack_category)
            df.attrs['has_ground_truth'] = True
            df.attrs['detected_target_col'] = 'label'
        else:
            # Unlabeled dataset - DO NOT inject fake ground truth!
            df.attrs['has_ground_truth'] = False
            df.attrs['detected_target_col'] = None

        df.attrs['profile'] = detect_dataset_profile(df)
        return df, None
    except pd.errors.EmptyDataError:
        return None, "Empty dataset: The uploaded file contains no data rows. Please upload a valid CSV or Excel file."
    except Exception as e:
        return None, str(e)


def get_dataset_summary(df: pd.DataFrame):
    """
    Returns comprehensive dataset overview stats and profile metadata.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # Class distribution if available
    class_dist = {}
    has_ground_truth = df.attrs.get('has_ground_truth', 'attack_category' in df.columns)
    if has_ground_truth and 'attack_category' in df.columns:
        class_dist = df['attack_category'].value_counts().to_dict()
    elif has_ground_truth and 'label' in df.columns:
        class_dist = df['label'].value_counts().to_dict()

    profile = df.attrs.get('profile', detect_dataset_profile(df))
        
    return {
        "rows": total_rows,
        "columns": total_cols,
        "missing_values": missing_count,
        "duplicates": duplicate_count,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_distribution": class_dist,
        "has_ground_truth": has_ground_truth,
        "detected_target_col": df.attrs.get('detected_target_col', None),
        "profile": profile
    }
