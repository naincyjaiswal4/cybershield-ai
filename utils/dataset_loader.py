"""
CyberShield AI - Dataset Loader & Parser
Loads NSL-KDD, CICIDS2017, and custom cybersecurity datasets with automatic schema detection.
"""

import os
import pandas as pd
import numpy as np
from data.dataset_generator import ensure_datasets, ATTACK_CATEGORIES, generate_nsl_kdd_dataset, generate_cicids2017_sample


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
    else:
        df = generate_nsl_kdd_dataset(n_samples=3000)
        
    return df


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


def parse_uploaded_dataset(uploaded_file, target_column_override=None):
    """
    Parses an uploaded CSV dataset file, auto-detects schema and normalizes label columns.
    """
    try:
        df = pd.read_csv(uploaded_file)
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        
        detected_col = target_column_override
        
        if not detected_col:
            # Priority candidate target column names
            priority_names = [
                'attack', 'attacks', 'attack_type', 'attack_cat', 'attacktype', 'attack_name',
                'label', 'labels', 'class', 'classes', 'target', 'targets',
                'attack_category', 'classification', 'category', 'categories',
                'threat_type', 'threat', 'outcome', 'result', 'status', 'activity'
            ]
            matching_cols = [cols_lower[p] for p in priority_names if p in cols_lower]
            
            # Prioritize candidates that contain multiple distinct classes (e.g. Normal + Attacks)
            # over legacy export columns that may only contain a single class (e.g. 'Normal')
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
                    if any(v in ['normal', 'benign', 'dos', 'probe', 'r2l', 'u2r', 'attack', 'anomaly'] for v in sample_vals):
                        if df[c].dropna().nunique() > 1:
                            heuristic_multi.append(c)
                        else:
                            heuristic_single.append(c)
            if heuristic_multi:
                detected_col = heuristic_multi[0]
            elif heuristic_single:
                detected_col = heuristic_single[0]
                        
        # Last column fallback if categorical or low-cardinality
        if not detected_col and len(df.columns) > 1:
            last_col = df.columns[-1]
            if df[last_col].nunique() <= 10:
                detected_col = last_col
                
        if detected_col:
            df['attack_category'] = df[detected_col].apply(normalize_attack_category)
            df.attrs['detected_target_col'] = detected_col
        elif 'attack_category' in df.columns:
            df['attack_category'] = df['attack_category'].apply(normalize_attack_category)
            df.attrs['detected_target_col'] = 'attack_category'
        else:
            df['attack_category'] = 'Normal'
            df.attrs['detected_target_col'] = None
            
        df['label'] = df['attack_category'].apply(lambda x: 0 if str(x).strip().lower() in ['normal', 'benign', '0'] else 1)
        return df, None
    except Exception as e:
        return None, str(e)


def get_dataset_summary(df: pd.DataFrame):
    """
    Returns comprehensive dataset overview stats.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # Class distribution if available
    class_dist = {}
    if 'attack_category' in df.columns:
        class_dist = df['attack_category'].value_counts().to_dict()
    elif 'label' in df.columns:
        class_dist = df['label'].value_counts().to_dict()
        
    return {
        "rows": total_rows,
        "columns": total_cols,
        "missing_values": missing_count,
        "duplicates": duplicate_count,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_distribution": class_dist
    }
