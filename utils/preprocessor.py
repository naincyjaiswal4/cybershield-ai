"""
CyberShield AI - Data Preprocessing & Feature Engineering Pipeline
Handles missing values, label encoding, scaling, train-test splitting, and feature selection.
Supports both labeled multi-class classification and unsupervised anomaly detection telemetry.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold


class CyberDataPreprocessor:
    """
    Modular preprocessor for NSL-KDD, CICIDS2017, and network threat datasets.
    Supports supervised multi-class classification and unsupervised telemetry ingestion.
    """
    def __init__(self, scaling_method="StandardScaler", n_features=20):
        self.scaling_method = scaling_method
        self.n_features = n_features
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = None
        self.feature_selector = None
        self.selected_feature_names = []
        self.feature_names_after_encoding = []
        self.is_fitted = False
        self.attack_classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.has_ground_truth = True

    def _get_scaler(self):
        if self.scaling_method == "MinMaxScaler":
            return MinMaxScaler()
        elif self.scaling_method == "RobustScaler":
            return RobustScaler()
        return StandardScaler()

    def fit_transform(self, df: pd.DataFrame, target_col="attack_category", select_features=True):
        """
        Fits encoders, scalers, and feature selectors, and returns processed train/test splits.
        Automatically switches to unsupervised mode if target_col is None or absent.
        """
        df_clean = df.copy()
        
        # 1. Drop non-feature SOC metadata and target/label columns to prevent leakage
        meta_cols = [
            'src_ip', 'dst_ip', 'dst_port', 'difficulty_level', 'flow_id', 'timestamp',
            'attack_category', 'attack_type', 'attack', 'Attack', 'attack_name',
            'label', 'Label', 'class', 'Class', 'target', 'Target',
            'classification', 'Classification', 'category', 'Category',
            'threat_type', 'threat', 'outcome', 'Outcome', 'status', 'Status',
            'result', 'Result', 'activity'
        ]
        
        # Determine if target column exists
        has_target = (target_col is not None and (target_col in df_clean.columns or target_col in df.columns))
        
        if has_target:
            self.has_ground_truth = True
            cols_to_drop = [c for c in meta_cols if c in df_clean.columns and c != target_col]
            df_features = df_clean.drop(columns=cols_to_drop, errors='ignore')
            
            if target_col in df_features.columns:
                y = df_features[target_col].copy()
                X = df_features.drop(columns=[target_col])
            else:
                y = df_clean[target_col].copy()
                X = df_features
        else:
            self.has_ground_truth = False
            cols_to_drop = [c for c in meta_cols if c in df_clean.columns]
            X = df_clean.drop(columns=cols_to_drop, errors='ignore')
            y = None
            
        # Ensure at least 1 feature column exists
        if X.shape[1] == 0:
            # Fallback: keep all non-target columns
            X = df_clean.copy()
            if has_target and target_col in X.columns:
                X = X.drop(columns=[target_col])
            
        # 2. Missing Value Imputation
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].median()
                X[col] = X[col].fillna(med if not np.isnan(med) else 0.0)
            else:
                mode_vals = X[col].mode()
                X[col] = X[col].fillna(mode_vals[0] if not mode_vals.empty else 'unknown')
                
        # 3. Categorical Encoding
        self.label_encoders = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
        self.feature_names_after_encoding = list(X.columns)
        n_samples = len(X)
        
        # 4. Target Encoding & Splitting
        if self.has_ground_truth and y is not None:
            y_encoded = self.target_encoder.fit_transform(y.astype(str))
            self.attack_classes = list(self.target_encoder.classes_)
            
            # Train-Test Split (handle tiny datasets gracefully)
            if n_samples < 5:
                X_train_raw = X.values
                X_test_raw = X.values
                y_train = y_encoded
                y_test = y_encoded
            else:
                class_counts = np.bincount(y_encoded) if len(y_encoded) > 0 else np.array([0])
                min_class_count = int(np.min(class_counts)) if len(class_counts) > 0 else 0
                stratify_param = y_encoded if min_class_count >= 2 and len(class_counts) > 1 else None
                
                test_sz = 0.20 if n_samples >= 10 else 0.25
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X.values, y_encoded, test_size=test_sz, random_state=42, stratify=stratify_param
                )
        else:
            self.attack_classes = []
            y_train = None
            y_test = None
            if n_samples < 5:
                X_train_raw = X.values
                X_test_raw = X.values
            else:
                test_sz = 0.20 if n_samples >= 10 else 0.25
                X_train_raw, X_test_raw = train_test_split(
                    X.values, test_size=test_sz, random_state=42
                )
        
        # 5. Feature Scaling
        self.scaler = self._get_scaler()
        X_train_scaled = self.scaler.fit_transform(X_train_raw.astype(np.float64))
        X_test_scaled = self.scaler.transform(X_test_raw.astype(np.float64))
        
        # 6. Feature Selection (Optional)
        if select_features and self.n_features and self.n_features < X.shape[1]:
            if self.has_ground_truth and len(np.unique(y_train)) >= 2:
                def safe_f_classif(X_arr, y_arr):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        scores, p_vals = f_classif(X_arr, y_arr)
                        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                        p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
                    return scores, p_vals

                k_features = min(self.n_features, X.shape[1])
                self.feature_selector = SelectKBest(score_func=safe_f_classif, k=k_features)
                X_train_final = self.feature_selector.fit_transform(X_train_scaled, y_train)
                X_test_final = self.feature_selector.transform(X_test_scaled)
                
                mask = self.feature_selector.get_support()
                self.selected_feature_names = [self.feature_names_after_encoding[i] for i, m in enumerate(mask) if m]
            else:
                # Unsupervised feature selection: top features by variance
                variances = np.var(X_train_scaled, axis=0)
                top_indices = np.argsort(variances)[::-1][:min(self.n_features, X.shape[1])]
                top_indices = sorted(top_indices)
                X_train_final = X_train_scaled[:, top_indices]
                X_test_final = X_test_scaled[:, top_indices]
                self.selected_feature_names = [self.feature_names_after_encoding[i] for i in top_indices]
                self.feature_selector = None
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            self.selected_feature_names = self.feature_names_after_encoding
            self.feature_selector = None
            
        self.is_fitted = True
        
        return {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "y_train": y_train,
            "y_test": y_test,
            "has_ground_truth": self.has_ground_truth,
            "feature_names": self.selected_feature_names,
            "target_classes": self.attack_classes,
            "original_features": self.feature_names_after_encoding
        }

    def transform_single(self, packet_dict: dict):
        """
        Transforms a single network packet for real-time threat inference.
        Ensures type safety, missing feature handling, and alignment.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform_single.")
            
        df_p = pd.DataFrame([packet_dict])
        
        # Encode categoricals
        for col, le in self.label_encoders.items():
            if col in df_p.columns:
                val = str(df_p[col].iloc[0])
                if val in le.classes_:
                    df_p[col] = le.transform([val])[0]
                else:
                    df_p[col] = 0
            else:
                df_p[col] = 0
                
        # Ensure all encoded features exist
        for col in self.feature_names_after_encoding:
            if col not in df_p.columns:
                df_p[col] = 0
                
        # Align columns
        df_p = df_p[self.feature_names_after_encoding]
        
        # Ensure all values are coerced to numeric float64
        for col in df_p.columns:
            df_p[col] = pd.to_numeric(df_p[col], errors='coerce').fillna(0.0)
            
        # Scale
        scaled = self.scaler.transform(df_p.values.astype(np.float64))
        
        # Select features if selector was used
        if self.feature_selector is not None:
            final_vec = self.feature_selector.transform(scaled)
        elif len(self.selected_feature_names) < len(self.feature_names_after_encoding):
            # Variance-based selection index alignment
            indices = [self.feature_names_after_encoding.index(f) for f in self.selected_feature_names if f in self.feature_names_after_encoding]
            final_vec = scaled[:, indices]
        else:
            final_vec = scaled
            
        return final_vec
