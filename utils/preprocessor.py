"""
CyberShield AI - Data Preprocessing & Feature Engineering Pipeline
Handles missing values, label encoding, scaling, train-test splitting, and feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class CyberDataPreprocessor:
    """
    Modular preprocessor for NSL-KDD, CICIDS2017, and network threat datasets.
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

    def _get_scaler(self):
        if self.scaling_method == "MinMaxScaler":
            return MinMaxScaler()
        elif self.scaling_method == "RobustScaler":
            return RobustScaler()
        return StandardScaler()

    def fit_transform(self, df: pd.DataFrame, target_col="attack_category", select_features=True):
        """
        Fits encoders, scalers, and feature selectors, and returns processed train/test splits.
        """
        df_clean = df.copy()
        
        # 1. Drop non-feature SOC metadata and all potential target/label columns to prevent leakage
        meta_cols = [
            'src_ip', 'dst_ip', 'dst_port', 'difficulty_level',
            'attack_category', 'attack_type', 'attack', 'Attack', 'attack_name',
            'label', 'Label', 'class', 'Class', 'target', 'Target',
            'classification', 'Classification', 'category', 'Category',
            'threat_type', 'threat', 'outcome', 'Outcome', 'status', 'Status',
            'result', 'Result', 'activity'
        ]
        cols_to_drop = [c for c in meta_cols if c in df_clean.columns and c != target_col]
        df_features = df_clean.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure target column is separated
        if target_col in df_features.columns:
            y = df_features[target_col].copy()
            X = df_features.drop(columns=[target_col])
        elif target_col in df_clean.columns:
            y = df_clean[target_col].copy()
            X = df_features
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
            
        # 2. Missing Value Imputation
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].median()
                X[col] = X[col].fillna(med if not np.isnan(med) else 0)
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
        
        # 4. Target Encoding (Multi-Class)
        y_encoded = self.target_encoder.fit_transform(y.astype(str))
        self.attack_classes = list(self.target_encoder.classes_)
        
        # 5. Train-Test Split (80/20 Stratified when valid)
        class_counts = np.bincount(y_encoded) if len(y_encoded) > 0 else np.array([0])
        min_class_count = int(np.min(class_counts)) if len(class_counts) > 0 else 0
        stratify_param = y_encoded if min_class_count >= 2 and len(class_counts) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=0.20, random_state=42, stratify=stratify_param
        )
        
        # 6. Feature Scaling
        self.scaler = self._get_scaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 7. Feature Selection (Optional, requires at least 2 distinct target classes)
        n_unique_classes = len(np.unique(y_train))
        if select_features and self.n_features and self.n_features < X.shape[1] and n_unique_classes >= 2:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X.shape[1]))
            X_train_final = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_final = self.feature_selector.transform(X_test_scaled)
            
            mask = self.feature_selector.get_support()
            self.selected_feature_names = [self.feature_names_after_encoding[i] for i, m in enumerate(mask) if m]
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            self.selected_feature_names = self.feature_names_after_encoding
            
        self.is_fitted = True
        
        return {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.selected_feature_names,
            "target_classes": self.attack_classes,
            "original_features": self.feature_names_after_encoding
        }

    def transform_single(self, packet_dict: dict):
        """
        Transforms a single network packet for real-time threat inference.
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
        
        # Scale
        scaled = self.scaler.transform(df_p.values)
        
        # Select features if selector was used
        if self.feature_selector is not None:
            final_vec = self.feature_selector.transform(scaled)
        else:
            final_vec = scaled
            
        return final_vec
