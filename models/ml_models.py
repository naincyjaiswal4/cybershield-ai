"""
CyberShield AI - Machine Learning Engine
Implements Random Forest, Support Vector Machine (SVM), Decision Tree, Logistic Regression, and XGBoost / Gradient Boosting.
Provides Accuracy, Precision, Recall, F1, ROC-AUC, Training Time, Confusion Matrix (TP, TN, FP, FN), and ROC curves.
"""

import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class CyberMLModelHub:
    """
    Unified manager for training, evaluating, and persisting cybersecurity ML models.
    """
    AVAILABLE_MODELS = [
        "Random Forest",
        "Support Vector Machine (SVM)",
        "Decision Tree",
        "Logistic Regression",
        "XGBoost / Gradient Boosting"
    ]

    def __init__(self, models_dir=None):
        if models_dir is None:
            self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        else:
            self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {}

    def get_model_instance(self, model_name: str, params: dict = None):
        """Initializes model instance with configurable hyperparameters."""
        params = params or {}
        if model_name == "Random Forest":
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 15)
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=-1
            )
        elif model_name == "Support Vector Machine (SVM)":
            C = params.get('C', 1.0)
            kernel = params.get('kernel', 'rbf')
            return SVC(C=C, kernel=kernel, probability=True, random_state=42)
        elif model_name == "Decision Tree":
            max_depth = params.get('max_depth', 12)
            criterion = params.get('criterion', 'gini')
            return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        elif model_name == "Logistic Regression":
            C = params.get('C', 1.0)
            max_iter = params.get('max_iter', 1000)
            return LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        elif model_name in ["XGBoost / Gradient Boosting", "XGBoost", "Gradient Boosting"]:
            if XGB_AVAILABLE:
                n_estimators = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', 6)
                lr = params.get('learning_rate', 0.1)
                return XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=lr, random_state=42, eval_metric='mlogloss', n_jobs=-1
                )
            else:
                n_estimators = params.get('n_estimators', 80)
                max_depth = params.get('max_depth', 5)
                return GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=42
                )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def train_and_evaluate(self, model_name: str, X_train, y_train, X_test, y_test, class_names=None, params=None):
        """
        Trains model, measures training time, and calculates full evaluation metrics:
        Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and TP, TN, FP, FN breakdowns.
        """
        clf = self.get_model_instance(model_name, params)
        
        # Measure training time
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time
        
        y_pred = clf.predict(X_test)
        
        # Probabilities for ROC-AUC
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC-AUC calculation & ROC curve coordinates
        roc_auc = None
        roc_curves = {}
        unique_classes = np.unique(np.concatenate([y_train, y_test]))
        n_classes = len(unique_classes)

        if y_proba is not None and n_classes >= 2:
            try:
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_curves["Binary"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
                else:
                    roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'))
                    y_test_bin = label_binarize(y_test, classes=unique_classes)
                    for i, cls_idx in enumerate(unique_classes):
                        cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else f"Class {cls_idx}"
                        if i < y_proba.shape[1]:
                            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                            auc_i = auc(fpr_i, tpr_i)
                            roc_curves[cls_name] = {"fpr": fpr_i.tolist(), "tpr": tpr_i.tolist(), "auc": float(auc_i)}
            except Exception:
                roc_auc = None
        else:
            roc_auc = None

        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate TP, TN, FP, FN (Multi-class One-vs-Rest aggregation)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        tp_sum = int(np.sum(TP))
        tn_sum = int(np.sum(TN))
        fp_sum = int(np.sum(FP))
        fn_sum = int(np.sum(FN))

        # Safe classification report supporting sparse class subsets
        labels_param = list(range(len(class_names))) if class_names else None
        try:
            report = classification_report(
                y_test, y_pred,
                labels=labels_param,
                target_names=class_names if labels_param else None,
                output_dict=True,
                zero_division=0
            )
        except Exception:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Feature importances if available
        feature_importance = None
        if hasattr(clf, 'feature_importances_'):
            feature_importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            feature_importance = np.mean(np.abs(clf.coef_), axis=0)

        self.models[model_name] = clf
        
        # Auto-save model artifact
        save_path = os.path.join(self.models_dir, f"{model_name.replace(' ', '_').replace('/', '_').lower()}.joblib")
        joblib.dump(clf, save_path)

        return {
            "model": clf,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "training_time": float(round(training_time, 4)),
            "confusion_matrix": cm,
            "tp": tp_sum,
            "tn": tn_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "tp_per_class": TP.tolist(),
            "fp_per_class": FP.tolist(),
            "fn_per_class": FN.tolist(),
            "tn_per_class": TN.tolist(),
            "roc_curves": roc_curves,
            "classification_report": report,
            "feature_importance": feature_importance,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
