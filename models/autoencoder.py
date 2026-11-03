"""
CyberShield AI - Autoencoder Anomaly Detection Engine
Trained on Normal network traffic; detects zero-day attacks & unknown intrusions via Reconstruction Error.
"""

import time
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support


class CyberAutoencoder:
    """
    Symmetric Autoencoder for Unsupervised Anomaly Detection.
    Architecture: Input -> Hidden1 -> Latent Bottleneck -> Hidden2 -> Reconstruction
    """
    def __init__(self, input_dim=20, latent_dim=6, hidden_dim=14, lr=0.01, epochs=35, batch_size=64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.05
        self.history = {"epoch": [], "loss": []}
        self.is_trained = False
        self.training_time = 0.0
        self._init_weights()

    def _init_weights(self):
        np.random.seed(42)
        # Encoder weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.latent_dim))
        
        # Decoder weights
        self.W3 = np.random.randn(self.latent_dim, self.hidden_dim) * np.sqrt(2.0 / self.latent_dim)
        self.b3 = np.zeros((1, self.hidden_dim))
        self.W4 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b4 = np.zeros((1, self.input_dim))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def encode(self, X):
        h1 = self._relu(np.dot(X, self.W1) + self.b1)
        latent = self._relu(np.dot(h1, self.W2) + self.b2)
        return latent

    def decode(self, latent):
        h2 = self._relu(np.dot(latent, self.W3) + self.b3)
        reconstruction = np.dot(h2, self.W4) + self.b4
        return reconstruction

    def reconstruct(self, X):
        latent = self.encode(X)
        return self.decode(latent)

    def compute_reconstruction_error(self, X):
        """Computes sample-wise Mean Squared Reconstruction Error."""
        reconstructed = self.reconstruct(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        return mse

    def fit(self, X_normal, epochs=None):
        """
        Trains Autoencoder strictly on Normal network traffic.
        """
        start_time = time.perf_counter()
        self.input_dim = X_normal.shape[1]
        self._init_weights()
        epochs = epochs or self.epochs
        
        n_samples = X_normal.shape[0]
        
        # Momentum velocity
        v_W1, v_b1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        v_W2, v_b2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        v_W3, v_b3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
        v_W4, v_b4 = np.zeros_like(self.W4), np.zeros_like(self.b4)
        beta = 0.9

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_normal[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                xb = X_shuffled[start_idx:start_idx + self.batch_size]
                m = xb.shape[0]

                # Forward
                z1 = np.dot(xb, self.W1) + self.b1
                a1 = self._relu(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                latent = self._relu(z2)
                z3 = np.dot(latent, self.W3) + self.b3
                a3 = self._relu(z3)
                x_rec = np.dot(a3, self.W4) + self.b4

                # Reconstruction Loss MSE
                d_out = 2.0 * (x_rec - xb) / m
                
                # Backprop
                dW4 = np.dot(a3.T, d_out)
                db4 = np.sum(d_out, axis=0, keepdims=True)
                
                da3 = np.dot(d_out, self.W4.T)
                dz3 = da3 * self._relu_deriv(z3)
                dW3 = np.dot(latent.T, dz3)
                db3 = np.sum(dz3, axis=0, keepdims=True)
                
                dlatent = np.dot(dz3, self.W3.T)
                dz2 = dlatent * self._relu_deriv(z2)
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)
                
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self._relu_deriv(z1)
                dW1 = np.dot(xb.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Gradient updates with momentum
                v_W4 = beta * v_W4 + (1 - beta) * dW4
                v_b4 = beta * v_b4 + (1 - beta) * db4
                v_W3 = beta * v_W3 + (1 - beta) * dW3
                v_b3 = beta * v_b3 + (1 - beta) * db3
                v_W2 = beta * v_W2 + (1 - beta) * dW2
                v_b2 = beta * v_b2 + (1 - beta) * db2
                v_W1 = beta * v_W1 + (1 - beta) * dW1
                v_b1 = beta * v_b1 + (1 - beta) * db1

                self.W4 -= self.lr * v_W4
                self.b4 -= self.lr * v_b4
                self.W3 -= self.lr * v_W3
                self.b3 -= self.lr * v_b3
                self.W2 -= self.lr * v_W2
                self.b2 -= self.lr * v_b2
                self.W1 -= self.lr * v_W1
                self.b1 -= self.lr * v_b1

            # Compute epoch MSE
            rec_all = self.reconstruct(X_normal)
            loss = np.mean(np.square(X_normal - rec_all))
            self.history["epoch"].append(epoch)
            self.history["loss"].append(float(loss))

        # Set Anomaly Threshold as the 95th percentile of normal reconstruction error
        train_errors = self.compute_reconstruction_error(X_normal)
        self.threshold = float(np.percentile(train_errors, 95))
        self.training_time = time.perf_counter() - start_time
        self.is_trained = True
        return self.history

    def detect_anomalies(self, X, custom_threshold=None):
        """
        Detects anomalies by comparing reconstruction error with calibrated threshold.
        Returns binary labels (0 = Normal, 1 = Anomaly/Attack) and continuous anomaly scores.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        errors = self.compute_reconstruction_error(X)
        predictions = (errors > threshold).astype(int)
        
        # Normalize anomaly score from 0 to 100
        max_err = max(threshold * 3.0, np.max(errors) if len(errors) > 0 else 1.0)
        anomaly_scores = np.clip((errors / max_err) * 100.0, 0.0, 100.0)
        
        return predictions, errors, anomaly_scores

    def evaluate(self, X_test, y_test_binary=None, custom_threshold=None):
        """Computes ROC-AUC, FPR, TPR, and classification metrics for anomaly detection. Gracefully handles unlabeled data."""
        preds, errors, scores = self.detect_anomalies(X_test, custom_threshold)
        threshold_val = float(custom_threshold if custom_threshold is not None else self.threshold)
        
        if y_test_binary is None:
            # Unlabeled evaluation
            normal_cnt = int(np.sum(preds == 0))
            anomaly_cnt = int(np.sum(preds == 1))
            return {
                "has_ground_truth": False,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "fpr": None,
                "tpr": None,
                "errors": errors,
                "predictions": preds,
                "anomaly_scores": scores,
                "threshold": threshold_val,
                "training_time": float(round(self.training_time, 4)),
                "normal_count": normal_cnt,
                "anomaly_count": anomaly_cnt,
                "tp": None, "tn": None, "fp": None, "fn": None
            }

        # Binary confusion components with ground truth
        tp = np.sum((preds == 1) & (y_test_binary == 1))
        tn = np.sum((preds == 0) & (y_test_binary == 0))
        fp = np.sum((preds == 1) & (y_test_binary == 0))
        fn = np.sum((preds == 0) & (y_test_binary == 1))
        total = tp + tn + fp + fn
        
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # ROC-AUC calculation (requires at least 2 target classes)
        unique_classes = np.unique(y_test_binary)
        if len(unique_classes) >= 2:
            try:
                fpr, tpr, _ = roc_curve(y_test_binary, errors)
                roc_auc = float(auc(fpr, tpr))
            except Exception:
                fpr, tpr, roc_auc = None, None, None
        else:
            fpr, tpr, roc_auc = None, None, None
        
        return {
            "has_ground_truth": True,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "errors": errors,
            "predictions": preds,
            "anomaly_scores": scores,
            "threshold": threshold_val,
            "training_time": float(round(self.training_time, 4)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
