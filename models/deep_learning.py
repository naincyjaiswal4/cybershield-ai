"""
CyberShield AI - Deep Learning Module (Deep Neural Network / MLP)
Architecture: Input Layer -> Dense(ReLU) -> Dropout -> Dense(ReLU) -> Dropout -> Softmax Output Layer.
Provides epoch-level loss/accuracy curves, validation tracking, and multi-class classification.
"""

import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class CyberDeepNeuralNetwork:
    """
    Modular Deep Neural Network (MLP) with explicit forward/backward propagation,
    ReLU activations, Dropout regularization, Softmax outputs, and loss/accuracy histories.
    """
    def __init__(self, input_dim=20, hidden_layers=(128, 64), num_classes=5, dropout_rate=0.25, lr=0.01, epochs=40, batch_size=64):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.is_trained = False
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) weight initialization for deep ReLU networks."""
        layer_dims = [self.input_dim] + list(self.hidden_layers) + [self.num_classes]
        self.weights = []
        self.biases = []
        np.random.seed(42)
        
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i+1]
            limit = np.sqrt(2.0 / n_in)
            W = np.random.randn(n_in, n_out) * limit
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]

    def _forward(self, X, training=True):
        activations = [X]
        zs = []
        dropout_masks = []
        
        A = X
        num_layers = len(self.weights)
        
        for i in range(num_layers - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            zs.append(Z)
            A = self._relu(Z)
            
            # Inverted Dropout
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                A = A * mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
                
            activations.append(A)
            
        # Output layer with Softmax
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        zs.append(Z_out)
        A_out = self._softmax(Z_out)
        activations.append(A_out)
        
        return activations, zs, dropout_masks

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Trains the Deep Neural Network using mini-batch Adam/SGD optimizer and tracks history.
        """
        self.input_dim = X_train.shape[1]
        self.num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else max(2, len(np.unique(y_train)))
        self._init_weights()
        
        y_train_oh = self._one_hot(y_train, self.num_classes)
        if X_val is not None and y_val is not None:
            y_val_oh = self._one_hot(y_val, self.num_classes)
        else:
            X_val, y_val = X_train, y_train
            y_val_oh = y_train_oh

        n_samples = X_train.shape[0]
        
        # Momentum velocity vectors
        v_W = [np.zeros_like(W) for W in self.weights]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta = 0.9

        for epoch in range(1, self.epochs + 1):
            # Shuffle mini-batches
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, zs, masks = self._forward(xb, training=True)
                y_pred = activations[-1]
                
                # Backward pass
                m = xb.shape[0]
                dZ = (y_pred - yb) / m
                
                dW = []
                db = []
                
                for l in reversed(range(len(self.weights))):
                    dW_l = np.dot(activations[l].T, dZ)
                    db_l = np.sum(dZ, axis=0, keepdims=True)
                    dW.insert(0, dW_l)
                    db.insert(0, db_l)
                    
                    if l > 0:
                        dA = np.dot(dZ, self.weights[l].T)
                        if masks[l-1] is not None:
                            dA = dA * masks[l-1]
                        dZ = dA * self._relu_deriv(zs[l-1])

                # Momentum gradient update
                for l in range(len(self.weights)):
                    v_W[l] = beta * v_W[l] + (1 - beta) * dW[l]
                    v_b[l] = beta * v_b[l] + (1 - beta) * db[l]
                    self.weights[l] -= self.lr * v_W[l]
                    self.biases[l] -= self.lr * v_b[l]

            # Compute Epoch Metrics
            train_acts, _, _ = self._forward(X_train, training=False)
            train_probs = train_acts[-1]
            train_loss = -np.mean(np.sum(y_train_oh * np.log(np.clip(train_probs, 1e-12, 1.0)), axis=1))
            train_acc = accuracy_score(y_train, np.argmax(train_probs, axis=1))

            val_acts, _, _ = self._forward(X_val, training=False)
            val_probs = val_acts[-1]
            val_loss = -np.mean(np.sum(y_val_oh * np.log(np.clip(val_probs, 1e-12, 1.0)), axis=1))
            val_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_acc"].append(float(val_acc))

        self.is_trained = True
        return self.history

    def predict_proba(self, X):
        activations, _, _ = self._forward(X, training=False)
        return activations[-1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X_test, y_test, class_names=None):
        """Calculates deep learning evaluation metrics."""
        probs = self.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "classification_report": report,
            "history": self.history,
            "y_pred": preds,
            "y_proba": probs
        }
