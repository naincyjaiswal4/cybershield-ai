"""
CyberShield AI - Explainable AI (XAI) & Threat Attribution Engine
Explains model decisions using feature attribution, deviation analysis, and forensic rule correlation.
"""

import numpy as np
import pandas as pd


class ThreatExplainer:
    """
    Decomposes predictions into feature importance, local risk contributions,
    and plain-language cybersecurity forensic explanations.
    """
    def __init__(self, feature_names=None, model=None):
        self.feature_names = feature_names or []
        self.model = model

    def explain_sample(self, sample_vec, feature_names, predicted_class, confidence, baseline_mean=None):
        """
        Computes feature deviations and generates an XAI attribution profile.
        """
        if baseline_mean is None:
            baseline_mean = np.zeros_like(sample_vec)

        # Feature contribution based on deviation from baseline normal
        diff = np.abs(sample_vec - baseline_mean)
        total_diff = np.sum(diff) if np.sum(diff) > 0 else 1.0
        contributions = (diff / total_diff) * 100.0

        # Sort top contributing features
        top_indices = np.argsort(contributions)[::-1][:7]
        top_features = []
        
        for idx in top_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            top_features.append({
                "feature": feat_name,
                "contribution_pct": round(float(contributions[idx]), 2),
                "sample_val": round(float(sample_vec[idx]), 4)
            })

        # Generate Human-Centric Forensic Cyber Reasoning
        narrative = self._generate_narrative(top_features, predicted_class, confidence)

        return {
            "top_features": top_features,
            "narrative": narrative,
            "confidence": confidence,
            "predicted_class": predicted_class
        }

    def _generate_narrative(self, top_features, predicted_class, confidence):
        """Generates plain-language SOC explanation for junior and senior cyber analysts."""
        if predicted_class == 'Normal':
            return (
                f"The system classified this packet flow as Normal ({confidence*100:.1f}% confidence). "
                "Network connection parameters (SYN handshake completion, standard byte ratios, "
                "and host error rates) are within normal operational baseline boundaries."
            )

        feat_names = [f["feature"] for f in top_features[:3]]
        
        explanations = {
            'DoS': (
                f"Flagged as a Denial of Service (DoS) attack ({confidence*100:.1f}% confidence). "
                f"Primary indicators driving this detection: high concentration in {feat_names[0]} and {feat_names[1]}. "
                "This behavior exhibits SYN flood/resource exhaustion characteristics where rapid unacknowledged "
                "connection attempts starve server sockets."
            ),
            'Probe': (
                f"Identified as Surveillance / Reconnaissance Probe ({confidence*100:.1f}% confidence). "
                f"Main anomaly drivers: {feat_names[0]} and {feat_names[1]}. "
                "The packet pattern matches port sweep scanning behavior aimed at enumerating active services."
            ),
            'R2L': (
                f"Detected as Remote-to-Local (R2L) Intrusion attempt ({confidence*100:.1f}% confidence). "
                f"Significant contributors: {feat_names[0]} and {feat_names[1]}. "
                "Indicates unauthorized external attempts to gain local access via credential guessing or buffer exploitation."
            ),
            'U2R': (
                f"CRITICAL: Flagged as User-to-Root (U2R) Privilege Escalation ({confidence*100:.1f}% confidence). "
                f"Key threat triggers: {feat_names[0]} and {feat_names[1]}. "
                "Telemetry shows root shell execution or compromised privilege boundaries."
            )
        }

        return explanations.get(predicted_class, (
            f"The packet flow has been classified as an attack ({predicted_class}) with {confidence*100:.1f}% confidence. "
            f"Key contributing metrics: {', '.join(feat_names)}."
        ))
