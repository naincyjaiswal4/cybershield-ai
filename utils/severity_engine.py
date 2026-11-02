"""
CyberShield AI - Threat Severity Engine & Risk Scoring Model
Calculates calibrated 0-100 Cyber Risk Scores and maps to standardized Threat Severity Tiers:
  - 0 - 25: Low (Green)
  - 26 - 50: Medium (Orange)
  - 51 - 75: High (Red)
  - 76 - 100: Critical (Dark Red)
"""


def calculate_risk_score(attack_category: str, confidence: float, packet_features: dict = None) -> dict:
    """
    Computes a calibrated 0-100 Threat Risk Score and assigns a Threat Severity level.
    """
    cat = str(attack_category).strip().title()
    
    # Base category risk points
    category_weights = {
        'Normal': 5.0,
        'Probe': 40.0,
        'Dos': 70.0,
        'R2L': 80.0,
        'U2R': 92.0
    }
    
    base_score = category_weights.get(cat, 10.0)
    
    # If normal, keep score strictly in Low tier (0 - 25)
    if cat == 'Normal':
        risk_score = round(base_score * (1.0 - confidence * 0.7), 1)
        risk_score = max(0.0, min(24.0, risk_score))
        return {
            "risk_score": risk_score,
            "severity": "Low",
            "color": "#16A34A",
            "mitre_technique": "T1071 (Standard Application Protocol)",
            "mitre_tactic": "Benign Operational Traffic",
            "impact_level": "None",
            "response_priority": "P4 - Informational"
        }
        
    # Scale by model confidence
    conf_factor = max(0.6, min(1.0, confidence))
    dynamic_score = base_score * conf_factor
    
    # Modifiers based on packet indicators
    bonus = 0.0
    if packet_features:
        flag = str(packet_features.get('flag', '')).upper()
        if flag in ['S0', 'REJ', 'RSTO']:
            bonus += 5.0
            
        serror = float(packet_features.get('serror_rate', 0.0))
        rerror = float(packet_features.get('rerror_rate', 0.0))
        if serror > 0.5:
            bonus += 5.0
        if rerror > 0.5:
            bonus += 4.0
            
        dst_port = int(packet_features.get('dst_port', 80))
        if dst_port in [22, 23, 445, 3389]:
            bonus += 6.0
            
        if int(packet_features.get('root_shell', 0)) == 1 or int(packet_features.get('su_attempted', 0)) > 0:
            bonus += 8.0
            
    final_score = min(100.0, max(26.0, dynamic_score + bonus))
    final_score = round(final_score, 1)
    
    # Tier assignment
    if final_score >= 76.0:
        severity = "Critical"
        color = "#991B1B"
        priority = "P1 - Critical Alert"
        impact = "Severe Network / Host Compromise Imminent"
    elif final_score >= 51.0:
        severity = "High"
        color = "#DC2626"
        priority = "P2 - High Urgency"
        impact = "Active Penetration / Resource Denial in Progress"
    elif final_score >= 26.0:
        severity = "Medium"
        color = "#F59E0B"
        priority = "P3 - Moderate Alert"
        impact = "Reconnaissance / Probe Scanning Activity Detected"
    else:
        severity = "Low"
        color = "#16A34A"
        priority = "P4 - Low Urgency"
        impact = "Minor Anomaly / Baseline Noise"
        
    # MITRE ATT&CK Mapping
    mitre_map = {
        'Dos': ("T1498 (Network Denial of Service)", "Impact"),
        'Probe': ("T1046 (Network Service Discovery)", "Discovery / Reconnaissance"),
        'R2L': ("T1110 (Brute Force / Credential Access)", "Initial Access / Credential Access"),
        'U2R': ("T1068 (Exploitation for Privilege Escalation)", "Privilege Escalation")
    }
    
    technique, tactic = mitre_map.get(cat, ("T1059 (Command Execution)", "Initial Access"))
    
    return {
        "risk_score": final_score,
        "severity": severity,
        "color": color,
        "mitre_technique": technique,
        "mitre_tactic": tactic,
        "impact_level": impact,
        "response_priority": priority
    }
