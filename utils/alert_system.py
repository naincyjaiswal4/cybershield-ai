"""
CyberShield AI - Cybersecurity Alert System & Incident Response Playbook
Generates SOC alerts, automated playbooks, mitigation scripts, and firewall defense rules.
"""

from datetime import datetime


def generate_threat_alert(packet_info: dict, classification: str, confidence: float, risk_details: dict) -> dict:
    """
    Constructs a structured cybersecurity incident alert with response recommendations.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    src_ip = packet_info.get('src_ip', '192.168.1.100')
    dst_ip = packet_info.get('dst_ip', '10.0.0.15')
    dst_port = packet_info.get('dst_port', 80)
    protocol = packet_info.get('protocol_type', 'tcp').upper()
    severity = risk_details.get('severity', 'Medium')
    risk_score = risk_details.get('risk_score', 50.0)
    
    # Generate contextual threat description & playbook
    if classification == 'DoS':
        title = "🚨 SYN Flood / High-Volume DoS Attack Detected"
        desc = f"Excessive packet stream targeting port {dst_port}. Connection state flag indicates malformed handshakes."
        playbook = [
            f"Apply firewall drop rule: iptables -A INPUT -s {src_ip} -j DROP",
            f"Enable SYN cookies on target gateway: sysctl -w net.ipv4.tcp_syncookies=1",
            f"Rate limit incoming TCP connections on port {dst_port}."
        ]
        ioc = f"Hostile Source: {src_ip} -> Target: {dst_ip}:{dst_port} (Protocol: {protocol})"
        
    elif classification == 'Probe':
        title = "🔍 Network Reconnaissance / Port Sweep Activity"
        desc = f"Sequential port scanning pattern observed from host {src_ip}. Reconnaissance attempt identified."
        playbook = [
            f"Temporarily quarantine IP {src_ip} for 60 minutes.",
            f"Update IDS/IPS Snort rule to alert on SYN-FIN probe scans from {src_ip}.",
            "Verify perimeter exposure and close unused external ports."
        ]
        ioc = f"Scanner IP: {src_ip} (Scan Type: Multi-port Sweep, Protocol: {protocol})"
        
    elif classification == 'R2L':
        title = "⚠️ Unauthorized Remote Access / Brute-Force Attempt"
        desc = f"Multiple failed authentications or buffer exploits targeting service on port {dst_port}."
        playbook = [
            f"Enforce fail2ban jail for IP: {src_ip}",
            "Mandate Multi-Factor Authentication (MFA) on remote administration endpoints.",
            "Inspect auth logs on destination server for compromised credentials."
        ]
        ioc = f"Target Endpoint: {dst_ip}:{dst_port} (Origin: {src_ip})"
        
    elif classification == 'U2R':
        title = "🛑 CRITICAL: Privilege Escalation (User-to-Root) Attempt"
        desc = f"Root shell spawn / buffer execution detected on host {dst_ip} initiated by {src_ip}."
        playbook = [
            f"IMMEDIATELY isolate target host {dst_ip} from local subnet.",
            "Terminate active privileged sessions (`pkill -u root` / revoke sudo tokens).",
            "Collect forensic snapshot of kernel memory and audit logs (`/var/log/audit/`)."
        ]
        ioc = f"Host Compromise Indicator on {dst_ip} (Source: {src_ip})"
        
    else:
        title = "✅ Normal Traffic Flow Verified"
        desc = "Standard network packet behavior consistent with baseline profile."
        playbook = ["Continue routine SIEM telemetry logging."]
        ioc = f"Session: {src_ip} -> {dst_ip}:{dst_port}"
        
    return {
        "timestamp": timestamp,
        "title": title,
        "classification": classification,
        "severity": severity,
        "risk_score": risk_score,
        "confidence": round(confidence * 100, 1),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "protocol": protocol,
        "description": desc,
        "playbook": playbook,
        "ioc": ioc
    }
