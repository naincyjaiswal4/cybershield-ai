"""
CyberShield AI - Dataset Generator & Pre-loader
Generates synthetic and realistic NSL-KDD and CICIDS2017 network traffic datasets.
Includes standard 41 NSL-KDD features, attack categories, labels, and SOC network attributes.
"""

import os
import numpy as np
import pandas as pd

# NSL-KDD Standard Features
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# Attack Classification Map (NSL-KDD 4 Attack Categories + Normal)
ATTACK_CATEGORIES = {
    'normal': 'Normal',
    # Denial of Service (DoS)
    'neptune': 'DoS',
    'smurf': 'DoS',
    'pod': 'DoS',
    'teardrop': 'DoS',
    'land': 'DoS',
    'back': 'DoS',
    'apache2': 'DoS',
    'udpstorm': 'DoS',
    'processtable': 'DoS',
    'mailbomb': 'DoS',
    # Surveillance / Probe
    'portsweep': 'Probe',
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'satan': 'Probe',
    'saint': 'Probe',
    'mscan': 'Probe',
    # Remote to Local (R2L)
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'guess_passwd': 'R2L',
    'ftp_write': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'imap': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'named': 'R2L',
    'xsnoop': 'R2L',
    'sendmail': 'R2L',
    # User to Root (U2R)
    'buffer_overflow': 'U2R',
    'rootkit': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R',
    'ps': 'U2R',
    'httptunnel': 'U2R'
}


def generate_nsl_kdd_dataset(n_samples=5000, random_state=42):
    """
    Generates a realistic benchmark NSL-KDD dataset with balanced multi-class distribution,
    standard connection telemetry, and cyber attack signatures.
    """
    np.random.seed(random_state)
    
    # Class ratios: Normal (52%), DoS (30%), Probe (12%), R2L (5%), U2R (1%)
    class_probs = [0.52, 0.30, 0.12, 0.05, 0.01]
    categories = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    selected_cats = np.random.choice(categories, size=n_samples, p=class_probs)
    
    records = []
    
    # IP pools for SOC dashboard simulation
    internal_ips = [f"192.168.1.{i}" for i in range(10, 60)] + [f"10.0.0.{i}" for i in range(2, 40)]
    external_attack_ips = [
        "185.220.101.5", "194.26.29.112", "45.154.255.89", "103.251.167.20",
        "91.240.118.172", "198.51.100.23", "203.0.113.88", "185.191.171.12",
        "193.32.162.155", "77.247.108.162", "141.98.11.89", "89.248.165.74"
    ]
    
    services = ['http', 'smtp', 'ftp', 'ftp_data', 'ssh', 'dns', 'telnet', 'pop_3', 'finger', 'other', 'private', 'eco_i', 'ecr_i', 'domain_u']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'RSTOS0', 'OTH']

    for i in range(n_samples):
        cat = selected_cats[i]
        
        # Default Normal profile
        duration = 0
        protocol = 'tcp'
        service = 'http'
        flag = 'SF'
        src_bytes = int(np.random.exponential(scale=350) + 40)
        dst_bytes = int(np.random.exponential(scale=2500) + 120)
        land = 0
        wrong_fragment = 0
        urgent = 0
        hot = 0
        num_failed_logins = 0
        logged_in = 1
        num_compromised = 0
        root_shell = 0
        su_attempted = 0
        num_root = 0
        num_file_creations = 0
        num_shells = 0
        num_access_files = 0
        num_outbound_cmds = 0
        is_host_login = 0
        is_guest_login = 0
        count = int(np.random.randint(1, 15))
        srv_count = int(np.random.randint(1, 15))
        serror_rate = float(np.random.uniform(0.0, 0.05))
        srv_serror_rate = float(np.random.uniform(0.0, 0.05))
        rerror_rate = float(np.random.uniform(0.0, 0.05))
        srv_rerror_rate = float(np.random.uniform(0.0, 0.05))
        same_srv_rate = float(np.random.uniform(0.9, 1.0))
        diff_srv_rate = float(np.random.uniform(0.0, 0.08))
        srv_diff_host_rate = float(np.random.uniform(0.0, 0.15))
        dst_host_count = int(np.random.randint(5, 255))
        dst_host_srv_count = int(np.random.randint(100, 255))
        dst_host_same_srv_rate = float(np.random.uniform(0.85, 1.0))
        dst_host_diff_srv_rate = float(np.random.uniform(0.0, 0.06))
        dst_host_same_src_port_rate = float(np.random.uniform(0.0, 0.1))
        dst_host_srv_diff_host_rate = float(np.random.uniform(0.0, 0.05))
        dst_host_serror_rate = float(np.random.uniform(0.0, 0.05))
        dst_host_srv_serror_rate = float(np.random.uniform(0.0, 0.05))
        dst_host_rerror_rate = float(np.random.uniform(0.0, 0.05))
        dst_host_srv_rerror_rate = float(np.random.uniform(0.0, 0.05))
        attack_sub = 'normal'
        
        # Cyber attack specific feature modifications
        if cat == 'DoS':
            attack_sub = np.random.choice(['neptune', 'smurf', 'back', 'teardrop', 'pod'])
            logged_in = 0
            count = int(np.random.randint(100, 512))
            srv_count = int(np.random.randint(80, 512))
            serror_rate = float(np.random.uniform(0.75, 1.0))
            srv_serror_rate = float(np.random.uniform(0.75, 1.0))
            dst_host_serror_rate = float(np.random.uniform(0.70, 1.0))
            dst_host_srv_serror_rate = float(np.random.uniform(0.70, 1.0))
            same_srv_rate = float(np.random.uniform(0.05, 0.4))
            diff_srv_rate = float(np.random.uniform(0.5, 0.95))
            flag = np.random.choice(['S0', 'REJ', 'RSTO'])
            protocol = np.random.choice(['tcp', 'icmp'], p=[0.75, 0.25])
            service = np.random.choice(['private', 'http', 'eco_i', 'ecr_i'])
            src_bytes = int(np.random.choice([0, 40, 1032, 42]))
            dst_bytes = 0
            
        elif cat == 'Probe':
            attack_sub = np.random.choice(['portsweep', 'ipsweep', 'nmap', 'satan'])
            logged_in = 0
            rerror_rate = float(np.random.uniform(0.60, 1.0))
            srv_rerror_rate = float(np.random.uniform(0.60, 1.0))
            dst_host_rerror_rate = float(np.random.uniform(0.60, 1.0))
            dst_host_srv_rerror_rate = float(np.random.uniform(0.60, 1.0))
            diff_srv_rate = float(np.random.uniform(0.70, 1.0))
            dst_host_diff_srv_rate = float(np.random.uniform(0.60, 1.0))
            flag = np.random.choice(['REJ', 'RSTO', 'RSTR', 'SH'])
            protocol = np.random.choice(['tcp', 'udp', 'icmp'], p=[0.6, 0.25, 0.15])
            service = np.random.choice(['private', 'other', 'eco_i', 'finger', 'telnet'])
            src_bytes = int(np.random.randint(0, 120))
            dst_bytes = int(np.random.randint(0, 80))
            
        elif cat == 'R2L':
            attack_sub = np.random.choice(['guess_passwd', 'warezclient', 'ftp_write', 'imap'])
            logged_in = np.random.choice([0, 1], p=[0.7, 0.3])
            num_failed_logins = int(np.random.randint(1, 5))
            hot = int(np.random.randint(1, 6))
            is_guest_login = int(np.random.choice([0, 1], p=[0.6, 0.4]))
            protocol = 'tcp'
            service = np.random.choice(['ftp', 'telnet', 'smtp', 'pop_3', 'imap'])
            flag = 'SF'
            duration = int(np.random.randint(2, 120))
            src_bytes = int(np.random.randint(200, 3500))
            dst_bytes = int(np.random.randint(400, 8000))
            
        elif cat == 'U2R':
            attack_sub = np.random.choice(['buffer_overflow', 'rootkit', 'loadmodule', 'sqlattack'])
            logged_in = 1
            root_shell = 1
            su_attempted = int(np.random.choice([1, 2]))
            num_root = int(np.random.randint(1, 10))
            num_file_creations = int(np.random.randint(1, 4))
            num_compromised = int(np.random.randint(1, 8))
            hot = int(np.random.randint(2, 10))
            protocol = 'tcp'
            service = np.random.choice(['telnet', 'ssh', 'ftp'])
            flag = 'SF'
            duration = int(np.random.randint(10, 300))
            src_bytes = int(np.random.randint(1500, 10000))
            dst_bytes = int(np.random.randint(3000, 25000))

        # SOC metadata for live stream simulation
        src_ip = np.random.choice(external_attack_ips) if cat != 'Normal' else np.random.choice(internal_ips)
        dst_ip = np.random.choice(internal_ips)
        dst_port = {'http': 80, 'ssh': 22, 'ftp': 21, 'ftp_data': 20, 'telnet': 23, 'smtp': 25, 'dns': 53, 'pop_3': 110, 'private': 445, 'other': 8080}.get(service, 80)
        
        records.append({
            'duration': duration,
            'protocol_type': protocol,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'land': land,
            'wrong_fragment': wrong_fragment,
            'urgent': urgent,
            'hot': hot,
            'num_failed_logins': num_failed_logins,
            'logged_in': logged_in,
            'num_compromised': num_compromised,
            'root_shell': root_shell,
            'su_attempted': su_attempted,
            'num_root': num_root,
            'num_file_creations': num_file_creations,
            'num_shells': num_shells,
            'num_access_files': num_access_files,
            'num_outbound_cmds': num_outbound_cmds,
            'is_host_login': is_host_login,
            'is_guest_login': is_guest_login,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': round(serror_rate, 4),
            'srv_serror_rate': round(srv_serror_rate, 4),
            'rerror_rate': round(rerror_rate, 4),
            'srv_rerror_rate': round(srv_rerror_rate, 4),
            'same_srv_rate': round(same_srv_rate, 4),
            'diff_srv_rate': round(diff_srv_rate, 4),
            'srv_diff_host_rate': round(srv_diff_host_rate, 4),
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': round(dst_host_same_srv_rate, 4),
            'dst_host_diff_srv_rate': round(dst_host_diff_srv_rate, 4),
            'dst_host_same_src_port_rate': round(dst_host_same_src_port_rate, 4),
            'dst_host_srv_diff_host_rate': round(dst_host_srv_diff_host_rate, 4),
            'dst_host_serror_rate': round(dst_host_serror_rate, 4),
            'dst_host_srv_serror_rate': round(dst_host_srv_serror_rate, 4),
            'dst_host_rerror_rate': round(dst_host_rerror_rate, 4),
            'dst_host_srv_rerror_rate': round(dst_host_srv_rerror_rate, 4),
            'attack_type': attack_sub,
            'attack_category': cat,
            'label': 0 if cat == 'Normal' else 1,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'dst_port': dst_port
        })
        
    df = pd.DataFrame(records)
    return df


def generate_cicids2017_sample(n_samples=2500, random_state=42):
    """
    Generates an authentic benchmark CICIDS2017 dataset format with realistic bidirectional
    flow metrics, multi-class attack patterns, and network traffic dynamics.
    """
    np.random.seed(random_state)
    labels = np.random.choice(
        ['BENIGN', 'DDoS', 'PortScan', 'Botnet', 'Infiltration', 'BruteForce'], 
        size=n_samples, 
        p=[0.55, 0.20, 0.12, 0.05, 0.05, 0.03]
    )
    
    rows = []
    for lbl in labels:
        if lbl == 'BENIGN':
            dur = int(np.random.exponential(scale=60000) + 5000)
            tfwd = int(np.random.randint(5, 40))
            tbwd = int(np.random.randint(6, 45))
            fbytes = float(np.random.uniform(2000, 45000))
            fpkts = float(np.random.uniform(40, 350))
            pktlen = float(np.random.uniform(250, 850))
            fiat = float(np.random.exponential(scale=8000))
            biat = float(np.random.exponential(scale=7000))
            act = float(np.random.uniform(1000, 5000))
            idle = float(np.random.uniform(10000, 60000))
        elif lbl == 'DDoS':
            dur = int(np.random.randint(5000, 45000))
            tfwd = int(np.random.randint(250, 1200))
            tbwd = int(np.random.randint(0, 4))
            fbytes = float(np.random.uniform(300000, 1500000))
            fpkts = float(np.random.uniform(8000, 45000))
            pktlen = float(np.random.uniform(64, 130))
            fiat = float(np.random.uniform(5, 80))
            biat = 0.0
            act = float(np.random.uniform(5000, 25000))
            idle = float(np.random.uniform(0, 300))
        elif lbl == 'PortScan':
            dur = int(np.random.randint(200, 2500))
            tfwd = int(np.random.randint(1, 3))
            tbwd = int(np.random.randint(0, 2))
            fbytes = float(np.random.uniform(80, 1500))
            fpkts = float(np.random.uniform(600, 3500))
            pktlen = float(np.random.uniform(40, 60))
            fiat = float(np.random.uniform(20, 200))
            biat = 0.0
            act = float(np.random.uniform(0, 200))
            idle = float(np.random.uniform(0, 500))
        elif lbl == 'Botnet':
            dur = int(np.random.randint(350000, 950000))
            tfwd = int(np.random.randint(12, 35))
            tbwd = int(np.random.randint(10, 30))
            fbytes = float(np.random.uniform(600, 4500))
            fpkts = float(np.random.uniform(15, 60))
            pktlen = float(np.random.uniform(120, 260))
            fiat = float(np.random.uniform(70000, 180000))
            biat = float(np.random.uniform(60000, 160000))
            act = float(np.random.uniform(500, 1800))
            idle = float(np.random.uniform(250000, 800000))
        elif lbl == 'Infiltration':
            dur = int(np.random.randint(120000, 500000))
            tfwd = int(np.random.randint(60, 180))
            tbwd = int(np.random.randint(90, 300))
            fbytes = float(np.random.uniform(90000, 400000))
            fpkts = float(np.random.uniform(250, 1200))
            pktlen = float(np.random.uniform(950, 1450))
            fiat = float(np.random.uniform(2000, 12000))
            biat = float(np.random.uniform(1500, 10000))
            act = float(np.random.uniform(12000, 45000))
            idle = float(np.random.uniform(8000, 25000))
        else:
            dur = int(np.random.randint(25000, 100000))
            tfwd = int(np.random.randint(35, 95))
            tbwd = int(np.random.randint(30, 85))
            fbytes = float(np.random.uniform(18000, 75000))
            fpkts = float(np.random.uniform(900, 2800))
            pktlen = float(np.random.uniform(180, 420))
            fiat = float(np.random.uniform(600, 2500))
            biat = float(np.random.uniform(500, 2200))
            act = float(np.random.uniform(2500, 9000))
            idle = float(np.random.uniform(1500, 7000))
            
        rows.append({
            'Flow Duration': dur,
            'Total Fwd Packets': tfwd,
            'Total Backward Packets': tbwd,
            'Flow Bytes/s': round(fbytes, 2),
            'Flow Packets/s': round(fpkts, 2),
            'Packet Length Mean': round(pktlen, 2),
            'Fwd IAT Mean': round(fiat, 2),
            'Bwd IAT Mean': round(biat, 2),
            'Active Mean': round(act, 2),
            'Idle Mean': round(idle, 2),
            'Label': lbl
        })
    df_cic = pd.DataFrame(rows)
    return df_cic


def ensure_datasets():
    """Ensures sample datasets exist in the data/ directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    
    nsl_path = os.path.join(base_dir, 'nsl_kdd_sample.csv')
    cic_path = os.path.join(base_dir, 'cicids2017_sample.csv')
    
    if not os.path.exists(nsl_path):
        print("Generating realistic NSL-KDD benchmark dataset...")
        df_nsl = generate_nsl_kdd_dataset(n_samples=5000)
        df_nsl.to_csv(nsl_path, index=False)
        print(f"Saved NSL-KDD to {nsl_path}")
        
    if not os.path.exists(cic_path):
        print("Generating CICIDS2017 benchmark dataset...")
        df_cic = generate_cicids2017_sample(n_samples=2500)
        df_cic.to_csv(cic_path, index=False)
        print(f"Saved CICIDS2017 to {cic_path}")


if __name__ == '__main__':
    ensure_datasets()
