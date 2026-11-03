"""
CyberShield AI - UI Styling & Visual Components
Enterprise Cybersecurity Design System: Dark Navy Sidebar, Clean Dashboard, and Cyber Blue Accents.
Ensures maximum contrast, readability, and consistency across all pages for faculty evaluation.
"""

import streamlit as st


def get_current_theme():
    """Returns standard cyber theme."""
    return "cyber"


def apply_cyber_theme():
    """Injects high-contrast enterprise cybersecurity theme CSS."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global Reset & Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }

        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1400px !important;
        }

        /* ==========================================================================
           1. ENTERPRISE SIDEBAR STYLING (Dark Navy Theme)
           ========================================================================== */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarContent"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #0B132B !important;
            background: #0B132B !important;
            border-right: 1px solid #1E293B !important;
        }

        section[data-testid="stSidebar"] * {
            color: #E2E8F0;
        }

        /* Sidebar Navigation Container */
        [data-testid="stSidebarNav"] {
            background: transparent !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }

        [data-testid="stSidebarNavItems"] {
            gap: 4px !important;
        }

        /* ==========================================================================
           PRESENTATION NAVIGATION CONFIGURATION:
           Temporarily hide 'Deep Learning' and 'Model Comparison' pages from the
           visible sidebar navigation as requested for presentation.
           The underlying files (pages/5_🧠_Deep_Learning.py and pages/9_⚖️_Model_Comparison.py),
           models, and backend logic remain 100% active and intact.
           'Anomaly Detection' remains visible as requested.
           ========================================================================== */
        /* Deep Learning */
        [data-testid="stSidebarNav"] li:has(a[href*="Deep_Learning"]),
        [data-testid="stSidebarNav"] li:has(a[href*="deep_learning"]),
        [data-testid="stSidebarNav"] li:has(a[href*="Deep-Learning"]),
        [data-testid="stSidebarNav"] li:has(a[href*="deep-learning"]),
        [data-testid="stSidebarNav"] li:has(a[href*="Deep%20Learning"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Deep_Learning"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="deep_learning"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Deep-Learning"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="deep-learning"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Deep%20Learning"]),
        [data-testid="stSidebarNav"] a[href*="Deep_Learning"],
        [data-testid="stSidebarNav"] a[href*="deep_learning"],
        [data-testid="stSidebarNav"] a[href*="Deep-Learning"],
        [data-testid="stSidebarNav"] a[href*="deep-learning"],
        [data-testid="stSidebarNav"] a[href*="Deep%20Learning"],

        /* Model Comparison */
        [data-testid="stSidebarNav"] li:has(a[href*="Model_Comparison"]),
        [data-testid="stSidebarNav"] li:has(a[href*="model_comparison"]),
        [data-testid="stSidebarNav"] li:has(a[href*="Model-Comparison"]),
        [data-testid="stSidebarNav"] li:has(a[href*="model-comparison"]),
        [data-testid="stSidebarNav"] li:has(a[href*="Model%20Comparison"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Model_Comparison"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="model_comparison"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Model-Comparison"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="model-comparison"]),
        [data-testid="stSidebarNav"] [data-testid="stSidebarNavLinkContainer"]:has(a[href*="Model%20Comparison"]),
        [data-testid="stSidebarNav"] a[href*="Model_Comparison"],
        [data-testid="stSidebarNav"] a[href*="model_comparison"],
        [data-testid="stSidebarNav"] a[href*="Model-Comparison"],
        [data-testid="stSidebarNav"] a[href*="model-comparison"],
        [data-testid="stSidebarNav"] a[href*="Model%20Comparison"] {
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Inactive Navigation Links */
        [data-testid="stSidebarNavLink"],
        [data-testid="stSidebarNav"] a {
            background-color: transparent !important;
            border-radius: 6px !important;
            margin: 2px 6px !important;
            padding: 8px 12px !important;
            transition: all 0.15s ease-in-out !important;
            color: #CBD5E1 !important;
            text-decoration: none !important;
            display: flex !important;
            align-items: center !important;
        }

        [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebarNavLinkText"],
        [data-testid="stSidebarNav"] a span {
            color: #CBD5E1 !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em !important;
        }

        [data-testid="stSidebarNavLink"] svg,
        [data-testid="stSidebarNav"] a svg {
            fill: #94A3B8 !important;
            stroke: #94A3B8 !important;
        }

        /* Hover Navigation Links */
        [data-testid="stSidebarNavLink"]:hover,
        [data-testid="stSidebarNav"] a:hover {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
        }

        [data-testid="stSidebarNavLink"]:hover span,
        [data-testid="stSidebarNav"] a:hover span {
            color: #FFFFFF !important;
        }

        [data-testid="stSidebarNavLink"]:hover svg,
        [data-testid="stSidebarNav"] a:hover svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        /* Active Navigation Link */
        [data-testid="stSidebarNavLink"][aria-current="page"],
        [data-testid="stSidebarNav"] a[aria-current="page"],
        li:has([aria-current="page"]) a {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4) !important;
        }

        [data-testid="stSidebarNavLink"][aria-current="page"] span,
        [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebarNavLink"][aria-current="page"] svg,
        [data-testid="stSidebarNav"] a[aria-current="page"] svg {
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }

        /* Sidebar Collapse Control Button */
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] button {
            color: #CBD5E1 !important;
            background: transparent !important;
        }

        button[data-testid="stSidebarCollapseButton"] svg,
        [data-testid="collapsedControl"] button svg {
            fill: #CBD5E1 !important;
            stroke: #CBD5E1 !important;
        }

        /* Sidebar Headings, Text & Badges */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #CBD5E1 !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: #1E293B !important;
            margin: 12px 0 !important;
        }

        section[data-testid="stSidebar"] .stAlert {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            color: #F8FAFC !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] {
            background-color: #1E293B !important;
            border: 1px solid #334155 !important;
            color: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] .stButton>button {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: 1px solid #3B82F6 !important;
            border-radius: 6px !important;
        }

        /* ==========================================================================
           2. MAIN DASHBOARD CONTENT & HIGH-CONTRAST TYPOGRAPHY
           ========================================================================== */
        .main h1 {
            font-size: 26px !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            color: #0F172A !important;
            margin-bottom: 6px !important;
        }

        .main h2 {
            font-size: 20px !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em !important;
            color: #1E293B !important;
            margin-top: 18px !important;
            margin-bottom: 12px !important;
        }

        .main h3 {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #1E293B !important;
            margin-top: 14px !important;
            margin-bottom: 8px !important;
        }

        .main h4, .main h5, .main h6 {
            font-size: 14px !important;
            font-weight: 600 !important;
            color: #334155 !important;
        }

        .main p, .main span, .main label, .main div {
            font-size: 14px;
            font-weight: 400;
        }

        /* Top Navbar Container */
        .top-navbar {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .navbar-logo {
            width: 32px;
            height: 32px;
            background: #2563EB;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 16px;
        }

        .navbar-title {
            font-size: 16px;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.01em;
        }

        .navbar-subtitle {
            font-size: 12px;
            color: #475569;
            font-weight: 400;
        }

        /* Metric KPI Cards */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            position: relative;
        }

        .kpi-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
        }

        .kpi-card.accent-blue { border-left: 4px solid #2563EB !important; }
        .kpi-card.accent-green { border-left: 4px solid #16A34A !important; }
        .kpi-card.accent-orange { border-left: 4px solid #F59E0B !important; }
        .kpi-card.accent-red { border-left: 4px solid #DC2626 !important; }
        .kpi-card.accent-purple { border-left: 4px solid #7C3AED !important; }

        .kpi-label {
            font-size: 13px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }

        .kpi-subtext {
            font-size: 12px;
            color: #64748B;
            margin-top: 4px;
            font-weight: 400;
        }

        /* Standard Cards & Containers */
        .content-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }

        /* Clean Page Header */
        .page-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }

        .page-header-title {
            font-size: 22px;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 4px;
            letter-spacing: -0.01em;
        }

        .page-header-subtitle {
            font-size: 13px;
            color: #475569;
            line-height: 1.5;
        }

        /* High-Contrast Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .badge-normal { background: #F0FDF4; color: #166534; border: 1px solid #86EFAC; }
        .badge-low { background: #EFF6FF; color: #1E40AF; border: 1px solid #93C5FD; }
        .badge-medium { background: #FFFBEB; color: #92400E; border: 1px solid #FCD34D; }
        .badge-high { background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }
        .badge-critical { background: #7F1D1D; color: #FEF2F2; border: 1px solid #991B1B; }

        /* Status Dot */
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }
        .status-dot.green { background-color: #16A34A; }
        .status-dot.blue { background-color: #2563EB; }
        .status-dot.red { background-color: #DC2626; }

        /* ==========================================================================
           3. BUTTONS & INTERACTIVE ELEMENTS
           ========================================================================== */
        .main .stButton>button {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #1D4ED8 !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            transition: all 0.15s ease !important;
        }

        .main .stButton>button:hover {
            background-color: #1D4ED8 !important;
            border-color: #1E40AF !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 4px 0 rgba(37, 99, 235, 0.2) !important;
        }

        .stDownloadButton>button {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #334155 !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            transition: all 0.15s ease !important;
        }

        .stDownloadButton>button:hover {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
        }

        /* Form Labels & Dropdowns */
        .main .stSelectbox label, 
        .main .stMultiSelect label, 
        .main .stSlider label, 
        .main .stRadio label, 
        .main .stTextInput label,
        .main .stFileUploader label {
            color: #0F172A !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            margin-bottom: 4px !important;
        }

        .main [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 6px !important;
            color: #0F172A !important;
        }

        .main [data-baseweb="select"] * {
            color: #0F172A !important;
        }

        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        }

        li[role="option"] {
            color: #0F172A !important;
            background-color: #FFFFFF !important;
        }

        li[role="option"]:hover, li[aria-selected="true"] {
            background-color: #EFF6FF !important;
            color: #2563EB !important;
            font-weight: 600 !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 1.5px dashed #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 14px !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #EFF6FF !important;
            color: #2563EB !important;
            border: 1px solid #BFDBFE !important;
            font-weight: 600 !important;
        }

        /* Dataframe & Tables */
        .stDataFrame {
            border: 1px solid #E2E8F0 !important;
            border-radius: 6px !important;
            overflow: hidden !important;
            background-color: #FFFFFF !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            border-bottom: 2px solid #E2E8F0 !important;
            background-color: transparent !important;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0 !important;
            padding: 9px 18px !important;
            color: #475569 !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            background: transparent !important;
            border: none !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #0F172A !important;
        }

        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 2.5px solid #2563EB !important;
            font-weight: 700 !important;
            background-color: #EFF6FF !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
        }

        [data-testid="stExpander"] summary {
            color: #0F172A !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }

        /* Alerts & Banners */
        .main .stAlert {
            border-radius: 8px !important;
            padding: 12px 16px !important;
            font-size: 13.5px !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_top_navbar(active_page_name="Dashboard"):
    """
    Renders clean, minimal top navbar inspired by Cisco/MS Defender dashboard.
    """
    dataset_name = st.session_state.get("dataset_name", "NSL-KDD Benchmark")
    
    col_nav1, col_nav2 = st.columns([3, 1])
    
    with col_nav1:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <div style="width: 32px; height: 32px; background: #2563EB; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 16px;">
                    🛡️
                </div>
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: #0F172A; line-height: 1.1;">
                        CyberShield AI
                    </div>
                    <div style="font-size: 12px; color: #475569;">
                        Machine Learning Based Cyber Threat Detection System
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_nav2:
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 4px;">
                <span class="badge badge-low"><span class="status-dot green"></span>{dataset_name}</span>
            </div>
        """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, status_tag: str = "Active Telemetry"):
    """
    Renders a standard clean academic page header.
    """
    render_top_navbar()
    st.markdown(f"""
        <div class="page-header">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
                <div>
                    <div class="page-header-title">{title}</div>
                    <div class="page-header-subtitle">{subtitle}</div>
                </div>
                <span class="badge badge-normal"><span class="status-dot green"></span>{status_tag}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", card_type: str = "blue"):
    """
    Renders a clean, professional KPI metric card.
    card_type: 'blue', 'green', 'orange', 'red', 'purple'
    """
    accent_class = f"accent-{card_type}" if card_type in ['blue', 'green', 'orange', 'red', 'purple'] else "accent-blue"
    st.markdown(f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-subtext">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def get_severity_badge_html(severity: str) -> str:
    """Returns styled academic badge for threat severity levels."""
    sev = str(severity).lower()
    if sev in ['critical', 'crit']:
        return '<span class="badge badge-critical">Critical</span>'
    elif sev in ['high']:
        return '<span class="badge badge-high">High</span>'
    elif sev in ['medium', 'med']:
        return '<span class="badge badge-medium">Medium</span>'
    elif sev in ['low']:
        return '<span class="badge badge-low">Low</span>'
    else:
        return '<span class="badge badge-normal">Normal</span>'


def get_plotly_theme_layout():
    """
    Returns consistent, clean Plotly layout configuration for charts.
    """
    bg_paper = "#FFFFFF"
    bg_plot = "#FAFAFA"
    grid_color = "#E2E8F0"
    text_color = "#0F172A"
    
    return dict(
        paper_bgcolor=bg_paper,
        plot_bgcolor=bg_plot,
        font=dict(family="Inter, sans-serif", size=12, color=text_color),
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        )
    )
