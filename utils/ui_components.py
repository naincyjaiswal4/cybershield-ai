"""
CyberShield AI - UI Styling & Visual Components
Clean, academic, and professional dashboard interface inspired by Google Analytics,
Microsoft Defender, and modern SaaS analytics platforms.
"""

import streamlit as st


def get_current_theme():
    """Returns current theme ('light' by default, or 'dark')."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    return st.session_state.theme_mode


def apply_cyber_theme():
    """Injects clean, academic, professional design system CSS."""
    theme = get_current_theme()
    is_dark = theme == "dark"

    # Color Tokens
    bg_color = "#0F172A" if is_dark else "#F8FAFC"
    card_bg = "#1E293B" if is_dark else "#FFFFFF"
    card_hover = "#243247" if is_dark else "#FFFFFF"
    border_color = "#334155" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    sidebar_bg = "#0B1120" if is_dark else "#FFFFFF"
    sidebar_border = "#1E293B" if is_dark else "#E2E8F0"
    tab_active_bg = "#1E293B" if is_dark else "#F1F5F9"
    shadow_card = "0 1px 3px 0 rgba(0, 0, 0, 0.25)" if is_dark else "0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Reset & Typography */
        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: {bg_color} !important;
            color: {text_primary} !important;
        }}

        /* Heading Hierarchy */
        h1 {{
            font-size: 28px !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: {text_primary} !important;
            margin-bottom: 6px !important;
        }}

        h2 {{
            font-size: 20px !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            color: {text_primary} !important;
            margin-top: 18px !important;
            margin-bottom: 12px !important;
        }}

        h3 {{
            font-size: 16px !important;
            font-weight: 600 !important;
            color: {text_primary} !important;
            margin-top: 14px !important;
            margin-bottom: 8px !important;
        }}

        h4, h5, h6 {{
            font-size: 14px !important;
            font-weight: 600 !important;
            color: {text_primary} !important;
        }}

        p, span, label, div {{
            font-size: 14px;
            font-weight: 400;
        }}

        /* Top Navbar Container */
        .top-navbar {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: {shadow_card};
        }}

        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .navbar-logo {{
            width: 28px;
            height: 28px;
            background: #2563EB;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 15px;
        }}

        .navbar-title {{
            font-size: 16px;
            font-weight: 700;
            color: {text_primary};
            letter-spacing: -0.01em;
        }}

        .navbar-subtitle {{
            font-size: 12px;
            color: {text_secondary};
            font-weight: 400;
        }}

        .navbar-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid {sidebar_border} !important;
            padding-top: 1rem !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown {{
            color: {text_primary};
        }}

        /* Clean Metric KPI Cards */
        .kpi-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: {shadow_card};
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            position: relative;
        }}

        .kpi-card:hover {{
            border-color: #CBD5E1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}

        .kpi-card.accent-blue {{ border-left: 4px solid #2563EB; }}
        .kpi-card.accent-green {{ border-left: 4px solid #16A34A; }}
        .kpi-card.accent-orange {{ border-left: 4px solid #F59E0B; }}
        .kpi-card.accent-red {{ border-left: 4px solid #DC2626; }}
        .kpi-card.accent-purple {{ border-left: 4px solid #7C3AED; }}

        .kpi-label {{
            font-size: 13px;
            font-weight: 500;
            color: {text_secondary};
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: {text_primary};
            line-height: 1.2;
            letter-spacing: -0.02em;
        }}

        .kpi-subtext {{
            font-size: 12px;
            color: {text_secondary};
            margin-top: 4px;
            font-weight: 400;
        }}

        /* Standard Cards & Containers */
        .content-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: {shadow_card};
        }}

        /* Clean Header */
        .page-header {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 20px;
            box-shadow: {shadow_card};
        }}

        .page-header-title {{
            font-size: 22px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 4px;
            letter-spacing: -0.01em;
        }}

        .page-header-subtitle {{
            font-size: 13px;
            color: {text_secondary};
            line-height: 1.4;
        }}

        /* Academic Status Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}

        .badge-normal {{ background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0; }}
        .badge-low {{ background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; }}
        .badge-medium {{ background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }}
        .badge-high {{ background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }}
        .badge-critical {{ background: #7F1D1D; color: #FEE2E2; border: 1px solid #991B1B; }}

        /* Status Dot */
        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }}
        .status-dot.green {{ background-color: #16A34A; }}
        .status-dot.blue {{ background-color: #2563EB; }}
        .status-dot.red {{ background-color: #DC2626; }}

        /* Streamlit Form Elements */
        .stButton>button {{
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #1D4ED8 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 7px 16px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            transition: background-color 0.15s ease !important;
        }}

        .stButton>button:hover {{
            background-color: #1D4ED8 !important;
            border-color: #1E40AF !important;
        }}

        .stDownloadButton>button {{
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #334155 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }}

        /* Dataframe styling */
        .stDataFrame {{
            border: 1px solid {border_color} !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {border_color};
            background-color: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            color: {text_secondary};
            font-size: 13px;
            font-weight: 500;
            background: transparent;
            border: none;
        }}

        .stTabs [aria-selected="true"] {{
            color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
            font-weight: 600 !important;
            background-color: {tab_active_bg} !important;
        }}
        </style>
    """, unsafe_allow_html=True)


def render_top_navbar(active_page_name="Dashboard"):
    """
    Renders clean, minimal top navbar inspired by Cisco/MS Defender dashboard.
    """
    dataset_name = st.session_state.get("dataset_name", "NSL-KDD Benchmark")
    
    col_nav1, col_nav2 = st.columns([3, 1])
    
    with col_nav1:
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <div style="width: 32px; height: 32px; background: #2563EB; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 16px;">
                    🛡️
                </div>
                <div>
                    <div style="font-size: 18px; font-weight: 700; color: inherit; line-height: 1.1;">
                        CyberShield AI
                    </div>
                    <div style="font-size: 12px; color: #64748B;">
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
    theme = get_current_theme()
    is_dark = theme == "dark"
    
    bg_paper = "#1E293B" if is_dark else "#FFFFFF"
    bg_plot = "#1E293B" if is_dark else "#FAFAFA"
    grid_color = "#334155" if is_dark else "#F1F5F9"
    text_color = "#F8FAFC" if is_dark else "#1E293B"
    
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
