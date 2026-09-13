"""
ResearchLens AI - Main Streamlit Application.
AI-Powered Research Paper Analysis, Comparison & Research Gap Detection.
"""

import os
import re
import streamlit as st

from src.utils import (
    validate_uploaded_file,
    render_badge,
    sanitize_filename,
    render_bullet_cards_html,
    markdown_to_html_spans,
    clean_html
)
from src.document_parser import extract_text

from src.ai_service import (
    analyze_paper,
    compare_papers,
    detect_research_gaps,
    resolve_api_key,
    resolve_gemini_api_key,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS
)
from src.comparator import (
    MIN_COMPARE_PAPERS,
    MAX_COMPARE_PAPERS
)
from src.report import (
    generate_single_paper_report_txt,
    generate_comparison_report_txt,
    generate_gap_report_txt,
    get_download_filename
)
from datetime import datetime
from src.database import init_db, get_user_by_id
from src.auth import (
    authenticate_user,
    register_user,
    reset_password,
    change_password,
    update_profile_name
)

# Page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="ResearchLens AI - Academic Paper Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite database
init_db()

# Custom Styling for Modern Academic Research UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Canvas Background for Entire Streamlit App */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1100px circle at 15% 10%, rgba(14, 165, 233, 0.12) 0%, transparent 60%),
            radial-gradient(1000px circle at 85% 15%, rgba(99, 102, 241, 0.13) 0%, transparent 55%),
            radial-gradient(900px circle at 50% 85%, rgba(168, 85, 247, 0.09) 0%, transparent 50%),
            radial-gradient(800px circle at 90% 75%, rgba(16, 185, 129, 0.07) 0%, transparent 45%),
            linear-gradient(180deg, #080C14 0%, #0D121F 50%, #080C14 100%) !important;
        background-attachment: fixed !important;
    }

    /* Micro-dot academic grid texture overlay */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: radial-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.85;
    }

    /* Top header bar */
    [data-testid="stHeader"] {
        background: rgba(8, 12, 20, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13, 18, 31, 0.95) 0%, rgba(8, 12, 20, 0.98) 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
    }

    /* =======================================================
       NAVBAR / SIDEBAR NAVIGATION - BOLD & BIGGER
       ======================================================= */
    
    /* Sidebar Brand Title */
    [data-testid="stSidebar"] h2 {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #38BDF8 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-top: 4px !important;
        margin-bottom: 6px !important;
    }

    /* Welcome User Message */
    .sidebar-user-welcome {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.01em !important;
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }

    /* Navigation Section Header */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label,
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #38BDF8 !important;
        margin-bottom: 12px !important;
        display: block !important;
    }

    /* Radio Group Container */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 10px !important;
    }

    /* Individual Navigation Items (Pills / Cards) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        margin-bottom: 6px !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Hover effect on Nav Items */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(56, 189, 248, 0.16) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25) !important;
    }

    /* Navigation Item Typography - Bold & Bigger */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] span {
        font-size: 1.22rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.01em !important;
        margin: 0 !important;
        line-height: 1.45 !important;
    }

    /* Selected / Active Navigation Item */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.3) 0%, rgba(99, 102, 241, 0.28) 100%) !important;
        border: 2px solid #38BDF8 !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateX(3px) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {
        color: #38BDF8 !important;
        font-weight: 800 !important;
    }

    /* Radio Indicator Dot Scaling */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] input + div,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
        transform: scale(1.2) !important;
        margin-right: 6px !important;
    }

    /* Page Hero Banners */
    .page-hero {
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .page-hero::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), rgba(129, 140, 248, 0.8), transparent);
    }

    .hero-home {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.90) 0%, rgba(30, 41, 59, 0.75) 50%, rgba(15, 23, 42, 0.92) 100%),
                    radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.20), transparent 50%),
                    radial-gradient(circle at 90% 80%, rgba(129, 140, 248, 0.18), transparent 50%);
        border-color: rgba(56, 189, 248, 0.25);
    }

    .hero-single {
        background: linear-gradient(135deg, rgba(13, 27, 30, 0.92) 0%, rgba(15, 35, 45, 0.78) 50%, rgba(10, 20, 30, 0.94) 100%),
                    radial-gradient(circle at 15% 25%, rgba(20, 184, 166, 0.22), transparent 50%),
                    radial-gradient(circle at 85% 75%, rgba(56, 189, 248, 0.18), transparent 50%);
        border-color: rgba(45, 212, 191, 0.3);
    }

    .hero-compare {
        background: linear-gradient(135deg, rgba(20, 24, 48, 0.92) 0%, rgba(28, 34, 68, 0.78) 50%, rgba(15, 18, 38, 0.94) 100%),
                    radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.24), transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(147, 51, 234, 0.20), transparent 50%);
        border-color: rgba(129, 140, 248, 0.3);
    }

    .hero-gap {
        background: linear-gradient(135deg, rgba(38, 24, 20, 0.92) 0%, rgba(45, 26, 46, 0.78) 50%, rgba(25, 16, 30, 0.94) 100%),
                    radial-gradient(circle at 15% 20%, rgba(245, 158, 11, 0.22), transparent 50%),
                    radial-gradient(circle at 85% 85%, rgba(192, 132, 252, 0.20), transparent 50%);
        border-color: rgba(251, 191, 36, 0.3);
    }

    .hero-about {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(20, 30, 48, 0.80) 50%, rgba(15, 23, 42, 0.96) 100%),
                    radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.16), transparent 50%),
                    radial-gradient(circle at 85% 80%, rgba(16, 185, 129, 0.16), transparent 50%);
        border-color: rgba(56, 189, 248, 0.25);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.55;
        margin-bottom: 14px;
        max-width: 880px;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-top: 4px;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #E2E8F0;
        backdrop-filter: blur(8px);
    }

    /* Frosted Glassmorphism Academic Cards */
    .academic-card {
        background: rgba(17, 24, 39, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(14px) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .academic-card:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 8px 30px -4px rgba(14, 165, 233, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-2px);
    }
    .card-header {
        font-size: 1.12rem;
        font-weight: 700;
        color: #F8FAFC !important;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-body {
        font-size: 0.95rem;
        color: #CBD5E1 !important;
        line-height: 1.65;
    }

    /* Modern Eye-Catching Bullet Cards */
    .bullet-card-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .bullet-card-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease !important;
    }
    .bullet-card-item:hover {
        transform: translateX(4px) !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.45) !important;
    }
    .bullet-icon-box {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        min-width: 32px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .bullet-content-box {
        flex: 1;
        font-size: 0.96rem;
        line-height: 1.65;
        color: #E2E8F0;
    }
    .bullet-content-box strong {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .bullet-content-box em {
        color: #CBD5E1 !important;
        font-style: italic !important;
    }
    .bullet-content-box u {
        color: #38BDF8 !important;
        text-decoration: underline !important;
        text-decoration-color: #38BDF8 !important;
        text-underline-offset: 4px !important;
        font-weight: 700 !important;
    }

    /* Executive Takeaway Cards Grid */
    .takeaway-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 14px;
        margin: 16px 0 24px 0;
    }
    .takeaway-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 18px -4px rgba(0, 0, 0, 0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .takeaway-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px -4px rgba(56, 189, 248, 0.25) !important;
    }
    .takeaway-title {
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .takeaway-desc {
        font-size: 0.92rem;
        line-height: 1.55;
        color: #E2E8F0;
    }

    /* Metric Containers */
    .metric-container {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        text-align: center !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease;
    }
    .metric-container:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-1px);
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #38BDF8 !important;
        letter-spacing: -0.01em;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 6px;
        font-weight: 600;
    }

    /* Disclaimers & Alerts */
    .disclaimer-box {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.10) 0%, rgba(99, 102, 241, 0.08) 100%) !important;
        border-left: 4px solid #38BDF8 !important;
        border-top: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
        padding: 14px 18px !important;
        border-radius: 0 12px 12px 0 !important;
        font-size: 0.9rem !important;
        color: #CBD5E1 !important;
        margin-top: 16px !important;
        backdrop-filter: blur(8px) !important;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.10) 0%, rgba(217, 119, 6, 0.08) 100%) !important;
        border-left: 4px solid #F59E0B !important;
        border-top: 1px solid rgba(245, 158, 11, 0.18) !important;
        border-right: 1px solid rgba(245, 158, 11, 0.18) !important;
        border-bottom: 1px solid rgba(245, 158, 11, 0.18) !important;
        padding: 14px 18px !important;
        border-radius: 0 12px 12px 0 !important;
        font-size: 0.9rem !important;
        color: #FDE68A !important;
        margin-top: 12px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.65) !important;
        background: rgba(15, 23, 42, 0.8) !important;
    }

    /* Tabs styling - Bolder & Bigger */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.75) !important;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #38BDF8 !important;
        font-weight: 800 !important;
        font-size: 1.08rem !important;
        border: 1.5px solid rgba(56, 189, 248, 0.45) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 50%, #4F46E5 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }

    /* Authentication Form & Card Styling */
    .auth-card {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 18px !important;
        padding: 24px 28px 18px 28px !important;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.65), 0 0 30px -5px rgba(14, 165, 233, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        margin-bottom: 20px !important;
    }
    .auth-brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 12px;
    }
    .auth-header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .auth-header-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    /* Profile Section Specific Styles */
    .hero-profile {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.94) 0%, rgba(30, 27, 75, 0.82) 50%, rgba(15, 23, 42, 0.96) 100%),
                    radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.22), transparent 50%),
                    radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.22), transparent 50%);
        border-color: rgba(168, 85, 247, 0.3);
    }
    .profile-header-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 28px;
        display: flex;
        align-items: center;
        gap: 22px;
        margin-bottom: 24px;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .profile-avatar {
        width: 72px;
        height: 72px;
        min-width: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0284C7 0%, #6366F1 50%, #A855F7 100%);
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45);
        border: 2px solid rgba(255, 255, 255, 0.25);
    }
    .profile-info-name {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .profile-info-meta {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .profile-info-email {
        color: #38BDF8;
        font-weight: 500;
    }
    .profile-detail-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .profile-detail-row:last-child {
        border-bottom: none;
    }
    .profile-detail-label {
        font-size: 0.90rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .profile-detail-value {
        font-size: 0.92rem;
        color: #F1F5F9;
        font-weight: 600;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)



# Initialize session states
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "auth_view" not in st.session_state:
    st.session_state["auth_view"] = "login"
if "auth_flash_msg" not in st.session_state:
    st.session_state["auth_flash_msg"] = None

if "single_analysis" not in st.session_state:
    st.session_state["single_analysis"] = None
if "single_paper_doc" not in st.session_state:
    st.session_state["single_paper_doc"] = None
if "compare_results" not in st.session_state:
    st.session_state["compare_results"] = None
if "gap_results" not in st.session_state:
    st.session_state["gap_results"] = None
if "comparison_papers" not in st.session_state:
    st.session_state["comparison_papers"] = []


# ==========================================
# AUTHENTICATION GATE
# ==========================================
if not st.session_state.get("authenticated", False):
    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
    _, auth_col, _ = st.columns([1, 1.7, 1])

    with auth_col:
        auth_view = st.session_state.get("auth_view", "login")

        if st.session_state.get("auth_flash_msg"):
            st.success(f"✅ {st.session_state['auth_flash_msg']}")
            st.session_state["auth_flash_msg"] = None

        if auth_view == "login":
            st.markdown("""
            <div class="auth-card">
                <div class="auth-brand-badge">🔬 ResearchLens AI</div>
                <div class="auth-header-title">Welcome back</div>
                <div class="auth-header-subtitle">Sign in to your account to access your research intelligence dashboard.</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email Address", placeholder="name@university.edu", key="login_email_input")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password_input")
                login_submit = st.form_submit_button("Login", type="primary", width="stretch")

                if login_submit:
                    ok, msg, user_data = authenticate_user(login_email, login_pwd)
                    if ok and user_data:
                        st.session_state["authenticated"] = True
                        st.session_state["user_id"] = user_data["id"]
                        st.session_state["user_name"] = user_data["name"]
                        st.session_state["user_email"] = user_data["email"]
                        st.session_state["auth_flash_msg"] = None
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                if st.button("Forgot Password?", type="secondary", width="stretch"):
                    st.session_state["auth_view"] = "forgot_password"
                    st.rerun()
            with col_sub2:
                if st.button("Don't have an account? Sign Up", type="secondary", width="stretch"):
                    st.session_state["auth_view"] = "signup"
                    st.rerun()

        elif auth_view == "signup":
            st.markdown("""
            <div class="auth-card">
                <div class="auth-brand-badge">🔬 ResearchLens AI</div>
                <div class="auth-header-title">Create Account</div>
                <div class="auth-header-subtitle">Join ResearchLens AI for deep technical paper extraction and gap analysis.</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("signup_form", clear_on_submit=False):
                su_name = st.text_input("Full Name", placeholder="Dr. Jane Doe", key="su_name_input")
                su_email = st.text_input("Email Address", placeholder="name@university.edu", key="su_email_input")
                su_pwd = st.text_input("Password", type="password", placeholder="Minimum 8 characters", key="su_pwd_input")
                su_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="su_confirm_input")
                su_submit = st.form_submit_button("Create Account", type="primary", width="stretch")

                if su_submit:
                    ok, msg = register_user(su_name, su_email, su_pwd, su_confirm)
                    if ok:
                        st.session_state["auth_flash_msg"] = msg
                        st.session_state["auth_view"] = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            if st.button("Already have an account? Log In", type="secondary", width="stretch"):
                st.session_state["auth_view"] = "login"
                st.rerun()

        elif auth_view == "forgot_password":
            st.markdown("""
            <div class="auth-card">
                <div class="auth-brand-badge">🔬 ResearchLens AI</div>
                <div class="auth-header-title">Reset Password</div>
                <div class="auth-header-subtitle">Enter your registered email and choose a new password.</div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("forgot_password_form", clear_on_submit=False):
                fp_email = st.text_input("Registered Email Address", placeholder="name@university.edu", key="fp_email_input")
                fp_new_pwd = st.text_input("New Password", type="password", placeholder="Minimum 8 characters", key="fp_new_pwd_input")
                fp_confirm_pwd = st.text_input("Confirm New Password", type="password", placeholder="Repeat your new password", key="fp_confirm_pwd_input")
                fp_submit = st.form_submit_button("Reset Password", type="primary", width="stretch")

                if fp_submit:
                    ok, msg = reset_password(fp_email, fp_new_pwd, fp_confirm_pwd)
                    if ok:
                        st.session_state["auth_flash_msg"] = msg
                        st.session_state["auth_view"] = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            if st.button("Back to Login", type="secondary", width="stretch"):
                st.session_state["auth_view"] = "login"
                st.rerun()

    st.stop()


# ==========================================
# SIDEBAR (AUTHENTICATED)
# ==========================================
with st.sidebar:
    st.markdown("## 🔬 ResearchLens AI")
    user_display_name = st.session_state.get("user_name") or "Researcher"
    st.markdown(f'<div class="sidebar-user-welcome">Welcome, {user_display_name} 👋</div>', unsafe_allow_html=True)
    user_email_display = st.session_state.get("user_email") or ""
    if user_email_display:
        st.caption(f"`{user_email_display}`")
    st.divider()

    # Navigation menu
    nav_selection = st.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "📄 Single Paper Analysis",
            "🔍 Paper Comparison",
            "🧠 Research Gaps",
            "👤 Profile",
            "ℹ️ About"
        ],
        index=0
    )

    st.divider()

    # AI Mode & Configuration
    st.markdown("### ⚙️ AI Engine & Configuration")

    ai_mode = st.radio(
        "AI Mode",
        options=["Gemini AI", "Demo Mode"],
        index=0,
        help="Gemini AI connects to the backend Google Gemini intelligence service. Demo Mode provides sample data without calling the backend service."
    )

    is_demo_mode = (ai_mode == "Demo Mode")

    if not is_demo_mode:
        model_choice = st.selectbox(
            "Gemini Model",
            options=["gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="gemini-3.6-flash delivers high speed and deep academic extraction on Google Gemini."
        )
    else:
        st.info("💡 **Demo Mode Active**\nPredefined academic research samples will be used without calling the backend service.")
        model_choice = "Demo Mode (Mock)"

    st.divider()

    # Platform Capabilities Info
    st.markdown("### 📋 Platform Specs")
    st.markdown("""
    - **Engine**: Google Gemini AI (Backend)
    - **Formats**: PDF, DOCX, TXT
    - **Max Comparison**: 2–5 Papers
    - **Context Strategy**: Head/Tail Preservation
    - **Fidelity**: Strict Evidence-Aware Mode
    """)

    st.caption("v1.0.0 • Production MVP")

    st.divider()

    # Logout Button
    if st.button("🚪 Logout", width="stretch", help="Sign out of your ResearchLens AI account"):
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["user_name"] = None
        st.session_state["user_email"] = None
        st.session_state["auth_view"] = "login"
        st.session_state["auth_flash_msg"] = "You have been logged out successfully."
        st.rerun()


# ==========================================
# PAGE 1: HOME
# ==========================================
if nav_selection == "🏠 Home":
    st.markdown("""
    <div class="page-hero hero-home">
        <div class="hero-title">🔬 ResearchLens AI</div>
        <div class="hero-subtitle">
            Understand papers deeply. Compare research methodologies across dimensions. Discover recurring research gaps and formulation opportunities.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">⚡ Evidence-Aware AI</span>
            <span class="hero-pill">🛡️ Zero Hallucination Guarantee</span>
            <span class="hero-pill">📊 Multi-Study Synthesis</span>
            <span class="hero-pill">🧠 Frontier Gap Detection</span>
            <span class="hero-pill">📄 PDF • DOCX • TXT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ResearchLens AI is a rigorous, evidence-grounded research acceleration platform built for students,
    researchers, and academic faculty. It automates deep technical extraction, cross-study comparative synthesis,
    and systematic research gap detection without hallucinations.
    """)

    st.markdown("### 🌟 Core Capabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="academic-card">
            <div class="card-header">📄 Paper Analysis</div>
            <div class="card-body">
                Upload a single research paper (PDF, DOCX, TXT) to extract 15+ standardized academic dimensions,
                including research problem, methodology, dataset pipeline, evaluation metrics, results, and explicit limitations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <div class="card-header">🧠 Research Gap Detection</div>
            <div class="card-body">
                Detect recurring blind spots, unaddressed edge cases, missing baselines, and methodological limitations
                across multiple papers. Generates a prioritized (HIGH / MEDIUM / LOW) frontier gap matrix.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="academic-card">
            <div class="card-header">🔍 Paper Comparison</div>
            <div class="card-body">
                Perform side-by-side comparative synthesis across 2 to 5 research papers. Automatically identify
                shared objectives, divergent architectures, empirical contradictions, and joint opportunities.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <div class="card-header">💡 Research Recommendations & Planner</div>
            <div class="card-body">
                Formulate concrete research directions and structured experimental protocols (hypothesis,
                independent/dependent variables, baselines, metrics, and risks) grounded directly in paper limitations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🔄 How It Works")
    step_cols = st.columns(6)
    steps = [
        ("1. Upload", "Submit PDF, DOCX, or TXT manuscripts"),
        ("2. Extract", "Parse text, clean structure, compute metrics"),
        ("3. Analyze", "AI extracts 15+ dimensions under zero-hallucination rules"),
        ("4. Compare", "Synthesize multi-paper comparative matrix"),
        ("5. Detect Gaps", "Discover recurring limitations & blind spots"),
        ("6. Plan", "Generate testable experimental designs")
    ]
    for col, (step_title, step_desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">{step_title}</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 6px;">{step_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic Standard Guarantee:</strong> ResearchLens AI strictly separates verbatim paper evidence
        from AI technical interpretation, actionable recommendations, and items requiring verification.
        Missing data points are explicitly flagged as <em>"Not clearly stated in the paper."</em>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: SINGLE PAPER ANALYSIS
# ==========================================
elif nav_selection == "📄 Single Paper Analysis":
    st.markdown("""
    <div class="page-hero hero-single">
        <div class="hero-title">📄 Single Paper Academic Analysis</div>
        <div class="hero-subtitle">
            Upload an individual research manuscript for comprehensive, evidence-aware structural extraction across 15+ standardized academic dimensions.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">🎯 Intelligent Head/Tail Retention</span>
            <span class="hero-pill">🔬 Structured JSON Schema</span>
            <span class="hero-pill">🛡️ Strict Evidence Separation</span>
            <span class="hero-pill">🧪 Testable Experimental Protocols</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Research Paper",
        type=["pdf", "docx", "txt"],
        help="Upload an academic paper in PDF, DOCX, or TXT format (max 25MB)."
    )

    if uploaded_file is not None:
        file_val = validate_uploaded_file(uploaded_file, uploaded_file.name)
        if not file_val["valid"]:
            st.error(f"❌ {file_val['error']}")
        else:
            with st.spinner("Extracting document structure and cleaning text..."):
                doc_data = extract_text(uploaded_file, uploaded_file.name)
                st.session_state["single_paper_doc"] = doc_data

            if doc_data.get("is_scanned"):
                st.warning(f"⚠️ {doc_data.get('error')}")
            elif doc_data.get("error"):
                st.error(f"❌ {doc_data.get('error')}")
            else:
                # Display Document Metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.markdown(clean_html(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['file_type']}</div>
                        <div class="metric-label">Document Type</div>
                    </div>
                    """), unsafe_allow_html=True)
                with mcol2:
                    st.markdown(clean_html(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['word_count']:,}</div>
                        <div class="metric-label">Word Count</div>
                    </div>
                    """), unsafe_allow_html=True)
                with mcol3:
                    st.markdown(clean_html(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['char_count']:,}</div>
                        <div class="metric-label">Character Count</div>
                    </div>
                    """), unsafe_allow_html=True)
                with mcol4:
                    st.markdown(clean_html(f"""
                    <div class="metric-container">
                        <div class="metric-value">~{doc_data['token_count']:,}</div>
                        <div class="metric-label">Est. Tokens</div>
                    </div>
                    """), unsafe_allow_html=True)

                st.write("")

                if st.button("🚀 Analyze Paper", type="primary", width="stretch"):
                    try:
                        with st.status("Performing evidence-aware analysis...", expanded=True) as status:
                            st.write("Checking token boundaries & intelligent head/tail retention...")
                            if is_demo_mode:
                                st.write("Loading pre-computed academic analysis sample (Demo Mode)...")
                            else:
                                st.write("Synthesizing deep academic extraction and evidence taxonomy...")
                            analysis_result = analyze_paper(
                                text=doc_data["text"],
                                filename=doc_data["filename"],
                                model=model_choice,
                                demo_mode=is_demo_mode
                            )
                            st.session_state["single_analysis"] = analysis_result
                            status.update(label="Analysis completed successfully!", state="complete", expanded=False)
                    except MissingAPIKeyError:
                        st.error("⚠️ AI intelligence service is not configured in backend secrets. Please contact administrator.")
                    except AnalysisError as ae:
                        st.error(f"❌ {str(ae)}")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {str(e)}")

    # Display Analysis Results
    if st.session_state.get("single_analysis") is not None:
        analysis = st.session_state["single_analysis"]
        meta = analysis.get("_metadata", {})

        st.divider()

        if meta.get("is_demo"):
            st.warning("⚠️ **Demo Mode — Results are sample data and are not generated by Gemini.**")

        # Metadata & Confidence Banner
        conf = analysis.get("confidence", {})
        overall_conf = conf.get("overall", "Medium")
        conf_reason = conf.get("reason", "N/A")

        banner_col1, banner_col2 = st.columns([3, 1])
        with banner_col1:
            st.markdown(f"### Analysis: **{meta.get('filename', 'Document')}**")
            if meta.get("was_truncated"):
                st.info(
                    f"ℹ️ Content was truncated from {meta.get('original_word_count'):,} words "
                    f"to {meta.get('analyzed_word_count'):,} words to fit model context. "
                    f"Introduction, methodology, findings, and conclusion were preserved."
                )
        with banner_col2:
            st.markdown(f"""
            <div style="text-align: right; margin-top: 10px;">
                {render_badge(f"Confidence: {overall_conf}", badge_type=overall_conf.lower())}
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">{conf_reason}</div>
            </div>
            """, unsafe_allow_html=True)

        # Quick-Glance Executive Takeaways Card Grid
        st.markdown("### ⚡ Executive Synthesis at a Glance")

        def _clean_preview_item(val, default="Not clearly stated in the paper."):
            if isinstance(val, list) and val:
                raw = str(val[0])
            elif isinstance(val, str) and val.strip():
                lines = [l.strip() for l in val.split("\n") if l.strip()]
                raw = lines[0] if lines else val
            else:
                raw = default
            clean = re.sub(r"^[•\-\*\d\.]+\s*", "", raw).strip()
            return clean or default

        takeaway_innovation = _clean_preview_item(analysis.get("executive_summary"))
        takeaway_arch = _clean_preview_item(analysis.get("methodology"))
        takeaway_result = _clean_preview_item(analysis.get("results"))
        takeaway_limit = _clean_preview_item(analysis.get("limitations") or analysis.get("research_gaps"))

        st.markdown(clean_html(f"""
        <div class="takeaway-grid">
            <div class="takeaway-card" style="border-left: 4px solid #38BDF8;">
                <div class="takeaway-title" style="color: #38BDF8;">💡 Core Innovation</div>
                <div class="takeaway-desc">{markdown_to_html_spans(takeaway_innovation)}</div>
            </div>
            <div class="takeaway-card" style="border-left: 4px solid #A855F7;">
                <div class="takeaway-title" style="color: #A855F7;">⚙️ System Architecture</div>
                <div class="takeaway-desc">{markdown_to_html_spans(takeaway_arch)}</div>
            </div>
            <div class="takeaway-card" style="border-left: 4px solid #34D399;">
                <div class="takeaway-title" style="color: #34D399;">📊 Empirical Finding</div>
                <div class="takeaway-desc">{markdown_to_html_spans(takeaway_result)}</div>
            </div>
            <div class="takeaway-card" style="border-left: 4px solid #F87171;">
                <div class="takeaway-title" style="color: #F87171;">⚠️ Primary Bottleneck / Gap</div>
                <div class="takeaway-desc">{markdown_to_html_spans(takeaway_limit)}</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # 12 Academic Analysis Tabs
        tabs = st.tabs([
            "1. Overview",
            "2. Problem & Objectives",
            "3. Methodology",
            "4. Dataset & Preprocessing",
            "5. Results",
            "6. Conclusion",
            "7. Strengths & Limitations",
            "8. Research Gaps",
            "9. Recommendations",
            "10. Future Work",
            "11. Experiments",
            "12. Evidence & Confidence"
        ])

        with tabs[0]:
            st.markdown("#### Executive Summary")
            st.markdown(render_bullet_cards_html(analysis.get('executive_summary'), default_icon="📑", accent="cyan"), unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("#### Research Problem & Core Bottleneck")
            st.markdown(render_bullet_cards_html(analysis.get('research_problem'), default_icon="🎯", accent="amber"), unsafe_allow_html=True)

            st.markdown("#### Objectives / Research Questions / Hypotheses")
            objs = analysis.get("objectives", [])
            st.markdown(render_bullet_cards_html(objs, default_icon="📌", accent="indigo"), unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("#### Technical Methodology & Architecture")
            st.markdown(render_bullet_cards_html(analysis.get('methodology'), default_icon="⚙️", accent="purple"), unsafe_allow_html=True)

        with tabs[3]:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Datasets Used")
                st.markdown(render_bullet_cards_html(analysis.get('dataset'), default_icon="🗄️", accent="indigo"), unsafe_allow_html=True)
            with col_d2:
                st.markdown("#### Preprocessing & Pipeline")
                st.markdown(render_bullet_cards_html(analysis.get('preprocessing'), default_icon="🔄", accent="cyan"), unsafe_allow_html=True)

        with tabs[4]:
            st.markdown("#### Evaluation Protocol & Baselines")
            st.markdown(render_bullet_cards_html(analysis.get('evaluation'), default_icon="📏", accent="purple"), unsafe_allow_html=True)

            st.markdown("#### Key Results & Quantitative Findings")
            results = analysis.get("results", [])
            st.markdown(render_bullet_cards_html(results, default_icon="📈", accent="emerald"), unsafe_allow_html=True)

        with tabs[5]:
            st.markdown("#### Authors' Conclusion")
            st.markdown(render_bullet_cards_html(analysis.get('conclusion'), default_icon="🏁", accent="cyan"), unsafe_allow_html=True)

        with tabs[6]:
            col_s, col_l = st.columns(2)
            with col_s:
                st.markdown("#### Demonstrated Strengths")
                strengths = analysis.get("strengths", [])
                st.markdown(render_bullet_cards_html(strengths, default_icon="✅", accent="emerald"), unsafe_allow_html=True)
            with col_l:
                st.markdown("#### Explicit Limitations")
                limits = analysis.get("limitations", [])
                st.markdown(render_bullet_cards_html(limits, default_icon="⚠️", accent="rose"), unsafe_allow_html=True)

        with tabs[7]:
            st.markdown("#### Potential Research Gaps (Identified from Paper)")
            gaps = analysis.get("research_gaps", [])
            st.markdown(render_bullet_cards_html(gaps, default_icon="🔍", accent="amber"), unsafe_allow_html=True)

        with tabs[8]:
            st.markdown("#### Actionable Research Recommendations")
            suggs = analysis.get("suggestions", [])
            st.markdown(render_bullet_cards_html(suggs, default_icon="💡", accent="amber"), unsafe_allow_html=True)

        with tabs[9]:
            st.markdown("#### Future Work Directions")
            fw = analysis.get("future_work", [])
            st.markdown(render_bullet_cards_html(fw, default_icon="🚀", accent="purple"), unsafe_allow_html=True)

        with tabs[10]:
            st.markdown("#### Evidence-Aware Experimental Designs")
            exps = analysis.get("experiments", [])
            if exps:
                for idx, exp in enumerate(exps, 1):
                    with st.expander(f"🧪 Experiment {idx}: {exp.get('title', 'Proposed Experiment')}", expanded=(idx == 1)):
                        st.markdown(f"**Hypothesis:** {markdown_to_html_spans(exp.get('hypothesis', 'N/A'))}", unsafe_allow_html=True)
                        st.markdown(f"**Independent Variables:** `{exp.get('independent_variables', 'N/A')}`")
                        st.markdown(f"**Dependent Variables:** `{exp.get('dependent_variables', 'N/A')}`")
                        st.markdown(f"**Recommended Dataset:** {markdown_to_html_spans(exp.get('dataset', 'N/A'))}", unsafe_allow_html=True)
                        st.markdown(f"**Baseline Models:** {markdown_to_html_spans(exp.get('baseline', 'N/A'))}", unsafe_allow_html=True)
                        st.markdown(f"**Evaluation Metrics:** `{exp.get('evaluation_metrics', 'N/A')}`")
                        st.markdown(f"**Expected Outcome:** {markdown_to_html_spans(exp.get('expected_outcome', 'N/A'))}", unsafe_allow_html=True)
                        st.markdown(f"**Risks & Limitations:** {markdown_to_html_spans(exp.get('risks_and_limitations', 'N/A'))}", unsafe_allow_html=True)
            else:
                st.write("No structured experiment plans synthesized.")

        with tabs[11]:
            st.markdown("#### Rigorous Taxonomy & Evidence Breakdown")
            st.caption("Ensures zero hallucination by categorizing extracted information.")
            eb = analysis.get("evidence_breakdown", {})

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown(f"### {render_badge('', 'evidence')} Paper Evidence", unsafe_allow_html=True)
                st.markdown(render_bullet_cards_html(eb.get("paper_evidence", []), default_icon="📄", accent="cyan"), unsafe_allow_html=True)

                st.write("")
                st.markdown(f"### {render_badge('', 'recommendation')} AI Recommendations", unsafe_allow_html=True)
                st.markdown(render_bullet_cards_html(eb.get("ai_recommendations", []), default_icon="💡", accent="emerald"), unsafe_allow_html=True)

            with col_e2:
                st.markdown(f"### {render_badge('', 'interpretation')} Technical Interpretation", unsafe_allow_html=True)
                st.markdown(render_bullet_cards_html(eb.get("interpretations", []), default_icon="🔎", accent="amber"), unsafe_allow_html=True)

                st.write("")
                st.markdown(f"### {render_badge('', 'verification')} Needs Verification", unsafe_allow_html=True)
                st.markdown(render_bullet_cards_html(eb.get("needs_verification", []), default_icon="⚠️", accent="rose"), unsafe_allow_html=True)

        st.divider()

        # Download Report
        report_txt = generate_single_paper_report_txt(analysis)
        download_name = get_download_filename("Analysis", meta.get("filename", "paper"))

        st.download_button(
            label="📥 Download Academic Analysis Report (TXT)",
            data=report_txt,
            file_name=download_name,
            mime="text/plain",
            type="secondary",
            width="stretch"
        )


# ==========================================
# PAGE 3: PAPER COMPARISON
# ==========================================
elif nav_selection == "🔍 Paper Comparison":
    st.markdown("""
    <div class="page-hero hero-compare">
        <div class="hero-title">🔍 Multi-Paper Comparative Synthesis</div>
        <div class="hero-subtitle">
            Side-by-side comparative analysis across 2 to 5 research papers to isolate shared goals, divergent architectures, and empirical contradictions.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">⚖️ 2–5 Paper Synthesis</span>
            <span class="hero-pill">⚡ Contradiction Discovery</span>
            <span class="hero-pill">📐 Dimension-by-Dimension Matrix</span>
            <span class="hero-pill">🚀 Joint Opportunity Formulation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    compare_files = st.file_uploader(
        "Upload 2 to 5 Research Papers",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload between 2 and 5 manuscripts in PDF, DOCX, or TXT format."
    )

    if compare_files:
        if len(compare_files) < MIN_COMPARE_PAPERS:
            st.warning(f"Please upload at least {MIN_COMPARE_PAPERS} papers to enable comparison (Currently uploaded: {len(compare_files)}).")
        elif len(compare_files) > MAX_COMPARE_PAPERS:
            st.error(f"Maximum limit is {MAX_COMPARE_PAPERS} papers. You uploaded {len(compare_files)}. Please remove some files.")
        else:
            st.success(f"✅ {len(compare_files)} papers uploaded and ready for extraction.")

            # Extract text from all files
            parsed_papers = []
            cols = st.columns(len(compare_files))
            for idx, (f, c) in enumerate(zip(compare_files, cols)):
                with c:
                    doc = extract_text(f, f.name)
                    parsed_papers.append(doc)
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-weight: 700; font-size: 0.9rem; word-break: break-all; color: #F8FAFC;">{doc['filename']}</div>
                        <div class="metric-value" style="font-size: 1.1rem; margin-top: 4px;">{doc['word_count']:,}</div>
                        <div class="metric-label">Words ({doc['file_type']})</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state["comparison_papers"] = parsed_papers

            st.write("")
            if st.button("⚖️ Compare Papers", type="primary", width="stretch"):
                try:
                    with st.status("Executing comparative synthesis...", expanded=True) as status:
                        st.write("Checking for duplicates and content overlap...")
                        if is_demo_mode:
                            st.write("Synthesizing multi-paper comparison (Demo Mode)...")
                        else:
                            st.write("Synthesizing cross-paper dimensions, similarities, and differences...")
                        comp_result = compare_papers(
                            papers=parsed_papers,
                            model=model_choice,
                            demo_mode=is_demo_mode
                        )
                        st.session_state["compare_results"] = comp_result
                        status.update(label="Comparison completed successfully!", state="complete", expanded=False)
                except MissingAPIKeyError:
                    st.error("⚠️ AI intelligence service is not configured in backend secrets. Please contact administrator.")
                except AnalysisError as ae:
                    st.error(f"❌ {str(ae)}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    if st.session_state.get("compare_results") is not None:
        comp = st.session_state["compare_results"]
        st.divider()

        if comp.get("is_demo"):
            st.warning("⚠️ **Demo Mode — Results are sample data and are not generated by Gemini.**")

        # Duplication warnings if any
        if comp.get("duplicate_warnings"):
            for warn in comp["duplicate_warnings"]:
                st.warning(f"⚠️ {warn}")

        st.markdown("### 📊 Dimension-by-Dimension Comparison")

        comp_table = comp.get("comparison_table", [])
        for row in comp_table:
            dim_name = row.get("dimension", "Dimension")
            papers_dict = row.get("papers", {})
            with st.expander(f"📌 Dimension: {dim_name}", expanded=True):
                p_cols = st.columns(len(papers_dict))
                for (p_name, p_val), p_col in zip(papers_dict.items(), p_cols):
                    with p_col:
                        st.markdown(f"**{p_name}**")
                        st.markdown(clean_html(f"""
                        <div class="academic-card">
                            <div class="card-body">{markdown_to_html_spans(p_val)}</div>
                        </div>
                        """), unsafe_allow_html=True)

        st.markdown("### 🔍 Comparative Synthesis Findings")
        col_s, col_d = st.columns(2)

        with col_s:
            st.markdown("#### Similarities Across Papers")
            st.markdown(render_bullet_cards_html(comp.get("similarities", []), default_icon="•", accent="cyan"), unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Contradictions & Inconsistencies")
            st.markdown(render_bullet_cards_html(comp.get("contradictions_inconsistencies", []), default_icon="⚡", accent="rose"), unsafe_allow_html=True)

        with col_d:
            st.markdown("#### Key Differences")
            st.markdown(render_bullet_cards_html(comp.get("differences", []), default_icon="•", accent="amber"), unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Unresolved Areas Across All Papers")
            st.markdown(render_bullet_cards_html(comp.get("unresolved_areas", []), default_icon="❓", accent="purple"), unsafe_allow_html=True)

        st.markdown("#### Combined Research Opportunity")
        st.markdown(clean_html(f"""
        <div class="academic-card" style="border-left: 4px solid #0284C7;">
            <div class="card-header">🚀 Synthesized Opportunity</div>
            <div class="card-body">{markdown_to_html_spans(comp.get('combined_research_opportunity', ''))}</div>
        </div>
        """), unsafe_allow_html=True)

        st.divider()

        # Download Comparison Report
        comp_report_txt = generate_comparison_report_txt(comp)
        download_name = get_download_filename("Comparison", "multi_paper")

        st.download_button(
            label="📥 Download Comparative Synthesis Report (TXT)",
            data=comp_report_txt,
            file_name=download_name,
            mime="text/plain",
            type="secondary",
            width="stretch"
        )



# ==========================================
# PAGE 4: RESEARCH GAPS
# ==========================================
elif nav_selection == "🧠 Research Gaps":
    st.markdown("""
    <div class="page-hero hero-gap">
        <div class="hero-title">🧠 Evidence-Aware Research Gap Detection</div>
        <div class="hero-subtitle">
            Synthesize recurring limitations, missing evaluations, and unaddressed scientific frontiers across multiple papers to formulate high-impact research directions.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">🎯 High / Medium / Low Prioritization</span>
            <span class="hero-pill">🧩 7-Dimension Visual Matrix</span>
            <span class="hero-pill">🛡️ Academic Humility Standard</span>
            <span class="hero-pill">🧪 Testable Experimental Protocols</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Academic Humility Rule:</strong> ResearchLens AI never claims a gap is globally novel.
        Outputs are strictly labeled as: <em>"Potential research gap based on the uploaded papers.
        Broader literature review is required to establish novelty."</em>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    gap_files = st.file_uploader(
        "Upload 2 to 5 Papers for Gap Detection",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="gap_uploader",
        help="Upload 2–5 papers to detect cross-paper research gaps."
    )

    if gap_files:
        if len(gap_files) < MIN_COMPARE_PAPERS:
            st.warning(f"Please upload at least {MIN_COMPARE_PAPERS} papers to analyze research gaps (Currently uploaded: {len(gap_files)}).")
        elif len(gap_files) > MAX_COMPARE_PAPERS:
            st.error(f"Maximum limit is {MAX_COMPARE_PAPERS} papers. You uploaded {len(gap_files)}.")
        else:
            parsed_papers = [extract_text(f, f.name) for f in gap_files]
            st.success(f"✅ {len(gap_files)} papers loaded.")

            if st.button("🔬 Detect Cross-Paper Research Gaps", type="primary", width="stretch"):
                try:
                    with st.status("Analyzing research frontier & detecting gaps...", expanded=True) as status:
                        st.write("Evaluating cross-paper datasets, models, and evaluation methods...")
                        if is_demo_mode:
                            st.write("Detecting prioritized research gaps (Demo Mode)...")
                        else:
                            st.write("Synthesizing frontier limitations and prioritizing research gaps...")
                        gap_res = detect_research_gaps(
                            papers=parsed_papers,
                            model=model_choice,
                            demo_mode=is_demo_mode
                        )
                        st.session_state["gap_results"] = gap_res
                        status.update(label="Gap detection complete!", state="complete", expanded=False)
                except MissingAPIKeyError:
                    st.error("⚠️ AI intelligence service is not configured in backend secrets. Please contact administrator.")
                except AnalysisError as ae:
                    st.error(f"❌ {str(ae)}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    if st.session_state.get("gap_results") is not None:
        gaps_data = st.session_state["gap_results"]
        st.divider()

        if gaps_data.get("is_demo"):
            st.warning("⚠️ **Demo Mode — Results are sample data and are not generated by Gemini.**")

        st.markdown("### 🌐 Collective Research Frontier Synthesis")
        st.markdown(render_bullet_cards_html(gaps_data.get('synthesis_summary', ''), default_icon="🌐", accent="purple"), unsafe_allow_html=True)

        # Visual Research Gap Matrix
        st.markdown("### 🧩 Visual Research Gap Matrix")
        st.caption("Standardized evaluation across 7 critical research dimensions.")

        matrix_items = gaps_data.get("matrix", [])
        for item in matrix_items:
            area_name = item.get("area", "Area")
            ratings = item.get("paper_ratings", {})
            potential_gap = item.get("potential_gap", "N/A")

            with st.expander(f"📐 Area: {area_name.upper()} — {potential_gap[:90]}...", expanded=False):
                r_cols = st.columns(len(ratings) + 1)
                for (p_name, p_val), col in zip(ratings.items(), r_cols[:-1]):
                    with col:
                        st.markdown(f"**{p_name}**")
                        st.info(p_val)
                with r_cols[-1]:
                    st.markdown("**Potential Gap Identified**")
                    st.warning(potential_gap)

        # Prioritized Gaps
        st.markdown("### 🎯 Prioritized Research Gaps")
        all_gaps = gaps_data.get("gaps", [])
        if all_gaps:
            for idx, g in enumerate(all_gaps, 1):
                prio = g.get("priority", "MEDIUM").upper()
                badge_html = render_badge(prio, prio.lower())
                conf_badge = render_badge(f"Confidence: {g.get('confidence', 'Medium')}", "default")
                verif_badge = render_badge("Needs Verification", "verification") if g.get("needs_verification") else ""

                st.markdown(clean_html(f"""
                <div class="academic-card" style="border-left: 5px solid {'#EF4444' if prio=='HIGH' else '#F59E0B' if prio=='MEDIUM' else '#0284C7'};">
                    <div class="card-header">
                        <span>Gap #{idx}: {g.get('title')}</span>
                        <div style="margin-left: auto;">
                            {badge_html} {conf_badge} {verif_badge}
                        </div>
                    </div>
                    <div class="card-body">
                        <p><strong>Description:</strong> {markdown_to_html_spans(g.get('description', ''))}</p>
                        <p><strong>Evidence from Uploaded Papers:</strong> {markdown_to_html_spans(g.get('evidence_from_papers', ''))}</p>
                        <p><strong>Why It Matters:</strong> {markdown_to_html_spans(g.get('why_it_matters', ''))}</p>
                        <p><strong>Suggested Research Direction:</strong> {markdown_to_html_spans(g.get('suggested_direction', ''))}</p>
                    </div>
                </div>
                """), unsafe_allow_html=True)
        else:
            st.info("No prioritized gaps generated.")

        st.divider()

        # Download Gap Report
        gap_report_txt = generate_gap_report_txt(gaps_data)
        download_name = get_download_filename("Gaps", "cross_paper")

        st.download_button(
            label="📥 Download Research Gap Report (TXT)",
            data=gap_report_txt,
            file_name=download_name,
            mime="text/plain",
            type="secondary",
            width="stretch"
        )


# ==========================================
# PAGE: RESEARCHER PROFILE
# ==========================================
elif nav_selection == "👤 Profile":
    user_id = st.session_state.get("user_id")
    user_record = get_user_by_id(user_id) if user_id else None

    current_name = (user_record.get("name") if user_record else None) or st.session_state.get("user_name") or "Researcher"
    current_email = (user_record.get("email") if user_record else None) or st.session_state.get("user_email") or "Not registered"
    created_at_raw = (user_record.get("created_at") if user_record else None) or ""

    formatted_date = "Active Session"
    member_since = "Member"
    if created_at_raw:
        try:
            dt = datetime.fromisoformat(created_at_raw)
            formatted_date = dt.strftime("%B %d, %Y (%I:%M %p UTC)")
            member_since = dt.strftime("%b %d, %Y")
        except Exception:
            formatted_date = created_at_raw
            member_since = created_at_raw[:10]

    # Generate initials for avatar
    name_parts = current_name.strip().split()
    if len(name_parts) >= 2:
        initials = (name_parts[0][0] + name_parts[-1][0]).upper()
    elif len(name_parts) == 1 and len(name_parts[0]) > 0:
        initials = name_parts[0][:2].upper()
    else:
        initials = "RL"

    user_code = f"#RL-{user_id:05d}" if user_id else "#RL-00000"

    # Hero Banner
    st.markdown(f"""
    <div class="page-hero hero-profile">
        <div class="hero-title">👤 Researcher Profile</div>
        <div class="hero-subtitle">
            Manage your academic credentials, security parameters, system preferences, and active analytical workspace.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">🛡️ Verified Academic Account</span>
            <span class="hero-pill">🔒 Salted Bcrypt Security</span>
            <span class="hero-pill">⚡ Active Authenticated Session</span>
            <span class="hero-pill">🆔 {user_code}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Avatar & Identity Banner Card
    st.markdown(f"""
    <div class="profile-header-card">
        <div class="profile-avatar">{initials}</div>
        <div>
            <div class="profile-info-name">{current_name}</div>
            <div class="profile-info-meta">
                <span class="profile-info-email">📧 {current_email}</span>
                <span>•</span>
                <span>🎓 Academic Scholar</span>
                <span>•</span>
                <span>📅 Joined {member_since}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row (4 columns)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #38BDF8; font-size: 1.4rem;">{user_code}</div>
            <div class="metric-label">Researcher ID</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value" style="color: #10B981; font-size: 1.4rem;">Verified</div>
            <div class="metric-label">Account Status</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #A855F7; font-size: 1.4rem;">{member_since}</div>
            <div class="metric-label">Member Since</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #F59E0B; font-size: 1.4rem;">{model_choice}</div>
            <div class="metric-label">Active AI Engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Main 2-column layout for Profile Details & Security
    prof_left_col, prof_right_col = st.columns([1.35, 1.15], gap="large")

    with prof_left_col:
        # Academic Identity Card
        st.markdown(f"""
        <div class="academic-card">
            <div class="card-header">🎓 Academic Identity & Credentials</div>
            <div class="card-body">
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Full Display Name</span>
                    <span class="profile-detail-value">{current_name}</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Registered Email</span>
                    <span class="profile-detail-value">{current_email}</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Academic Role</span>
                    <span class="profile-detail-value">Academic Researcher / AI Scholar</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Account Created</span>
                    <span class="profile-detail-value">{formatted_date}</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Identity Encryption</span>
                    <span class="profile-detail-value" style="color: #10B981;">Bcrypt 12-Rounds Salted</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Update Display Name Form
        with st.expander("✏️ Edit Display Name", expanded=False):
            with st.form("update_name_form", clear_on_submit=False):
                new_display_name = st.text_input(
                    "New Display Name",
                    value=current_name,
                    placeholder="Enter full name",
                    key="new_display_name_input"
                )
                update_name_submit = st.form_submit_button("Save Name Changes", type="primary")

                if update_name_submit:
                    if user_id:
                        ok, msg = update_profile_name(user_id, new_display_name)
                        if ok:
                            st.session_state["user_name"] = new_display_name.strip()
                            st.success(f"✅ {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.error("❌ Cannot update name for unverified user ID.")

        # Active Session & Workspace State Card
        has_single = st.session_state.get("single_analysis") is not None
        single_title = st.session_state.get("single_paper_doc", {}).get("filename", "Active Paper") if has_single else "None"
        comp_count = len(st.session_state.get("comparison_papers", []))
        has_gaps = st.session_state.get("gap_results") is not None
        single_status_label = f"🟢 {single_title}" if has_single else "⚪ Idle (No paper loaded)"
        gaps_status_label = "🟢 Yes (Gaps Detected)" if has_gaps else "⚪ Not yet run"

        st.markdown(f"""
        <div class="academic-card">
            <div class="card-header">🔬 Workspace & Session State</div>
            <div class="card-body">
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Single Analysis in Memory</span>
                    <span class="profile-detail-value">{single_status_label}</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Comparison Papers Staged</span>
                    <span class="profile-detail-value">{comp_count} paper(s)</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Research Gaps Synthesized</span>
                    <span class="profile-detail-value">{gaps_status_label}</span>
                </div>
                <div class="profile-detail-row">
                    <span class="profile-detail-label">Memory Retention Protocol</span>
                    <span class="profile-detail-value" style="color: #38BDF8;">Strict Ephemeral (RAM only)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with prof_right_col:
        # Account & Session Controls
        st.markdown("""
        <div class="academic-card" style="border-color: rgba(239, 68, 68, 0.25);">
            <div class="card-header" style="color: #F87171 !important;">🚪 Session & Account Controls</div>
            <div class="card-body">
                Terminate your authenticated session. All ephemeral paper extractions, comparisons, and cached analyses will be cleared from memory.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Sign Out of ResearchLens AI", type="secondary", width="stretch", key="profile_logout_btn"):
            st.session_state["authenticated"] = False
            st.session_state["user_id"] = None
            st.session_state["user_name"] = None
            st.session_state["user_email"] = None
            st.session_state["auth_view"] = "login"
            st.session_state["auth_flash_msg"] = "You have been signed out successfully."
            st.rerun()


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif nav_selection == "ℹ️ About":
    st.markdown("""
    <div class="page-hero hero-about">
        <div class="hero-title">ℹ️ About ResearchLens AI</div>
        <div class="hero-subtitle">
            Architectural framework, evidence-aware design principles, ephemeral security protocols, and academic disclaimers.
        </div>
        <div class="hero-badges">
            <span class="hero-pill">🛡️ Strict Context Fidelity</span>
            <span class="hero-pill">🔒 Ephemeral Memory Processing</span>
            <span class="hero-pill">🔍 Zero-Hallucination Taxonomy</span>
            <span class="hero-pill">🤝 Academic Decision Support</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="academic-card">
        <div class="card-header">🎯 Purpose & Mission</div>
        <div class="card-body">
            Modern researchers and students face exponential literature growth, making it difficult to keep track of
            technical nuances, contradictory findings, and genuine research gaps. <strong>ResearchLens AI</strong> was engineered
            to act as an evidence-aware pair-researcher that extracts exact technical dimensions, compares competing architectures,
            and isolates high-value research opportunities without hallucinating facts.
        </div>
    </div>

    <div class="academic-card">
        <div class="card-header">🛡️ Evidence-Aware Design Principles</div>
        <div class="card-body">
            <p><strong>1. Strict Context Fidelity:</strong> The system will never invent benchmarks, fabricate citations, or claim a dataset exists if omitted in the manuscript.</p>
            <p><strong>2. Explicit Uncertainty Handling:</strong> Omissions are cleanly surfaced as <em>"Not clearly stated in the paper."</em></p>
            <p><strong>3. Academic Humility:</strong> Rather than declaring global novelty, the system specifies <em>"Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."</em></p>
            <p><strong>4. Rigorous Evidence Taxonomy:</strong></p>
            <ul>
                <li>📄 <strong>Paper Evidence:</strong> Direct facts reported by the authors.</li>
                <li>🔎 <strong>Interpretation:</strong> Logical conclusions synthesized from reported facts.</li>
                <li>💡 <strong>AI Recommendation:</strong> Proposed next steps and research formulations.</li>
                <li>⚠️ <strong>Needs Verification:</strong> Assumptions requiring external literature validation.</li>
            </ul>
        </div>
    </div>

    <div class="academic-card">
        <div class="card-header">🔒 Security, Privacy & Temporary Processing</div>
        <div class="card-body">
            <ul>
                <li>Uploaded papers are held strictly in ephemeral memory for the duration of the analysis session.</li>
                <li>No permanent database storage or public sharing of papers occurs in the MVP.</li>
                <li>API keys are handled securely via Streamlit secrets or OS environment variables and are never transmitted to external services other than Google's official Gemini API endpoint.</li>
            </ul>
        </div>
    </div>

    <div class="academic-card">
        <div class="card-header">⚠️ System Scope & Boundaries</div>
        <div class="card-body">
            <ul>
                <li><strong>No OCR in MVP:</strong> Scanned or pure-image PDFs are identified and rejected with an explanatory message.</li>
                <li><strong>Context Bounds:</strong> Very large documents (>12,000 words) are truncated using intelligent head/tail preservation (preserving Abstract, Introduction, Methodology, Results, and Conclusions).</li>
                <li><strong>Literature Boundary:</strong> Comparison and gap detection reflect strictly the 2–5 papers provided by the user.</li>
            </ul>
        </div>
    </div>

    <div class="disclaimer-box">
        <strong>Academic Disclaimer:</strong> ResearchLens AI provides AI-assisted decision support.
        It does not replace peer review, domain expertise, or independent verification of the original research literature.
    </div>
    """, unsafe_allow_html=True)

