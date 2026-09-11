"""
ResearchLens AI - Main Streamlit Application.
AI-Powered Research Paper Analysis, Comparison & Research Gap Detection.
"""

import os
import streamlit as st

from src.utils import (
    validate_uploaded_file,
    render_badge,
    sanitize_filename
)
from src.document_parser import extract_text
from src.analyzer import (
    analyze_paper,
    resolve_api_key,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_MODEL
)
from src.comparator import (
    compare_papers,
    detect_research_gaps,
    MIN_COMPARE_PAPERS,
    MAX_COMPARE_PAPERS
)
from src.report import (
    generate_single_paper_report_txt,
    generate_comparison_report_txt,
    generate_gap_report_txt,
    get_download_filename
)

# Page configuration
st.set_page_config(
    page_title="ResearchLens AI - Academic Paper Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Modern Academic Research UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .academic-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .academic-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.06);
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-body {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.6;
    }
    .metric-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 3px solid #0284C7 !important;
        color: #0284C7 !important;
        font-weight: 700 !important;
    }
    .tag-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 6px;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 16px;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session states
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
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🔬 ResearchLens AI")
    st.caption("**AI-Powered Research Paper Analysis, Comparison & Research Gap Detection**")
    st.divider()

    # Navigation menu
    nav_selection = st.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "📄 Single Paper Analysis",
            "🔍 Paper Comparison",
            "🧠 Research Gaps",
            "ℹ️ About"
        ],
        index=0
    )

    st.divider()

    # API Key Configuration
    st.markdown("### ⚙️ API Configuration")
    has_system_key = False
    try:
        resolve_api_key()
        has_system_key = True
    except MissingAPIKeyError:
        has_system_key = False

    user_api_key = ""
    if has_system_key:
        st.success("🟢 OpenAI API Key active (System/Secrets)")
    else:
        st.warning("🟡 No API Key found in secrets or environment.")
        user_api_key = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your API key is used strictly for this session and is never stored permanently."
        )

    model_choice = st.selectbox(
        "AI Engine Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini offers rapid, cost-effective high-context extraction. gpt-4o provides maximal analytical depth."
    )

    st.divider()

    # Platform Capabilities Info
    st.markdown("### 📋 Platform Specs")
    st.markdown("""
    - **Formats**: PDF, DOCX, TXT
    - **Max Comparison**: 2–5 Papers
    - **Context Strategy**: Head/Tail Preservation
    - **Fidelity**: Strict Evidence-Aware Mode
    """)

    st.caption("v1.0.0 • Production MVP")


# Helper function to get active API key
def get_current_api_key() -> str:
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    try:
        return resolve_api_key()
    except MissingAPIKeyError as e:
        raise e


# ==========================================
# PAGE 1: HOME
# ==========================================
if nav_selection == "🏠 Home":
    st.markdown('<h1 class="main-title">ResearchLens AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Understand papers. Compare research. Discover what\'s missing.</p>',
        unsafe_allow_html=True
    )

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
                <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem;">{step_title}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">{step_desc}</div>
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
    st.markdown('<h1 class="main-title">📄 Single Paper Academic Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Upload an individual research manuscript for comprehensive, evidence-aware structural extraction.</p>',
        unsafe_allow_html=True
    )

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
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['file_type']}</div>
                        <div class="metric-label">Document Type</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['word_count']:,}</div>
                        <div class="metric-label">Word Count</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{doc_data['char_count']:,}</div>
                        <div class="metric-label">Character Count</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mcol4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">~{doc_data['token_count']:,}</div>
                        <div class="metric-label">Est. Tokens</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")

                if st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
                    try:
                        active_key = get_current_api_key()
                        with st.status("Performing evidence-aware analysis...", expanded=True) as status:
                            st.write("Checking token boundaries & intelligent head/tail retention...")
                            st.write(f"Querying OpenAI ({model_choice}) with structured JSON schema...")
                            analysis_result = analyze_paper(
                                text=doc_data["text"],
                                filename=doc_data["filename"],
                                api_key=active_key,
                                model=model_choice
                            )
                            st.session_state["single_analysis"] = analysis_result
                            status.update(label="Analysis completed successfully!", state="complete", expanded=False)
                    except MissingAPIKeyError as me:
                        st.error(f"🔑 {str(me)}")
                    except AnalysisError as ae:
                        st.error(f"❌ {str(ae)}")
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {str(e)}")

    # Display Analysis Results
    if st.session_state.get("single_analysis") is not None:
        analysis = st.session_state["single_analysis"]
        meta = analysis.get("_metadata", {})

        st.divider()

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
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">{conf_reason}</div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="academic-card">
                <div class="card-body">{analysis.get('executive_summary')}</div>
            </div>
            """, unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("#### Research Problem & Core Bottleneck")
            st.markdown(f"""
            <div class="academic-card">
                <div class="card-body">{analysis.get('research_problem')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Objectives / Research Questions / Hypotheses")
            objs = analysis.get("objectives", [])
            if objs:
                for obj in objs:
                    st.markdown(f"- {obj}")
            else:
                st.write("Not clearly stated in the paper.")

        with tabs[2]:
            st.markdown("#### Technical Methodology & Architecture")
            st.markdown(f"""
            <div class="academic-card">
                <div class="card-body">{analysis.get('methodology')}</div>
            </div>
            """, unsafe_allow_html=True)

        with tabs[3]:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### Datasets Used")
                st.markdown(f"""
                <div class="academic-card">
                    <div class="card-body">{analysis.get('dataset')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_d2:
                st.markdown("#### Preprocessing & Pipeline")
                st.markdown(f"""
                <div class="academic-card">
                    <div class="card-body">{analysis.get('preprocessing')}</div>
                </div>
                """, unsafe_allow_html=True)

        with tabs[4]:
            st.markdown("#### Evaluation Protocol & Baselines")
            st.markdown(f"""
            <div class="academic-card">
                <div class="card-body">{analysis.get('evaluation')}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Key Results & Quantitative Findings")
            results = analysis.get("results", [])
            if results:
                for r in results:
                    st.markdown(f"- {r}")
            else:
                st.write("Not clearly stated in the paper.")

        with tabs[5]:
            st.markdown("#### Authors' Conclusion")
            st.markdown(f"""
            <div class="academic-card">
                <div class="card-body">{analysis.get('conclusion')}</div>
            </div>
            """, unsafe_allow_html=True)

        with tabs[6]:
            col_s, col_l = st.columns(2)
            with col_s:
                st.markdown("#### Demonstrated Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for s in strengths:
                        st.markdown(f"✅ {s}")
                else:
                    st.write("None specifically documented.")
            with col_l:
                st.markdown("#### Explicit Limitations")
                limits = analysis.get("limitations", [])
                if limits:
                    for l in limits:
                        st.markdown(f"⚠️ {l}")
                else:
                    st.write("None specifically documented.")

        with tabs[7]:
            st.markdown("#### Potential Research Gaps (Identified from Paper)")
            gaps = analysis.get("research_gaps", [])
            if gaps:
                for g in gaps:
                    st.markdown(f"""
                    <div class="academic-card">
                        <div class="card-body">🔍 <strong>{g}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No distinct research gaps isolated.")

        with tabs[8]:
            st.markdown("#### Actionable Research Recommendations")
            suggs = analysis.get("suggestions", [])
            if suggs:
                for idx, s in enumerate(suggs, 1):
                    st.markdown(f"💡 **Recommendation {idx}:** {s}")
            else:
                st.write("No recommendations generated.")

        with tabs[9]:
            st.markdown("#### Future Work Directions")
            fw = analysis.get("future_work", [])
            if fw:
                for item in fw:
                    st.markdown(f"📌 {item}")
            else:
                st.write("Not clearly stated in the paper.")

        with tabs[10]:
            st.markdown("#### Evidence-Aware Experimental Designs")
            exps = analysis.get("experiments", [])
            if exps:
                for idx, exp in enumerate(exps, 1):
                    with st.expander(f"🧪 Experiment {idx}: {exp.get('title', 'Proposed Experiment')}", expanded=(idx == 1)):
                        st.markdown(f"**Hypothesis:** {exp.get('hypothesis', 'N/A')}")
                        st.markdown(f"**Independent Variables:** `{exp.get('independent_variables', 'N/A')}`")
                        st.markdown(f"**Dependent Variables:** `{exp.get('dependent_variables', 'N/A')}`")
                        st.markdown(f"**Recommended Dataset:** {exp.get('dataset', 'N/A')}")
                        st.markdown(f"**Baseline Models:** {exp.get('baseline', 'N/A')}")
                        st.markdown(f"**Evaluation Metrics:** `{exp.get('evaluation_metrics', 'N/A')}`")
                        st.markdown(f"**Expected Outcome:** {exp.get('expected_outcome', 'N/A')}")
                        st.markdown(f"**Risks & Limitations:** {exp.get('risks_and_limitations', 'N/A')}")
            else:
                st.write("No structured experiment plans synthesized.")

        with tabs[11]:
            st.markdown("#### Rigorous Taxonomy & Evidence Breakdown")
            st.caption("Ensures zero hallucination by categorizing extracted information.")
            eb = analysis.get("evidence_breakdown", {})

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown(f"### {render_badge('', 'evidence')} Paper Evidence", unsafe_allow_html=True)
                for item in eb.get("paper_evidence", []):
                    st.markdown(f"• {item}")

                st.write("")
                st.markdown(f"### {render_badge('', 'recommendation')} AI Recommendations", unsafe_allow_html=True)
                for item in eb.get("ai_recommendations", []):
                    st.markdown(f"• {item}")

            with col_e2:
                st.markdown(f"### {render_badge('', 'interpretation')} Technical Interpretation", unsafe_allow_html=True)
                for item in eb.get("interpretations", []):
                    st.markdown(f"• {item}")

                st.write("")
                st.markdown(f"### {render_badge('', 'verification')} Needs Verification", unsafe_allow_html=True)
                for item in eb.get("needs_verification", []):
                    st.markdown(f"• {item}")

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
            use_container_width=True
        )


# ==========================================
# PAGE 3: PAPER COMPARISON
# ==========================================
elif nav_selection == "🔍 Paper Comparison":
    st.markdown('<h1 class="main-title">🔍 Multi-Paper Comparative Synthesis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Upload 2 to 5 research papers to perform side-by-side comparative analysis across core dimensions.</p>',
        unsafe_allow_html=True
    )

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
                        <div style="font-weight: 700; font-size: 0.9rem; word-break: break-all;">{doc['filename']}</div>
                        <div class="metric-value" style="font-size: 1.1rem; margin-top: 4px;">{doc['word_count']:,}</div>
                        <div class="metric-label">Words ({doc['file_type']})</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state["comparison_papers"] = parsed_papers

            st.write("")
            if st.button("⚖️ Compare Papers", type="primary", use_container_width=True):
                try:
                    active_key = get_current_api_key()
                    with st.status("Executing comparative synthesis...", expanded=True) as status:
                        st.write("Checking for duplicates and content overlap...")
                        st.write(f"Running multi-paper comparison using {model_choice}...")
                        comp_result = compare_papers(
                            papers=parsed_papers,
                            api_key=active_key,
                            model=model_choice
                        )
                        st.session_state["compare_results"] = comp_result
                        status.update(label="Comparison completed successfully!", state="complete", expanded=False)
                except MissingAPIKeyError as me:
                    st.error(f"🔑 {str(me)}")
                except AnalysisError as ae:
                    st.error(f"❌ {str(ae)}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    if st.session_state.get("compare_results") is not None:
        comp = st.session_state["compare_results"]
        st.divider()

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
                        st.markdown(f"""
                        <div class="academic-card">
                            <div class="card-body">{p_val}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Comparative Synthesis Findings")
        col_s, col_d = st.columns(2)

        with col_s:
            st.markdown("#### Similarities Across Papers")
            for sim in comp.get("similarities", []):
                st.markdown(f"• {sim}")

            st.write("")
            st.markdown("#### Contradictions & Inconsistencies")
            for con in comp.get("contradictions_inconsistencies", []):
                st.markdown(f"⚡ {con}")

        with col_d:
            st.markdown("#### Key Differences")
            for diff in comp.get("differences", []):
                st.markdown(f"• {diff}")

            st.write("")
            st.markdown("#### Unresolved Areas Across All Papers")
            for unres in comp.get("unresolved_areas", []):
                st.markdown(f"❓ {unres}")

        st.markdown("#### Combined Research Opportunity")
        st.markdown(f"""
        <div class="academic-card" style="border-left: 4px solid #0284C7;">
            <div class="card-header">🚀 Synthesized Opportunity</div>
            <div class="card-body">{comp.get('combined_research_opportunity')}</div>
        </div>
        """, unsafe_allow_html=True)

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
            use_container_width=True
        )


# ==========================================
# PAGE 4: RESEARCH GAPS
# ==========================================
elif nav_selection == "🧠 Research Gaps":
    st.markdown('<h1 class="main-title">🧠 Evidence-Aware Research Gap Detection</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Synthesize recurring limitations, missing evaluations, and unaddressed scientific frontiers across papers.</p>',
        unsafe_allow_html=True
    )

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

            if st.button("🔬 Detect Cross-Paper Research Gaps", type="primary", use_container_width=True):
                try:
                    active_key = get_current_api_key()
                    with st.status("Analyzing research frontier & detecting gaps...", expanded=True) as status:
                        st.write("Evaluating cross-paper datasets, models, and evaluation methods...")
                        st.write("Grouping recurring limitations into prioritized gap categories...")
                        gap_res = detect_research_gaps(
                            papers=parsed_papers,
                            api_key=active_key,
                            model=model_choice
                        )
                        st.session_state["gap_results"] = gap_res
                        status.update(label="Gap detection complete!", state="complete", expanded=False)
                except MissingAPIKeyError as me:
                    st.error(f"🔑 {str(me)}")
                except AnalysisError as ae:
                    st.error(f"❌ {str(ae)}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

    if st.session_state.get("gap_results") is not None:
        gaps_data = st.session_state["gap_results"]
        st.divider()

        st.markdown("### 🌐 Collective Research Frontier Synthesis")
        st.markdown(f"""
        <div class="academic-card">
            <div class="card-body">{gaps_data.get('synthesis_summary')}</div>
        </div>
        """, unsafe_allow_html=True)

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

                st.markdown(f"""
                <div class="academic-card" style="border-left: 5px solid {'#EF4444' if prio=='HIGH' else '#F59E0B' if prio=='MEDIUM' else '#0284C7'};">
                    <div class="card-header">
                        <span>Gap #{idx}: {g.get('title')}</span>
                        <div style="margin-left: auto;">
                            {badge_html} {conf_badge} {verif_badge}
                        </div>
                    </div>
                    <div class="card-body">
                        <p><strong>Description:</strong> {g.get('description')}</p>
                        <p><strong>Evidence from Uploaded Papers:</strong> {g.get('evidence_from_papers')}</p>
                        <p><strong>Why It Matters:</strong> {g.get('why_it_matters')}</p>
                        <p><strong>Suggested Research Direction:</strong> {g.get('suggested_direction')}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
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
            use_container_width=True
        )


# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif nav_selection == "ℹ️ About":
    st.markdown('<h1 class="main-title">ℹ️ About ResearchLens AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Architectural design, evidence-aware principles, and academic disclaimers.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎯 Purpose & Mission
    Modern researchers and students face exponential literature growth, making it difficult to keep track of
    technical nuances, contradictory findings, and genuine research gaps. **ResearchLens AI** was engineered
    to act as an evidence-aware pair-researcher that extracts exact technical dimensions, compares competing architectures,
    and isolates high-value research opportunities without hallucinating facts.

    ---

    ### 🛡️ Evidence-Aware Design Principles
    1. **Strict Context Fidelity**: The system will never invent benchmarks, fabricate citations, or claim a dataset exists if omitted in the manuscript.
    2. **Explicit Uncertainty Handling**: Omissions are cleanly surfaced as *"Not clearly stated in the paper."*
    3. **Academic Humility**: Rather than declaring global novelty, the system specifies *"Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."*
    4. **Rigorous Taxonomy**:
       - 📄 **Paper Evidence**: Direct facts reported by the authors.
       - 🔎 **Interpretation**: Logical conclusions synthesized from reported facts.
       - 💡 **AI Recommendation**: Proposed next steps.
       - ⚠️ **Needs Verification**: Assumptions needing external literature verification.

    ---

    ### 🔒 Security, Privacy & Temporary Processing
    - Uploaded papers are held strictly in ephemeral memory for the duration of the analysis.
    - No permanent database storage or public sharing of papers occurs in the MVP.
    - API keys are handled securely via Streamlit secrets or OS environment variables and are never transmitted to external services other than OpenAI's official endpoint.

    ---

    ### ⚠️ Limitations
    - **No OCR in MVP**: Scanned or pure-image PDFs are identified and rejected with an explanatory message.
    - **Context Bounds**: Very large documents (>12,000 words) are truncated using intelligent head/tail preservation (preserving Abstract, Introduction, Methodology, Results, and Conclusions).
    - **Literature Boundary**: Comparison and gap detection reflect only the 2–5 papers provided by the user.

    ---

    ### 📜 Academic Disclaimer
    <div class="disclaimer-box">
        <strong>Disclaimer:</strong> ResearchLens AI provides AI-assisted decision support.
        It does not replace peer review, domain expertise, or verification of the original research literature.
    </div>
    """, unsafe_allow_html=True)
