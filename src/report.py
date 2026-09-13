"""
Report generation module for ResearchLens AI.
Generates comprehensive, professionally formatted TXT reports for:
- Single paper academic analysis
- Multi-paper comparative analysis
- Cross-paper research gap detection and matrix
"""

import re
from datetime import datetime
from typing import Dict, Any, List
from src.utils import sanitize_filename

DISCLAIMER_TEXT = (
    "DISCLAIMER:\n"
    "ResearchLens AI provides AI-assisted academic decision support.\n"
    "It does not replace peer review, domain expertise, or independent\n"
    "verification of the original research literature."
)


def strip_html_tags(text: Any) -> str:
    """Strip HTML formatting tags (e.g. <u>, <b>, <span>) for pure plain-text reports."""
    if not text:
        return ""
    s = str(text)
    # Remove HTML tags
    cleaned = re.sub(r"</?[a-zA-Z0-9]+[^>]*>", "", s)
    return cleaned.strip()


def format_list_items(items: Any, prefix: str = "• ") -> str:
    """Helper to cleanly format list items into indented bullet points without HTML tags."""
    if not items:
        return "  None reported or not clearly stated in the paper."
    if isinstance(items, str):
        cleaned = strip_html_tags(items)
        return f"  {prefix}{cleaned}"
    if isinstance(items, list):
        formatted = []
        for item in items:
            if isinstance(item, dict):
                cleaned_dict = ", ".join(f"{k}: {strip_html_tags(v)}" for k, v in item.items())
                formatted.append(f"  {prefix}{cleaned_dict}")
            else:
                formatted.append(f"  {prefix}{strip_html_tags(str(item))}")
        return "\n".join(formatted)
    return f"  {strip_html_tags(str(items))}"


def generate_single_paper_report_txt(
    analysis: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> str:
    """
    Generate a complete, structured plain-text academic analysis report for a single paper.
    """
    meta = metadata or analysis.get("_metadata", {})
    filename = meta.get("filename", "Unknown Document")
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    orig_words = meta.get("original_word_count", "N/A")
    analyzed_words = meta.get("analyzed_word_count", "N/A")
    truncated = meta.get("was_truncated", False)
    model = meta.get("model_used", "gemini-2.5-flash")

    confidence = analysis.get("confidence", {})
    overall_conf = confidence.get("overall", "Medium")
    conf_reason = confidence.get("reason", "N/A")

    sep_double = "=" * 80
    sep_single = "-" * 80

    lines = [
        sep_double,
        "RESEARCHLENS AI - ACADEMIC PAPER ANALYSIS REPORT",
        sep_double,
        f"Date of Analysis  : {analysis_date}",
        f"Paper Name        : {filename}",
        f"Original Words    : {orig_words:,}" if isinstance(orig_words, int) else f"Original Words    : {orig_words}",
        f"Analyzed Words    : {analyzed_words:,}" if isinstance(analyzed_words, int) else f"Analyzed Words    : {analyzed_words}",
        f"Content Truncated : {'Yes (head/tail preserved)' if truncated else 'No (complete document analyzed)'}",
        f"AI Engine Model   : {model}",
        f"Overall Confidence: {overall_conf} ({conf_reason})",
        sep_double,
        "",
        "1. EXECUTIVE SUMMARY",
        sep_single,
        strip_html_tags(analysis.get("executive_summary", "Not clearly stated in the paper.")),
        "",
        "2. RESEARCH PROBLEM & BOTTLENECK",
        sep_single,
        strip_html_tags(analysis.get("research_problem", "Not clearly stated in the paper.")),
        "",
        "3. OBJECTIVES / RESEARCH QUESTIONS / HYPOTHESES",
        sep_single,
        format_list_items(analysis.get("objectives", [])),
        "",
        "4. METHODOLOGY & SYSTEM ARCHITECTURE",
        sep_single,
        strip_html_tags(analysis.get("methodology", "Not clearly stated in the paper.")),
        "",
        "5. DATASET & PREPROCESSING",
        sep_single,
        f"[Dataset Source & Splits]:\n{strip_html_tags(analysis.get('dataset', 'Not clearly stated in the paper.'))}\n",
        f"[Preprocessing & Pipeline]:\n{strip_html_tags(analysis.get('preprocessing', 'Not clearly stated in the paper.'))}",
        "",
        "6. EVALUATION PROTOCOL & BASELINES",
        sep_single,
        strip_html_tags(analysis.get("evaluation", "Not clearly stated in the paper.")),
        "",
        "7. RESULTS & QUANTITATIVE FINDINGS",
        sep_single,
        format_list_items(analysis.get("results", [])),
        "",
        "8. AUTHORS' CONCLUSION",
        sep_single,
        strip_html_tags(analysis.get("conclusion", "Not clearly stated in the paper.")),
        "",
        "9. STRENGTHS & LIMITATIONS",
        sep_single,
        "[Paper Strengths]:",
        format_list_items(analysis.get("strengths", [])),
        "",
        "[Paper Limitations]:",
        format_list_items(analysis.get("limitations", [])),
        "",
        "10. POTENTIAL RESEARCH GAPS",
        sep_single,
        format_list_items(analysis.get("research_gaps", [])),
        "",
        "11. ACTIONABLE RESEARCH RECOMMENDATIONS",
        sep_single,
        format_list_items(analysis.get("suggestions", [])),
        "",
        "12. FUTURE WORK DIRECTIONS (AUTHOR-STATED / DIRECT)",
        sep_single,
        format_list_items(analysis.get("future_work", [])),
        "",
        "13. EVIDENCE-AWARE EXPERIMENTAL DESIGNS",
        sep_single
    ]

    experiments = analysis.get("experiments", [])
    if experiments:
        for idx, exp in enumerate(experiments, 1):
            lines.extend([
                f"Experiment {idx}: {exp.get('title', 'Proposed Test')}",
                f"  • Hypothesis            : {exp.get('hypothesis', 'N/A')}",
                f"  • Independent Variables : {exp.get('independent_variables', 'N/A')}",
                f"  • Dependent Variables   : {exp.get('dependent_variables', 'N/A')}",
                f"  • Dataset Recommended   : {exp.get('dataset', 'N/A')}",
                f"  • Comparative Baseline  : {exp.get('baseline', 'N/A')}",
                f"  • Evaluation Metrics    : {exp.get('evaluation_metrics', 'N/A')}",
                f"  • Expected Outcome      : {exp.get('expected_outcome', 'N/A')}",
                f"  • Risks / Limitations   : {exp.get('risks_and_limitations', 'N/A')}",
                ""
            ])
    else:
        lines.append("  No specific experiment designs synthesized.")
        lines.append("")

    # Evidence breakdown
    eb = analysis.get("evidence_breakdown", {})
    lines.extend([
        "14. EVIDENCE-AWARE TAXONOMY BREAKDOWN",
        sep_single,
        "[📄 Paper Evidence - Verbatim & Direct Findings]:",
        format_list_items(eb.get("paper_evidence", [])),
        "",
        "[🔎 Technical Interpretation]:",
        format_list_items(eb.get("interpretations", [])),
        "",
        "[💡 AI Recommendations]:",
        format_list_items(eb.get("ai_recommendations", [])),
        "",
        "[⚠️ Needs External Verification / Literature Review]:",
        format_list_items(eb.get("needs_verification", [])),
        "",
        sep_double,
        DISCLAIMER_TEXT,
        sep_double
    ])

    return "\n".join(lines)


def generate_comparison_report_txt(comparison: Dict[str, Any]) -> str:
    """
    Generate a formatted plain-text multi-paper comparison report.
    """
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    paper_names = comparison.get("paper_names", [])

    sep_double = "=" * 80
    sep_single = "-" * 80

    lines = [
        sep_double,
        "RESEARCHLENS AI - MULTI-PAPER COMPARATIVE ANALYSIS REPORT",
        sep_double,
        f"Date of Comparison: {analysis_date}",
        f"Compared Papers   : {len(paper_names)} documents",
        f"Document Names    : {', '.join(paper_names)}",
        sep_double,
        ""
    ]

    # Duplicate warnings
    dup_warnings = comparison.get("duplicate_warnings", [])
    if dup_warnings:
        lines.extend([
            "POTENTIAL DUPLICATION WARNINGS:",
            format_list_items(dup_warnings, prefix="⚠ "),
            ""
        ])

    lines.extend([
        "1. COMPARATIVE DIMENSIONS MATRIX",
        sep_single
    ])

    for row in comparison.get("comparison_table", []):
        dimension = row.get("dimension", "Unknown Dimension")
        papers_dict = row.get("papers", {})
        lines.append(f"\n[Dimension: {dimension.upper()}]")
        for p_name, val in papers_dict.items():
            lines.append(f"  • {p_name}:\n      {strip_html_tags(val)}")

    lines.extend([
        "",
        "2. CORE SIMILARITIES ACROSS PAPERS",
        sep_single,
        format_list_items(comparison.get("similarities", [])),
        "",
        "3. KEY METHODOLOGICAL & EMPIRICAL DIFFERENCES",
        sep_single,
        format_list_items(comparison.get("differences", [])),
        "",
        "4. CONTRADICTIONS & INCONSISTENCIES",
        sep_single,
        format_list_items(comparison.get("contradictions_inconsistencies", [])),
        "",
        "5. UNRESOLVED RESEARCH AREAS",
        sep_single,
        format_list_items(comparison.get("unresolved_areas", [])),
        "",
        "6. SYNTHESIZED COMBINED RESEARCH OPPORTUNITY",
        sep_single,
        strip_html_tags(comparison.get("combined_research_opportunity", "Not clearly stated.")),
        "",
        sep_double,
        DISCLAIMER_TEXT,
        sep_double
    ])

    return "\n".join(lines)


def generate_gap_report_txt(gap_data: Dict[str, Any]) -> str:
    """
    Generate a formatted plain-text research gap detection report.
    """
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    paper_names = gap_data.get("paper_names", [])

    sep_double = "=" * 80
    sep_single = "-" * 80

    lines = [
        sep_double,
        "RESEARCHLENS AI - CROSS-PAPER RESEARCH GAP & FRONTIER REPORT",
        sep_double,
        f"Date of Analysis: {analysis_date}",
        f"Analyzed Papers : {', '.join(paper_names)}",
        sep_double,
        "",
        "COLLECTIVE SYNTHESIS SUMMARY",
        sep_single,
        strip_html_tags(gap_data.get("synthesis_summary", "Synthesis completed.")),
        "",
        "1. RESEARCH GAP EVALUATION MATRIX",
        sep_single
    ]

    for item in gap_data.get("matrix", []):
        area = item.get("area", "Unknown Area")
        lines.append(f"\n[Domain Area: {area.upper()}]")
        ratings = item.get("paper_ratings", {})
        for p_name, rating in ratings.items():
            lines.append(f"  • {p_name}: {rating}")
        lines.append(f"  → Potential Gap in {area}: {item.get('potential_gap', 'N/A')}")

    lines.extend([
        "",
        "2. PRIORITIZED RESEARCH GAPS",
        sep_single
    ])

    gaps = gap_data.get("gaps", [])
    if gaps:
        for idx, g in enumerate(gaps, 1):
            lines.extend([
                f"\nGap #{idx} [{g.get('priority', 'MEDIUM')} PRIORITY] - {g.get('title', 'Research Gap')}",
                f"  • Description          : {g.get('description', 'N/A')}",
                f"  • Evidence from Papers : {g.get('evidence_from_papers', 'N/A')}",
                f"  • Scientific Rationale : {g.get('why_it_matters', 'N/A')}",
                f"  • Suggested Direction  : {g.get('suggested_direction', 'N/A')}",
                f"  • Confidence Level     : {g.get('confidence', 'Medium')}",
                f"  • Verification Status  : {'Requires broader literature verification' if g.get('needs_verification') else 'Validated in papers'}"
            ])
    else:
        lines.append("  No high-confidence research gaps detected among provided papers.")

    lines.extend([
        "",
        sep_double,
        "CAUTIONARY NOTE:",
        "Potential research gaps are based strictly on the uploaded papers.",
        "Broader literature review is required to establish true academic novelty.",
        sep_double,
        DISCLAIMER_TEXT,
        sep_double
    ])

    return "\n".join(lines)


def get_download_filename(prefix: str, paper_name: str) -> str:
    """Generate safe, timestamped download filename for reports."""
    safe_name = sanitize_filename(paper_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ResearchLens_{prefix}_{safe_name}_{timestamp}.txt"
