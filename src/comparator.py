"""
Comparator and Research Gap Detection module for ResearchLens AI.
Delegates to the unified src.ai_service layer powered by the official Google Gemini API.
Compares 2 to 5 research papers across standard academic dimensions.
Detects recurring limitations, missing benchmarks, and synthesizes evidence-grounded research gaps.
"""

from typing import List, Dict, Any, Optional

from src.ai_service import (
    compare_papers,
    detect_research_gaps,
    check_for_duplicates,
    format_papers_summary,
    validate_comparison_schema,
    validate_research_gap_schema,
    STANDARD_DIMENSIONS,
    GAP_AREAS,
    get_demo_comparison,
    get_demo_gaps,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL
)

MIN_COMPARE_PAPERS = 2
MAX_COMPARE_PAPERS = 5

__all__ = [
    "compare_papers",
    "detect_research_gaps",
    "check_for_duplicates",
    "format_papers_summary",
    "validate_comparison_schema",
    "validate_research_gap_schema",
    "MIN_COMPARE_PAPERS",
    "MAX_COMPARE_PAPERS",
    "STANDARD_DIMENSIONS",
    "GAP_AREAS",
    "get_demo_comparison",
    "get_demo_gaps",
    "MissingAPIKeyError",
    "AnalysisError",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_MODEL"
]
