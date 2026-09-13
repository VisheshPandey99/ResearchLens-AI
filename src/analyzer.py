"""
AI Analyzer module for ResearchLens AI.
Delegates to the unified src.ai_service layer powered by the official Google Gemini API.
Maintains 100% backward compatibility for existing callers and workflows.
"""

from typing import Dict, Any, Optional, List

from src.ai_service import (
    analyze_paper,
    generate_recommendations,
    generate_experiments,
    validate_analysis_schema,
    resolve_gemini_api_key,
    resolve_api_key,
    get_gemini_client,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL,
    SUPPORTED_GEMINI_MODELS,
    get_demo_single_analysis
)

__all__ = [
    "analyze_paper",
    "generate_recommendations",
    "generate_experiments",
    "validate_analysis_schema",
    "resolve_gemini_api_key",
    "resolve_api_key",
    "get_gemini_client",
    "MissingAPIKeyError",
    "AnalysisError",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_MODEL",
    "SUPPORTED_GEMINI_MODELS",
    "get_demo_single_analysis"
]
