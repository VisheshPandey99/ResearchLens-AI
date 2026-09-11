"""
AI Analyzer module for ResearchLens AI.
Interfaces with OpenAI API using structured JSON mode.
Enforces robust error handling, schema validation, and intelligent context window management.
"""

import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIError

from src.prompts import (
    SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
    SINGLE_PAPER_ANALYSIS_PROMPT,
    RECOMMENDATION_PROMPT,
    EXPERIMENT_PROMPT
)
from src.utils import truncate_text_intelligently, clean_json_markdown

DEFAULT_MODEL = "gpt-4o-mini"


class MissingAPIKeyError(Exception):
    """Raised when no OpenAI API key is detected in secrets, environment, or arguments."""
    pass


class AnalysisError(Exception):
    """Raised when analysis fails due to API or validation errors."""
    pass


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """
    Resolve API key in order of priority:
    1. Explicit function argument
    2. Streamlit secrets (if available)
    3. Environment variable OPENAI_API_KEY
    """
    if api_key and api_key.strip():
        return api_key.strip()

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    # Try OS environment variable
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip()

    raise MissingAPIKeyError(
        "OpenAI API key not found. Please set your key in '.streamlit/secrets.toml', "
        "export the OPENAI_API_KEY environment variable, or enter it in the application sidebar."
    )


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Instantiate and return an OpenAI client with resolved credentials."""
    resolved_key = resolve_api_key(api_key)
    return OpenAI(api_key=resolved_key)


def validate_analysis_schema(data: Any) -> Dict[str, Any]:
    """
    Validate and normalize the structure of single paper analysis JSON.
    Ensures all expected keys and list structures exist to prevent UI indexing errors.
    """
    if not isinstance(data, dict):
        raise ValueError("Model response is not a valid JSON object dictionary.")

    defaults = {
        "executive_summary": "Not clearly stated in the paper.",
        "research_problem": "Not clearly stated in the paper.",
        "objectives": [],
        "methodology": "Not clearly stated in the paper.",
        "dataset": "Not clearly stated in the paper.",
        "preprocessing": "Not clearly stated in the paper.",
        "evaluation": "Not clearly stated in the paper.",
        "results": [],
        "conclusion": "Not clearly stated in the paper.",
        "strengths": [],
        "limitations": [],
        "research_gaps": [],
        "suggestions": [],
        "future_work": [],
        "experiments": [],
        "confidence": {
            "overall": "Medium",
            "reason": "Evaluated based on extracted machine-readable text."
        },
        "evidence_breakdown": {
            "paper_evidence": [],
            "interpretations": [],
            "ai_recommendations": [],
            "needs_verification": []
        }
    }

    normalized = {}
    for key, default_val in defaults.items():
        if key not in data or data[key] is None:
            normalized[key] = default_val
        elif isinstance(default_val, list) and not isinstance(data[key], list):
            normalized[key] = [str(data[key])] if data[key] else []
        elif isinstance(default_val, dict) and not isinstance(data[key], dict):
            normalized[key] = default_val
        else:
            normalized[key] = data[key]

    # Ensure nested dict keys exist
    if not isinstance(normalized["confidence"], dict):
        normalized["confidence"] = defaults["confidence"]
    else:
        normalized["confidence"].setdefault("overall", "Medium")
        normalized["confidence"].setdefault("reason", "Analysis completed.")

    if not isinstance(normalized["evidence_breakdown"], dict):
        normalized["evidence_breakdown"] = defaults["evidence_breakdown"]
    else:
        for subkey in ["paper_evidence", "interpretations", "ai_recommendations", "needs_verification"]:
            if subkey not in normalized["evidence_breakdown"] or not isinstance(normalized["evidence_breakdown"][subkey], list):
                normalized["evidence_breakdown"][subkey] = []

    return normalized


def analyze_paper(
    text: str,
    filename: str = "Research Paper",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_words: int = 12000
) -> Dict[str, Any]:
    """
    Perform deep academic analysis of a single research paper.
    Intelligently truncates text if exceeding limit and requests structured JSON output.
    """
    if not text or not text.strip():
        raise AnalysisError("No readable text available to analyze. The document is empty or non-extractable.")

    # Intelligent truncation
    processed_text, was_truncated, orig_words, final_words = truncate_text_intelligently(
        text, max_words=max_words
    )

    try:
        client = get_openai_client(api_key)

        prompt = SINGLE_PAPER_ANALYSIS_PROMPT.format(
            filename=filename,
            text=processed_text
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ACADEMIC_EVIDENCE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=4000
        )

        raw_content = response.choices[0].message.content or "{}"
        cleaned_content = clean_json_markdown(raw_content)
        parsed = json.loads(cleaned_content)
        validated = validate_analysis_schema(parsed)

        # Attach document processing metadata
        validated["_metadata"] = {
            "filename": filename,
            "was_truncated": was_truncated,
            "original_word_count": orig_words,
            "analyzed_word_count": final_words,
            "model_used": model
        }

        return validated

    except MissingAPIKeyError as me:
        raise me
    except AuthenticationError:
        raise AnalysisError("Invalid OpenAI API key. Please check your credentials in settings.")
    except RateLimitError:
        raise AnalysisError("OpenAI API rate limit exceeded or balance exhausted. Please try again shortly.")
    except (APITimeoutError, APIConnectionError):
        raise AnalysisError("Network connection to OpenAI timed out or failed. Please check your internet connection.")
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse structured response from AI model: {str(je)}")
    except APIError as ae:
        raise AnalysisError(f"OpenAI service error: {ae.message if hasattr(ae, 'message') else str(ae)}")
    except Exception as e:
        raise AnalysisError(f"Unexpected analysis error: {str(e)}")


def generate_recommendations(
    analysis_data: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generate evidence-grounded practical research recommendations based on paper analysis.
    """
    try:
        client = get_openai_client(api_key)

        prompt = RECOMMENDATION_PROMPT.format(
            analysis_json=json.dumps({
                "problem": analysis_data.get("research_problem"),
                "methodology": analysis_data.get("methodology"),
                "limitations": analysis_data.get("limitations", []),
                "results": analysis_data.get("results", []),
                "research_gaps": analysis_data.get("research_gaps", [])
            }, indent=2)
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ACADEMIC_EVIDENCE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(clean_json_markdown(raw_content))
        return parsed.get("recommendations", [])

    except Exception as e:
        # Fallback to existing suggestions from analysis if secondary call fails
        return [
            {
                "recommendation": s,
                "reason": "Derived from paper limitation analysis.",
                "supporting_evidence": "Documented in analyzed limitations.",
                "expected_benefit": "Addresses primary bottleneck identified in paper.",
                "priority": "Medium",
                "confidence": "Medium"
            }
            for s in analysis_data.get("suggestions", [])
        ]


def generate_experiments(
    analysis_data: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Formulate structured experiment plans addressing paper limitations.
    """
    try:
        client = get_openai_client(api_key)

        prompt = EXPERIMENT_PROMPT.format(
            analysis_json=json.dumps({
                "problem": analysis_data.get("research_problem"),
                "methodology": analysis_data.get("methodology"),
                "dataset": analysis_data.get("dataset"),
                "limitations": analysis_data.get("limitations", []),
                "future_work": analysis_data.get("future_work", [])
            }, indent=2)
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ACADEMIC_EVIDENCE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2500
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(clean_json_markdown(raw_content))
        return parsed.get("experiments", [])

    except Exception:
        # Return base experiments from initial analysis
        return analysis_data.get("experiments", [])
