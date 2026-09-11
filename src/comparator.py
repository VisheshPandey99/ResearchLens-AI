"""
Comparator and Research Gap Detection module for ResearchLens AI.
Compares 2 to 5 research papers across standard academic dimensions.
Detects recurring limitations, missing benchmarks, and synthesizes evidence-grounded research gaps.
"""

import json
from typing import List, Dict, Any, Optional
from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIError

from src.prompts import (
    SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
    PAPER_COMPARISON_PROMPT,
    RESEARCH_GAP_PROMPT
)
from src.analyzer import (
    get_openai_client,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_MODEL
)
from src.utils import clean_json_markdown, truncate_text_intelligently

MIN_COMPARE_PAPERS = 2
MAX_COMPARE_PAPERS = 5

STANDARD_DIMENSIONS = [
    "Research Problem",
    "Objectives",
    "Methodology",
    "Dataset",
    "Preprocessing",
    "Evaluation",
    "Results",
    "Strengths",
    "Limitations"
]

GAP_AREAS = [
    "Dataset",
    "Model",
    "Explainability",
    "Robustness",
    "Real-world testing",
    "Evaluation",
    "Generalization"
]


def check_for_duplicates(papers: List[Dict[str, Any]]) -> List[str]:
    """
    Detect duplicate papers by comparing names and text content.
    Returns a list of warning messages for detected duplicates.
    """
    warnings = []
    n = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            name_i = papers[i].get("filename", f"Paper {i+1}").strip().lower()
            name_j = papers[j].get("filename", f"Paper {j+1}").strip().lower()

            text_i = papers[i].get("text", "")[:500].strip().lower()
            text_j = papers[j].get("text", "")[:500].strip().lower()

            if name_i == name_j:
                warnings.append(
                    f"Papers '{papers[i].get('filename')}' and '{papers[j].get('filename')}' have identical filenames."
                )
            elif text_i and text_j and text_i == text_j:
                warnings.append(
                    f"Papers '{papers[i].get('filename')}' and '{papers[j].get('filename')}' appear to have identical content."
                )

    return warnings


def format_papers_summary(papers: List[Dict[str, Any]], words_per_paper: int = 3500) -> str:
    """
    Format and truncate paper texts into a concise multi-paper summary for LLM prompt context.
    """
    summaries = []
    for idx, p in enumerate(papers, 1):
        name = p.get("filename", f"Paper_{idx}")
        raw_text = p.get("text", "")
        # Truncate each paper so total context stays well within model window
        truncated, _, _, _ = truncate_text_intelligently(raw_text, max_words=words_per_paper)
        summaries.append(
            f"--- PAPER {idx}: {name} ---\n{truncated}\n"
        )
    return "\n".join(summaries)


def validate_comparison_schema(data: Any, paper_names: List[str]) -> Dict[str, Any]:
    """
    Ensure the returned comparison structure conforms to required keys and dimensions.
    """
    if not isinstance(data, dict):
        raise ValueError("Model comparison output is not a dictionary.")

    comparison_table = data.get("comparison_table", [])
    if not isinstance(comparison_table, list) or not comparison_table:
        # Build fallback table structure
        comparison_table = [
            {
                "dimension": dim,
                "papers": {name: "Not clearly stated." for name in paper_names}
            }
            for dim in STANDARD_DIMENSIONS
        ]
    else:
        # Verify all standard dimensions are covered
        existing_dims = {entry.get("dimension") for entry in comparison_table if isinstance(entry, dict)}
        for dim in STANDARD_DIMENSIONS:
            if dim not in existing_dims:
                comparison_table.append({
                    "dimension": dim,
                    "papers": {name: "Not analyzed." for name in paper_names}
                })

    return {
        "comparison_table": comparison_table,
        "similarities": data.get("similarities", []) if isinstance(data.get("similarities"), list) else [],
        "differences": data.get("differences", []) if isinstance(data.get("differences"), list) else [],
        "contradictions_inconsistencies": (
            data.get("contradictions_inconsistencies", [])
            if isinstance(data.get("contradictions_inconsistencies"), list)
            else ["No obvious contradictions detected."]
        ),
        "unresolved_areas": (
            data.get("unresolved_areas", [])
            if isinstance(data.get("unresolved_areas"), list)
            else []
        ),
        "combined_research_opportunity": data.get(
            "combined_research_opportunity",
            "Synthesized multi-paper research opportunity based on uploaded papers."
        )
    }


def validate_research_gap_schema(data: Any, paper_names: List[str]) -> Dict[str, Any]:
    """
    Ensure the returned research gap structure conforms to matrix and priority gaps format.
    """
    if not isinstance(data, dict):
        raise ValueError("Model gap output is not a valid dictionary.")

    matrix = data.get("matrix", [])
    if not isinstance(matrix, list) or not matrix:
        matrix = [
            {
                "area": area,
                "paper_ratings": {name: "Evaluated in study" for name in paper_names},
                "potential_gap": f"Further cross-validation needed for {area.lower()}."
            }
            for area in GAP_AREAS
        ]

    raw_gaps = data.get("gaps", [])
    validated_gaps = []
    if isinstance(raw_gaps, list):
        for g in raw_gaps:
            if isinstance(g, dict):
                priority = str(g.get("priority", "MEDIUM")).upper()
                if priority not in ("HIGH", "MEDIUM", "LOW"):
                    priority = "MEDIUM"

                validated_gaps.append({
                    "title": g.get("title", "Potential Research Gap"),
                    "priority": priority,
                    "description": g.get("description", "Not clearly detailed."),
                    "evidence_from_papers": g.get("evidence_from_papers", "Derived from paper limitations."),
                    "why_it_matters": g.get("why_it_matters", "Significance to the domain."),
                    "suggested_direction": g.get("suggested_direction", "Suggested future investigation."),
                    "confidence": g.get("confidence", "Medium"),
                    "needs_verification": g.get("needs_verification", True)
                })

    # Sort gaps by priority: HIGH -> MEDIUM -> LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    validated_gaps.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return {
        "matrix": matrix,
        "gaps": validated_gaps,
        "synthesis_summary": data.get(
            "synthesis_summary",
            "Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."
        )
    }


def compare_papers(
    papers: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Compare 2 to 5 research papers across standard academic dimensions.
    Returns structured comparison table, similarities, differences, contradictions, and joint opportunities.
    """
    if len(papers) < MIN_COMPARE_PAPERS:
        raise ValueError(f"At least {MIN_COMPARE_PAPERS} papers are required for comparison. Received {len(papers)}.")
    if len(papers) > MAX_COMPARE_PAPERS:
        raise ValueError(f"A maximum of {MAX_COMPARE_PAPERS} papers can be compared at once. Received {len(papers)}.")

    paper_names = [p.get("filename", f"Paper_{i+1}") for i, p in enumerate(papers)]
    duplicate_warnings = check_for_duplicates(papers)

    # Budget words per paper based on count (e.g. 5 papers -> ~2,400 words each; 2 papers -> ~5,000 words each)
    budget = int(12000 / len(papers))
    papers_summary_text = format_papers_summary(papers, words_per_paper=budget)

    try:
        client = get_openai_client(api_key)

        prompt = PAPER_COMPARISON_PROMPT.format(
            papers_summary=papers_summary_text,
            paper_names_example='", "'.join(paper_names)
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
        parsed = json.loads(clean_json_markdown(raw_content))
        validated = validate_comparison_schema(parsed, paper_names)
        validated["duplicate_warnings"] = duplicate_warnings
        validated["paper_names"] = paper_names

        return validated

    except MissingAPIKeyError as me:
        raise me
    except AuthenticationError:
        raise AnalysisError("Invalid OpenAI API key. Please check your credentials in settings.")
    except RateLimitError:
        raise AnalysisError("OpenAI API rate limit exceeded. Please try again shortly.")
    except (APITimeoutError, APIConnectionError):
        raise AnalysisError("Network connection to OpenAI timed out. Please check your internet connection.")
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse structured comparison output: {str(je)}")
    except APIError as ae:
        raise AnalysisError(f"OpenAI service error: {ae.message if hasattr(ae, 'message') else str(ae)}")
    except Exception as e:
        raise AnalysisError(f"Comparison error: {str(e)}")


def detect_research_gaps(
    papers: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Perform cross-paper research gap detection and generate the Research Gap Matrix.
    """
    if len(papers) < MIN_COMPARE_PAPERS:
        raise ValueError(f"At least {MIN_COMPARE_PAPERS} papers are required for research gap detection.")
    if len(papers) > MAX_COMPARE_PAPERS:
        raise ValueError(f"A maximum of {MAX_COMPARE_PAPERS} papers can be analyzed for research gaps.")

    paper_names = [p.get("filename", f"Paper_{i+1}") for i, p in enumerate(papers)]
    duplicate_warnings = check_for_duplicates(papers)

    budget = int(12000 / len(papers))
    papers_summary_text = format_papers_summary(papers, words_per_paper=budget)

    try:
        client = get_openai_client(api_key)

        prompt = RESEARCH_GAP_PROMPT.format(
            papers_summary=papers_summary_text,
            paper_names_example='", "'.join(paper_names)
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_ACADEMIC_EVIDENCE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
            max_tokens=4000
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(clean_json_markdown(raw_content))
        validated = validate_research_gap_schema(parsed, paper_names)
        validated["duplicate_warnings"] = duplicate_warnings
        validated["paper_names"] = paper_names

        return validated

    except MissingAPIKeyError as me:
        raise me
    except AuthenticationError:
        raise AnalysisError("Invalid OpenAI API key. Please check your credentials in settings.")
    except RateLimitError:
        raise AnalysisError("OpenAI API rate limit exceeded. Please try again shortly.")
    except (APITimeoutError, APIConnectionError):
        raise AnalysisError("Network connection to OpenAI timed out. Please check your internet connection.")
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse research gap response: {str(je)}")
    except APIError as ae:
        raise AnalysisError(f"OpenAI service error: {ae.message if hasattr(ae, 'message') else str(ae)}")
    except Exception as e:
        raise AnalysisError(f"Research gap detection error: {str(e)}")
