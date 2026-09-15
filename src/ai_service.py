"""
AI Service module for ResearchLens AI.
Interfaces with official Google Gemini API using the google-genai Python SDK.
Enforces evidence-aware structured JSON mode, schema validation, API error handling,
context window management, and zero-API Demo Mode.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError, ClientError

logger = logging.getLogger(__name__)

from src.prompts import (
    SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
    SINGLE_PAPER_ANALYSIS_PROMPT,
    PAPER_COMPARISON_PROMPT,
    RESEARCH_GAP_PROMPT,
    RECOMMENDATION_PROMPT,
    EXPERIMENT_PROMPT
)
from src.utils import truncate_text_intelligently, clean_json_markdown

# Default Gemini model available on Google AI Studio
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_MODEL = DEFAULT_GEMINI_MODEL

# Verified Flash models supported by current Google GenAI API for fallback
FALLBACK_FLASH_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest"
]

SUPPORTED_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

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


class MissingAPIKeyError(Exception):
    """Raised when no Gemini API key is detected in secrets, environment, or arguments."""
    pass


class AnalysisError(Exception):
    """Raised when analysis fails due to API, quota, model, or validation errors."""
    pass


def resolve_gemini_api_key(api_key: Optional[str] = None) -> str:
    """
    Resolve Gemini API key in order of priority:
    1. Explicit function argument
    2. Streamlit secrets (st.secrets["GEMINI_API_KEY"])
    3. Environment variable GEMINI_API_KEY
    """
    if api_key and api_key.strip():
        return api_key.strip()

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    # Try OS environment variable
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key and env_key.strip():
        return env_key.strip()

    raise MissingAPIKeyError(
        "Gemini API key is not configured. Please add GEMINI_API_KEY to Streamlit secrets "
        "('.streamlit/secrets.toml'), export the GEMINI_API_KEY environment variable, "
        "or enter it in the application sidebar."
    )


# Alias for backward compatibility
resolve_api_key = resolve_gemini_api_key


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Instantiate and return the official Google GenAI client with resolved credentials."""
    resolved_key = resolve_gemini_api_key(api_key)
    return genai.Client(api_key=resolved_key)


def map_gemini_error(e: Exception, model: str) -> AnalysisError:
    """
    Map Google Gemini API exceptions into clear, friendly, and secure error messages.
    Ensures no internal credentials, stack traces, or secret keys are exposed.
    """
    err_str = str(e).lower()

    if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str:
        return AnalysisError("Gemini API quota/rate limit has been reached. Please try again later.")

    if "400" in err_str or "403" in err_str or "api_key_invalid" in err_str or "invalid api key" in err_str or "unregistered" in err_str:
        return AnalysisError("Invalid Gemini API key. Please check your credentials in settings.")

    if "404" in err_str or "not_found" in err_str or "is not found for api version" in err_str or ("model" in err_str and "not found" in err_str):
        return AnalysisError(f"The configured Gemini model '{model}' is currently unavailable or invalid. Please select a supported model (e.g. gemini-2.5-flash).")

    if "timeout" in err_str or "timed out" in err_str or "connection" in err_str:
        return AnalysisError("Network connection to Google Gemini API timed out or failed. Please check your internet connection.")

    if "503" in err_str or "unavailable" in err_str or "500" in err_str:
        return AnalysisError("Google Gemini service is temporarily unavailable. Please try again shortly.")

    return AnalysisError(f"Gemini service error: {str(e)}")


def is_temporary_capacity_error(exc: Exception) -> bool:
    """
    Determines if an exception indicates a temporary server capacity or 503 error,
    which is eligible for multi-model fallback.
    Explicitly excludes authentication (401/403), invalid API keys,
    bad requests (400), not found (404), and quota exhaustion (429).
    """
    if hasattr(exc, "code") and getattr(exc, "code") in (500, 502, 503, 504):
        return True

    err_str = str(exc).lower()

    # Explicitly do NOT treat auth, permission, quota or bad request as temporary capacity errors
    if any(auth_err in err_str for auth_err in [
        "401", "403", "unauthenticated", "permission_denied",
        "invalid api key", "api_key_invalid", "unregistered"
    ]):
        return False

    if any(quota_err in err_str for quota_err in [
        "429", "resource_exhausted", "quota", "rate limit"
    ]):
        return False

    if "400" in err_str or "bad request" in err_str:
        return False

    # Check for temporary availability / capacity signals
    temporary_signals = [
        "503",
        "unavailable",
        "high demand",
        "experiencing high demand",
        "capacity",
        "overloaded",
        "temporarily unavailable",
        "500",
        "internal server error",
        "service unavailable"
    ]
    return any(sig in err_str for sig in temporary_signals)


def call_gemini_json(
    prompt: str,
    system_instruction: str = SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.2,
    fallback_models: Optional[List[str]] = None,
    backoff_seconds: float = 1.0
) -> str:
    """
    Call official Google Gemini API expecting a structured JSON response.
    Applies system instruction, JSON mime-type configuration, and safe error handling.
    Implements automatic multi-model fallback for temporary 503/capacity errors:
    - Attempts the user-selected primary model first.
    - If temporary 503 capacity error occurs, retries with backoff across supported Flash models.
    - Non-capacity errors (auth, quota, bad requests) fail immediately without fallback.
    - If all fallback models are exhausted, raises a clear consolidated AnalysisError.
    """
    client = get_gemini_client(api_key)

    primary_model = model or DEFAULT_GEMINI_MODEL
    fallbacks = fallback_models if fallback_models is not None else FALLBACK_FLASH_MODELS
    model_queue: List[str] = [primary_model]
    for m in fallbacks:
        if m != primary_model and m not in model_queue:
            model_queue.append(m)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature
    )

    last_error: Optional[Exception] = None
    attempted_models: List[str] = []

    for idx, current_model in enumerate(model_queue):
        attempted_models.append(current_model)
        try:
            if idx > 0:
                logger.info(
                    "Retrying with fallback Gemini model: '%s' (attempt %d/%d)",
                    current_model, idx + 1, len(model_queue)
                )

            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=config
            )

            raw_text = response.text or "{}"
            if idx > 0:
                logger.info("Successfully received response using fallback model: '%s'", current_model)
            return clean_json_markdown(raw_text)

        except MissingAPIKeyError as me:
            raise me
        except Exception as e:
            last_error = e

            # Only fallback for temporary capacity / 503 errors
            if is_temporary_capacity_error(e):
                logger.warning(
                    "Gemini model '%s' encountered temporary capacity error (%s).",
                    current_model, str(e)[:150]
                )
                if idx < len(model_queue) - 1:
                    next_model = model_queue[idx + 1]
                    logger.info(
                        "Applying backoff (%.1fs) before switching to fallback model '%s'...",
                        backoff_seconds, next_model
                    )
                    time.sleep(backoff_seconds)
                    continue
                else:
                    logger.error(
                        "All candidate Gemini models failed with temporary capacity errors: %s",
                        attempted_models
                    )
                    raise AnalysisError(
                        "Google Gemini service is temporarily unavailable due to high demand across all "
                        f"tested models ({', '.join(attempted_models)}). Please try again shortly or switch to Demo Mode."
                    )
            else:
                # Non-temporary errors (auth, quota, bad request, not found) must not be masked
                logger.error(
                    "Non-capacity error encountered on model '%s': %s",
                    current_model, str(e)
                )
                raise map_gemini_error(e, current_model)

    if last_error:
        raise map_gemini_error(last_error, primary_model)
    raise AnalysisError("No response received from Gemini API.")


# =========================================================================
# SCHEMA VALIDATION
# =========================================================================

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

    # Preserve metadata if present
    if "_metadata" in data and isinstance(data["_metadata"], dict):
        normalized["_metadata"] = data["_metadata"]

    return normalized


def validate_comparison_schema(data: Any, paper_names: List[str]) -> Dict[str, Any]:
    """
    Ensure the returned comparison structure conforms to required keys and dimensions.
    """
    if not isinstance(data, dict):
        raise ValueError("Model comparison output is not a dictionary.")

    comparison_table = data.get("comparison_table", [])
    if not isinstance(comparison_table, list) or not comparison_table:
        comparison_table = [
            {
                "dimension": dim,
                "papers": {name: "Not clearly stated." for name in paper_names}
            }
            for dim in STANDARD_DIMENSIONS
        ]
    else:
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
            else ["No obvious contradictions detected based on provided texts."]
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
                "potential_gap": f"Potential research gap in cross-paper {area.lower()} generalization."
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

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    validated_gaps.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return {
        "matrix": matrix,
        "gaps": validated_gaps,
        "synthesis_summary": data.get(
            "synthesis_summary",
            "Potential research gap based on the supplied papers. Broader literature review is required to establish novelty."
        )
    }


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
        truncated, _, _, _ = truncate_text_intelligently(raw_text, max_words=words_per_paper)
        summaries.append(
            f"--- PAPER {idx}: {name} ---\n{truncated}\n"
        )
    return "\n".join(summaries)


# =========================================================================
# DEMO MODE SAMPLE DATA GENERATORS
# =========================================================================

def get_demo_single_analysis(filename: str = "Attention Is All You Need.pdf") -> Dict[str, Any]:
    """Return high-fidelity, evidence-grounded sample analysis without making API calls."""
    data = {
        "executive_summary": (
            "• <u>Core Architectural Shift:</u> Introduces the **Transformer architecture**, completely replacing *recurrent* and *convolutional* neural networks with **multi-head self-attention**.\n"
            "• <u>Computational Efficiency:</u> Eliminates sequential training dependencies, enabling **massive matrix parallelization** on modern GPU accelerators.\n"
            "• <u>Empirical Benchmark:</u> Achieved state-of-the-art **28.4 BLEU** on *WMT 2014 English-to-German* (outperforming prior ensembles by <u>+2.0 BLEU</u>) and **41.8 BLEU** on *WMT 2014 English-to-French*.\n"
            "• <u>Transfer Generalization:</u> Demonstrates high zero-shot transfer capability on *Penn Treebank* syntactic constituency parsing with *minimal task-specific tuning*."
        ),
        "research_problem": (
            "• <u>Sequential Bottleneck:</u> Recurrent models (*RNNs, LSTMs, GRUs*) inherently compute representations sequentially along token positions.\n"
            "• <u>Hardware Inefficiency:</u> Precludes **parallelization within training examples**, creating severe compute and memory bottlenecks over *extended sequence lengths*."
        ),
        "objectives": [
            "Eliminate sequential recurrence in sequence transduction models using **multi-head self-attention**.",
            "Enable **massive computational parallelization** across long token contexts.",
            "Establish new state-of-the-art translation benchmarks with <u>drastically reduced training compute</u>."
        ],
        "methodology": (
            "• <u>Encoder-Decoder Stack:</u> Composed of **6 identical encoder** and **6 decoder layers** with *d_model=512* and **8 parallel attention heads**.\n"
            "• <u>Scaled Dot-Product Attention:</u> Computes affinities via <u>Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V</u>, yielding constant path length *O(1)* across arbitrary token distances.\n"
            "• <u>Positional Encodings:</u> Employs fixed *sinusoidal wave functions* to inject token sequence order without recurrent connections."
        ),
        "dataset": (
            "• <u>WMT 2014 English-to-German:</u> Standard benchmark containing **4.5 million sentence pairs**.\n"
            "• <u>WMT 2014 English-to-French:</u> Large-scale corpus consisting of **36 million sentence pairs**.\n"
            "• <u>Shared Vocabulary:</u> Combined vocabulary of **~37,000 subword tokens** using *Byte-Pair Encoding (BPE)*."
        ),
        "preprocessing": (
            "• <u>Byte-Pair Encoding (BPE):</u> Subword segmentation ensuring a compact, out-of-vocabulary resistant vocabulary.\n"
            "• <u>Dynamic Batching:</u> Sentence pairs batched by approximate sequence length with *dynamic padding tokens*."
        ),
        "evaluation": (
            "• <u>Primary Metric:</u> Evaluated using standard **case-sensitive SacreBLEU** on *newstest2014*.\n"
            "• <u>Decoding Strategy:</u> Beam search with **beam width 4** and length penalty *alpha = 0.6*.\n"
            "• <u>Hardware & Training:</u> Trained on **8 NVIDIA P100 GPUs** for *3.5 days* (Transformer Base)."
        ),
        "results": [
            "Achieved **28.4 BLEU** on *WMT 2014 English-to-German*, establishing a new state-of-the-art by <u>+2.0 BLEU</u>.",
            "Attained **41.8 BLEU** on *WMT 2014 English-to-French* with **substantially lower training cost** than preceding architectures.",
            "Demonstrated competitive parsing accuracy of **91.3 F1** on the *Penn Treebank* dataset in zero-shot transfer."
        ],
        "conclusion": (
            "• <u>Self-Attention Sufficiency:</u> Proves that self-attention mechanisms *alone* can replace recurrence in sequence transduction.\n"
            "• <u>Efficiency & Quality:</u> Establishes both **superior translation fidelity** and **significantly faster training convergence**."
        ),
        "strengths": [
            "Complete elimination of sequential compute bottlenecks enables full **GPU matrix parallelization**.",
            "Direct path length **O(1)** between any two token positions minimizes *vanishing gradient difficulties*.",
            "Multi-head mechanism allows joint attendance across **distinct representation subspaces**."
        ],
        "limitations": [
            "Quadratic **O(N^2)** memory and time complexity relative to sequence length *N*.",
            "Absence of native inductive bias for *relative token proximity* or *hierarchical syntax trees*.",
            "High sensitivity to *learning rate warmup schedules* and *dropout calibration*."
        ],
        "research_gaps": [
            "<u>Long-Context Tractability:</u> Efficient scaling for document-level contexts beyond *512–1024 tokens*.",
            "<u>Low-Resource Generalization:</u> Cross-lingual transfer robustness under *sparse parallel training corpora*."
        ],
        "suggestions": [
            "Investigate **linear-attention** or **sparse attention kernels** to alleviate quadratic memory constraints.",
            "Incorporate **adaptive relative positional embeddings** (e.g. RoPE or ALiBi) rather than fixed sinusoids."
        ],
        "future_work": [
            "Extending attention architectures to non-text modalities including *audio*, *images*, and *video*.",
            "Exploring local and restricted attention spans for *streaming infinite sequence inputs*."
        ],
        "experiments": [
            {
                "title": "Sparse Attention Kernel for Long Document Translation",
                "hypothesis": "Restricting attention to local windows with random global routing will maintain BLEU while halving memory footprint.",
                "independent_variables": "Attention sparsity pattern (dense vs block-sparse).",
                "dependent_variables": "BLEU score on WMT14, Peak GPU VRAM, Training throughput.",
                "dataset": "WMT 2014 English-to-German extended with paragraph contexts.",
                "baseline": "Standard 6-layer Transformer Base (Vaswani et al.).",
                "evaluation_metrics": "SacreBLEU, Memory allocation (GB), Inference latency (ms).",
                "expected_outcome": "Memory reduction from quadratic to O(N sqrt(N)) with less than 0.3 BLEU degradation.",
                "risks_and_limitations": "Engineering complex CUDA kernels for irregular sparse memory access."
            }
        ],
        "confidence": {
            "overall": "High",
            "reason": "Complete architectural specifications, benchmarks, and hyperparameter tables clearly stated in the paper."
        },
        "evidence_breakdown": {
            "paper_evidence": [
                "Authors report 28.4 BLEU on WMT 2014 English-to-German using 8 P100 GPUs.",
                "Model uses 6 encoder and 6 decoder layers with d_model=512 and 8 parallel attention heads.",
                "Positional encodings use fixed sine and cosine functions of varying frequencies."
            ],
            "interpretations": [
                "Self-attention acts as an implicit graph neural network where edge weights are query-key similarities.",
                "The reliance on residual layer normalization suggests deep attention representations can destabilize without skip paths."
            ],
            "ai_recommendations": [
                "Apply rotary positional embeddings (RoPE) to generalize effectively to out-of-distribution sequence lengths.",
                "Benchmark quantized integer inference for edge device deployment."
            ],
            "needs_verification": [
                "Whether self-attention alone captures linguistic compositional hierarchies without pretraining on syntactic trees."
            ]
        },
        "_metadata": {
            "filename": filename,
            "was_truncated": False,
            "original_word_count": 6420,
            "analyzed_word_count": 6420,
            "model_used": "Demo Mode (Mock)",
            "is_demo": True
        }
    }
    return validate_analysis_schema(data)


def get_demo_comparison(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return high-fidelity sample comparison without making API calls."""
    names = [p.get("filename", f"Paper {i+1}") for i, p in enumerate(papers)]
    p1 = names[0] if len(names) > 0 else "Paper 1"
    p2 = names[1] if len(names) > 1 else "Paper 2"

    table = [
        {
            "dimension": "Research Problem",
            "papers": {
                p1: "Sequential training bottlenecks in recurrent sequence transducers.",
                p2: "Inefficient contextual pretraining in unidirectional language models."
            }
        },
        {
            "dimension": "Objectives",
            "papers": {
                p1: "Eliminate recurrence using multi-head self-attention.",
                p2: "Pretrain deep bidirectional representations using masked language modeling."
            }
        },
        {
            "dimension": "Methodology",
            "papers": {
                p1: "Encoder-Decoder multi-head dot-product attention with sinusoidal position encoding.",
                p2: "Multi-layer bidirectional Transformer encoder with masked LM and next-sentence prediction."
            }
        },
        {
            "dimension": "Dataset",
            "papers": {
                p1: "WMT 2014 English-German (4.5M) and English-French (36M) sentence pairs.",
                p2: "BooksCorpus (800M words) and English Wikipedia (2,500M words)."
            }
        },
        {
            "dimension": "Preprocessing",
            "papers": {
                p1: "BPE with 37k shared token vocabulary.",
                p2: "WordPiece tokenization with 30k vocabulary."
            }
        },
        {
            "dimension": "Evaluation",
            "papers": {
                p1: "BLEU score on WMT newstest2014 with beam search.",
                p2: "GLUE benchmark (9 tasks), SQuAD v1.1/v2.0, and SWAG."
            }
        },
        {
            "dimension": "Results",
            "papers": {
                p1: "28.4 BLEU on En-De; 41.8 BLEU on En-Fr.",
                p2: "GLUE score of 80.5% (Base) and 82.1% (Large); SQuAD F1 93.2%."
            }
        },
        {
            "dimension": "Strengths",
            "papers": {
                p1: "Full parallelization and constant maximum path length between tokens.",
                p2: "True bidirectionality transfers zero-shot to diverse classification and QA downstream tasks."
            }
        },
        {
            "dimension": "Limitations",
            "papers": {
                p1: "Quadratic memory scaling O(N^2) with sequence length.",
                p2: "High pretraining compute cost and pretrain-finetune discrepancy from [MASK] tokens."
            }
        }
    ]

    return {
        "comparison_table": table,
        "similarities": [
            "Both architectures rely on multi-head dot-product self-attention as the core computational primitive.",
            "Both approaches replace recurrent gating (LSTM/GRU) to achieve parallel tensor operations during training.",
            "Both evaluate representations on established public NLP benchmarks (WMT and GLUE)."
        ],
        "differences": [
            f"'{p1}' employs an autoregressive Encoder-Decoder for sequence generation, whereas '{p2}' focuses on bidirectional encoder representations for language understanding.",
            f"'{p1}' trains from scratch on supervised translation pairs, while '{p2}' leverages self-supervised pretraining on unlabeled corpora."
        ],
        "contradictions_inconsistencies": [
            "No direct factual contradictions identified. The studies address complementary aspects of representation learning (generation vs understanding)."
        ],
        "unresolved_areas": [
            "Memory complexity scaling when combining bidirectional pretraining with generative decoder sequence lengths beyond 4096 tokens."
        ],
        "combined_research_opportunity": (
            "A unified encoder-decoder architecture that merges masked span corruption with autoregressive decoding to support "
            "both high-throughput classification and long-form scientific text generation."
        ),
        "duplicate_warnings": check_for_duplicates(papers),
        "paper_names": names,
        "is_demo": True
    }


def get_demo_gaps(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return high-fidelity sample research gap detection without making API calls."""
    names = [p.get("filename", f"Paper {i+1}") for i, p in enumerate(papers)]
    p1 = names[0] if len(names) > 0 else "Paper 1"
    p2 = names[1] if len(names) > 1 else "Paper 2"

    matrix = [
        {
            "area": "Dataset",
            "paper_ratings": {p1: "General Web & News corpora", p2: "Books & Wikipedia"},
            "potential_gap": "Potential research gap: Lack of specialized scientific and mathematical domain benchmark evaluation."
        },
        {
            "area": "Model",
            "paper_ratings": {p1: "Encoder-Decoder", p2: "Encoder-only"},
            "potential_gap": "Potential research gap: Unified parameter-efficient architectures that handle both reasoning and extraction."
        },
        {
            "area": "Explainability",
            "paper_ratings": {p1: "Attention head weight heatmaps", p2: "Layer probing"},
            "potential_gap": "Potential research gap: Mechanistic interpretability of factual recall vs spurious correlation."
        },
        {
            "area": "Robustness",
            "paper_ratings": {p1: "Adversarial noise not tested", p2: "Evaluated on out-of-domain GLUE"},
            "potential_gap": "Potential research gap: Vulnerability to adversarial perturbations and synthetic hallucinations."
        },
        {
            "area": "Real-world testing",
            "paper_ratings": {p1: "Synthetic benchmark only", p2: "Static academic evaluations"},
            "potential_gap": "Potential research gap: Deployment latency on edge hardware and resource-constrained devices."
        },
        {
            "area": "Evaluation",
            "paper_ratings": {p1: "BLEU score", p2: "Accuracy, F1 score"},
            "potential_gap": "Potential research gap: Reliance on surface n-gram overlap metrics rather than semantic fidelity."
        },
        {
            "area": "Generalization",
            "paper_ratings": {p1: "High resource pairs only", p2: "English-only pretraining"},
            "potential_gap": "Potential research gap: Cross-lingual zero-shot transfer to morphologically rich low-resource languages."
        }
    ]

    gaps = [
        {
            "title": "Quadratic Context Complexity Bottleneck in Multi-Document Reasoning",
            "priority": "HIGH",
            "description": "Both architectures suffer from O(N^2) memory scaling, preventing direct end-to-end reasoning over entire multi-chapter academic papers or literature corpora.",
            "evidence_from_papers": "Self-attention requires computing all pair-wise token affinities, limiting context length to 512 tokens in original implementations.",
            "why_it_matters": "Real-world research synthesis requires processing 20+ pages simultaneously without heuristic truncation.",
            "suggested_direction": "Integrate hierarchical state-space models (SSMs) or linear attention approximations.",
            "confidence": "High",
            "needs_verification": True
        },
        {
            "title": "Absence of Semantic Attribution Grounding in Token Prediction",
            "priority": "MEDIUM",
            "description": "Models predict tokens statistically without verifiable citations pointing back to source paragraph coordinate offsets.",
            "evidence_from_papers": "Neither study provides a mechanism to verify which training sentence justified a specific generated token.",
            "why_it_matters": "Crucial for academic integrity, peer review, and preventing fabricated citations.",
            "suggested_direction": "Implement retrieved-evidence masking with differentiable citation pointer networks.",
            "confidence": "Medium",
            "needs_verification": True
        },
        {
            "title": "Low-Resource Multilingual Alignment Disparity",
            "priority": "LOW",
            "description": "Evaluations are restricted to high-resource languages (English, German, French), leaving low-resource language performance unmeasured.",
            "evidence_from_papers": "WMT benchmarks used contain millions of sentence pairs, unrepresentative of languages with sparse parallel data.",
            "why_it_matters": "Limits global access to automated scientific literature discovery.",
            "suggested_direction": "Explore cross-lingual transfer using shared universal lexical anchors.",
            "confidence": "Medium",
            "needs_verification": True
        }
    ]

    return {
        "matrix": matrix,
        "gaps": gaps,
        "synthesis_summary": (
            "Potential research gap based on the supplied papers. Broader literature review is required to establish novelty. "
            "Across the examined papers, the primary opportunities center on scaling beyond quadratic attention limits and "
            "enforcing strict citation provenance for factual claims."
        ),
        "duplicate_warnings": check_for_duplicates(papers),
        "paper_names": names,
        "is_demo": True
    }


# =========================================================================
# PUBLIC AI SERVICE METHODS
# =========================================================================

def analyze_paper(
    text: str,
    filename: str = "Research Paper",
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    max_words: int = 12000,
    demo_mode: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform deep academic analysis of a single research paper using Google Gemini API or Demo Mode.
    Intelligently truncates text if exceeding limit and requests structured JSON output.
    """
    if demo_mode:
        return get_demo_single_analysis(filename=filename)

    if not text or not text.strip():
        raise AnalysisError("No readable text available to analyze. The document is empty or non-extractable.")

    # Intelligent head/tail preservation
    processed_text, was_truncated, orig_words, final_words = truncate_text_intelligently(
        text, max_words=max_words
    )

    prompt = SINGLE_PAPER_ANALYSIS_PROMPT.format(
        filename=filename,
        text=processed_text
    )

    json_str = call_gemini_json(
        prompt=prompt,
        system_instruction=SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
        api_key=api_key,
        model=model,
        temperature=0.2
    )

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse structured JSON response from Gemini: {str(je)}")

    validated = validate_analysis_schema(parsed)

    validated["_metadata"] = {
        "filename": filename,
        "was_truncated": was_truncated,
        "original_word_count": orig_words,
        "analyzed_word_count": final_words,
        "model_used": model,
        "is_demo": False
    }

    return validated


def compare_papers(
    papers: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    demo_mode: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare 2 to 5 research papers across standard academic dimensions using Google Gemini API or Demo Mode.
    Returns structured comparison table, similarities, differences, contradictions, and joint opportunities.
    """
    if len(papers) < 2:
        raise ValueError(f"At least 2 papers are required for comparison. Received {len(papers)}.")
    if len(papers) > 5:
        raise ValueError(f"A maximum of 5 papers can be compared at once. Received {len(papers)}.")

    if demo_mode:
        return get_demo_comparison(papers=papers)

    paper_names = [p.get("filename", f"Paper_{i+1}") for i, p in enumerate(papers)]
    duplicate_warnings = check_for_duplicates(papers)

    # Budget words per paper based on count to keep prompt concise
    budget = int(12000 / len(papers))
    papers_summary_text = format_papers_summary(papers, words_per_paper=budget)

    prompt = PAPER_COMPARISON_PROMPT.format(
        papers_summary=papers_summary_text,
        paper_names_example='", "'.join(paper_names)
    )

    json_str = call_gemini_json(
        prompt=prompt,
        system_instruction=SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
        api_key=api_key,
        model=model,
        temperature=0.2
    )

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse structured comparison output from Gemini: {str(je)}")

    validated = validate_comparison_schema(parsed, paper_names)
    validated["duplicate_warnings"] = duplicate_warnings
    validated["paper_names"] = paper_names
    validated["is_demo"] = False

    return validated


def detect_research_gaps(
    papers: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    demo_mode: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform cross-paper research gap detection and generate the Research Gap Matrix using Google Gemini API or Demo Mode.
    """
    if len(papers) < 2:
        raise ValueError(f"At least 2 papers are required for research gap detection. Received {len(papers)}.")
    if len(papers) > 5:
        raise ValueError(f"A maximum of 5 papers can be analyzed for research gaps. Received {len(papers)}.")

    if demo_mode:
        return get_demo_gaps(papers=papers)

    paper_names = [p.get("filename", f"Paper_{i+1}") for i, p in enumerate(papers)]
    duplicate_warnings = check_for_duplicates(papers)

    budget = int(12000 / len(papers))
    papers_summary_text = format_papers_summary(papers, words_per_paper=budget)

    prompt = RESEARCH_GAP_PROMPT.format(
        papers_summary=papers_summary_text,
        paper_names_example='", "'.join(paper_names)
    )

    json_str = call_gemini_json(
        prompt=prompt,
        system_instruction=SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
        api_key=api_key,
        model=model,
        temperature=0.25
    )

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as je:
        raise AnalysisError(f"Failed to parse research gap response from Gemini: {str(je)}")

    validated = validate_research_gap_schema(parsed, paper_names)
    validated["duplicate_warnings"] = duplicate_warnings
    validated["paper_names"] = paper_names
    validated["is_demo"] = False

    return validated


def generate_recommendations(
    analysis_data: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    demo_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate evidence-grounded practical research recommendations based on paper analysis.
    """
    if demo_mode:
        return [
            {
                "recommendation": "Integrate sparse-attention kernel benchmarks on multi-page PDF contexts.",
                "reason": "Alleviates O(N^2) memory limits evidenced in the paper's self-attention limitations.",
                "supporting_evidence": "Documented in paper limitations and quadratic complexity section.",
                "expected_benefit": "Enables processing long-sequence academic papers with 50% lower VRAM.",
                "priority": "High",
                "confidence": "High"
            },
            {
                "recommendation": "Apply calibrated relative positional encodings (e.g. RoPE or ALiBi).",
                "reason": "Improves length extrapolation when sequence exceeds training horizon.",
                "supporting_evidence": "Authors noted sensitivity to fixed sinusoidal frequencies.",
                "expected_benefit": "Maintains perplexity and BLEU on long out-of-domain sequences.",
                "priority": "Medium",
                "confidence": "Medium"
            }
        ]

    try:
        prompt = RECOMMENDATION_PROMPT.format(
            analysis_json=json.dumps({
                "problem": analysis_data.get("research_problem"),
                "methodology": analysis_data.get("methodology"),
                "limitations": analysis_data.get("limitations", []),
                "results": analysis_data.get("results", []),
                "research_gaps": analysis_data.get("research_gaps", [])
            }, indent=2)
        )

        json_str = call_gemini_json(
            prompt=prompt,
            system_instruction=SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
            api_key=api_key,
            model=model,
            temperature=0.3
        )
        parsed = json.loads(json_str)
        return parsed.get("recommendations", [])

    except Exception:
        # Graceful fallback to existing suggestions from analysis if secondary call fails
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
    model: str = DEFAULT_GEMINI_MODEL,
    demo_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Formulate structured experiment plans addressing paper limitations.
    """
    if demo_mode:
        return analysis_data.get("experiments", [])

    try:
        prompt = EXPERIMENT_PROMPT.format(
            analysis_json=json.dumps({
                "problem": analysis_data.get("research_problem"),
                "methodology": analysis_data.get("methodology"),
                "dataset": analysis_data.get("dataset"),
                "limitations": analysis_data.get("limitations", []),
                "future_work": analysis_data.get("future_work", [])
            }, indent=2)
        )

        json_str = call_gemini_json(
            prompt=prompt,
            system_instruction=SYSTEM_ACADEMIC_EVIDENCE_PROMPT,
            api_key=api_key,
            model=model,
            temperature=0.3
        )
        parsed = json.loads(json_str)
        return parsed.get("experiments", [])

    except Exception:
        return analysis_data.get("experiments", [])
