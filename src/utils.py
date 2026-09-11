"""
Utility functions for ResearchLens AI.
Handles validation, token estimation, intelligent truncation, filename sanitization,
JSON markdown stripping, and UI badge formatting.
"""

import re
import html
from typing import Tuple, Dict, Any, Optional

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension from filename."""
    dot_idx = filename.rfind(".")
    if dot_idx != -1:
        return filename[dot_idx:].lower()
    return ""


def validate_uploaded_file(file_obj, filename: str, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> Dict[str, Any]:
    """
    Validate uploaded file format and size.
    Returns dict with 'valid': bool, 'error': str, 'ext': str, 'size_bytes': int.
    """
    ext = get_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Unsupported file extension '{ext}'. Only PDF (.pdf), Word (.docx), and Plain Text (.txt) files are supported.",
            "ext": ext,
            "size_bytes": 0
        }

    # Check size
    try:
        # Streamlit UploadedFile has .size or file_obj can be seeked
        if hasattr(file_obj, "size"):
            size_bytes = file_obj.size
        elif hasattr(file_obj, "getbuffer"):
            size_bytes = len(file_obj.getbuffer())
        elif hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
            curr = file_obj.tell()
            file_obj.seek(0, 2)
            size_bytes = file_obj.tell()
            file_obj.seek(curr)
        else:
            size_bytes = 0
    except Exception:
        size_bytes = 0

    if size_bytes == 0 and hasattr(file_obj, "size") and file_obj.size == 0:
        return {
            "valid": False,
            "error": "The uploaded file is empty (0 bytes). Please upload a valid document.",
            "ext": ext,
            "size_bytes": 0
        }

    if size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        return {
            "valid": False,
            "error": f"File size ({actual_mb:.1f} MB) exceeds maximum allowed size of {max_mb:.0f} MB.",
            "ext": ext,
            "size_bytes": size_bytes
        }

    return {
        "valid": True,
        "error": "",
        "ext": ext,
        "size_bytes": size_bytes
    }


def estimate_token_count(text: str) -> int:
    """
    Estimate token count using the standard ~0.75 words/token heuristic (approx 4 chars per token).
    """
    if not text:
        return 0
    words = len(text.split())
    # 1 word ~ 1.33 tokens
    return int(words * 1.33)


def truncate_text_intelligently(text: str, max_words: int = 12000) -> Tuple[str, bool, int, int]:
    """
    Intelligently truncate long documents preserving the introduction/methodology
    at the start and results/conclusion at the end.

    Returns:
        (truncated_text, was_truncated, original_word_count, final_word_count)
    """
    if not text:
        return "", False, 0, 0

    words = text.split()
    total_words = len(words)

    if total_words <= max_words:
        return text, False, total_words, total_words

    # Preserve 60% head (abstract, intro, methodology) and 40% tail (results, discussion, conclusion)
    head_count = int(max_words * 0.6)
    tail_count = max_words - head_count

    head_words = words[:head_count]
    tail_words = words[-tail_count:]

    omitted_count = total_words - (head_count + tail_count)
    delimiter = (
        f"\n\n[... TRUNCATION NOTICE: {omitted_count:,} words from the middle body were omitted "
        f"to comply with AI context limits. The paper's beginning ({head_count:,} words: Abstract, "
        f"Introduction, Problem Formulation) and conclusion ({tail_count:,} words: Results, Discussion, "
        f"Limitations, Conclusion) have been strictly preserved. ...]\n\n"
    )

    final_text = " ".join(head_words) + delimiter + " ".join(tail_words)
    final_word_count = len(final_text.split())

    return final_text, True, total_words, final_word_count


def clean_json_markdown(raw_str: str) -> str:
    """
    Remove markdown code block wrappers (```json ... ```) from LLM output if present.
    """
    s = raw_str.strip()
    if s.startswith("```"):
        # Match ```json or ``` and end ```
        match = re.search(r"^```(?:json)?\s*([\s\S]*?)\s*```$", s, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return s


def sanitize_filename(name: str, max_len: int = 40) -> str:
    """
    Sanitize a filename for safe download across operating systems.
    """
    # Remove file extension if present
    base = name.rsplit(".", 1)[0]
    # Keep alphanumeric, underscores and dashes
    clean = re.sub(r"[^\w\-]", "_", base)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "research_paper"
    return clean[:max_len]


def render_badge(label: str, badge_type: str = "default") -> str:
    """
    Generate an HTML badge snippet for Streamlit UI rendering.
    badge_type can be: 'evidence', 'interpretation', 'recommendation', 'verification',
                       'high', 'medium', 'low', 'default'
    """
    color_map = {
        "evidence": ("#E0F2FE", "#0369A1", "📄 Paper Evidence"),
        "interpretation": ("#FEF3C7", "#92400E", "🔎 Interpretation"),
        "recommendation": ("#DCFCE7", "#15803D", "💡 AI Recommendation"),
        "verification": ("#FEE2E2", "#B91C1C", "⚠️ Needs Verification"),
        "high": ("#FEE2E2", "#991B1B", "HIGH"),
        "medium": ("#FEF3C7", "#92400E", "MEDIUM"),
        "low": ("#E0F2FE", "#075985", "LOW"),
        "default": ("#F1F5F9", "#475569", label)
    }

    bg, fg, default_text = color_map.get(badge_type.lower(), ("#F1F5F9", "#475569", label))
    display_text = default_text if badge_type.lower() in ("evidence", "interpretation", "recommendation", "verification") else label

    return (
        f'<span style="background-color: {bg}; color: {fg}; '
        f'padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; '
        f'font-weight: 600; display: inline-block; margin-right: 6px; '
        f'border: 1px solid {fg}33;">{html.escape(display_text)}</span>'
    )
