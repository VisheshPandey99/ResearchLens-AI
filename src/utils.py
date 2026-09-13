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


import textwrap


def clean_html(html_str: str) -> str:
    """
    Dedent and strip HTML strings to ensure Streamlit st.markdown
    never interprets leading spaces as a markdown code block.
    """
    if not html_str:
        return ""
    return textwrap.dedent(html_str).strip()


def markdown_to_html_spans(text: str) -> str:
    """
    Convert markdown bold (**), italic (*), and underline (<u>) into styled HTML spans.
    Preserves existing <u> tags and enhances them with high-contrast academic colors.
    Converts HTML tags (<b>, <strong>, <i>, <em>) into unified visual styling.
    Escapes all other HTML characters to prevent rendering bugs and injection.
    """
    if not text:
        return ""

    # Normalize internal newlines to spaces to prevent multi-line indentation issues
    cleaned_text = re.sub(r"[\r\n]+", " ", str(text))

    # Strip dangerous or disruptive block tags
    cleaned_text = re.sub(r"</?(?:p|div|section|article|blockquote)\b[^>]*>", " ", cleaned_text)
    cleaned_text = re.sub(r"<br\s*/?>", " ", cleaned_text)

    # Normalize HTML formatting tags to standard markdown syntax before escaping
    cleaned_text = re.sub(r"</?(?:b|strong)\b[^>]*>", "**", cleaned_text)
    cleaned_text = re.sub(r"</?(?:i|em)\b[^>]*>", "*", cleaned_text)
    cleaned_text = re.sub(r"</?code\b[^>]*>", "`", cleaned_text)

    # Split preserving <u> and </u> tags
    parts = re.split(r"(</?[uU]>)", cleaned_text)
    escaped_parts = []
    for p in parts:
        if p.lower() in ("<u>", "</u>"):
            escaped_parts.append(p.lower())
        else:
            escaped_parts.append(html.escape(p))
    out = "".join(escaped_parts)

    # Style <u> tags with eye-catching underline and cyan accent
    out = re.sub(
        r"<u>(.*?)</u>",
        r'<u style="color: #38BDF8; text-decoration: underline; text-decoration-color: #38BDF8; text-underline-offset: 4px; font-weight: 700;">\1</u>',
        out,
        flags=re.IGNORECASE
    )

    # Convert **bold** to crisp bright white
    out = re.sub(
        r"\*\*(.*?)\*\*",
        r'<strong style="color: #FFFFFF; font-weight: 700;">\1</strong>',
        out
    )

    # Convert *italic* to subtle slate
    out = re.sub(
        r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)",
        r'<em style="color: #CBD5E1; font-style: italic;">\1</em>',
        out
    )

    # Convert `code` to formatted badge
    out = re.sub(
        r"`(.*?)`",
        r'<code style="background: rgba(56, 189, 248, 0.12); color: #38BDF8; padding: 2px 6px; border-radius: 6px; font-family: monospace; font-size: 0.88em;">\1</code>',
        out
    )

    return out.strip()


def render_bullet_cards_html(content: Any, default_icon: str = "•", accent: str = "cyan") -> str:
    """
    Convert text or list into modern glassmorphic bullet cards with bold, italic, and underline styling.
    Renders pure inline/block HTML without leading whitespace or blank lines to avoid markdown code-block bugs.
    """
    accent_styles = {
        "cyan": {"border": "#38BDF8", "bg": "rgba(56, 189, 248, 0.14)", "text": "#38BDF8"},
        "emerald": {"border": "#34D399", "bg": "rgba(52, 211, 153, 0.14)", "text": "#34D399"},
        "amber": {"border": "#FBBF24", "bg": "rgba(251, 191, 36, 0.14)", "text": "#FBBF24"},
        "purple": {"border": "#A855F7", "bg": "rgba(168, 85, 247, 0.14)", "text": "#A855F7"},
        "rose": {"border": "#F87171", "bg": "rgba(248, 113, 113, 0.14)", "text": "#F87171"},
        "indigo": {"border": "#818CF8", "bg": "rgba(129, 140, 248, 0.14)", "text": "#818CF8"}
    }
    style = accent_styles.get(accent, accent_styles["cyan"])

    raw_items = []

    def _add_cleaned_line(line_str: str):
        cleaned = re.sub(r"^[•\-\*\d\.]+\s*", "", line_str.strip()).strip()
        if cleaned:
            raw_items.append(cleaned)

    if isinstance(content, list):
        for it in content:
            if isinstance(it, str) and it.strip():
                sub_lines = [l.strip() for l in it.strip().split("\n") if l.strip()]
                for sl in sub_lines:
                    _add_cleaned_line(sl)
            elif isinstance(it, dict):
                formatted_dict = ", ".join(f"**{k}:** {v}" for k, v in it.items())
                raw_items.append(formatted_dict)
    elif isinstance(content, str) and content.strip():
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        has_bullets = any(line.startswith(("•", "-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")) for line in lines)
        if has_bullets or len(lines) > 1:
            for l in lines:
                _add_cleaned_line(l)
        else:
            # If single paragraph, split by sentence boundaries if extensive
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", content.strip())
            if len(sentences) > 1 and len(content.strip().split()) > 35:
                raw_items = [s.strip() for s in sentences if s.strip()]
            else:
                raw_items = [content.strip()]

    if not raw_items:
        return '<div style="color: #94A3B8; font-style: italic; padding: 12px 16px; background: rgba(15, 23, 42, 0.5); border-radius: 10px;">Not clearly stated in the paper.</div>'

    cards = []
    for item in raw_items:
        clean_item = re.sub(r"\s+", " ", item).strip()
        formatted = markdown_to_html_spans(clean_item)
        cards.append(
            f'<div class="bullet-card-item" style="border-left: 3.5px solid {style["border"]};">'
            f'<div class="bullet-icon-box" style="background: {style["bg"]}; color: {style["text"]};">{default_icon}</div>'
            f'<div class="bullet-content-box">{formatted}</div>'
            f'</div>'
        )

    return f'<div class="bullet-card-list">{"".join(cards)}</div>'
