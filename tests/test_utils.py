"""
Unit tests for src/utils.py formatting and bullet cards rendering.
Ensures no markdown code-block indentation triggers exist and styling produces refined output.
"""

import re
import pytest
from src.utils import (
    markdown_to_html_spans,
    render_bullet_cards_html,
    clean_html,
    estimate_token_count,
    truncate_text_intelligently,
    clean_json_markdown,
    sanitize_filename,
    render_badge
)
from src.report import strip_html_tags


def test_clean_html():
    indented = """
        <div class="test">
            <p>Hello</p>
        </div>
    """
    cleaned = clean_html(indented)
    assert cleaned.startswith('<div class="test">')
    assert not cleaned.startswith("    ")


def test_markdown_to_html_spans_basic():
    text = "This is **bold** and *italic* and `code`."
    out = markdown_to_html_spans(text)
    assert '<strong style="color: #FFFFFF; font-weight: 700;">bold</strong>' in out
    assert '<em style="color: #CBD5E1; font-style: italic;">italic</em>' in out
    assert '<code style="' in out
    assert "\n" not in out


def test_markdown_to_html_spans_underline():
    text = "• <u>Core Bottleneck:</u> The primary issue."
    out = markdown_to_html_spans(text)
    assert '<u style="color: #38BDF8; text-decoration: underline;' in out
    assert "Core Bottleneck:</u>" in out


def test_markdown_to_html_spans_html_tag_normalization():
    # LLM might use <b> or <i> or <p> tags
    text = "<p>Findings show <b>high performance</b> in <i>athletic</i> tests.</p>"
    out = markdown_to_html_spans(text)
    assert "<p>" not in out
    assert "</p>" not in out
    assert '<strong style="color: #FFFFFF; font-weight: 700;">high performance</strong>' in out
    assert '<em style="color: #CBD5E1; font-style: italic;">athletic</em>' in out


def test_render_bullet_cards_html_list_of_strings():
    items = [
        "Core Bottleneck: Performance scaling issue.",
        "Limitation: Requires extensive memory."
    ]
    html_out = render_bullet_cards_html(items, default_icon="🎯", accent="amber")
    assert html_out.startswith('<div class="bullet-card-list">')
    assert html_out.endswith('</div>')
    # Count bullet-card-item occurrences
    assert html_out.count('class="bullet-card-item"') == 2
    assert html_out.count("🎯") == 2
    # Verify NO leading whitespace indentation on lines that would trigger markdown code blocks
    lines = html_out.split("\n")
    for line in lines:
        assert not line.startswith("    "), f"Line has 4+ spaces of indentation: {repr(line)}"
    # Verify no blank lines inside the HTML
    assert "\n\n" not in html_out


def test_render_bullet_cards_html_string_with_bullets():
    content = (
        "• <u>Core Bottleneck:</u> Determining whether gains from **creatine** translate.\n"
        "• <u>Limitation Addressed:</u> Evaluating long-term adaptations."
    )
    html_out = render_bullet_cards_html(content, default_icon="🎯", accent="amber")
    assert html_out.count('class="bullet-card-item"') == 2
    assert "•" not in html_out.replace('default_icon = "•"', '') # Bullet prefix stripped from card content
    assert '<u style="color: #38BDF8;' in html_out
    assert '<strong style="color: #FFFFFF;' in html_out
    # Ensure no indented code block triggers
    assert not re.search(r"\n[ \t]{4,}<div", html_out)


def test_render_bullet_cards_html_objectives_multiple_items():
    objs = [
        "Synthesize research on the effects of short-term **creatine supplementation**.",
        "Evaluate long-term adaptations to **creatine supplementation**.",
        "Assess the overall safety profile and therapeutic potential."
    ]
    html_out = render_bullet_cards_html(objs, default_icon="📌", accent="indigo")
    assert html_out.count('class="bullet-card-item"') == 3
    assert html_out.count("📌") == 3
    assert "\n\n" not in html_out
    assert not re.search(r"\n\s*\n\s*<div", html_out)


def test_render_bullet_cards_html_empty():
    html_out = render_bullet_cards_html([])
    assert "Not clearly stated in the paper." in html_out
    html_out_none = render_bullet_cards_html(None)
    assert "Not clearly stated in the paper." in html_out_none


def test_strip_html_tags():
    raw = "• <u>Core Bottleneck:</u> Determines <b>long-term</b> adaptations."
    cleaned = strip_html_tags(raw)
    assert cleaned == "• Core Bottleneck: Determines long-term adaptations."
    assert "<" not in cleaned
    assert ">" not in cleaned
