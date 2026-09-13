"""
Unit tests for AI Analyzer module (Google Gemini client mocking, schema validation, error handling, Demo Mode).
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.analyzer import (
    analyze_paper,
    generate_recommendations,
    generate_experiments,
    validate_analysis_schema,
    resolve_api_key,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_MODEL,
    DEFAULT_GEMINI_MODEL
)
from src.utils import truncate_text_intelligently


MOCK_VALID_ANALYSIS = {
    "executive_summary": "A study on sparse attention mechanisms.",
    "research_problem": "Quadratic memory scaling in full self-attention.",
    "objectives": ["Reduce memory complexity to O(N log N)"],
    "methodology": "Sparse block-diagonal attention matrices.",
    "dataset": "WikiText-103",
    "preprocessing": "Standard BPE tokenization.",
    "evaluation": "Perplexity and latency benchmarks.",
    "results": ["Achieved 18.2 perplexity with 40% less memory."],
    "conclusion": "Sparse attention offers linear scalability with minimal quality loss.",
    "strengths": ["Scalable to 16k context window"],
    "limitations": ["Requires custom CUDA kernels"],
    "research_gaps": ["Unproven on multi-modal tasks"],
    "suggestions": ["Benchmark on visual transformers"],
    "future_work": ["Extend to cross-attention"],
    "experiments": [
        {
            "title": "Vision Transformer Test",
            "hypothesis": "Sparse attention will reduce memory on ImageNet-1k.",
            "independent_variables": "Sparsity factor k",
            "dependent_variables": "Top-1 accuracy and peak VRAM",
            "dataset": "ImageNet-1k",
            "baseline": "ViT-Base",
            "evaluation_metrics": "Top-1 Acc, Memory MB",
            "expected_outcome": "Within 0.5% accuracy with 30% lower VRAM",
            "risks_and_limitations": "Training stability issues"
        }
    ],
    "confidence": {
        "overall": "High",
        "reason": "Complete reporting of methodology and benchmarks."
    },
    "evidence_breakdown": {
        "paper_evidence": ["Achieved 18.2 perplexity on WikiText-103."],
        "interpretations": ["Memory savings will be more pronounced at longer context lengths."],
        "ai_recommendations": ["Evaluate on bidirectional translation."],
        "needs_verification": ["Claim of constant memory overhead requires verification."]
    }
}


class TestSchemaValidation:
    def test_validate_complete_schema(self):
        validated = validate_analysis_schema(MOCK_VALID_ANALYSIS)
        assert validated["research_problem"] == "Quadratic memory scaling in full self-attention."
        assert len(validated["objectives"]) == 1
        assert validated["confidence"]["overall"] == "High"

    def test_validate_partial_schema_fills_defaults(self):
        partial = {"executive_summary": "Quick summary."}
        validated = validate_analysis_schema(partial)
        assert validated["executive_summary"] == "Quick summary."
        assert validated["research_problem"] == "Not clearly stated in the paper."
        assert isinstance(validated["objectives"], list)
        assert validated["confidence"]["overall"] == "Medium"
        assert "paper_evidence" in validated["evidence_breakdown"]


class TestIntelligentTruncation:
    def test_truncation_preserves_head_and_tail(self):
        words = [f"word_{i}" for i in range(1000)]
        text = " ".join(words)

        truncated, was_trunc, orig_count, final_count = truncate_text_intelligently(text, max_words=200)
        assert was_trunc is True
        assert orig_count == 1000
        assert "word_0" in truncated  # Head preserved
        assert "word_999" in truncated  # Tail preserved
        assert "TRUNCATION NOTICE" in truncated

    def test_truncation_skips_short_text(self):
        text = "Short academic text with twenty words or less."
        truncated, was_trunc, orig_count, final_count = truncate_text_intelligently(text, max_words=100)
        assert was_trunc is False
        assert truncated == text


class TestAnalyzerAPIExecution:
    def test_missing_api_key_raises_error(self):
        with patch("streamlit.secrets", {}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(MissingAPIKeyError):
                    resolve_api_key(None)

    def test_analyze_paper_empty_text_raises_error(self):
        with pytest.raises(AnalysisError) as exc_info:
            analyze_paper(text="", api_key="valid-key")
        assert "No readable text" in str(exc_info.value)

    def test_analyze_paper_success_mocked(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_VALID_ANALYSIS)

        with patch("src.ai_service.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = analyze_paper(
                text="Sample paper text on sparse attention.",
                filename="paper_test.txt",
                api_key="test-key"
            )

            assert result["research_problem"] == "Quadratic memory scaling in full self-attention."
            assert result["_metadata"]["filename"] == "paper_test.txt"
            assert result["_metadata"]["was_truncated"] is False
            assert result["_metadata"]["is_demo"] is False

    def test_analyze_paper_demo_mode(self):
        # Demo mode should succeed without any client mocking or API keys
        result = analyze_paper(
            text="Any text",
            filename="DemoPaper.pdf",
            demo_mode=True
        )
        assert result["_metadata"]["is_demo"] is True
        assert "Transformer architecture" in result["executive_summary"]
        assert len(result["objectives"]) >= 2
        assert result["confidence"]["overall"] == "High"

    def test_analyze_paper_handles_auth_error(self):
        with patch("src.ai_service.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("403 API_KEY_INVALID: User not registered")
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                analyze_paper(text="Some text", api_key="bad-key")
            assert "Invalid Gemini API key" in str(exc_info.value)

    def test_analyze_paper_handles_rate_limit(self):
        with patch("src.ai_service.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                analyze_paper(text="Some text", api_key="test-key")
            assert "quota/rate limit" in str(exc_info.value).lower()

    def test_analyze_paper_handles_malformed_json(self):
        mock_response = MagicMock()
        mock_response.text = "This is not json { [ broken"

        with patch("src.ai_service.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                analyze_paper(text="Some text", api_key="test-key")
            assert "failed to parse structured json response" in str(exc_info.value).lower()

    def test_generate_recommendations_and_experiments_demo_mode(self):
        recs = generate_recommendations(MOCK_VALID_ANALYSIS, demo_mode=True)
        assert isinstance(recs, list)
        assert len(recs) >= 1
        assert "recommendation" in recs[0]

        exps = generate_experiments(MOCK_VALID_ANALYSIS, demo_mode=True)
        assert isinstance(exps, list)
        assert len(exps) == 1
