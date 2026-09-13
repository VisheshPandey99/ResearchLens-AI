"""
Unit tests for src/ai_service.py (Google Gemini API integration, key resolution, error mapping, and Demo Mode).
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from src.ai_service import (
    resolve_gemini_api_key,
    get_gemini_client,
    map_gemini_error,
    call_gemini_json,
    get_demo_single_analysis,
    get_demo_comparison,
    get_demo_gaps,
    MissingAPIKeyError,
    AnalysisError,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS
)


class TestKeyResolution:
    def test_explicit_argument_takes_precedence(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key"}):
            key = resolve_gemini_api_key("explicit_key")
            assert key == "explicit_key"

    def test_env_var_fallback(self):
        with patch("streamlit.secrets", {}):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTestKeyFromEnv"}):
                key = resolve_gemini_api_key(None)
                assert key == "AIzaSyTestKeyFromEnv"

    def test_missing_key_raises_informative_error(self):
        with patch("streamlit.secrets", {}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(MissingAPIKeyError) as exc_info:
                    resolve_gemini_api_key(None)
                assert "GEMINI_API_KEY" in str(exc_info.value)
                assert "secrets.toml" in str(exc_info.value)


class TestErrorMapping:
    def test_map_quota_error(self):
        err = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric")
        mapped = map_gemini_error(err, "gemini-2.5-flash")
        assert "quota/rate limit" in str(mapped).lower()

    def test_map_invalid_key_error(self):
        err = Exception("403 API_KEY_INVALID: API key not valid")
        mapped = map_gemini_error(err, "gemini-2.5-flash")
        assert "invalid gemini api key" in str(mapped).lower()

    def test_map_model_not_found(self):
        err = Exception("404 NOT_FOUND: models/gemini-obsolete is not found")
        mapped = map_gemini_error(err, "gemini-obsolete")
        assert "currently unavailable" in str(mapped).lower()

    def test_map_timeout_error(self):
        err = Exception("Connection timed out after 30.0s")
        mapped = map_gemini_error(err, "gemini-2.5-flash")
        assert "timed out or failed" in str(mapped).lower()


class TestCallGeminiJson:
    def test_call_gemini_json_success(self):
        mock_response = MagicMock()
        mock_response.text = '```json\n{"summary": "Test Paper"}\n```'

        with patch("src.ai_service.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_get_client.return_value = mock_client

            result_str = call_gemini_json("Analyze paper", api_key="dummy-key")
            parsed = json.loads(result_str)
            assert parsed["summary"] == "Test Paper"


class TestDemoModeGenerators:
    def test_demo_single_analysis_format(self):
        analysis = get_demo_single_analysis("Attention Is All You Need.pdf")
        assert analysis["_metadata"]["is_demo"] is True
        assert "Transformer" in analysis["executive_summary"]
        assert len(analysis["results"]) >= 2
        assert len(analysis["evidence_breakdown"]["paper_evidence"]) >= 1

    def test_demo_comparison_format(self):
        papers = [{"filename": "P1.pdf"}, {"filename": "P2.pdf"}]
        comp = get_demo_comparison(papers)
        assert comp["is_demo"] is True
        assert len(comp["comparison_table"]) == 9
        assert "combined_research_opportunity" in comp

    def test_demo_gaps_format(self):
        papers = [{"filename": "P1.pdf"}, {"filename": "P2.pdf"}]
        gaps = get_demo_gaps(papers)
        assert gaps["is_demo"] is True
        assert len(gaps["matrix"]) == 7
        assert len(gaps["gaps"]) >= 3
        # Strict academic wording check
        assert "Potential research gap based on the supplied papers" in gaps["synthesis_summary"]
