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
    SUPPORTED_GEMINI_MODELS,
    FALLBACK_FLASH_MODELS,
    is_temporary_capacity_error
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


class TestCapacityErrorDetection:
    def test_detects_503_and_high_demand(self):
        assert is_temporary_capacity_error(Exception("503 UNAVAILABLE")) is True
        assert is_temporary_capacity_error(Exception("This model is currently experiencing high demand.")) is True
        assert is_temporary_capacity_error(Exception("Server overloaded, please try again shortly")) is True

    def test_rejects_auth_and_quota_errors(self):
        assert is_temporary_capacity_error(Exception("401 Unauthorized")) is False
        assert is_temporary_capacity_error(Exception("403 Forbidden: Invalid API Key")) is False
        assert is_temporary_capacity_error(Exception("API_KEY_INVALID")) is False
        assert is_temporary_capacity_error(Exception("429 Resource has been exhausted (quota limit reached)")) is False
        assert is_temporary_capacity_error(Exception("400 Bad Request")) is False


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

    def test_call_gemini_json_fallback_succeeds_on_503(self):
        mock_503_exc = Exception("503 UNAVAILABLE: This model is currently experiencing high demand.")
        mock_success_response = MagicMock()
        mock_success_response.text = '{"summary": "Fallback Paper"}'

        with patch("src.ai_service.get_gemini_client") as mock_get_client, \
             patch("time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = [
                mock_503_exc,
                mock_success_response
            ]
            mock_get_client.return_value = mock_client

            result_str = call_gemini_json(
                "Analyze paper",
                api_key="dummy-key",
                model="gemini-3.6-flash",
                fallback_models=["gemini-3.5-flash"],
                backoff_seconds=0.01
            )
            parsed = json.loads(result_str)
            assert parsed["summary"] == "Fallback Paper"
            assert mock_client.models.generate_content.call_count == 2
            mock_sleep.assert_called_once()

    def test_call_gemini_json_auth_error_fails_immediately_without_fallback(self):
        auth_exc = Exception("403 Forbidden: Invalid API Key")

        with patch("src.ai_service.get_gemini_client") as mock_get_client, \
             patch("time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = auth_exc
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                call_gemini_json(
                    "Analyze paper",
                    api_key="dummy-key",
                    model="gemini-3.6-flash",
                    fallback_models=["gemini-3.5-flash"]
                )
            assert "invalid gemini api key" in str(exc_info.value).lower()
            assert mock_client.models.generate_content.call_count == 1
            mock_sleep.assert_not_called()

    def test_call_gemini_json_quota_error_fails_immediately_without_fallback(self):
        quota_exc = Exception("429 Resource Exhausted: Quota exceeded")

        with patch("src.ai_service.get_gemini_client") as mock_get_client, \
             patch("time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = quota_exc
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                call_gemini_json(
                    "Analyze paper",
                    api_key="dummy-key",
                    model="gemini-3.6-flash",
                    fallback_models=["gemini-3.5-flash"]
                )
            assert "quota/rate limit" in str(exc_info.value).lower()
            assert mock_client.models.generate_content.call_count == 1
            mock_sleep.assert_not_called()

    def test_call_gemini_json_all_fallbacks_fail_raises_consolidated_error(self):
        mock_503_exc = Exception("503 UNAVAILABLE: Model overloaded.")

        with patch("src.ai_service.get_gemini_client") as mock_get_client, \
             patch("time.sleep"):
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = mock_503_exc
            mock_get_client.return_value = mock_client

            with pytest.raises(AnalysisError) as exc_info:
                call_gemini_json(
                    "Analyze paper",
                    api_key="dummy-key",
                    model="gemini-3.6-flash",
                    fallback_models=["gemini-3.5-flash"],
                    backoff_seconds=0.01
                )
            err_msg = str(exc_info.value).lower()
            assert "temporarily unavailable" in err_msg
            assert "high demand" in err_msg
            assert mock_client.models.generate_content.call_count == 2


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
