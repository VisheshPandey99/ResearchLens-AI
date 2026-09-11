"""
Unit tests for Paper Comparator and Research Gap Detection modules.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.comparator import (
    compare_papers,
    detect_research_gaps,
    check_for_duplicates,
    validate_comparison_schema,
    validate_research_gap_schema
)

MOCK_PAPERS = [
    {
        "filename": "paper_A.txt",
        "text": "Paper A proposes Convolutional Networks for image classification on CIFAR-10."
    },
    {
        "filename": "paper_B.txt",
        "text": "Paper B introduces Vision Transformers with multi-head self-attention on ImageNet."
    },
    {
        "filename": "paper_C.txt",
        "text": "Paper C studies hybrid CNN-Transformer architectures for edge devices."
    },
    {
        "filename": "paper_D.txt",
        "text": "Paper D proposes quantization-aware training for neural networks on mobile GPUs."
    },
    {
        "filename": "paper_E.txt",
        "text": "Paper E evaluates pruning and sparsity techniques for long-context vision models."
    }
]

MOCK_COMPARE_RESPONSE = {
    "comparison_table": [
        {
            "dimension": "Research Problem",
            "papers": {"paper_A.txt": "CNN efficiency", "paper_B.txt": "ViT scaling"}
        }
    ],
    "similarities": ["Both focus on computer vision."],
    "differences": ["Paper A uses convolutions while Paper B uses attention."],
    "contradictions_inconsistencies": ["None detected."],
    "unresolved_areas": ["Few-shot generalization under severe domain shift."],
    "combined_research_opportunity": "Hybrid attention-convolution on resource-constrained robotics."
}

MOCK_GAP_RESPONSE = {
    "matrix": [
        {
            "area": "Robustness",
            "paper_ratings": {"paper_A.txt": "CIFAR-C evaluated", "paper_B.txt": "Not evaluated"},
            "potential_gap": "Lack of standardized adversarial evaluation."
        }
    ],
    "gaps": [
        {
            "title": "Adversarial Robustness Gap",
            "priority": "HIGH",
            "description": "Neither paper addresses adversarial perturbations.",
            "evidence_from_papers": "Omitted in baseline tables.",
            "why_it_matters": "Real-world reliability.",
            "suggested_direction": "Evaluate against AutoAttack benchmarks.",
            "confidence": "High",
            "needs_verification": True
        },
        {
            "title": "Low Precision Calibration",
            "priority": "LOW",
            "description": "Quantization effects were not tested.",
            "evidence_from_papers": "Single precision FP32 used exclusively.",
            "why_it_matters": "Edge deployment feasibility.",
            "suggested_direction": "INT8 post-training quantization.",
            "confidence": "Medium",
            "needs_verification": True
        }
    ],
    "synthesis_summary": "Collective focus on accuracy rather than robustness."
}


class TestDuplicateDetection:
    def test_detect_identical_filenames(self):
        dups = [
            {"filename": "study.pdf", "text": "Content 1"},
            {"filename": "study.pdf", "text": "Content 2"}
        ]
        warnings = check_for_duplicates(dups)
        assert len(warnings) == 1
        assert "identical filenames" in warnings[0].lower()

    def test_detect_identical_content(self):
        dups = [
            {"filename": "study_v1.pdf", "text": "Identical paper text that has duplicate content."},
            {"filename": "study_v2.pdf", "text": "Identical paper text that has duplicate content."}
        ]
        warnings = check_for_duplicates(dups)
        assert len(warnings) == 1
        assert "identical content" in warnings[0].lower()

    def test_distinct_papers_no_warning(self):
        warnings = check_for_duplicates(MOCK_PAPERS[:2])
        assert len(warnings) == 0


class TestPaperCountValidation:
    def test_reject_less_than_2_papers(self):
        with pytest.raises(ValueError) as exc:
            compare_papers([MOCK_PAPERS[0]], api_key="key")
        assert "At least 2 papers" in str(exc.value)

    def test_reject_more_than_5_papers(self):
        six_papers = MOCK_PAPERS + [{"filename": "paper_F.txt", "text": "Extra"}]
        with pytest.raises(ValueError) as exc:
            compare_papers(six_papers, api_key="key")
        assert "maximum of 5" in str(exc.value).lower()


class TestCompareExecutionMocked:
    def test_compare_2_papers(self):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(MOCK_COMPARE_RESPONSE)
        mock_response = MagicMock(choices=[mock_choice])

        with patch("src.comparator.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            res = compare_papers(MOCK_PAPERS[:2], api_key="test-key")
            assert len(res["similarities"]) == 1
            assert len(res["differences"]) == 1
            assert "combined_research_opportunity" in res
            assert len(res["paper_names"]) == 2

    def test_compare_3_papers(self):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(MOCK_COMPARE_RESPONSE)
        mock_response = MagicMock(choices=[mock_choice])

        with patch("src.comparator.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            res = compare_papers(MOCK_PAPERS[:3], api_key="test-key")
            assert len(res["paper_names"]) == 3

    def test_compare_5_papers(self):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(MOCK_COMPARE_RESPONSE)
        mock_response = MagicMock(choices=[mock_choice])

        with patch("src.comparator.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            res = compare_papers(MOCK_PAPERS[:5], api_key="test-key")
            assert len(res["paper_names"]) == 5


class TestResearchGapsMocked:
    def test_detect_research_gaps(self):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(MOCK_GAP_RESPONSE)
        mock_response = MagicMock(choices=[mock_choice])

        with patch("src.comparator.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            res = detect_research_gaps(MOCK_PAPERS[:2], api_key="test-key")
            assert len(res["gaps"]) == 2
            # HIGH priority should be first
            assert res["gaps"][0]["priority"] == "HIGH"
            assert res["gaps"][1]["priority"] == "LOW"
            assert len(res["matrix"]) >= 1
