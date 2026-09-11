"""
Prompt architecture for ResearchLens AI.
Enforces evidence-aware, hallucination-resistant structured JSON outputs.
Strictly separates Paper Evidence, Interpretation, AI Recommendations, and Verification flags.
"""

SYSTEM_ACADEMIC_EVIDENCE_PROMPT = """You are ResearchLens AI, a specialized research paper analysis system designed for researchers, scientists, and academics.

CRITICAL EVIDENCE & FIDELITY RULES:
1. Ground every claim STRICTLY in the provided text. Do not extrapolate, hallucinate, or fabricate facts, datasets, citations, metrics, or experimental baselines.
2. If any piece of information is missing, ambiguous, or not stated in the paper, explicitly write: "Not clearly stated in the paper."
3. Do NOT make ungrounded claims that something is globally novel. Always specify: "Based on the provided paper..." or "Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."
4. Rigorously separate:
   - [Paper Evidence]: Verbatim or direct factual statements from the authors.
   - [Interpretation]: Logical deduction from stated facts.
   - [AI Recommendation]: Practical suggestions or next steps inferred by AI.
   - [Needs Verification]: Assumptions or unvalidated claims requiring external literature review.
5. Always output valid, parseable JSON conforming strictly to the requested schema. Do not enclose in any markdown blocks unless instructed.
"""

SINGLE_PAPER_ANALYSIS_PROMPT = """Analyze the provided research paper text according to strict academic standards.

PAPER METADATA / FILENAME: {filename}

PAPER CONTENT:
\"\"\"
{text}
\"\"\"

Return a valid JSON object matching the exact structure below:
{{
  "executive_summary": "A comprehensive 2-3 paragraph synthesis summarizing the core premise, methodology, key findings, and implications.",
  "research_problem": "The precise problem or bottleneck the paper addresses.",
  "objectives": [
    "Primary research question, hypothesis, or objective 1",
    "Objective 2"
  ],
  "methodology": "Detailed description of the technical methodology, architecture, framework, or algorithm used.",
  "dataset": "Datasets used, source, size, split, domain, or 'Not clearly stated in the paper.'",
  "preprocessing": "Data cleaning, tokenization, augmentation, or preprocessing pipeline, or 'Not clearly stated in the paper.'",
  "evaluation": "Evaluation protocol, metrics used (e.g. Accuracy, F1, BLEU, latency), baselines compared against.",
  "results": [
    "Quantitative or qualitative result finding 1 with exact numbers if stated in paper",
    "Result finding 2"
  ],
  "conclusion": "The main conclusions reached by the authors.",
  "strengths": [
    "Key strength 1 demonstrated in the paper",
    "Key strength 2"
  ],
  "limitations": [
    "Explicit limitation 1 acknowledged by the authors or evidenced in the methodology",
    "Limitation 2"
  ],
  "research_gaps": [
    "Specific research gap 1 identifiable from this paper's scope or omissions",
    "Research gap 2"
  ],
  "suggestions": [
    "Actionable suggestion 1 for improving or extending this study",
    "Suggestion 2"
  ],
  "future_work": [
    "Author-stated or direct continuation future direction 1",
    "Future direction 2"
  ],
  "experiments": [
    {{
      "title": "Title of proposed experiment",
      "hypothesis": "Testable hypothesis directly addressing a paper limitation",
      "independent_variables": "What is manipulated or changed",
      "dependent_variables": "What is measured or evaluated",
      "dataset": "Specific dataset or benchmark recommended",
      "baseline": "State of the art or existing model to compare against",
      "evaluation_metrics": "Specific evaluation metrics to record",
      "expected_outcome": "Expected outcome and scientific significance",
      "risks_and_limitations": "Potential failure modes or computational challenges"
    }}
  ],
  "confidence": {{
    "overall": "High | Medium | Low",
    "reason": "Clear justification for confidence rating based on document clarity, completeness of reporting, and text extractability."
  }},
  "evidence_breakdown": {{
    "paper_evidence": [
      "Direct stated fact or data point from the paper 1",
      "Direct stated fact or data point from the paper 2"
    ],
    "interpretations": [
      "Synthesized technical insight 1",
      "Synthesized technical insight 2"
    ],
    "ai_recommendations": [
      "Constructive recommendation 1 based on limitations",
      "Constructive recommendation 2"
    ],
    "needs_verification": [
      "Claim or assumption requiring verification against external literature 1",
      "Claim requiring verification 2"
    ]
  }}
}}

Remember: If any information is absent in the text, use "Not clearly stated in the paper." Output ONLY the JSON object.
"""

PAPER_COMPARISON_PROMPT = """You are comparing between 2 and 5 research papers.
Analyze each paper's extracted content and synthesize a systematic comparative analysis.

PAPERS OVERVIEW:
{papers_summary}

Return a valid JSON object matching this schema:
{{
  "comparison_table": [
    {{
      "dimension": "Research Problem",
      "papers": {{
        "{paper_names_example}": "Summary of problem for paper"
      }}
    }},
    {{
      "dimension": "Objectives",
      "papers": {{}}
    }},
    {{
      "dimension": "Methodology",
      "papers": {{}}
    }},
    {{
      "dimension": "Dataset",
      "papers": {{}}
    }},
    {{
      "dimension": "Preprocessing",
      "papers": {{}}
    }},
    {{
      "dimension": "Evaluation",
      "papers": {{}}
    }},
    {{
      "dimension": "Results",
      "papers": {{}}
    }},
    {{
      "dimension": "Strengths",
      "papers": {{}}
    }},
    {{
      "dimension": "Limitations",
      "papers": {{}}
    }}
  ],
  "similarities": [
    "Key methodological, theoretical, or empirical similarity 1",
    "Similarity 2"
  ],
  "differences": [
    "Key difference in approach, scope, or findings 1",
    "Difference 2"
  ],
  "contradictions_inconsistencies": [
    "Contradictory findings or inconsistent conclusions between papers (or 'No direct contradictions detected among the evaluated papers.')"
  ],
  "unresolved_areas": [
    "Research question or challenge left unaddressed by all examined papers"
  ],
  "combined_research_opportunity": "A synthesized multi-paper research opportunity combining the strengths or addressing the joint weaknesses of the papers."
}}

RULES:
- Maintain strict objectivity.
- Populate each dimension for all provided papers.
- Output ONLY valid JSON.
"""

RESEARCH_GAP_PROMPT = """Perform an evidence-aware cross-paper research gap detection across the provided papers.

PAPERS OVERVIEW:
{papers_summary}

INSTRUCTIONS:
1. Analyze recurring limitations, missing benchmarks, unaddressed edge cases, and methodological blind spots across these papers.
2. Group findings into a structured matrix evaluating common dimensions:
   - Dataset
   - Model
   - Explainability
   - Robustness
   - Real-world testing
   - Evaluation
   - Generalization
3. Identify distinct research gaps and rank their priority: HIGH, MEDIUM, or LOW.
4. For each gap, cite specific evidence from the uploaded papers and provide actionable suggestions.
5. Use cautionary academic language: "Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."

Return a valid JSON object matching this schema:
{{
  "matrix": [
    {{
      "area": "Dataset",
      "paper_ratings": {{
        "{paper_names_example}": "Assessment of dataset dimension for this paper"
      }},
      "potential_gap": "Synthesis of what is missing or weakly addressed in datasets across the papers"
    }},
    {{
      "area": "Model",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }},
    {{
      "area": "Explainability",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }},
    {{
      "area": "Robustness",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }},
    {{
      "area": "Real-world testing",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }},
    {{
      "area": "Evaluation",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }},
    {{
      "area": "Generalization",
      "paper_ratings": {{}},
      "potential_gap": "..."
    }}
  ],
  "gaps": [
    {{
      "title": "Concise title of potential gap",
      "priority": "HIGH | MEDIUM | LOW",
      "description": "Clear description of what is missing or unresolved.",
      "evidence_from_papers": "Specific paper references, omissions, or cited limitations.",
      "why_it_matters": "Theoretical or practical significance of resolving this gap.",
      "suggested_direction": "Constructive research recommendation to address it.",
      "confidence": "High | Medium | Low",
      "needs_verification": true
    }}
  ],
  "synthesis_summary": "Comprehensive overview of the collective research frontier represented by these papers."
}}

Output ONLY the valid JSON object.
"""

RECOMMENDATION_PROMPT = """Based STRICTLY on the evidence and limitations identified in the following paper analysis, formulate practical research recommendations.

PAPER ANALYSIS:
{analysis_json}

Return a valid JSON object:
{{
  "recommendations": [
    {{
      "recommendation": "Concrete, actionable research recommendation",
      "reason": "Detailed rationale explaining why this recommendation is justified",
      "supporting_evidence": "Specific evidence/limitation cited in the paper supporting this",
      "expected_benefit": "Anticipated improvement, theoretical gain, or real-world benefit",
      "priority": "High | Medium | Low",
      "confidence": "High | Medium | Low"
    }}
  ]
}}

Output ONLY the valid JSON object.
"""

EXPERIMENT_PROMPT = """Design rigorous, testable experiment plans based on the limitations and future work of the analyzed research.

PAPER ANALYSIS:
{analysis_json}

Return a valid JSON object:
{{
  "experiments": [
    {{
      "title": "Title of proposed experiment",
      "hypothesis": "Testable scientific hypothesis",
      "independent_variables": "Specific factors manipulated",
      "dependent_variables": "Metrics and behaviors measured",
      "dataset": "Recommended dataset or benchmark",
      "baseline": "Comparative baseline models or algorithms",
      "evaluation_metrics": "Primary and secondary evaluation metrics",
      "expected_outcome": "Anticipated results",
      "risks_and_limitations": "Known limitations, computational risks, or confounding variables"
    }}
  ]
}}

Output ONLY the valid JSON object.
"""
