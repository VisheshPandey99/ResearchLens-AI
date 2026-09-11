# 🔬 ResearchLens AI
> **AI-Powered Research Paper Analysis, Comparison & Research Gap Detection**

ResearchLens AI is a production-quality academic research assistant engineered for students, researchers, and university faculty. It provides deep structural extraction, evidence-aware synthesis, cross-study comparative matrices, and scientific gap detection across research manuscripts without hallucinations.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Evidence-Aware AI Philosophy](#evidence-aware-ai-philosophy)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)
- [Sample Evaluation Workflow](#sample-evaluation-workflow)
- [Security, Privacy & Data Handling](#security-privacy--data-handling)
- [Current MVP Scope & Limitations](#current-mvp-scope--limitations)
- [Future Scope](#future-scope)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [License & Academic Disclaimer](#license--academic-disclaimer)

---

## 🎯 Overview

Academic literature is expanding faster than researchers can digest. Typical conversational AI tools hallucinate citations, invent datasets, or overgeneralize claims. 

**ResearchLens AI** solves this bottleneck by adhering to **strict evidence-grounding constraints**:
- It extracts **15+ standardized academic dimensions** from individual manuscripts.
- It normalizes and compares **2 to 5 papers** across shared technical axes.
- It identifies recurring blind spots, unaddressed edge cases, and missing evaluations to construct a **prioritized Research Gap Matrix** (HIGH / MEDIUM / LOW).
- It formats testable, hypothesis-driven **experimental protocols** directly addressing paper limitations.
- It provides one-click exports of clean plain-text (**TXT**) academic dossiers.

---

## ✨ Key Features

### 1. Document Extraction & Cleaning
- Supports **PDF**, **DOCX**, and **TXT** files (up to 25MB).
- Extracts text, paragraph blocks, and tabular baseline data.
- **Scanned PDF Safeguard**: Intelligently identifies image-only or scanned PDFs lacking machine-readable text and notifies the user rather than fabricating analysis.
- Computes real-time document metrics: word count, character count, and estimated token usage.

### 2. Evidence-Aware Single Paper Analysis (12-Tab Dossier)
1. **Overview**: Executive summary, confidence rating, and justification.
2. **Problem & Objectives**: Core research bottlenecks, hypotheses, and research questions.
3. **Methodology**: Technical approach, system architecture, and algorithmic formulations.
4. **Dataset & Preprocessing**: Data sources, training/validation splits, tokenization, and augmentation pipelines.
5. **Results**: Key quantitative findings, accuracy/perplexity benchmarks, and empirical speedups.
6. **Conclusion**: Authors' core deductions.
7. **Strengths & Limitations**: Explicit strengths and acknowledged shortcomings.
8. **Research Gaps**: Potential gaps directly identifiable from this paper's scope.
9. **Recommendations**: Concrete, actionable improvements.
10. **Future Work**: Author-stated future research trajectories.
11. **Experiments**: Structured experiment planner (Hypothesis, Independent/Dependent Variables, Dataset, Baselines, Metrics, Risks).
12. **Evidence & Confidence**: Strict 4-way taxonomy:
    - 📄 **Paper Evidence** (direct facts from paper)
    - 🔎 **Interpretation** (technical deduction)
    - 💡 **AI Recommendation** (inferred next step)
    - ⚠️ **Needs Verification** (unverified claims requiring broader review)

### 3. Multi-Paper Comparative Synthesis (2–5 Papers)
- Side-by-side comparative matrices across 9 core dimensions:
  - *Research Problem, Objectives, Methodology, Dataset, Preprocessing, Evaluation, Results, Strengths, Limitations*.
- Identifies **Similarities**, **Key Differences**, **Contradictions / Inconsistencies**, and **Unresolved Research Areas**.
- Synthesizes an integrated **Combined Research Opportunity**.
- Automatic duplicate detection (identifies duplicate filenames or identical paper text).

### 4. Cross-Paper Research Gap Detection
- Visual **Research Gap Matrix** evaluating 7 dimensions:
  - *Dataset, Model, Explainability, Robustness, Real-World Testing, Evaluation, Generalization*.
- Prioritized research gaps classified as **HIGH**, **MEDIUM**, or **LOW** priority.
- Grounded with specific paper evidence, rationales, and suggested directions.
- Clear academic humility labeling: *"Potential research gap based on the uploaded papers. Broader literature review is required to establish novelty."*

### 5. Plain-Text Report Export
- Generate and download comprehensive, publication-ready plain-text (`.txt`) reports for single analysis, multi-paper comparison, and research gaps.
- Safe, OS-compatible sanitized filenames with timestamps.

---

## 🛡️ Evidence-Aware AI Philosophy

ResearchLens AI does **not** behave like a conversational chatbot. It enforces strict academic constraints:
1. **Grounding**: The AI operates exclusively on the uploaded document's contents.
2. **Zero Fabrication**: It will never invent datasets, fabricate metrics, or simulate non-existent citations.
3. **Explicit Handling of Omissions**: If an item is missing in the paper, it writes:
   > *"Not clearly stated in the paper."*
4. **No Unchecked Novelty Claims**: Novelty is explicitly conditioned on the provided papers, reminding researchers that external literature verification is necessary.

---

## 🏗️ Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                 Streamlit Web UI (app.py)                   │
│  [Home]   [Single Analysis]   [Compare]   [Gaps]   [About]  │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌─────────────────────────┐     ┌───────────────────────────┐
  │ src/document_parser.py  │     │       src/utils.py        │
  │ • pypdf                 │     │ • Validation & Sanitizer  │
  │ • python-docx           │     │ • Token / Word Counter    │
  │ • Text cleaner & normal │     │ • Intelligent Truncation  │
  └────────────┬────────────┘     └─────────────┬─────────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │        src/prompts.py         │
               │ • Strict academic schemas     │
               │ • Zero-hallucination rules    │
               └───────────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌─────────────────────────┐     ┌───────────────────────────┐
  │     src/analyzer.py     │     │     src/comparator.py     │
  │ • OpenAI API (JSON Mode)│     │ • Multi-paper synthesizer │
  │ • Schema normalizer     │     │ • Research gap matrix     │
  │ • Error & quota handler │     │ • Duplicate detector      │
  └────────────┬────────────┘     └─────────────┬─────────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │         src/report.py         │
               │ • Academic TXT dossiers       │
               │ • Safe download generators    │
               └───────────────────────────────┘
```

- **Frontend / UI**: Streamlit with custom CSS (modern academic typography, card containers, and badges).
- **Backend / Core**: Python 3.10+
- **Document Parsing**: `pypdf`, `python-docx`, native UTF-8/Latin-1 text parsers.
- **AI Intelligence**: Official `openai` Python SDK (Structured JSON Mode with `gpt-4o-mini` and `gpt-4o`).
- **Testing**: `pytest` with unit tests and mocked API responses.

---

## 📁 Project Directory Structure

```
researchlens-ai/
│
├── app.py                      # Main Streamlit web application & UI
├── requirements.txt            # Minimal, production dependencies
├── README.md                   # Comprehensive documentation & setup
├── .gitignore                  # Git exclusions (secrets, caches, venv)
│
├── .streamlit/
│   └── secrets.toml.example    # Template for API keys
│
├── src/
│   ├── __init__.py             # Package declaration
│   ├── document_parser.py      # PDF, DOCX, TXT parsers & cleaning
│   ├── analyzer.py             # Single paper AI service & error handling
│   ├── comparator.py           # Multi-paper comparative analysis & gaps
│   ├── prompts.py              # Zero-hallucination academic prompts
│   ├── report.py               # Plain-text academic report generators
│   └── utils.py                # Validation, truncation, and UI badges
│
├── tests/
│   ├── __init__.py             # Test package declaration
│   ├── test_parser.py          # Document parser & edge case tests
│   ├── test_analyzer.py        # Analyzer, JSON schema, & mock API tests
│   └── test_comparator.py      # Comparator, gap ranking, & duplicate tests
│
└── sample_papers/
    ├── README.md               # Guide to sample test papers
    ├── paper_1_sparse_transformers.txt
    ├── paper_2_linear_attention_mechanisms.txt
    └── paper_3_state_space_models.txt
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python **3.10** or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/researchlens-ai.git
cd researchlens-ai
```

### 3. Create and Activate a Virtual Environment
**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

ResearchLens AI connects to OpenAI via official endpoints. The API key can be supplied in three ways (checked in priority order):

### Option A: Streamlit Secrets (Recommended for local dev & cloud deployment)
1. Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Open `.streamlit/secrets.toml` and add your key:
   ```toml
   OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
   ```

### Option B: Environment Variable
```bash
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
# On Windows PowerShell:
$env:OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### Option C: UI Sidebar Input
If no key is configured in secrets or the environment, the application will display a secure password input field directly in the sidebar for temporary session use.

---

## 🚀 Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The application will start and automatically open in your default browser at:
```
http://localhost:8501
```

If launched without an API key configured, the app will display a friendly setup reminder in the sidebar and allow you to test document parsing immediately.

---

## 🧪 Running the Test Suite

The test suite contains automated tests covering document extraction, error handling, intelligent truncation, duplicate detection, and schema validation. **Tests do not require an active OpenAI API key** (all network requests are mocked).

Run all tests:
```bash
pytest -v
```

To run a specific test module:
```bash
pytest tests/test_parser.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_comparator.py -v
```

---

## 💡 Sample Evaluation Workflow

Try ResearchLens AI in less than 2 minutes using the bundled sample papers:

1. Launch `streamlit run app.py`.
2. Navigate to **Single Paper Analysis** in the sidebar.
3. Upload `sample_papers/paper_1_sparse_transformers.txt`.
4. Click **Analyze Paper**. Explore the 12 tabs (Methodology, Dataset, Results, Limitations, and Evidence Breakdown).
5. Click **Download Academic Analysis Report (TXT)** to export the dossier.
6. Navigate to **Paper Comparison** in the sidebar.
7. Upload `paper_1_sparse_transformers.txt` and `paper_2_linear_attention_mechanisms.txt`.
8. Click **Compare Papers** to view the comparative dimensions and joint research opportunity.
9. Navigate to **Research Gaps**, upload all 3 sample papers, and click **Detect Cross-Paper Research Gaps** to view the visual Research Gap Matrix.

---

## 🔒 Security, Privacy & Data Handling

- **In-Memory Ephemeral Processing**: Uploaded documents are processed entirely in memory during the active session. The application stores no files permanently on disk.
- **Zero Exposure**: API keys are never exposed to the client interface or logged in output traces.
- **Controlled Transmission**: Document text is transmitted strictly to OpenAI's encrypted API endpoints for analysis.
- **Input Sanitization**: File sizes and types are verified prior to processing, preventing buffer overruns and unhandled binary parsing.

---

## ⚠️ Current MVP Scope & Limitations

The current release is an MVP focused on core analytical fidelity:
- **No OCR**: Scanned documents or image-only PDFs are detected and rejected with a helpful message.
- **Context Boundaries**: Very long documents (>12,000 words) undergo intelligent head/tail truncation to retain critical sections (Abstract, Intro, Method, Results, Conclusion) within model context limits.
- **Comparative Bound**: Supports 2 to 5 papers simultaneously in the comparator.
- **No User Database**: No user logins, external databases (e.g. Postgres), or persistent user history are included in the MVP.

---

## 🔮 Future Scope

The modular structure is designed to seamlessly accommodate future extensions:
- [ ] OCR integration (Tesseract / EasyOCR / PDF image layer extraction)
- [ ] Semantic Scholar & CrossRef API integration for automatic citation verification
- [ ] Vector embeddings & persistent ChromaDB / PGVector store for multi-paper semantic search
- [ ] Interactive citation graph visualization (NetworkX / PyVis)
- [ ] User authentication and project workspace persistence (PostgreSQL / Supabase)
- [ ] Formatted PDF/DOCX report exports with academic LaTeX styling

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: ResearchLens AI MVP"
   git remote add origin https://github.com/your-username/researchlens-ai.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Under **Advanced Settings** -> **Secrets**, add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
   ```
5. Click **Deploy!**

---

## 📜 License & Academic Disclaimer

Distributed under the MIT License. See `LICENSE` for more information.

> **Academic Disclaimer:**  
> ResearchLens AI provides AI-assisted decision support. It does not replace peer review, domain expertise, or independent verification of original research literature.
