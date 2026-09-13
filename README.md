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

### 6. Authentication & User Management
- **Security-First Architecture**: Built-in SQLite user database (`researchlens.db`) with parameterized queries preventing SQL injection.
- **Bcrypt Password Hashing**: Passwords salted and hashed with bcrypt (never plain text).
- **Session Management**: Native Streamlit `session_state` protection gating all analysis tabs and reports behind authentication.
- **Modern SaaS UI**: Custom academic glassmorphic cards for Login, Registration, and Forgot Password flows.
- **Safe Password Reset**: Secure MVP password reset flow without user email enumeration.
- **Sidebar Profile & Logout**: Shows user welcome badge (`Welcome, [User Name] 👋`) and one-click session logout.

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
│  [Login/Signup Gate] -> [Home] [Single] [Compare] [Gaps]    │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
  ┌─────────────────────────┐ ┌───────────────┐ ┌─────────────────────────┐
  │      src/auth.py        │ │src/database.py│ │ src/document_parser.py  │
  │ • bcrypt password hash  │ │ • SQLite      │ │ • pypdf                 │
  │ • input validations     │ │ • users table │ │ • python-docx           │
  │ • register / auth / reset│ │ • parameterized│ │ • Text cleaner & normal │
  └────────────┬────────────┘ └───────┬───────┘ └────────────┬────────────┘
               │                      │                      │
               └───────────────┬──────┴──────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │        src/prompts.py         │
               │ • Strict academic schemas     │
               │ • Zero-hallucination rules    │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │       src/ai_service.py       │
               │ • Official Google GenAI SDK   │
               │ • Gemini 2.5 Flash / 2.0 / 1.5│
               │ • Interactive Demo Mode       │
               │ • Quota & error mapping       │
               └───────────────┬───────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌─────────────────────────┐     ┌───────────────────────────┐
  │     src/analyzer.py     │     │     src/comparator.py     │
  │ • Single paper analyzer │     │ • Multi-paper synthesizer │
  │ • Schema normalizer     │     │ • Research gap matrix     │
  │ • Backward-compat layer │     │ • Duplicate detector      │
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
- **Authentication & Database**: SQLite (`researchlens.db`), `bcrypt` password hashing.
- **Document Parsing**: `pypdf`, `python-docx`, native UTF-8/Latin-1 text parsers.
- **AI Intelligence**: Official Google Gemini API (`google-genai` SDK) with structured JSON generation (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`).
- **Interactive Demo Mode**: Zero-key offline testing mode with precomputed realistic academic analysis, comparisons, and research gaps.
- **Testing**: `pytest` with 95 automated unit and integration tests (mocked Gemini API, isolated SQLite fixtures, and Streamlit AppTest).

---

## 📁 Project Directory Structure

```
researchlens-ai/
│
├── app.py                      # Main Streamlit web application & UI (Auth gate + Dashboard)
├── requirements.txt            # Minimal, production dependencies (includes bcrypt)
├── README.md                   # Comprehensive documentation & setup
├── .gitignore                  # Git exclusions (secrets, caches, venv, sqlite db)
├── researchlens.db             # Local SQLite database (auto-generated on startup)
│
├── .streamlit/
│   └── secrets.toml.example    # Template for API keys
│
├── src/
│   ├── __init__.py             # Package declaration
│   ├── ai_service.py           # Core Google Gemini AI client, schemas, error mapping & Demo Mode
│   ├── auth.py                 # Registration, login, bcrypt hashing, validation & reset
│   ├── database.py             # SQLite connection, users table & parameterized CRUD
│   ├── document_parser.py      # PDF, DOCX, TXT parsers & cleaning
│   ├── analyzer.py             # Single paper analysis adapter (delegates to ai_service)
│   ├── comparator.py           # Multi-paper comparative analysis adapter (delegates to ai_service)
│   ├── prompts.py              # Zero-hallucination academic prompts
│   ├── report.py               # Plain-text academic report generators
│   └── utils.py                # Validation, truncation, and UI badges
│
├── tests/
│   ├── __init__.py             # Test package declaration
│   ├── test_ai_service.py      # Gemini client, error mapping, and demo mode tests
│   ├── test_auth.py            # Authentication, bcrypt, input validation & reset tests
│   ├── test_database.py        # SQLite schema, CRUD, and constraint tests
│   ├── test_app_auth_flow.py   # Streamlit AppTest UI authentication integration tests
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

ResearchLens AI connects to the Google Gemini API using the official `google-genai` SDK. You can get a free API key at [Google AI Studio](https://aistudio.google.com/). The API key can be supplied in three ways (checked in priority order):

### Option A: Streamlit Secrets (Recommended for local dev & cloud deployment)
1. Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Open `.streamlit/secrets.toml` and add your Google Gemini key:
   ```toml
   GEMINI_API_KEY = "AIzaSyYourGeminiApiKeyHere"
   ```

### Option B: Environment Variable
```bash
export GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
# On Windows PowerShell:
$env:GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
```

### Option C: UI Sidebar Input
If no key is configured in secrets or the environment, you can enter your Gemini API key directly into the secure password field in the sidebar.

---

## 🎮 Interactive Demo Mode

Want to test ResearchLens AI immediately without configuring an API key?
In the application sidebar, select:
- **`○ Demo Mode`** under **AI Mode**

In Demo Mode, the application instantly returns realistic, evidence-grounded sample analyses, comparative matrices, and prioritized research gaps for demonstration and review.

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

If launched without an API key configured, the app will offer you the option to switch to **Demo Mode** or enter your key directly in the sidebar.

---

## 🧪 Running the Test Suite

The test suite contains **95 automated tests** covering Google Gemini client integration, error mapping, demo generators, document extraction, error handling, intelligent truncation, duplicate detection, schema validation, SQLite user persistence, bcrypt password hashing, and Streamlit AppTest authentication UI flows. **Tests do not require an active API key** (all network requests are mocked and databases use temporary in-memory/isolated fixtures).

Run all 95 tests:
```bash
pytest -v
```

To run a specific test module:
```bash
pytest tests/test_ai_service.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_comparator.py -v
pytest tests/test_auth.py -v
pytest tests/test_database.py -v
pytest tests/test_app_auth_flow.py -v
pytest tests/test_parser.py -v
```

---

## 💡 Sample Evaluation Workflow

Try ResearchLens AI in less than 2 minutes using the bundled sample papers:

1. Launch `streamlit run app.py`.
2. Select **Demo Mode** or enter your Gemini API key in the sidebar.
3. Navigate to **Single Paper Analysis** in the sidebar.
4. Upload `sample_papers/paper_1_sparse_transformers.txt`.
5. Click **Analyze Paper**. Explore the 12 tabs (Methodology, Dataset, Results, Limitations, and Evidence Breakdown).
6. Click **Download Academic Analysis Report (TXT)** to export the dossier.
7. Navigate to **Paper Comparison** in the sidebar.
8. Upload `paper_1_sparse_transformers.txt` and `paper_2_linear_attention_mechanisms.txt`.
9. Click **Compare Papers** to view the comparative dimensions and joint research opportunity.
10. Navigate to **Research Gaps**, upload all 3 sample papers, and click **Detect Cross-Paper Research Gaps** to view the visual Research Gap Matrix.

---

## 🔒 Security, Privacy & Data Handling

- **In-Memory Ephemeral Processing**: Uploaded documents are processed entirely in memory during the active session. The application stores no files permanently on disk.
- **Zero Exposure**: API keys are never exposed to the client interface or logged in output traces.
- **Controlled Transmission**: When Gemini AI Mode is active, document text is transmitted strictly to Google Gemini's encrypted API endpoints for analysis. In Demo Mode, no external network requests are made.
- **Input Sanitization**: File sizes and types are verified prior to processing, preventing buffer overruns and unhandled binary parsing.

---

## ⚠️ Current MVP Scope & Limitations

The current release is an MVP focused on core analytical fidelity:
- **No OCR**: Scanned documents or image-only PDFs are detected and rejected with a helpful message.
- **Context Boundaries**: Very long documents (>12,000 words) undergo intelligent head/tail truncation to retain critical sections (Abstract, Intro, Method, Results, Conclusion) within model context limits.
- **Comparative Bound**: Supports 2 to 5 papers simultaneously in the comparator.

---

## 🔮 Future Scope

The modular structure is designed to seamlessly accommodate future extensions:
- [ ] OCR integration (Tesseract / EasyOCR / PDF image layer extraction)
- [ ] Semantic Scholar & CrossRef API integration for automatic citation verification
- [ ] Vector embeddings & persistent ChromaDB / PGVector store for multi-paper semantic search
- [ ] Interactive citation graph visualization (NetworkX / PyVis)
- [ ] Formatted PDF/DOCX report exports with academic LaTeX styling

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Migrate to Google Gemini API and add Demo Mode"
   git remote add origin https://github.com/your-username/researchlens-ai.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Under **Advanced Settings** -> **Secrets**, add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "AIzaSyYourGeminiApiKeyHere"
   ```
5. Click **Deploy!**

---

## 📜 License & Academic Disclaimer

Distributed under the MIT License. See `LICENSE` for more information.

> **Academic Disclaimer:**  
> ResearchLens AI provides AI-assisted decision support. It does not replace peer review, domain expertise, or independent verification of original research literature.
