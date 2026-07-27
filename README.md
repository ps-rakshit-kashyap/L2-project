# Resume Tailoring Agent (MCP + AI Agent)

An autonomous AI agent that tailors resumes to job descriptions using **MCP (Model Context Protocol)**, **Ollama** (Qwen 3.5), **Python**, and **Streamlit**.

The agent reads a resume PDF, analyzes a job description, performs skill gap analysis, calculates ATS scores, ranks projects, and generates a tailored resume — all through autonomous ReAct-style reasoning.

## Architecture

```
resume-agent/
├── app.py                    # Streamlit frontend (UI)
├── requirements.txt          # Python dependencies
├── agent/
│   ├── agent_controller.py   # ReAct agent loop + MCP client
│   └── memory.py             # Agent step history & context
├── mcp_server/
│   ├── server.py             # FastMCP server (6 tools)
│   ├── resume_tools.py       # read_resume implementation
│   ├── analysis_tools.py     # analyze_jd, skill_gap, ats_score
│   └── tailoring_tools.py    # rank_projects, tailor_resume
├── models/
│   └── ollama_client.py      # Ollama HTTP client (Qwen 3.5)
├── utils/
│   ├── pdf_parser.py         # PyMuPDF resume parser
│   ├── skill_extractor.py    # Keyword-based skill detection
│   └── ats.py                # ATS scoring & recommendations
├── output/                   # Uploaded PDFs saved here
├── prompt.txt                # Original LLM prompt used to build this project
├── README.md                 # This file
├── UNDERSTANDING.md           # Deep-dive project documentation
└── .gitignore
```

## Agent Workflow

```
User Goal: "Tailor my resume for this job."

  Step 1: read_resume           → Parse PDF, extract sections & skills
  Step 2: analyze_job_description → Extract required skills & keywords
  Step 3: skill_gap_analysis    → Compare resume vs job skills → score
  Step 4: ats_score             → Calculate ATS compatibility + tips
  Step 5: rank_projects         → Rank projects by keyword overlap
  Step 6: tailor_resume         → Generate tailored summary & skills
  Step 7: Final Report          → All results assembled
```

Each step is decided by the LLM (ReAct reasoning via Ollama) or via a deterministic fallback plan if Ollama is unavailable.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Qwen 3.5 model pulled: `ollama pull qwen3.5:2b-q4_K_M`

## Setup & Run

### 1. Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ensure Ollama is Running

```powershell
ollama serve
```

In a separate terminal, verify the model is available:

```powershell
ollama pull qwen3.5:2b-q4_K_M
```

### 3. Run the App

```powershell
streamlit run app.py
```

### 4. Use

1. Upload a resume PDF.
2. Paste a job description.
3. Click **Analyze & Tailor Resume**.
4. View the results in the tabs.

> **Note:** The agent works **with or without Ollama**. Without it, the LLM reasoning step is skipped and the agent follows a deterministic fallback plan through all 6 tools.

## Tech Stack

| Component      | Technology                               |
|----------------|------------------------------------------|
| Frontend       | Streamlit                                |
| LLM            | Qwen 3.5 (via Ollama)                    |
| Protocol       | MCP (Model Context Protocol)             |
| PDF Parsing    | PyMuPDF (fitz)                           |
| Validation     | Pydantic                                 |
| HTTP Client    | httpx                                    |
| Language       | Python 3.11+                             |
