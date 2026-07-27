# Resume Tailoring Agent (MCP + AI Agent)

An autonomous AI agent that tailors resumes to job descriptions using **MCP (Model Context Protocol)**, **Hugging Face Inference API** (Mistral 7B), **Python**, and **Streamlit**.

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
│   ├── hf_client.py          # Hugging Face Inference API client **(active)**
│   └── ollama_client.py      # DEPRECATED — local Ollama client (reference only)
├── utils/
│   ├── pdf_parser.py         # PyMuPDF resume parser
│   ├── skill_extractor.py    # Keyword-based skill detection
│   └── ats.py                # ATS scoring & recommendations
└── output/                   # Uploaded PDFs saved here
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

Each step is decided by the LLM (ReAct reasoning via Hugging Face Inference API) or via a deterministic fallback plan if the API is unavailable.

## Prerequisites

- Python 3.11+
- [Hugging Face API token](https://huggingface.co/settings/tokens) (free — paste in app sidebar)

## Setup & Run

### 1. Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the App

```powershell
streamlit run app.py
```

### 3. Use

1. Paste your **Hugging Face API token** in the sidebar.
2. Upload a resume PDF.
3. Paste a job description.
4. Click **Analyze & Tailor Resume**.
5. View the results in the tabs.

> **Note:** The agent works **with or without an API key**. Without it, the LLM reasoning step is skipped and the agent follows a deterministic fallback plan through all 6 tools.

## Tech Stack

| Component      | Technology                               |
|----------------|------------------------------------------|
| Frontend       | Streamlit                                |
| LLM            | Mistral 7B (via Hugging Face Inference)  |
| Protocol       | MCP (Model Context Protocol)             |
| PDF Parsing    | PyMuPDF (fitz)                           |
| Validation     | Pydantic                                 |
| HTTP Client    | httpx / huggingface-hub                  |
| Language       | Python 3.11+                             |

## License

MIT
