# Resume Tailoring Agent — Understanding the Project

This document explains the entire project in detail: what every file does, how data flows through the system, how the MCP protocol works, and how the ReAct agent thinks. Read this to understand, modify, or extend the code.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [The MCP Protocol](#3-the-mcp-protocol)
4. [The ReAct Agent Pattern](#4-the-react-agent-pattern)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [Data Flow](#6-data-flow)
7. [Error Handling & Fallbacks](#7-error-handling--fallbacks)
8. [Extending the Project](#8-extending-the-project)

---

## 1. What This Project Does

The Resume Tailoring Agent is an autonomous AI system that:

1. **Reads** your resume (PDF).
2. **Analyzes** a job description you paste.
3. **Compares** your skills against what the job requires.
4. **Scores** your resume for ATS (Applicant Tracking System) compatibility.
5. **Ranks** your projects by how relevant they are to the job.
6. **Generates** a tailored version of your resume optimized for that specific job.
7. **Shows** its entire reasoning process so you can verify its work.

It is **not** a simple chatbot. It is a true **agent** that plans multiple steps, calls tools, observes results, and decides what to do next — all autonomously.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)               │
│  Upload PDF │ Paste JD │ Button │ Progress │ Results Tabs   │
└──────────────────────────┬──────────────────────────────────┘
                           │ calls
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agent Controller (agent_controller.py)      │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │ Planner │  │ Executor │  │  Memory    │  │ HF API   │  │
│  │(ReAct)  │  │(MCP call)│  │(steps/ctx) │  │(Mistral)│  │
│  └─────────┘  └────┬─────┘  └────────────┘  └───────────┘  │
└─────────────────────┼───────────────────────────────────────┘
                      │ MCP protocol (JSON-RPC over stdio)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (server.py)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ ResumeTools  │  │AnalysisTools │  │ TailoringTools   │   │
│  │ read_resume  │  │ analyze_jd   │  │ rank_projects    │   │
│  │              │  │ skill_gap    │  │ tailor_resume    │   │
│  │              │  │ ats_score    │  │                  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
└─────────┼─────────────────┼───────────────────┼──────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
     ┌────────┐      ┌────────────┐       ┌──────────┐
     │PyMuPDF │      │SkillExtract│       │ ATSScorer│
     │parser  │      │or          │       │          │
     └────────┘      └────────────┘       └──────────┘
```

### Three Layers

| Layer | Process | Description |
|-------|---------|-------------|
| **Frontend** | Streamlit (app.py) | UI for upload, input, progress, results |
| **Agent** | agent_controller.py | ReAct loop: thinks, calls MCP tools, stores results |
| **MCP Server** | server.py + tools | 6 tools running as a subprocess, connected via MCP protocol |

### Key Design Decision: Why MCP?

MCP (Model Context Protocol) is an open standard developed by Anthropic for connecting AI models to external tools and data sources. Instead of the agent calling tools directly (tight coupling), the tools are exposed through a standardized protocol:

- **Tools run in a separate process** — isolation, fault tolerance.
- **Standardized interface** — any MCP-compatible client can use these tools.
- **The agent speaks MCP** — it connects to the server, lists tools, and calls them with JSON parameters.

---

## 3. The MCP Protocol

MCP is JSON-RPC 2.0 over stdio. The agent controller spawns `mcp_server/server.py` as a child process and communicates via stdin/stdout.

### How the MCP Server Starts

In `agent_controller.py`:

```python
server_params = StdioServerParameters(
    command=sys.executable,          # python.exe
    args=["mcp_server/server.py"],   # the server script
)
```

The agent uses `mcp.client.stdio.stdio_client` to spawn the subprocess and create a bidirectional communication channel.

### What Happens Inside the MCP Server

`server.py` creates a `FastMCP` instance and registers 6 tools with the `@mcp.tool()` decorator:

```python
mcp = FastMCP("Resume Tailoring Agent")

@mcp.tool()
def read_resume(file_path: str) -> dict:
    return ResumeTools.read_resume(file_path)

@mcp.tool()
def analyze_job_description(job_description: str) -> dict:
    return AnalysisTools.analyze_job_description(job_description)

# ... 4 more tools
```

When the agent calls `session.call_tool("read_resume", {"file_path": "..."})`, the MCP server:
1. Receives a JSON-RPC request via stdin.
2. Routes it to the `read_resume` function.
3. Executes the function.
4. Sends the JSON result back via stdout.

### The 6 MCP Tools

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `read_resume` | `file_path` (str) | `{summary, skills, projects, experience, education}` | Parse PDF resume |
| `analyze_job_description` | `job_description` (str) | `{role, required_skills, preferred_skills, keywords}` | Extract JD requirements |
| `skill_gap_analysis` | `resume_data`, `job_data` | `{matched_skills, missing_skills, score}` | Compare skills |
| `ats_score` | `resume_data`, `job_data` | `{score, recommendations}` | ATS compatibility |
| `rank_projects` | `projects`, `job_description` | `{ranked_projects}` | Sort by relevance |
| `tailor_resume` | `resume_data`, `job_data` | `{professional_summary, tailored_skills, ...}` | Generate tailored content |

---

## 4. The ReAct Agent Pattern

ReAct (Reasoning + Acting) is an agent architecture where the LLM alternates between:

1. **Thought**: Reasoning about the current state and what to do next.
2. **Action**: Calling a tool with specific parameters.
3. **Observation**: Seeing the result of the tool call.
4. **Repeat**: Using the observation to inform the next thought.

### The Loop in Code

In `agent_controller.py`, the `run()` method:

```python
for step_idx in range(max_steps):
    # 1. Build prompt with current state (what data is available)
    prompt = self._build_react_prompt(tool_names)
    
    # 2. Ask LLM what to do
    llm_response = self.llm.generate(prompt, system=REACT_SYSTEM_PROMPT)
    action, action_input = self._parse_action(llm_response)
    
    # 3. Execute via MCP
    mcp_result = await session.call_tool(action, filled_input)
    observation = self._parse_mcp_result(mcp_result)
    
    # 4. Store result and update context
    self._store_result(action, observation)
    self.memory.add_step(step)
```

### What the LLM Sees

The `_build_react_prompt` method constructs a prompt like:

```
Goal: Tailor my resume for this job.

Current data state:
- resume_data: available
- job_data: not available
- skill_gap: not available
- ats_result: not available
- ranked_projects: not available
- tailored_resume: not available

Steps completed so far:
Step: read_resume
  Observation: {summary: "...", skills: ["Python", "SQL"], ...}

Resume skills: ['Python', 'SQL', 'Git']

What is the next action?
Action: tool_name
Action Input: {"param": "value"}
```

The LLM (Mistral 7B via HF Inference API) then outputs:

```
Action: analyze_job_description
Action Input: {"job_description": "..."}
```

### Fallback Plan (No LLM)

If the HF Inference API is unreachable (no API key, network issue, rate limit), the agent doesn't crash. It uses `_fallback_decision()` which follows a hardcoded plan:

```python
def _fallback_decision(self, step_idx):
    ordered_tools = [
        ("read_resume", {}),
        ("analyze_job_description", {}),
        ("skill_gap_analysis", {}),
        ("ats_score", {}),
        ("rank_projects", {}),
        ("tailor_resume", {}),
    ]
    if step_idx < len(ordered_tools):
        return ordered_tools[step_idx]
    return ("FINAL", {})
```

This makes the agent **resilient** — it works with or without the LLM.

---

## 5. File-by-File Breakdown

### `app.py` — Streamlit Frontend

**What it does:** The user interface. Handles file upload, text input, button clicks, progress display, and results rendering.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `init_session_state()` | Sets up Streamlit session variables |
| `progress_callback()` | Called by agent after each step — updates UI |
| `_render_status()` | Shows ✅/⏳ step indicators |
| `run_agent()` | Creates AgentController and runs it asynchronously |
| `display_results()` | Renders the 5-tab results view |
| `main()` | Top-level UI layout and event handler |

**Streamlit components used:**
- `st.file_uploader` — Resume PDF upload (accepts only .pdf).
- `st.text_area` — Job description input.
- `st.button` — "Analyze & Tailor Resume" action.
- `st.status` — Collapsible agent activity panel.
- `st.progress` — Progress bar (0% → 100%).
- `st.tabs` — Results organized in 5 tabs.
- `st.download_button` — Export report as .md.

**Async handling:** Streamlit is synchronous. The agent uses `asyncio`. We bridge them by creating a new event loop and calling `loop.run_until_complete(run_agent(...))`. This keeps the UI responsive during agent execution.

---

### `agent/agent_controller.py` — The Agent Brain

**What it does:** Orchestrates everything. Connects to MCP server, runs the ReAct loop, manages data state, generates the final report.

**Class: `AgentController`**

**Key methods:**

| Method | Purpose |
|--------|---------|
| `run()` | Main entry — spawn MCP server, connect, loop, report |
| `_build_react_prompt()` | Constructs the LLM prompt with current state |
| `_fallback_decision()` | Deterministic plan when LLM is down |
| `_parse_action()` | Extracts tool name/input from LLM output |
| `_fill_tool_inputs()` | Auto-fills missing tool params from stored context |
| `_parse_mcp_result()` | Converts MCP response to Python dict |
| `_store_result()` | Saves tool output to the correct field |
| `_generate_final_report()` | Assembles all results into final dict |
| `_build_report_text()` | Creates Markdown report string |

**State management:** The controller stores results from each tool call in dedicated fields (`self.resume_data`, `self.job_data`, etc.). These fields are:
1. **Populated** by `_store_result()` after each MCP call.
2. **Read** by `_fill_tool_inputs()` for subsequent tool calls.
3. **Assembled** by `_generate_final_report()` at the end.

---

### `agent/memory.py` — Agent Memory

**What it does:** Records every step the agent takes. Each `AgentStep` stores the thought, action, input, and observation. `AgentMemory` maintains the ordered list and can produce summaries.

**Classes:**

- **`AgentStep`** — One ReAct cycle (thought + action + observation).
- **`AgentMemory`** — Collection of steps, plus a generic context dict.

The memory serves two purposes:
1. **LLM context** — `step.summary()` generates truncated text for the ReAct prompt.
2. **UI display** — `memory.get_summary()` generates Markdown for the "Workflow Log" tab.

---

### `mcp_server/server.py` — MCP Server Entry Point

**What it does:** Creates a `FastMCP` server and registers 6 tools. When run directly, starts listening on stdio for JSON-RPC messages.

**How FastMCP works:**

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Server Name")

@mcp.tool()
def my_tool(param: str) -> dict:
    return {"result": param}

if __name__ == "__main__":
    mcp.run()  # Starts stdio transport
```

The `@mcp.tool()` decorator:
1. Registers the function as an MCP tool.
2. Uses type hints to generate the JSON schema for tool parameters.
3. Handles JSON-RPC serialization/deserialization automatically.

---

### `mcp_server/resume_tools.py` — Resume Reading

**What it does:** Implements `read_resume` — parses a PDF, extracts text, splits sections, detects skills.

**Flow:**
1. `PDFParser.parse_resume(file_path)` → `{raw_text, summary, skills, experience, education, projects}`
2. `raw_text` is passed to `SkillExtractor.extract_skills()` to detect all known skills.
3. Returns structured dict without `raw_text` (not needed downstream).

**Error handling:** If the PDF can't be parsed (wrong format, file not found), it returns empty data with an `"error"` key instead of crashing.

---

### `mcp_server/analysis_tools.py` — Analysis Tools

**What it does:** Three tools for understanding the job requirements and comparing them against the resume.

**`analyze_job_description(jd)`:**
- Runs `SkillExtractor.extract_skills()` on the JD text.
- Returns the detected skills as `required_skills` and lowercase versions as `keywords`.

**`skill_gap_analysis(resume_data, job_data)`:**
- Converts both skill lists to lowercase sets.
- Computes `matched = resume_set & job_set` and `missing = job_set - resume_set`.
- Score = `len(matched) / len(job_set) * 100`.

**`ats_score(resume_data, job_data)`:**
- Delegates scoring to `ATSScorer.calculate_score()`.
- Generates recommendations via `ATSScorer.get_recommendations()`.

---

### `mcp_server/tailoring_tools.py` — Tailoring Tools

**What it does:** Two tools for the final output — ranking projects and generating tailored content.

**`rank_projects(projects, job_description)`:**
- Tokenizes the JD into words.
- For each project, counts overlapping words between project text and JD.
- Sorts by overlap count descending.

**`tailor_resume(resume_data, job_data)`:**
- Computes matched/missing skills.
- Builds tailored skills list: matched skills first, then up to 5 missing skills.
- Generates a simple professional summary highlighting matched skills.

---

### `models/hf_client.py` — LLM Client (Active)

**What it does:** HTTP client for Hugging Face Inference API's chat completions endpoint.

**Key details:**
- Uses `huggingface-hub`'s `InferenceClient` under the hood.
- Default model: `mistralai/Mistral-7B-Instruct-v0.3`.
- Endpoint: `https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions`.
- Timeout: 30 seconds (HF infra is fast).
- Error handling: clear messages for auth errors, missing model, rate limits.

**Why HF Inference API instead of Ollama?** The original Ollama-based client (`models/ollama_client.py`, now deprecated) ran Qwen 3.5 locally but repeatedly timed out (60s+ per request) on the user's hardware, making ReAct reasoning non-functional. HF Inference API provides fast hosted inference (1-5s per request) with access to stronger models, no local GPU required, and a free tier.

### `models/ollama_client.py` — LLM Client (Deprecated / Reference)

**What it does:** HTTP client for Ollama's `/api/generate` endpoint. **No longer imported or used.**

**Why deprecated:** Local Qwen 3.5 via Ollama repeatedly timed out (60s+ per request), causing every ReAct step to fall through to the fallback plan. Replaced by `hf_client.py` (Hugging Face Inference API).

**Reference details (kept for local-only setups):**
- Uses `httpx` for HTTP with connection pooling.
- Default model: `qwen3.5:2b-q4_K_M`.
- Default endpoint: `http://localhost:11434`.
- Error handling: clear messages for connection refused and 404 (model not pulled).

---

### `utils/pdf_parser.py` — PDF Parser

**What it does:** Extracts text from PDF resumes using PyMuPDF and splits it into sections.

**Section detection:** Uses regex patterns to find common resume headings:

```python
SECTION_PATTERNS = {
    "summary": r"(?i)(professional\s*summary|summary|profile|about\s*me)...",
    "experience": r"(?i)(experience|employment|work\s*history)...",
    "education": r"(?i)(education|academic|qualifications)...",
    "skills": r"(?i)(skills|technical\s*skills|core\s*competencies)...",
    "projects": r"(?i)(projects|personal\s*projects|key\s*projects)...",
}
```

Each pattern uses a lookahead `(?=...)` to find the next section heading without consuming it, so subsequent patterns can match correctly.

**Section parsing:** For list-type sections (skills, projects, experience, education), the content is split into lines. Skills are further split by commas and pipes for fine-grained extraction.

**Fallback:** If no sections are detected (unusual resume format), the first 10 lines are used as a summary.

---

### `utils/skill_extractor.py` — Skill Detection

**What it does:** Detects technology and professional skills in arbitrary text.

**How it works:** Contains a comprehensive list of ~130 skill keywords covering programming languages, frameworks, databases, cloud platforms, DevOps tools, methodologies, and domains. For each keyword, it performs a case-insensitive regex search. Results are sorted by length (longest first) to prioritize specific matches over general ones.

**Example:** If text contains "React", both "React" and "R" could match. Since results are sorted by length, "React" appears first.

---

### `utils/ats.py` — ATS Scoring

**What it does:** Calculates ATS compatibility scores and generates recommendations.

**Scoring formula:** `(matched_skills / total_job_skills) * 100`, capped at 100.

**Recommendations:** Generated based on the gap analysis:
1. If skills are missing, suggests adding them.
2. If fewer than 3 skills matched, suggests increasing keyword density.
3. Always includes general best practices (action verbs, quantify achievements, consistent formatting, etc.).

---

## 6. Data Flow

### End-to-End Flow

```
User uploads PDF + pastes JD
        │
        ▼
app.py saves PDF to output/resume.pdf
        │
        ▼
app.py calls run_agent(pdf_path, jd)
        │
        ▼
AgentController.run():
  ┌─────────────────────────────────────────────────────┐
  │ 1. Spawn MCP server (subprocess)                    │
  │ 2. Connect via stdio_client                         │
  │ 3. Initialize MCP session                           │
  │ 4. List tools (verify connection)                   │
  │ 5. LOOP (up to 12 iterations):                      │
  │    a. Build ReAct prompt with current state         │
  │    b. Ask LLM: "What next?" (or use fallback)       │
  │    c. Parse action from LLM response                │
  │    d. Fill missing inputs from stored context       │
  │    e. Call MCP tool via session.call_tool()         │
  │    f. Parse MCP result (JSON → dict)                │
  │    g. Store result (resume_data, job_data, etc.)    │
  │    h. Record step in memory                         │
  │    i. Notify progress_callback (updates Streamlit)  │
  │ 6. Generate final report                            │
  │ 7. Return results dict                              │
  └─────────────────────────────────────────────────────┘
        │
        ▼
app.py receives results dict
        │
        ▼
display_results(): Renders 5 tabs with all data
```

### Data Dependencies Between Tools

```
read_resume
    │
    ▼
resume_data ────────────┬──────────────────────────────┐
                        │                              │
analyze_job_description │                              │
    │                   │                              │
    ▼                   ▼                              │
job_data ──────────── skill_gap_analysis ──────────────┤
                        │                              │
                        ├── ats_score                  │
                        │                              │
                        └── tailor_resume              │
                                                       │
rank_projects ←─────────────────────────────────────── resum├e_data.projects + job_description
```

Tool calls that depend on prior data:
- `skill_gap_analysis` needs both `resume_data` and `job_data`.
- `ats_score` needs both `resume_data` and `job_data`.
- `rank_projects` needs `resume_data.projects` and the raw `job_description`.
- `tailor_resume` needs both `resume_data` and `job_data`.

The agent controller handles this automatically via `_fill_tool_inputs()` — if a tool is called before its dependencies are available, the controller will wait (the LLM naturally follows the correct order because the ReAct prompt shows what's available and what's not).

---

## 7. Error Handling & Fallbacks

### If the HF Inference API is unavailable

The `HFClient.generate()` call raises an exception (network error, invalid token, rate limit). The agent controller catches it:

```python
try:
    if not self._llm_available:
        raise RuntimeError("LLM previously failed, skip")
    prompt = self._build_react_prompt(tool_names)
    llm_response = self.llm.generate(prompt, system=REACT_SYSTEM_PROMPT)
    action, action_input = self._parse_action(llm_response)
except Exception as e:
    self._llm_available = False
    logger.warning(f"LLM unavailable, using fallback plan: {e}")
    action, action_input = self._fallback_decision(step_idx)
```

Key details:
- After the **first** failure, `_llm_available` is set to `False`, skipping all subsequent LLM calls.
- Remaining steps use the deterministic fallback plan (instant, no waiting).
- Without any API key, all steps use the fallback (the agent still works).

### If the PDF is invalid

`read_resume` catches parsing errors and returns empty data with an `"error"` key. Downstream tools handle missing data gracefully (default empty lists, zero scores).

### If an MCP tool call fails

The `session.call_tool()` call is wrapped in try-except. On failure, the observation is set to `{"error": str(e)}` and the agent continues with the next step.

### If the MCP server fails to start

The `stdio_client` context manager throws an exception, caught by the outer try-except in `run()`. Returns `{"error": str(e)}` along with any partial results.

---

## 8. Extending the Project

### Adding a New MCP Tool

1. **Implement the tool logic** in the appropriate file under `mcp_server/` (or create a new file).
2. **Register it** in `mcp_server/server.py` with `@mcp.tool()`.
3. **Update the agent** in `agent_controller.py`:
   - Add the tool to `TOOL_INPUT_MAP` if it needs auto-filled inputs.
   - Add a storage field (e.g., `self.new_tool_result`).
   - Add a case in `_store_result()`.
   - Add a case in `_fill_tool_inputs()` if needed.
   - Update `_generate_final_report()` to include the new data.
4. **Update the frontend** in `app.py` — add a new results tab or section.

### Changing the LLM Model

In the Streamlit sidebar, select a different model from the dropdown, or in `app.py`:

```python
controller = AgentController(
    hf_model="meta-llama/Meta-Llama-3-8B-Instruct",
    hf_api_key="hf_..."
)
```

To switch back to local Ollama (if your hardware supports it), swap the client in `agent_controller.py`:

```python
# from models.hf_client import HFClient    # comment out
from models.ollama_client import OllamaClient  # restore

# In __init__:
# self.llm = HFClient(...)   # comment out
self.llm = OllamaClient(model="qwen3.5:2b-q4_K_M")
```

### Improving Skill Detection

Edit `utils/skill_extractor.py` — add or remove entries from `SKILL_KEYWORDS`. The extractor does case-insensitive regex matching, so entries should use proper casing for display.

### Adding PDF Export

The `report_text` field is Markdown. Streamlit's `st.download_button` already allows downloading it. To add direct PDF generation, you could integrate `weasyprint` or `pdfkit` in `app.py`.

---

## Summary Table: Which File Does What

| File | Layer | Purpose |
|------|-------|---------|
| `app.py` | Frontend | Streamlit UI — upload, input, progress, results |
| `agent/agent_controller.py` | Agent | ReAct loop, MCP client, state management, report |
| `agent/memory.py` | Agent | Step tracking, context storage |
| `mcp_server/server.py` | MCP Server | FastMCP server with 6 tool registrations |
| `mcp_server/resume_tools.py` | MCP Tool | read_resume implementation |
| `mcp_server/analysis_tools.py` | MCP Tools | analyze_jd, skill_gap, ats_score implementations |
| `mcp_server/tailoring_tools.py` | MCP Tools | rank_projects, tailor_resume implementations |
| `models/hf_client.py` | LLM Client **(active)** | Hugging Face Inference API (Mistral 7B) |
| `models/ollama_client.py` | LLM Client (deprecated) | Local Ollama — kept as reference |
| `utils/pdf_parser.py` | Utility | PDF text extraction and section splitting |
| `utils/skill_extractor.py` | Utility | Keyword-based skill detection |
| `utils/ats.py` | Utility | ATS scoring and recommendation generation |
| `requirements.txt` | Config | Python package dependencies |
| `README.md` | Docs | Setup and run instructions |
| `UNDERSTANDING.md` | Docs | This document |
