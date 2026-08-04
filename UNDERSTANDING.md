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
│  │ Planner │  │ Executor │  │  Memory    │  │ Ollama   │  │
│  │(ReAct)  │  │(MCP call)│  │(steps/ctx) │  │(Qwen 3.5)│  │
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
| `analyze_job_description` | `job_description` (str) | `{role, required_skills, keywords}` | Extract JD requirements |
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
    action, action_input, thought = self._parse_action(llm_response)
    
    # 3. Validate the choice — override invalid/repeated/out-of-order tools
    action, action_input = self._validate_decision(
        action, action_input, tool_names, completed_tools
    )
    
    # 4. Execute via MCP
    mcp_result = await session.call_tool(action, filled_input)
    observation = self._parse_mcp_result(mcp_result)
    
    # 5. Store result and update context
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
Thought: I need the job requirements to compare against.
Action: analyze_job_description
Action Input: {"job_description": "..."}
```

The LLM (Qwen 3.5 via Ollama) then outputs:

```
Thought: I need the job requirements to compare against.
Action: analyze_job_description
Action Input: {"job_description": "..."}
```

The `Thought:` line is parsed and stored in agent memory — it appears in the "Workflow Log" tab so you can see the model's actual reasoning.

### Fallback Plan (No LLM) & Guided Validation

**Fallback when Ollama is unreachable** (not running, model not pulled, timeout, empty response): the agent doesn't crash. `_fallback_decision()` walks the `ORDERED_TOOLS` list, which is the hardcoded plan:

```python
ORDERED_TOOLS = [
    "read_resume",
    "analyze_job_description",
    "skill_gap_analysis",
    "ats_score",
    "rank_projects",
    "tailor_resume",
]

def _fallback_decision(self, step_idx):
    if step_idx < len(ORDERED_TOOLS):
        tool = ORDERED_TOOLS[step_idx]
        return (tool, {}, f"Using fallback plan (Ollama unavailable): calling {tool}.")
    return ("FINAL", {}, "")
```

**Guided validation even when the LLM IS available:** a small local model (Qwen 3.5 2B) frequently proposes invalid, repeated, or out-of-order tool calls. `_validate_decision()` only accepts the LLM's choice when it is a known, not-yet-completed tool whose data dependencies are satisfied. Otherwise it falls back to `_next_planned_tool()` — the first pending tool in `ORDERED_TOOLS`.

```python
def _validate_decision(self, action, action_input, tool_names, completed_tools):
    if action not in tool_names or action in completed_tools or not self._deps_satisfied(action):
        return self._next_planned_tool(completed_tools)
    return (action, action_input)
```

`_deps_satisfied()` is what prevents the bug where `skill_gap_analysis` runs before `analyze_job_description` (no `job_data` yet):

```python
def _deps_satisfied(self, action):
    if action in ("read_resume", "analyze_job_description"):
        return True
    if action in ("skill_gap_analysis", "ats_score", "tailor_resume"):
        return bool(self.resume_data and self.job_data)
    if action == "rank_projects":
        return bool(self.resume_data and self.resume_data.get("projects"))
    return False
```

The full decision ladder:

| Situation | What happens |
|---|---|
| Ollama up, model picks a valid pending tool | Executed as-is |
| Model picks an invalid / repeated / out-of-order tool | Overridden → next pending tool in `ORDERED_TOOLS` |
| Model returns garbage / no recognizable tool | Same override (garbage "FINAL" is also overridden) |
| Ollama down / timeout / empty response | `_llm_available=False` → `_fallback_decision(step_idx)` |
| All 6 tools completed | `_next_planned_tool()` returns `FINAL` → loop breaks |

This makes the agent **resilient** — it completes the full workflow with or without the LLM.

---

## 5. File-by-File Breakdown

### `app.py` — Streamlit Frontend

**What it does:** The user interface. Handles file upload, text input, button clicks, progress display, and results rendering.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `init_session_state()` | Sets up Streamlit session variables |
| `progress_callback()` | Called by agent after each step — updates UI |
| `_render_status()` | Shows step indicators |
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

**Live per-step results:** `progress_callback()` stores each MCP observation in `st.session_state.step_results`, and `_render_status()` shows it inline under each completed step in the activity panel (an expandable "Result" JSON snippet, truncated to 800 chars). The full history remains available in the "Workflow Log" tab after completion.

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
| `_parse_action()` | Extracts tool name/input/thought from LLM output |
| `_fill_tool_inputs()` | Auto-fills missing tool params from stored context |
| `_parse_mcp_result()` | Converts MCP response to Python dict |
| `_store_result()` | Saves tool output to the correct field |
| `_validate_decision()` | Overrides invalid/repeated/out-of-order LLM choices |
| `_next_planned_tool()` | Returns the next pending tool in `ORDERED_TOOLS` |
| `_deps_satisfied()` | Checks a tool's data dependencies are available |
| `_generate_final_report()` | Assembles all results into final dict |
| `_build_report_text()` | Creates Markdown report string |

**State management:** The controller stores results from each tool call in dedicated fields (`self.resume_data`, `self.job_data`, etc.). These fields are:
1. **Populated** by `_store_result()` after each MCP call.
2. **Read** by `_fill_tool_inputs()` for subsequent tool calls.
3. **Assembled** by `_generate_final_report()` at the end.

**TOOL_INPUT_MAP:** Maps tool names to their required context dependencies. Used by `_fill_tool_inputs()` to auto-populate parameters:

```python
TOOL_INPUT_MAP = {
    "read_resume": "file_path",
    "analyze_job_description": "job_description",
    "skill_gap_analysis": "resume_data+job_data",
    "ats_score": "resume_data+job_data",
    "rank_projects": "projects+job_description",
    "tailor_resume": "resume_data+job_data",
}
```

**Why this matters:** The LLM only needs to say "call skill_gap_analysis" — the controller automatically attaches the resume_data and job_data from its context. The LLM never needs to pass large data blobs.

**ORDERED_TOOLS & guided validation:** Because Qwen 3.5 2B frequently proposes invalid or repeated tool calls, the controller also maintains `ORDERED_TOOLS` (the standard 6-step sequence). `_validate_decision()` accepts the LLM's choice only when it is valid and its dependencies are met (`_deps_satisfied()`); otherwise `_next_planned_tool()` supplies the next pending tool. See [§4 Fallback Plan & Guided Validation](#4-the-react-agent-pattern) for the full decision ladder.

---

### `agent/memory.py` — Agent Memory

**What it does:** Records every step the agent takes. Each `AgentStep` stores the thought, action, input, and observation. `AgentMemory` maintains the ordered list and can produce summaries.

**Classes:**

- **`AgentStep`** — One ReAct cycle (thought + action + observation). Timestamped for traceability.
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

**Important:** The server never runs standalone in production. It is always spawned as a subprocess by `agent_controller.py` via `StdioServerParameters`.

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
- Returns detected skills as `required_skills` and lowercase versions as `keywords`.

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
- Computes matched/missing skills via case-insensitive set comparison.
- Builds tailored skills list: matched skills first, then up to 5 missing skills.
- Generates a simple professional summary highlighting matched expertise.

---

### `models/ollama_client.py` — LLM Client (Active)

**What it does:** HTTP client for Ollama's `/api/generate` endpoint.

**Key details:**
- Uses `httpx` for HTTP with connection pooling.
- Default model: `qwen3.5:2b-q4_K_M`.
- Default endpoint: `http://localhost:11434`.
- Timeout: 600 seconds — Ollama on CPU is slow, and Qwen 3.5 is a 2.3B model. A full generation can take several minutes, so the timeout must comfortably exceed that or the agent falls back unnecessarily.
- `think: False` — **critical for Qwen3 models.** By default Qwen3 runs in "thinking mode" and spends all its output tokens on internal reasoning, returning an **empty** response. Disabling thinking makes it respond in ~4s instead of timing out with nothing.
- `num_predict: 256` — ReAct responses are short (`Thought:` + `Action:`), so 256 tokens is plenty and keeps generation fast (was 2048).
- Pydantic models: `GenerationConfig` and `OllamaResponse` for request/response validation.

**Error handling:**
- `ConnectionError` — Ollama not running (provide clear message to run `ollama serve`).
- `ValueError` — Model not found (provide clear message to run `ollama pull`).
- `RuntimeError` — Other HTTP errors, **including an empty response**. If the model returns no text, the client raises instead of returning `""` — this prevents the agent from silently treating an empty reply as "done" and producing an empty report.

**Why Ollama instead of a cloud API?** The project was designed to run fully locally with no cloud dependencies. Qwen 3.5 provides reasonable reasoning capability for the ReAct loop while running on consumer hardware. If Ollama is too slow on a given machine, the agent auto-falls back to the deterministic plan.

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

**Section parsing:** For list-type sections (skills, projects, experience, education), the content is split into lines. Skills are further split by commas, pipes, and slashes for fine-grained extraction.

**Fallback:** If no sections are detected (unusual resume format), the first 10 lines are used as a summary.

---

### `utils/skill_extractor.py` — Skill Detection

**What it does:** Detects technology and professional skills in arbitrary text.

**How it works:** Contains a comprehensive list of ~130 skill keywords covering programming languages, frameworks, databases, cloud platforms, DevOps tools, methodologies, and domains. For each keyword, it performs a case-insensitive regex search. Results are sorted by length (longest first) to prioritize specific matches over general ones.

**Example:** If text contains "React", both "React" and "R" could match. Since results are sorted by length, "React" appears first.

---

### `utils/ats.py` — ATS Scoring

**What it does:** Calculates ATS compatibility scores and generates recommendations.

**Scoring formula:** `(len(matched_skills) / len(total_job_skills)) * 100`, capped at 100.

**Recommendations:** Generated based on the gap analysis:
1. If skills are missing, suggests adding them with specific skill names.
2. If fewer than 3 skills matched, suggests increasing keyword density.
3. Always includes general best practices (action verbs, quantify achievements, consistent formatting, 1-2 pages, standard headings).

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
rank_projects ←────────────────────────────────────── resume_data.projects + job_description
```

Tool calls that depend on prior data:
- `skill_gap_analysis` needs both `resume_data` and `job_data`.
- `ats_score` needs both `resume_data` and `job_data`.
- `rank_projects` needs `resume_data.projects` and the raw `job_description`.
- `tailor_resume` needs both `resume_data` and `job_data`.

The agent controller handles this automatically via `_fill_tool_inputs()` — if a tool is called before its dependencies are available, the controller will fill them from context. In practice, the LLM naturally follows the correct order because the ReAct prompt shows what's available and what's not.

---

## 7. Error Handling & Fallbacks

### If Ollama is Unavailable

The `OllamaClient.generate()` call raises an exception (connection refused, model not found, timeout, empty response). The agent controller catches it:

```python
try:
    if not self._llm_available:
        raise RuntimeError("LLM previously failed, skip")
    prompt = self._build_react_prompt(tool_names)
    llm_response = self.llm.generate(prompt, system=REACT_SYSTEM_PROMPT)
    action, action_input, thought = self._parse_action(llm_response)
except Exception as e:
    self._llm_available = False
    logger.warning(f"LLM unavailable, using fallback plan: {e}")
    action, action_input, thought = self._fallback_decision(step_idx)
```

Key details:
- After the **first** failure, `_llm_available` is set to `False`, skipping all subsequent LLM calls.
- Remaining steps use the deterministic fallback plan (instant, no waiting).
- Without Ollama running, all steps use the fallback (the agent still works).

### If the LLM Makes a Bad Decision (still available)

Even when Ollama responds, a small model may choose the wrong tool. `_validate_decision()` catches this *after* the LLM call succeeds:
- Invalid tool name → overridden to the next pending tool in `ORDERED_TOOLS`.
- Already-completed tool (e.g. calling `read_resume` twice) → overridden.
- Tool whose dependencies aren't ready (e.g. `skill_gap_analysis` before `analyze_job_description`) → overridden via `_deps_satisfied()`.

This is a second safety net that keeps the workflow deterministic even when the LLM is "working" but unreliable.

### If the PDF is Invalid

`read_resume` catches parsing errors and returns empty data with an `"error"` key. Downstream tools handle missing data gracefully (default empty lists, zero scores).

### If an MCP Tool Call Fails

The `session.call_tool()` call is wrapped in try-except. On failure, the observation is set to `{"error": str(e)}` and the agent continues with the next step.

### If the MCP Server Fails to Start

The `stdio_client` context manager throws an exception, caught by the outer try-except in `run()`. Returns `{"error": str(e)}` along with any partial results.

---

## 8. Extending the Project

### Adding a New MCP Tool

1. **Implement the tool logic** in the appropriate file under `mcp_server/` (or create a new file).
2. **Register it** in `mcp_server/server.py` with `@mcp.tool()`.
3. **Update the agent** in `agent_controller.py`:
   - Add the tool to `TOOL_INPUT_MAP` if it needs auto-filled inputs.
   - Add the tool to `ORDERED_TOOLS` if it should be part of the standard workflow sequence.
   - Add a storage field (e.g., `self.new_tool_result`).
   - Add a case in `_store_result()`.
   - Add a case in `_fill_tool_inputs()` if needed.
   - Add a case in `_deps_satisfied()` if the tool has data dependencies.
   - Update `_generate_final_report()` to include the new data.
4. **Update the frontend** in `app.py` — add a new results tab or section.

### Changing the LLM Model

In `agent_controller.py`, change the default model in `__init__`:

```python
controller = AgentController(llm_model="llama3.2:3b")
```

Or pull a different model with Ollama:

```powershell
ollama pull llama3.2:3b
```

### Using a Cloud LLM Instead of Ollama

To switch from local Ollama to a cloud API (e.g., Hugging Face, OpenAI):

1. Create a new client in `models/` (e.g., `models/hf_client.py`) that implements a `generate(prompt, system, temperature)` method.
2. Swap the client in `agent_controller.py`:

```python
# from models.ollama_client import OllamaClient
from models.hf_client import HFClient

# In __init__:
# self.llm = OllamaClient(model=llm_model)
self.llm = HFClient(api_key="hf_...", model="mistralai/Mistral-7B-Instruct-v0.3")
```

### Improving Skill Detection

Edit `utils/skill_extractor.py` — add or remove entries from `SKILL_KEYWORDS`. The extractor does case-insensitive regex matching, so entries should use proper casing for display.

### `prompt.txt` — Original Build Prompt

The file `prompt.txt` at the project root contains the exact prompt that was given to an LLM to generate this entire project. It serves as:
- A reference for the original requirements.
- A specification document showing what was asked for vs what was built.
- A starting point for regenerating or re-architecting the project.

---

## Summary Table: Which File Does What

| File | Layer | Purpose |
|------|-------|---------|
| `app.py` | Frontend | Streamlit UI — upload, input, progress, results; live per-step result snippets |
| `agent/agent_controller.py` | Agent | ReAct loop, guided plan validation, MCP client, state management, report |
| `agent/memory.py` | Agent | Step tracking, context storage |
| `mcp_server/server.py` | MCP Server | FastMCP server with 6 tool registrations |
| `mcp_server/resume_tools.py` | MCP Tool | read_resume implementation |
| `mcp_server/analysis_tools.py` | MCP Tools | analyze_jd, skill_gap, ats_score implementations |
| `mcp_server/tailoring_tools.py` | MCP Tools | rank_projects, tailor_resume implementations |
| `models/ollama_client.py` | LLM Client | Ollama HTTP client (Qwen 3.5) — `think: False`, 600s timeout, empty-response guard |
| `utils/pdf_parser.py` | Utility | PDF text extraction and section splitting |
| `utils/skill_extractor.py` | Utility | Keyword-based skill detection (~130 skills) |
| `utils/ats.py` | Utility | ATS scoring and recommendation generation |
| `prompt.txt` | Docs | Original LLM prompt that generated this project |
| `requirements.txt` | Config | Python package dependencies |
| `README.md` | Docs | Setup and run instructions |
| `UNDERSTANDING.md` | Docs | This document |
