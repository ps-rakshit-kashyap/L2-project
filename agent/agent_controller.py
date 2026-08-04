# agent/agent_controller.py
# The heart of the system — orchestrates the ReAct (Reasoning + Acting) loop.
#
# Architecture:
# 1. Spawns the MCP server as a subprocess (via StdioServerParameters).
# 2. Connects to it using mcp.client.stdio.stdio_client.
# 3. Runs a ReAct loop where the LLM (Qwen 3.5) decides the next action.
# 4. Executes the decided action via the MCP client session.
# 5. Stores observations and builds context for the next iteration.
# 6. If the LLM is unavailable, uses a deterministic fallback plan.
# 7. Generates a final report from all collected data.
#
# The ReAct cycle at each step:
#   Thought: "I need resume details."
#   Action: read_resume(file_path="...")
#   Observation: {summary, skills, projects, ...}
#   (feed back to LLM for next decision)

import asyncio
import json
import sys
import os
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple, Set

# Add project root to path for imports when running as subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

# ── LLM Client ──────────────────────────────────────────────────────────────
# Uses Ollama with a local model (qwen3.5:2b-q4_K_M) for ReAct reasoning.
# If Ollama is unavailable or times out, falls back to the deterministic plan.
from models.ollama_client import OllamaClient
from agent.memory import AgentMemory, AgentStep

logger = logging.getLogger(__name__)

# System prompt for the ReAct agent — tells the LLM how to behave,
# what tools are available, and the expected output format.
# The {{...}} are literal braces in the prompt (escaped for f-string compatibility).
REACT_SYSTEM_PROMPT = """You are a Resume Tailoring Agent. Your job is to tailor a resume for a specific job description.

You have these tools available:

1. **read_resume** — Parse a resume PDF. Returns summary, skills, projects, experience, education.
2. **analyze_job_description** — Extract requirements from a job description. Returns required_skills, keywords.
3. **skill_gap_analysis** — Compare resume skills vs job requirements. Returns matched_skills, missing_skills, score.
4. **ats_score** — Calculate ATS compatibility. Returns score, recommendations.
5. **rank_projects** — Rank projects by relevance to the job. Returns ranked_projects.
6. **tailor_resume** — Generate tailored resume content. Returns professional_summary, tailored_skills, etc.

Work through the steps in order:
1. read_resume
2. analyze_job_description
3. skill_gap_analysis
4. ats_score
5. rank_projects
6. tailor_resume
7. Output Final Answer

IMPORTANT: The ONLY valid tool names are:
read_resume, analyze_job_description, skill_gap_analysis, ats_score, rank_projects, tailor_resume.
Never invent or use any other tool name.

Output format (one action per response):
Thought: <one sentence explaining why>
Action: tool_name
Action Input: {{"param": "value"}}

When all steps are complete, output:
Final Answer: {{"report": "..."}}
"""

# Maps each tool to the data it needs from the agent's context.
# Used by _fill_tool_inputs to automatically supply correct parameters.
TOOL_INPUT_MAP: Dict[str, str] = {
    "read_resume": "file_path",
    "analyze_job_description": "job_description",
    "skill_gap_analysis": "resume_data+job_data",
    "ats_score": "resume_data+job_data",
    "rank_projects": "projects+job_description",
    "tailor_resume": "resume_data+job_data",
}

# The standard workflow order. Used to enforce a valid sequence even when
# the (small local) LLM proposes an invalid or repeated tool call.
ORDERED_TOOLS: List[str] = [
    "read_resume",
    "analyze_job_description",
    "skill_gap_analysis",
    "ats_score",
    "rank_projects",
    "tailor_resume",
]


class AgentController:
    """Orchestrates the ReAct agent loop with MCP tools and LLM reasoning.
    
    Maintains internal state (resume_data, job_data, etc.) that gets
    populated as each MCP tool is called. The LLM decides which tool
    to call next based on what data is still needed.
    """

    def __init__(self, llm_model: str = "qwen3.5:2b-q4_K_M"):
        # Initialize LLM client via local Ollama.
        # Falls back to deterministic plan if Ollama is unreachable or times out.
        self.llm = OllamaClient(model=llm_model)
        self.memory = AgentMemory()

        # Data store — populated by MCP tool results
        self.resume_data: Optional[Dict[str, Any]] = None
        self.job_data: Optional[Dict[str, Any]] = None
        self.skill_gap: Optional[Dict[str, Any]] = None
        self.ats_result: Optional[Dict[str, Any]] = None
        self.ranked_projects: Optional[Dict[str, Any]] = None
        self.tailored_resume: Optional[Dict[str, Any]] = None

        # Input parameters from the user
        self.resume_path: str = ""
        self.job_description: str = ""
        # After first LLM failure, skip subsequent LLM calls to avoid repeated timeouts
        self._llm_available: bool = True

    async def run(
        self,
        resume_path: str,
        job_description: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the full agent workflow.
        
        Steps:
        1. Start MCP server as subprocess.
        2. Connect via MCP client session.
        3. Run ReAct loop (up to 12 iterations).
        4. Each iteration: LLM decides action → execute via MCP → store result.
        5. Generate final report after loop completes.
        
        Args:
            resume_path: Path to the uploaded resume PDF.
            job_description: Job description text from the user.
            progress_callback: Optional function(status, data) for Streamlit UI updates.
            
        Returns:
            Dict with all analysis results and the final report text.
        """
        self.resume_path = resume_path
        self.job_description = job_description

        # Configure the MCP server subprocess — run server.py with the same Python interpreter
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server", "server.py")],
        )

        result: Dict[str, Any] = {}

        try:
            # Connect to MCP server via stdio transport
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List available tools to verify the connection
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    logger.info(f"Connected to MCP server. Tools: {tool_names}")

                    if progress_callback:
                        progress_callback("initialized", {"status": "connected"})

                    # ReAct loop — try up to 12 iterations
                    max_steps = 12
                    completed_tools: Set[str] = set()
                    for step_idx in range(max_steps):
                        thought = ""
                        try:
                            if not self._llm_available:
                                raise RuntimeError("LLM previously failed, skip")
                            prompt = self._build_react_prompt(tool_names)
                            llm_response = self.llm.generate(
                                prompt,
                                system=REACT_SYSTEM_PROMPT,
                                temperature=0.1 ,
                            )
                            action, action_input, thought = self._parse_action(llm_response)
                        except Exception as e:
                            self._llm_available = False
                            logger.warning(f"LLM unavailable, using fallback plan: {e}")
                            action, action_input, thought = self._fallback_decision(step_idx)

                        # Enforce the plan: only accept valid, pending tools whose
                        # dependencies are satisfied; otherwise pick the next planned tool.
                        proposed_action = action
                        action, action_input = self._validate_decision(
                            action, action_input, tool_names, completed_tools
                        )

                        # If the plan overrode the LLM's choice, don't show the LLM's
                        # thought (it reasoned about a different action).
                        if action != proposed_action:
                            thought = ""

                        # All steps are done
                        if action == "FINAL":
                            if progress_callback:
                                progress_callback("finalizing", {})
                            break

                        if not thought:
                            thought = f"Calling {action} to gather more information."

                        # Fill in any missing input parameters from agent context
                        filled_input = self._fill_tool_inputs(action, action_input)

                        if not filled_input:
                            completed_tools.add(action)
                            continue

                        # Execute the tool via MCP
                        try:
                            mcp_result = await session.call_tool(action, filled_input)
                            observation = self._parse_mcp_result(mcp_result)
                        except Exception as e:
                            logger.error(f"Tool call {action} failed: {e}")
                            observation = {"error": str(e)}
                            if progress_callback:
                                progress_callback("error", {"action": action, "error": str(e)})

                        # Store the result in the appropriate field
                        self._store_result(action, observation)
                        completed_tools.add(action)

                        # Record the step in agent memory
                        step = AgentStep(thought, action, filled_input, observation)
                        self.memory.add_step(step)

                        if progress_callback:
                            progress_callback(action, observation)

                    # Generate final structured report
                    report = await self._generate_final_report()
                    result = report

        except Exception as e:
            logger.error(f"Agent controller failed: {e}", exc_info=True)
            result = {
                "error": str(e),
                "resume_data": self.resume_data,
                "job_data": self.job_data,
            }

        finally:
            self.llm.close()

        return result

    def _build_react_prompt(self, tool_names: List[str]) -> str:
        """Build the ReAct prompt showing current state for LLM decision.
        
        Includes:
        - The user's goal
        - Current data availability (what's been collected, what's missing)
        - Previous steps with observations
        - Key data summaries (resume skills, job skills)
        - Expected output format
        
        Args:
            tool_names: List of available MCP tool names.
            
        Returns:
            Formatted prompt string for the LLM.
        """
        parts = [f"Goal: Tailor my resume for this job."]
        parts.append(f"Resume file: {self.resume_path}")
        parts.append("")

        # Show which data is already available and what's still needed
        state_parts = []
        state_parts.append("Current data state:")
        state_parts.append(f"- resume_data: {'available' if self.resume_data else 'not available'}")
        state_parts.append(f"- job_data: {'available' if self.job_data else 'not available'}")
        state_parts.append(f"- skill_gap: {'available' if self.skill_gap else 'not available'}")
        state_parts.append(f"- ats_result: {'available' if self.ats_result else 'not available'}")
        state_parts.append(f"- ranked_projects: {'available' if self.ranked_projects else 'not available'}")
        state_parts.append(f"- tailored_resume: {'available' if self.tailored_resume else 'not available'}")
        parts.append("\n".join(state_parts))
        parts.append("")

        # Show previous steps with truncated observations
        if self.memory.steps:
            parts.append("Steps completed so far:")
            for step in self.memory.steps:
                summary = step.summary(max_obs_length=200)
                parts.append(summary)

        # Show key data summaries
        if self.resume_data:
            skills = self.resume_data.get("skills", [])
            parts.append(f"Resume skills: {skills[:10]}")

        if self.job_data:
            req = self.job_data.get("required_skills", [])
            parts.append(f"Job required skills: {req[:10]}")

        # Output format instructions
        parts.append("\nWhat is the next action? Output exactly in this format:")
        parts.append("Action: tool_name")
        parts.append('Action Input: {"param": "value"}')
        parts.append("Or if complete: Final Answer: {}")

        return "\n".join(parts)

    def _fallback_decision(self, step_idx: int) -> Tuple[str, Dict[str, Any], str]:
        """Deterministic fallback plan when LLM is unavailable.
        
        Follows the standard 6-step workflow in order:
        1. read_resume → 2. analyze_job_description → 3. skill_gap_analysis
        4. ats_score → 5. rank_projects → 6. tailor_resume → FINAL
        
        Args:
            step_idx: Current iteration index in the ReAct loop.
            
        Returns:
            Tuple of (action_name, action_input_dict, thought_text).
        """
        if step_idx < len(ORDERED_TOOLS):
            tool = ORDERED_TOOLS[step_idx]
            return (tool, {}, f"Using fallback plan (Ollama unavailable): calling {tool}.")
        return ("FINAL", {}, "")

    def _validate_decision(
        self,
        action: str,
        action_input: Dict[str, Any],
        tool_names: List[str],
        completed_tools: Set[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Enforce the workflow plan regardless of the LLM's choice.
        
        A small local model (e.g. Qwen 3.5 2B) frequently proposes invalid,
        repeated, or out-of-order tool calls. This method accepts the LLM's
        decision only when it is a known, not-yet-completed tool whose data
        dependencies are satisfied. Otherwise it returns the next pending
        tool in the standard workflow order.
        
        Args:
            action: Proposed action from the LLM or fallback plan.
            action_input: Proposed input parameters.
            tool_names: List of available MCP tool names.
            completed_tools: Set of tools already executed this run.
            
        Returns:
            Tuple of (valid_action, action_input).
        """
        if action not in tool_names or action in completed_tools or not self._deps_satisfied(action):
            return self._next_planned_tool(completed_tools)
        return (action, action_input)

    def _next_planned_tool(self, completed_tools: Set[str]) -> Tuple[str, Dict[str, Any]]:
        """Return the first not-yet-completed tool in the standard workflow order."""
        for tool in ORDERED_TOOLS:
            if tool not in completed_tools:
                return (tool, {})
        return ("FINAL", {})

    def _deps_satisfied(self, action: str) -> bool:
        """Check that all data dependencies for a tool are currently available.
        
        Prevents out-of-order tool calls (e.g. running skill_gap_analysis
        before the job description has been analyzed).
        
        Args:
            action: The MCP tool name.
            
        Returns:
            True if the tool can run with the data collected so far.
        """
        if action in ("read_resume", "analyze_job_description"):
            return True
        if action in ("skill_gap_analysis", "ats_score", "tailor_resume"):
            return bool(self.resume_data and self.job_data)
        if action == "rank_projects":
            return bool(self.resume_data and self.resume_data.get("projects"))
        return False

    def _parse_action(self, llm_response: str) -> Tuple[str, Dict[str, Any], str]:
        """Parse the LLM's response to extract the next action.
        
        Supports two formats:
        1. "Thought: ...\nAction: tool_name\nAction Input: {...}" — tool call
        2. "Final Answer: ..." — completion signal
        
        Falls back to heuristic tool name detection if the format is off.
        
        Args:
            llm_response: Raw text output from the LLM.
            
        Returns:
            Tuple of (action_name, action_input_dict, thought_text).
        """
        import re

        # Check for final answer
        if "Final Answer:" in llm_response or "FINAL" in llm_response:
            return ("FINAL", {}, "")

        # Capture the LLM's reasoning for the workflow log
        thought = ""
        thought_match = re.search(r"Thought:\s*(.+)", llm_response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Try to extract structured action/input
        action_match = re.search(r"Action:\s*(\w+)", llm_response)
        input_match = re.search(r"Action Input:\s*(\{.*\})", llm_response, re.DOTALL)

        if not action_match:
            # Fallback: search for any known tool name in the response
            for tool in [
                "read_resume", "analyze_job_description", "skill_gap_analysis",
                "ats_score", "rank_projects", "tailor_resume",
            ]:
                if tool in llm_response:
                    return (tool, {}, thought)
            return ("FINAL", {}, thought)

        action = action_match.group(1)
        action_input: Dict[str, Any] = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}

        return (action, action_input, thought)

    def _fill_tool_inputs(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-fill required tool inputs from the agent's context.
        
        For example, skill_gap_analysis needs resume_data and job_data —
        this method supplies them from the agent's stored state without
        requiring the LLM to pass the actual data.
        
        Args:
            action: The MCP tool name.
            action_input: Partial input dict from the LLM (may be empty).
            
        Returns:
            Complete input dict with all required parameters filled in.
        """
        if action == "read_resume":
            if "file_path" not in action_input:
                action_input["file_path"] = self.resume_path
        elif action == "analyze_job_description":
            if "job_description" not in action_input:
                action_input["job_description"] = self.job_description
        elif action in ("skill_gap_analysis", "ats_score", "tailor_resume"):
            if "resume_data" not in action_input and self.resume_data:
                action_input["resume_data"] = self.resume_data
            if "job_data" not in action_input and self.job_data:
                action_input["job_data"] = self.job_data
        elif action == "rank_projects":
            if "projects" not in action_input and self.resume_data:
                action_input["projects"] = self.resume_data.get("projects", [])
            if "job_description" not in action_input:
                action_input["job_description"] = self.job_description
        return action_input

    def _parse_mcp_result(self, result) -> Any:
        """Parse the MCP tool call result into a Python dict.
        
        MCP results contain a list of Content objects (usually TextContent).
        This method extracts the text and parses it as JSON.
        Also detects error conditions from result.isError.
        
        Args:
            result: CallToolResult from the MCP client session.
            
        Returns:
            Parsed dictionary from the tool's JSON response.
        """
        try:
            is_error = getattr(result, "isError", False)
            if hasattr(result, "content") and result.content:
                for content_item in result.content:
                    if hasattr(content_item, "text"):
                        try:
                            parsed = json.loads(content_item.text)
                            if is_error:
                                parsed["_mcp_error"] = True
                            return parsed
                        except json.JSONDecodeError:
                            # Tool returned non-JSON (e.g. error message)
                            return {"raw_text": content_item.text, "_mcp_error": is_error}
            return {"raw": str(result)}
        except (TypeError, AttributeError) as e:
            return {"raw": str(result), "_parse_error": str(e)}

    def _store_result(self, action: str, observation: Dict[str, Any]) -> None:
        """Store a tool's result in the appropriate field.
        
        Maps tool names to their corresponding storage fields:
        - read_resume → self.resume_data
        - analyze_job_description → self.job_data
        - etc.
        
        Args:
            action: The name of the MCP tool that was called.
            observation: The parsed result from that tool.
        """
        if action == "read_resume":
            self.resume_data = observation
        elif action == "analyze_job_description":
            self.job_data = observation
        elif action == "skill_gap_analysis":
            self.skill_gap = observation
        elif action == "ats_score":
            self.ats_result = observation
        elif action == "rank_projects":
            self.ranked_projects = observation
        elif action == "tailor_resume":
            self.tailored_resume = observation

    async def _generate_final_report(self) -> Dict[str, Any]:
        """Assemble the final structured report from all collected data.
        
        Aggregates results from all 6 MCP tool calls into a comprehensive
        report dictionary and a human-readable Markdown text report.
        
        Returns:
            Dict containing:
            - match_score, ats_score
            - matched_skills, missing_skills
            - ats_recommendations
            - ranked_projects
            - tailored_summary, tailored_skills, tailored_experience, tailored_projects
            - report_text (Markdown)
            - workflow_history (Markdown)
        """
        # Safely extract values from tool results (which may be None)
        match_score = 0
        if self.skill_gap:
            match_score = self.skill_gap.get("score", 0)

        matched_skills = []
        if self.skill_gap:
            matched_skills = self.skill_gap.get("matched_skills", [])

        missing_skills = []
        if self.skill_gap:
            missing_skills = self.skill_gap.get("missing_skills", [])

        ats_recommendations = []
        if self.ats_result:
            ats_recommendations = self.ats_result.get("recommendations", [])

        ats_score_val = 0
        if self.ats_result:
            ats_score_val = self.ats_result.get("score", 0)

        ranked_projects = []
        if self.ranked_projects:
            ranked_projects = self.ranked_projects.get("ranked_projects", [])

        tailored_summary = ""
        tailored_skills = []
        tailored_experience = []
        tailored_projects = []
        if self.tailored_resume:
            tailored_summary = self.tailored_resume.get("professional_summary", "")
            tailored_skills = self.tailored_resume.get("tailored_skills", [])
            tailored_experience = self.tailored_resume.get("tailored_experience", [])
            tailored_projects = self.tailored_resume.get("tailored_projects", [])

        report_text = self._build_report_text(
            match_score, matched_skills, missing_skills,
            ats_recommendations, ranked_projects,
            tailored_summary, tailored_skills,
            tailored_experience, tailored_projects,
        )

        return {
            # Analysis results
            "match_score": match_score,
            "ats_score": ats_score_val,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "ats_recommendations": ats_recommendations,
            # Tailoring results
            "ranked_projects": ranked_projects,
            "tailored_summary": tailored_summary,
            "tailored_skills": tailored_skills,
            "tailored_experience": tailored_experience,
            "tailored_projects": tailored_projects,
            # Formatted outputs
            "report_text": report_text,
            "workflow_history": self.memory.get_summary(),
        }

    def _build_report_text(
        self,
        match_score: int,
        matched_skills: List[str],
        missing_skills: List[str],
        ats_recommendations: List[str],
        ranked_projects: List[Any],
        tailored_summary: str,
        tailored_skills: List[str],
        tailored_experience: List[Any],
        tailored_projects: List[Any],
    ) -> str:
        """Build a human-readable Markdown report from all analysis data.
        
        This is the final output the user can download as a .md file.
        Covers all sections: score, skills, recommendations, projects,
        summary, experience, and projects.
        
        Args:
            match_score: Skill match percentage.
            matched_skills: Skills present in both resume and JD.
            missing_skills: Skills missing from resume.
            ats_recommendations: List of ATS improvement tips.
            ranked_projects: Projects sorted by relevance.
            tailored_summary: Optimized professional summary.
            tailored_skills: Optimized skills list.
            tailored_experience: Experience section.
            tailored_projects: Projects section.
            
        Returns:
            Complete Markdown report as a single string.
        """
        lines = [
            "# Resume Analysis Report",
            "",
            f"## Overall Match Score: {match_score}%",
            "",
            "## Strong Skills",
        ]
        for s in matched_skills:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("## Missing Skills")
        for s in missing_skills:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("## ATS Recommendations")
        for r in ats_recommendations:
            lines.append(f"- {r}")
        lines.append("")
        lines.append("## Relevant Projects (Ranked)")
        for p in ranked_projects:
            if isinstance(p, dict):
                lines.append(f"- {p.get('name', str(p))}")
            else:
                lines.append(f"- {p}")
        lines.append("")
        lines.append("## Tailored Professional Summary")
        lines.append(tailored_summary)
        lines.append("")
        lines.append("## Tailored Skills")
        for s in tailored_skills:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("## Tailored Experience")
        for e in tailored_experience[:5]:
            lines.append(f"- {e}")
        lines.append("")
        lines.append("## Tailored Projects")
        for p in tailored_projects[:5]:
            if isinstance(p, dict):
                lines.append(f"- {p.get('name', str(p))}")
            else:
                lines.append(f"- {p}")

        return "\n".join(lines)
