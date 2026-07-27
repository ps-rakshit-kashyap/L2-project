# app.py
# Streamlit frontend for the Resume Tailoring Agent.
# Provides the user interface for uploading resumes, entering job
# descriptions, triggering the agent, and viewing results.
#
# Key sections:
# 1. Layout: Title, sidebar instructions, file uploader, text area, button.
# 2. Agent Execution: Spawns the agent controller, tracks progress via
#    callbacks that update Streamlit status components.
# 3. Results Display: Tabbed view of skills, ATS, projects, tailored resume.
# 4. Download: Export full report as Markdown.

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from agent.agent_controller import AgentController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Directory where uploaded PDFs are saved before processing
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Resume Tailoring Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Human-readable labels for each agent workflow step
STEP_LABELS = {
    "initialized": " Connecting to MCP Server",
    "read_resume": " Reading Resume PDF",
    "analyze_job_description": " Analyzing Job Description",
    "skill_gap_analysis": " Comparing Skills (Gap Analysis)",
    "ats_score": " Calculating ATS Score",
    "rank_projects": " Ranking Projects by Relevance",
    "tailor_resume": " Tailoring Resume Content",
    "finalizing": " Generating Final Report",
    "error": " Error Occurred",
}

# The expected order of steps — used for progress bar calculation
STEP_ORDER = [
    "initialized",
    "read_resume",
    "analyze_job_description",
    "skill_gap_analysis",
    "ats_score",
    "rank_projects",
    "tailor_resume",
    "finalizing",
]


def init_session_state() -> None:
    """Initialize Streamlit session state variables.
    
    Must be called at the start of each script run to ensure
    all state keys exist before they're accessed.
    """
    if "status_items" not in st.session_state:
        st.session_state.status_items = {}
    if "agent_running" not in st.session_state:
        st.session_state.agent_running = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = None


def progress_callback(action: str, data: Dict[str, Any]) -> None:
    """Callback invoked by the agent controller after each step.
    
    Updates Streamlit's progress bar, status indicators, and step labels.
    This function runs in the same thread as Streamlit (synchronous),
    so it can safely access st.session_state.
    
    Args:
        action: The action/step name (e.g. "read_resume", "error").
        data: Additional data from the agent (observations, errors).
    """
    if action == "error":
        st.session_state.current_status = "error"
        st.error(f"Error during step: {data.get('action', 'unknown')}: {data.get('error', 'unknown error')}")
        return

    # Calculate and update progress bar
    step_idx = STEP_ORDER.index(action) if action in STEP_ORDER else -1
    if step_idx >= 0:
        st.session_state.current_step = max(st.session_state.current_step, step_idx + 1)
        progress_val = st.session_state.current_step / len(STEP_ORDER)
        if st.session_state.progress_bar is not None:
            st.session_state.progress_bar.progress(progress_val)

    # Mark this step as complete
    st.session_state.status_items[action] = "complete"

    # Update the status display (placeholder container in the UI)
    status_placeholder = st.session_state.get("status_placeholder")
    if status_placeholder is not None:
        with status_placeholder.container():
            _render_status(st.session_state.status_items)


def _render_status(status_items: Dict[str, str]) -> None:
    """Render the step-by-step status indicators.
    
    Shows each step with a status icon:
    - ✅ Completed steps (success)
    - ℹ️ Currently running steps (info)
    - ⏳ Future steps (text)
    
    Args:
        status_items: Dict mapping step names to their status ("complete" or missing).
    """
    for step_name in STEP_ORDER:
        label = STEP_LABELS.get(step_name, step_name)
        status = status_items.get(step_name)
        if status == "complete":
            st.success(f"{label}")
        elif status == "running":
            st.info(f"{label}")
        else:
            st.text(f"⏳ {label}")


async def run_agent(resume_path: str, job_description: str) -> Dict[str, Any]:
    """Create and run the agent controller with local Ollama.

    This is the async entry point that Streamlit calls via
    asyncio.run_until_complete().

    Args:
        resume_path: Path to the saved resume PDF.
        job_description: Job description text.

    Returns:
        Dict with all analysis results.
    """
    controller = AgentController()
    result = await controller.run(
        resume_path=resume_path,
        job_description=job_description,
        progress_callback=progress_callback,
    )
    return result


def display_results(results: Dict[str, Any]) -> None:
    """Display the agent's results in a tabbed layout.
    
    Tabs:
    1. Skills Analysis — Matched vs missing skills.
    2. ATS Report — Score and recommendations.
    3. Ranked Projects — Projects sorted by relevance.
    4. Tailored Resume — Summary, skills, experience, projects.
    5. Workflow Log — Full agent step history.
    
    Args:
        results: The complete results dict from the agent controller.
    """
    if "error" in results and results.get("error"):
        st.error(f"Agent encountered an error: {results['error']}")
        return

    # Top-level summary: overall fit score and skill counts
    col1, col2 = st.columns(2)

    with col1:
        match_score = results.get("match_score", 0)
        ats_score_val = results.get("ats_score", 0)
        avg_score = (match_score + ats_score_val) // 2

        st.markdown("### Overall Fit Score")
        score_color = "green" if avg_score >= 70 else ("orange" if avg_score >= 40 else "red")
        st.markdown(
            f"<h1 style='color: {score_color};'>{avg_score}%</h1>",
            unsafe_allow_html=True,
        )
        st.caption(f"Skills Match: {match_score}% | ATS Score: {ats_score_val}%")

    with col2:
        matched = results.get("matched_skills", [])
        missing = results.get("missing_skills", [])
        st.markdown("### Skills Summary")
        st.markdown(f"**Strong Skills:** {len(matched)}")
        st.markdown(f"**Missing Skills:** {len(missing)}")

    st.divider()

    # Tabbed detailed view
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Skills Analysis",
        " ATS Report",
        " Ranked Projects",
        " Tailored Resume",
        " Workflow Log",
    ])

    # Tab 1: Skills gap analysis
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Strong Skills (Matched)")
            matched_skills = results.get("matched_skills", [])
            if matched_skills:
                for skill in matched_skills:
                    st.markdown(f"- ✅ {skill}")
            else:
                st.info("No matching skills found.")

        with col_b:
            st.markdown("#### Missing Skills")
            missing_skills = results.get("missing_skills", [])
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f"- ❌ {skill}")
            else:
                st.success("All required skills are covered!")

    # Tab 2: ATS compatibility
    with tab2:
        st.markdown("#### ATS Score")
        st.metric("ATS Compatibility", f"{results.get('ats_score', 0)}%")

        st.markdown("#### Recommendations")
        recommendations = results.get("ats_recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.info("No specific recommendations.")

    # Tab 3: Ranked projects
    with tab3:
        st.markdown("#### Projects Ranked by Relevance")
        ranked = results.get("ranked_projects", [])
        if ranked:
            for i, proj in enumerate(ranked, 1):
                if isinstance(proj, dict):
                    name = proj.get("name", f"Project {i}")
                    desc = proj.get("description", "")
                    tech = proj.get("technologies", "")
                    st.markdown(f"**{i}. {name}**")
                    if desc:
                        st.caption(desc)
                    if tech:
                        st.caption(f"Tech: {tech}")
                else:
                    st.markdown(f"**{i}.** {proj}")
        else:
            st.info("No projects to rank.")

    # Tab 4: Tailored resume content
    with tab4:
        st.markdown("#### Tailored Professional Summary")
        summary = results.get("tailored_summary", "")
        if summary:
            st.info(summary)
        else:
            st.info("No tailored summary available.")

        st.markdown("#### Tailored Skills Section")
        tailored_skills = results.get("tailored_skills", [])
        if tailored_skills:
            cols = st.columns(3)
            for i, skill in enumerate(tailored_skills):
                cols[i % 3].markdown(f"- {skill}")
        else:
            st.info("No tailored skills.")

        st.markdown("#### Tailored Experience Section")
        exp = results.get("tailored_experience", [])
        if exp:
            for e in exp[:5]:
                st.markdown(f"- {e}")
        else:
            st.info("No experience entries.")

        st.markdown("#### Tailored Projects Section")
        proj = results.get("tailored_projects", [])
        if proj:
            for p in proj[:5]:
                if isinstance(p, dict):
                    st.markdown(f"- {p.get('name', str(p))}")
                else:
                    st.markdown(f"- {p}")
        else:
            st.info("No project entries.")

    # Tab 5: Agent workflow log
    with tab5:
        st.markdown("#### Agent Workflow History")
        workflow = results.get("workflow_history", "")
        if workflow:
            st.text(workflow)
        else:
            st.info("No workflow history.")


def main() -> None:
    """Main Streamlit application entry point.
    
    Renders the UI, handles button clicks, runs the agent,
    and displays results.
    """
    init_session_state()

    # Page title and description
    st.title(" Resume Tailoring Agent (MCP + AI Agent)")
    st.markdown(
        "Upload your resume PDF and paste a job description. The AI agent will analyze, "
        "score, and tailor your resume using autonomous reasoning and MCP tools."
    )

    # Sidebar instructions
    with st.sidebar:
        st.markdown("### How it works")
        st.markdown(
            """
1. **Upload** your resume PDF
2. **Paste** the job description
3. **Click** 'Analyze & Tailor'
4. The agent **plans & executes** steps autonomously
5. **Review** the tailored results

**Agent Steps:**
- Read & parse resume
- Analyze job requirements
- Compare skills (gap analysis)
- Calculate ATS score
- Rank relevant projects
- Generate tailored resume
- Produce final report
            """
        )
        st.divider()
        st.caption("Powered by Qwen 3.5 via Ollama + MCP")

    # Input columns: file upload + job description
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            help="Select your resume in PDF format.",
        )

    with col2:
        job_description = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the full job description here...",
            help="Paste the job posting you want to tailor your resume for.",
        )

    # Main action button
    run_button = st.button(
        " Analyze & Tailor Resume",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.agent_running,
    )

    if run_button:
        # Validate inputs
        if not uploaded_file:
            st.warning("Please upload a resume PDF.")
            st.stop()

        if not job_description or len(job_description.strip()) < 20:
            st.warning("Please paste a valid job description (at least 20 characters).")
            st.stop()

        # Reset state for new run
        st.session_state.agent_running = True
        st.session_state.results = None
        st.session_state.status_items = {}
        st.session_state.current_step = 0

        # Save uploaded PDF to disk
        pdf_path = str(OUTPUT_DIR / "resume.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Resume saved: {uploaded_file.name}")

        # Create UI placeholders for progress tracking
        status_placeholder = st.empty()
        st.session_state.status_placeholder = status_placeholder

        progress_bar = st.progress(0.0, text="Initializing agent...")
        st.session_state.progress_bar = progress_bar

        status_container = st.status("Agent is working...", expanded=True)
        st.session_state.status_container = status_container

        with status_container:
            st.markdown("### Agent Activity")
            status_area = st.empty()
            st.session_state.status_area = status_area

        # Run the agent (synchronous wrapper around async code)
        try:
            with st.spinner("AI Agent is analyzing your resume..."):
                # Create a fresh event loop for the async MCP client
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        run_agent(pdf_path, job_description)
                    )
                finally:
                    loop.close()

            # Store results and update UI
            st.session_state.results = results
            st.session_state.status_items["finalizing"] = "complete"
            progress_bar.progress(1.0)

            status_container.update(
                label=" Agent completed all steps!",
                state="complete",
                expanded=False,
            )

            st.balloons()

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            status_container.update(
                label=" Agent encountered an error",
                state="error",
                expanded=True,
            )
            st.error(f"Agent failed: {e}")
            st.info(
                "Make sure Ollama is running with `ollama serve` and the model "
                "is pulled: `ollama pull qwen3.5:2b-q4_K_M`"
            )

        finally:
            st.session_state.agent_running = False

    # Display results if available (also shown on re-runs after completion)
    if st.session_state.results:
        st.divider()
        st.markdown("## 📋 Results")
        display_results(st.session_state.results)

        # Download button for the full report
        report_text = st.session_state.results.get("report_text", "")
        if report_text:
            st.download_button(
                label=" Download Full Report",
                data=report_text,
                file_name="resume_tailoring_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Footer
    st.divider()
    st.caption(
        "Resume Tailoring Agent | MCP Protocol | Qwen 3.5 via Ollama | Streamlit"
    )


if __name__ == "__main__":
    main()
