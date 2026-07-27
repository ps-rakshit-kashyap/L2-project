# mcp_server/server.py
# Main entry point for the MCP (Model Context Protocol) server.
# This is a FastMCP server that exposes 6 tools for resume analysis and tailoring.
#
# How it works:
# 1. Creates a FastMCP instance named "Resume Tailoring Agent".
# 2. Registers 6 tools using the @mcp.tool() decorator.
# 3. Each tool delegates to the corresponding implementation class.
# 4. When run directly, starts the MCP server on stdio transport.
#
# The server runs as a child process spawned by the agent controller.
# Communication happens via JSON-RPC over stdio (the MCP protocol).
# The agent controller uses mcp.client.stdio.stdio_client to connect.

import sys
import os
import logging

# Add project root to Python path so imports work from subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp_server.resume_tools import ResumeTools
from mcp_server.analysis_tools import AnalysisTools
from mcp_server.tailoring_tools import TailoringTools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp-server")

# Create the FastMCP server instance
mcp = FastMCP("Resume Tailoring Agent")


@mcp.tool()
def read_resume(file_path: str) -> dict:
    """Parse a resume PDF and extract structured data.
    
    Input: file_path (str) — Path to the PDF resume.
    Output: Dict with summary, skills, projects, experience, education.
    """
    logger.info(f"Tool: read_resume({file_path})")
    return ResumeTools.read_resume(file_path)


@mcp.tool()
def analyze_job_description(job_description: str) -> dict:
    """Extract skill requirements from a job description.
    
    Input: job_description (str) — Full text of the job posting.
    Output: Dict with role, required_skills, preferred_skills, keywords.
    """
    logger.info("Tool: analyze_job_description")
    return AnalysisTools.analyze_job_description(job_description)


@mcp.tool()
def skill_gap_analysis(resume_data: dict, job_data: dict) -> dict:
    """Compare resume skills against job requirements.
    
    Input: resume_data (dict), job_data (dict).
    Output: Dict with matched_skills, missing_skills, score.
    """
    logger.info("Tool: skill_gap_analysis")
    return AnalysisTools.skill_gap_analysis(resume_data, job_data)


@mcp.tool()
def ats_score(resume_data: dict, job_data: dict) -> dict:
    """Calculate ATS compatibility score.
    
    Input: resume_data (dict), job_data (dict).
    Output: Dict with score (0-100) and recommendations list.
    """
    logger.info("Tool: ats_score")
    return AnalysisTools.ats_score(resume_data, job_data)


@mcp.tool()
def rank_projects(projects: list, job_description: str) -> dict:
    """Rank resume projects by relevance to the job.
    
    Input: projects (list), job_description (str).
    Output: Dict with ranked_projects list.
    """
    logger.info("Tool: rank_projects")
    return TailoringTools.rank_projects(projects, job_description)


@mcp.tool()
def tailor_resume(resume_data: dict, job_data: dict) -> dict:
    """Generate a tailored resume for the target job.
    
    Input: resume_data (dict), job_data (dict).
    Output: Dict with professional_summary, tailored_skills,
            tailored_experience, tailored_projects.
    """
    logger.info("Tool: tailor_resume")
    return TailoringTools.tailor_resume(resume_data, job_data)


if __name__ == "__main__":
    # Start the MCP server — listens on stdio for JSON-RPC messages
    logger.info("Starting MCP server: Resume Tailoring Agent")
    mcp.run()
