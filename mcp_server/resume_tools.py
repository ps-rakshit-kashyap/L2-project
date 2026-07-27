# mcp_server/resume_tools.py
# Implements the read_resume MCP tool.
# This tool accepts a file path to a PDF resume, parses it using
# PyMuPDF, extracts text, and runs skill detection.
# Returns structured resume data including summary, skills,
# projects, experience, and education sections.
#
# Registered as an MCP tool in server.py via the @mcp.tool() decorator.

from typing import Dict, Any
from utils.pdf_parser import PDFParser
from utils.skill_extractor import SkillExtractor
import logging

logger = logging.getLogger(__name__)


class ResumeTools:
    """MCP tool implementations for resume reading and parsing."""

    @staticmethod
    def read_resume(file_path: str) -> Dict[str, Any]:
        """Parse a resume PDF and extract structured information.
        
        Steps:
        1. Parse the PDF using PDFParser to extract sections.
        2. Remove raw_text from the output (not needed downstream).
        3. Run skill extraction on the full raw text.
        4. Return clean structured data for the agent.
        
        Args:
            file_path: Absolute or relative path to the PDF resume.
            
        Returns:
            Dict with keys: summary, skills, projects, experience, education.
            On error, returns the same keys with empty values plus an "error" key.
        """
        logger.info(f"Parsing resume: {file_path}")
        try:
            parsed = PDFParser.parse_resume(file_path)
            raw_text = parsed.pop("raw_text", "")
            skills = SkillExtractor.extract_skills(raw_text)

            return {
                "summary": parsed.get("summary", ""),
                "skills": skills,
                "projects": parsed.get("projects", []),
                "experience": parsed.get("experience", []),
                "education": parsed.get("education", []),
            }
        except Exception as e:
            # Graceful error handling — return empty data instead of crashing
            logger.error(f"Failed to read resume: {e}")
            return {
                "summary": "",
                "skills": [],
                "projects": [],
                "experience": [],
                "education": [],
                "error": str(e),
            }
